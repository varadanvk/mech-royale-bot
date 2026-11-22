//! Optional fight tactics (each off by default, switched on by its parameter): sidestepping incoming shots,
//! breaking off to hidden ground when outnumbered, choosing a fight position, holding a quiet home spot late in
//! the match, choosing whom to engage, and grabbing cubes mid-fight.
use super::{d2, norm2, wrap, yaw_to, Brain};
use crate::proto::{Actor, Obs};
use std::f32::consts::{FRAC_PI_2, TAU};

/// Weapon and step sounds worth turning toward.
const LOUD: &[&str] = &["gun", "laser", "laser_charge", "lightning", "mortar", "step", "boost"];

fn hp_frac(m: &Actor) -> f32 {
    if m.max_hp > 0.0 {
        m.hp / m.max_hp
    } else {
        1.0
    }
}

fn hp_abs(m: &Actor) -> f32 {
    if m.max_hp > 0.0 {
        m.hp
    } else {
        1400.0
    }
}

/// Whether `m`'s torso points at `at`, within the width of a mech at that distance plus a little.
fn aimed_at(m: &Actor, at: &[f32; 3]) -> bool {
    let d = d2(&m.pos, at).max(1.0);
    wrap(m.aim[0] - yaw_to(&m.pos, at)).abs() < (4.0 / d).atan() + 0.05
}

/// How long an enemy took to turn its torso to us once hit from behind (`flank`).
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TurnWatch {
    /// Since when it has been facing away from us after a hit from behind.
    since: Option<f32>,
    /// When it was last watched.
    seen: f32,
    /// Average seconds taken to turn to us.
    secs: Option<f32>,
}

/// How far round behind `facing`, as seen from `c`, the point `q` is (0 = straight ahead, PI = right behind).
pub(super) fn behind(c: &[f32], facing: f32, q: &[f32]) -> f32 {
    wrap(yaw_to(c, q) - facing).abs()
}

impl Brain {
    /// Time how long the target takes to bring its torso round to us after we hit it from behind.
    pub(super) fn watch_turn(&mut self, o: &Obs, t: &Actor) {
        let off = behind(&t.pos, t.aim[0], &o.me.pos);
        let hit = self.hit_t.get(&t.id).map_or(false, |h| o.t - h < 0.3);
        let w = self.turns.entry(t.id).or_default();
        if o.t - w.seen > 0.5 {
            w.since = None;
        }
        w.seen = o.t;
        if off > FRAC_PI_2 {
            if hit && w.since.is_none() {
                w.since = Some(o.t);
            }
        } else if off < 0.35 {
            if let Some(s) = w.since.take() {
                let d = o.t - s;
                w.secs = Some(w.secs.map_or(d, |v| 0.5 * v + 0.5 * d));
            }
        }
    }

    /// The target's facing when we should work round to its back: practice matches only, the target facing
    /// more than 90 degrees away from us, and not known to turn quickly (`flank`).
    pub(super) fn flank_facing(&self, o: &Obs, t: &Actor) -> Option<f32> {
        let p = &self.params;
        if p.flank <= 0.0 || self.ffa_on() || t.kind != "mech" || behind(&t.pos, t.aim[0], &o.me.pos) <= FRAC_PI_2 {
            return None;
        }
        let w = self.turns.get(&t.id);
        let slow = w.and_then(|w| w.secs).map_or(true, |s| s >= p.flank_slow) || w.and_then(|w| w.since).map_or(false, |s| o.t - s >= p.flank_slow);
        slow.then_some(t.aim[0])
    }

    /// The sideways sign (step `[-to[1] * s, to[0] * s]`, `to` toward the target) that takes us further round
    /// behind a target with this facing.
    pub(super) fn flank_sign(&self, o: &Obs, t: &Actor, facing: f32) -> f32 {
        let pos = o.me.pos;
        let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
        let step = |s: f32| [pos[0] - to[1] * s * 5.0, pos[1] + to[0] * s * 5.0];
        if behind(&t.pos, facing, &step(1.0)) >= behind(&t.pos, facing, &step(-1.0)) {
            1.0
        } else {
            -1.0
        }
    }

    /// Point the strafe round toward the target's back for this step (`flank`).
    pub(super) fn flank_strafe(&mut self, o: &Obs, t: &Actor) {
        if let Some(f) = self.flank_facing(o, t) {
            self.strafe = self.flank_sign(o, t, f);
            self.strafe_until = o.t + self.params.strafe_min;
        }
    }

    /// Remember whom we hit, where hits came from, and where the last loud sound came from.
    pub(super) fn track_contacts(&mut self, o: &Obs) {
        for e in &o.events {
            if e.kind == "hit" {
                if let Some(t) = e.target.filter(|t| *t < 100) {
                    self.hit_t.insert(t, o.t);
                }
            }
            if e.kind == "damaged" && e.by.map_or(false, |b| b < 100) {
                if let Some(b) = e.bearing {
                    self.hit_bearing = Some((b, o.t));
                }
            }
        }
        self.hit_t.retain(|_, t| o.t - *t < 10.0);
        if let Some(sd) = o.sounds.iter().filter(|s| LOUD.contains(&s.kind.as_str())).min_by(|x, y| x.age.partial_cmp(&y.age).unwrap_or(std::cmp::Ordering::Equal)) {
            self.sound_bearing = Some((sd.bearing, o.t - sd.age));
        }
    }

    /// Enemies worth engaging. Late (`late`): within 60 m, hit us in the last 4 s, charging a laser at us within
    /// 120 m, under 350 HP within 120 m, or anyone once at most two opponents are left. Before that: within
    /// 60 m, hit us in the last 6 s, closing faster than 3 m/s within 180 m, under 40% within 160 m, or aimed
    /// at us within 190 m.
    pub(super) fn engage_ok(&self, o: &Obs, m: &Actor, late: bool) -> bool {
        let pos = o.me.pos;
        let d = d2(&m.pos, &pos);
        let hurt_s = |s: f32| self.attackers.get(&m.id).map_or(false, |t| o.t - t < s);
        if late {
            o.alive <= if self.params.ext_late > 0.0 { 2 } else { 3 }
                || d < 60.0
                || hurt_s(4.0)
                || (m.charging && aimed_at(m, &pos) && d < 120.0)
                || (m.max_hp > 0.0 && m.hp < 350.0 && d < 120.0)
        } else {
            let to_me = norm2(pos[0] - m.pos[0], pos[1] - m.pos[1]);
            let closing = m.vel[0] * to_me[0] + m.vel[1] * to_me[1];
            (self.params.ext_target > 0.0 && o.alive <= 3) || d < 60.0 || hurt_s(6.0) || (closing > 3.0 && d < 180.0) || (hp_frac(m) < 0.4 && d < 160.0) || (aimed_at(m, &pos) && d < 190.0)
        }
    }

    /// Target score for `engage_filter` (lower is better): distance plus 0.18 per HP point, much lower for a
    /// nearly dead mech and for the current target (held 2 s at a time, or while we keep hitting it), a little
    /// lower for a recent attacker and for a mech closing on us.
    pub(super) fn engage_score(&self, o: &Obs, m: &Actor) -> f32 {
        let d = d2(&m.pos, &o.me.pos);
        let mut s = d + 0.18 * hp_abs(m);
        if hp_frac(m) < 0.3 {
            s -= 200.0;
        }
        let ext = self.params.ext_target > 0.0;
        let held = ext && o.t - self.focus_t < 2.0;
        if self.focus == Some(m.id) && (held || self.hit_t.get(&m.id).map_or(false, |t| o.t - t < 2.0)) {
            s -= 200.0;
        }
        if self.attackers.get(&m.id).map_or(false, |t| o.t - t < if ext { 4.0 } else { 6.0 }) {
            s -= 40.0;
        }
        if ext {
            let to_me = norm2(o.me.pos[0] - m.pos[0], o.me.pos[1] - m.pos[1]);
            if m.vel[0] * to_me[0] + m.vel[1] * to_me[1] > 1.0 {
                s -= 15.0;
            }
        }
        if self.params.focus_laser > 0.0 && m.weapons.iter().any(|w| w == "laser") {
            s -= self.params.focus_laser;
            if m.charging && aimed_at(m, &o.me.pos) {
                s -= self.params.focus_laser;
            }
        }
        s + self.reinforcements(m)
    }

    /// Whether a mech standing at `c` (ground height `z`) is hidden from every eye at head and mid-body height.
    fn hidden_at(&self, c: [f32; 2], z: f32, eyes: &[[f32; 3]]) -> bool {
        eyes.iter().all(|e| !self.clear(*e, [c[0], c[1], z + 4.5]) && !self.clear(*e, [c[0], c[1], z + 8.5]))
    }

    /// Every enemy we know of for the break-off: seen or remembered (`memory_s`, moved on by its last speed for
    /// up to 3 s), the last threat, and believed enemies we are fairly sure of; nearest first.
    pub(super) fn cb_foes(&self, o: &Obs, threats: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let me = o.me.pos;
        let mut v: Vec<[f32; 3]> = self
            .known
            .values()
            .filter(|k| o.t - k.seen_t < self.params.memory_s && d2(&k.pos, &me) < 320.0)
            .map(|k| {
                let dt = (o.t - k.seen_t).min(3.0);
                [k.pos[0] + k.vel[0] * dt, k.pos[1] + k.vel[1] * dt, k.pos[2]]
            })
            .collect();
        for t in threats {
            if !v.iter().any(|q| d2(q, t) < 20.0) {
                v.push(*t);
            }
        }
        for b in self.beliefs.iter().filter(|b| b.conf >= 0.5 && b.r <= 40.0) {
            if d2(&b.pos, &me) < 320.0 && !v.iter().any(|q| d2(q, &b.pos) < 20.0) {
                v.push([b.pos[0], b.pos[1], self.ground(b.pos[0], b.pos[1], me[2])]);
            }
        }
        v.sort_by(|a, b| d2(a, &me).partial_cmp(&d2(b, &me)).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Whether going straight from `from` to `to` keeps clear of every enemy in `foes`: 25 m, or as far as we
    /// already are from one closer than that.
    pub(super) fn cb_line_clear(from: [f32; 2], to: [f32; 2], foes: &[[f32; 2]]) -> bool {
        let d = [to[0] - from[0], to[1] - from[1]];
        let l2 = d[0] * d[0] + d[1] * d[1];
        foes.iter().all(|f| {
            let now = d2(f, &from);
            let s = if l2 > 1e-6 { (((f[0] - from[0]) * d[0] + (f[1] - from[1]) * d[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let c = [from[0] + d[0] * s, from[1] + d[1] * s];
            d2(f, &c) >= now.min(25.0) - 0.5
        })
    }

    /// Whether a goal at `g` lies away from the enemies in `foes`: farther than we are from their centre (those
    /// within 200 m) and from the nearest, and not within 60 degrees of one within 80 m. Returns the extra cost
    /// of heading that way (directions toward enemies within 150 m cost more), or `None` when it is not away.
    pub(super) fn cb_away_cost(here: [f32; 2], g: [f32; 2], foes: &[[f32; 2]]) -> Option<f32> {
        if foes.is_empty() {
            return Some(0.0);
        }
        let near: Vec<&[f32; 2]> = foes.iter().filter(|f| d2(*f, &here) < 200.0).collect();
        let pool: Vec<&[f32; 2]> = if near.is_empty() { foes.iter().collect() } else { near };
        let n = pool.len() as f32;
        let cen = [pool.iter().map(|f| f[0]).sum::<f32>() / n, pool.iter().map(|f| f[1]).sum::<f32>() / n];
        let nearest = foes.iter().fold((f32::MAX, here), |b, f| {
            let d = d2(f, &here);
            if d < b.0 {
                (d, *f)
            } else {
                b
            }
        });
        if d2(&g, &cen) <= d2(&here, &cen) || d2(&g, &nearest.1) <= nearest.0 {
            return None;
        }
        let dir = norm2(g[0] - here[0], g[1] - here[1]);
        let mut cost = 0.0;
        for f in foes {
            let d = d2(f, &here);
            if d > 150.0 {
                continue;
            }
            let to = norm2(f[0] - here[0], f[1] - here[1]);
            let c = dir[0] * to[0] + dir[1] * to[1];
            if d < 80.0 && c > 0.5 {
                return None;
            }
            cost += 12.0 * c.max(0.0) * (1.0 - d / 150.0);
        }
        Some(cost)
    }

    /// Nearest walkable cell, searching the walk grid outward up to `cr_steps` steps, that is hidden from
    /// `threats` (at most four; six with `cb_known`), inside the zone and not right next to an enemy. `Some(here)`
    /// when we are hidden already. With `cb_away` / `cb_path`, only cells away from the enemies in `all` and
    /// reached without passing near one of them count.
    pub(super) fn retreat_cell(&self, o: &Obs, threats: &[[f32; 3]], all: &[[f32; 2]]) -> Option<[f32; 2]> {
        let pos = o.me.pos;
        let here = [pos[0], pos[1]];
        let n_eyes = if self.params.cb_known > 0.0 { 6 } else { 4 };
        let away_on = self.params.cb_away > 0.0;
        let path_on = self.params.cb_path > 0.0;
        let eyes: Vec<[f32; 3]> = threats.iter().take(n_eyes).map(|t| [t[0], t[1], t[2] + 8.0]).collect();
        if eyes.is_empty() {
            return None;
        }
        if self.hidden_at([pos[0], pos[1]], pos[2], &eyes) {
            return Some([pos[0], pos[1]]);
        }
        let nav = self.nav.as_ref()?;
        let start = nav.nearest_walkable(pos[0], pos[1])?;
        let z = &o.zone;
        let foes: Vec<[f32; 2]> = threats.iter().take(n_eyes).map(|t| [t[0], t[1]]).collect();
        // The search does not pass through ground near an enemy (closer than 25 m, or than we already are).
        let blocked = |q: [f32; 2]| path_on && all.iter().any(|f| d2(f, &q) < d2(f, &here).min(25.0) - 0.5);
        let mut seen = vec![false; nav.nx * nav.ny];
        seen[start] = true;
        let mut frontier = vec![start];
        let mut checks = 0;
        let max_steps = self.params.cr_steps.max(1.0) as usize;
        // With `ext_lose`: cost = 2 per step + a zone penalty - 0.25 per metre from the nearest enemy (up to
        // 80 m), searched until no further cell could beat the best found, over at most 350 cells.
        let weighed = self.params.ext_lose > 0.0;
        let mut best_cost: Option<([f32; 2], f32)> = None;
        for step in 0..max_steps {
            if weighed {
                if let Some(b) = best_cost {
                    if 2.0 * (step + 1) as f32 - 20.0 >= b.1 {
                        return Some(b.0);
                    }
                }
            }
            let mut next = Vec::new();
            for &c in &frontier {
                for n in nav.neighbors(c) {
                    if !seen[n] {
                        seen[n] = true;
                        if blocked(nav.center(n)) {
                            continue;
                        }
                        next.push(n);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            let mut best: Option<([f32; 2], f32)> = None;
            for &c in &next {
                let q = nav.center(c);
                if d2(&q, &z.center) > z.radius - 8.0 {
                    continue;
                }
                let near = foes.iter().map(|f| d2(f, &q)).fold(f32::MAX, f32::min);
                if near < 25.0 {
                    continue;
                }
                let mut extra = 0.0;
                if away_on {
                    match Self::cb_away_cost(here, q, all) {
                        Some(c) => extra = c,
                        None => continue,
                    }
                }
                if path_on && !Self::cb_line_clear(here, q, all) {
                    continue;
                }
                // The sight table errs toward "seen": a cell it calls hidden from nobody is skipped cheaply.
                if let Some(st) = self.sight.as_ref() {
                    if foes.iter().any(|f| st.visible(q, *f) && st.cell_at(q[0], q[1]).is_some()) {
                        continue;
                    }
                }
                checks += 1;
                if checks > if weighed { 350 } else { 600 } {
                    return if weighed { best_cost.map(|b| b.0) } else { best.map(|b| b.0) };
                }
                let qz = self.ground(q[0], q[1], pos[2]);
                if weighed {
                    if self.hidden_at(q, qz, &eyes) {
                        let out_next = (d2(&q, &z.next_center) - z.next_radius).max(0.0);
                        let cost = 2.0 * (step + 1) as f32 + 0.5 * out_next.min(40.0) - 0.25 * near.min(80.0) + extra;
                        if best_cost.map_or(true, |b| cost < b.1) {
                            best_cost = Some((q, cost));
                        }
                    }
                    continue;
                }
                if self.hidden_at(q, qz, &eyes) {
                    // Among cells at the same step count, prefer the one furthest from the enemies.
                    if best.map_or(true, |b| near > b.1) {
                        best = Some((q, near));
                    }
                }
            }
            if let Some(b) = best {
                return Some(b.0);
            }
            frontier = next;
        }
        best_cost.map(|b| b.0)
    }

    /// Move toward the best of: staying put, or one of 16 points 6 m away. Score: closeness to the preferred
    /// distance, little movement, a line to the target while a weapon is ready, few other enemies within
    /// 170 m with a line to it (with `map_expose`: also believed enemies out of sight), and distance kept from
    /// lightning and mortar holders. When staying put wins, strafe
    /// slowly across the target, reversing every 3 to 5 s (with `ext_fpos`: every `fp_flip_lo`..`fp_flip_hi` s).
    pub(super) fn fight_pos_move(&mut self, o: &Obs, t: &Actor, pref: f32, ready: bool) -> [f32; 2] {
        let pos = o.me.pos;
        let p = &self.params;
        let muzzle_h = o.me.stat("muzzle_height", 7.0);
        let tc = [t.pos[0], t.pos[1], t.pos[2] + 4.5];
        let others: Vec<&Actor> = o
            .mechs
            .iter()
            .filter(|m| m.kind == "mech" && m.id != t.id && !(m.max_hp > 0.0 && m.hp <= 0.0) && d2(&m.pos, &pos) < 176.0)
            .collect();
        // Other enemies whose line to a point counts against it: those in sight, and with `ext_fpos` also those
        // seen in the last 4 s.
        let mut watchers: Vec<[f32; 3]> = others.iter().map(|m| m.pos).collect();
        // Enemies counted above, left out of the believed ones (`map_expose`).
        let mut counted: Vec<u32> = others.iter().map(|m| m.id).collect();
        if p.ext_fpos > 0.0 {
            for (id, k) in self.known.iter() {
                if *id != t.id && o.t - k.seen_t < 4.0 && o.t - k.seen_t > 0.05 && d2(&k.pos, &pos) < 176.0 && !others.iter().any(|m| m.id == *id) {
                    watchers.push(k.pos);
                    counted.push(*id);
                }
            }
        }
        let keep_of = |m: &Actor| -> f32 {
            let mut k = 0.0f32;
            if m.weapons.iter().any(|w| w == "lightning") {
                k = k.max(p.keep_ltg);
            }
            if m.weapons.iter().any(|w| w == "mortar") {
                k = k.max(p.keep_mortar);
            }
            k
        };
        let z = &o.zone;
        // With two or more enemies about (`pincer`): keep them all in front, a wall behind, out of the open.
        let pin = self.pin_on();
        // A target facing away and slow to turn: points further round behind it score higher (`flank`).
        let flank = self.flank_facing(o, t);
        let ex_here = if pin { self.exposure_at([pos[0], pos[1]]) } else { None };
        let score = |b: &Brain, q: [f32; 2], moved: f32| -> f32 {
            let qz = b.ground(q[0], q[1], pos[2]);
            let mut s = -1.1 * (d2(&q, &t.pos) - pref).abs() - 0.4 * moved;
            if ready && b.clear([q[0], q[1], qz + muzzle_h], tc) {
                s += 45.0;
            }
            for m in &watchers {
                if d2(m, &q) >= 170.0 {
                    continue;
                }
                let sees = match b.sight.as_ref() {
                    Some(st) if st.cell_at(q[0], q[1]).is_some() => st.visible(q, [m[0], m[1]]),
                    _ => b.clear([m[0], m[1], m[2] + 8.0], [q[0], q[1], qz + 5.0]),
                };
                if sees {
                    s -= 30.0;
                }
            }
            s -= b.hidden_watch_cost(q, qz, t.id, &counted);
            for m in others.iter().copied().chain(std::iter::once(t)) {
                let k = keep_of(m);
                let d = d2(&m.pos, &q);
                if k > 0.0 && d < k {
                    s -= 3.0 * (k - d);
                }
            }
            let out = d2(&q, &z.center) - (z.radius - 10.0);
            if out > 0.0 {
                s -= out * 5.0;
            }
            if let Some(f) = flank {
                s += 25.0 * p.flank * behind(&t.pos, f, &q);
            }
            if pin {
                let ps = b.pin_score(o, [pos[0], pos[1]], q, t.pos, ex_here);
                s += if p.pin_cap > 0.0 { ps.clamp(-p.pin_cap, p.pin_cap) } else { ps };
            }
            s
        };
        let here = [pos[0], pos[1]];
        let mut best = (here, score(self, here, 0.0), true);
        let rings = self.pin_rings();
        for &r in &rings {
            for k in 0..16 {
                let a = k as f32 * TAU / 16.0;
                let q = [pos[0] + a.cos() * r, pos[1] + a.sin() * r];
                if !self.straight(&pos, &q) {
                    continue;
                }
                let s = score(self, q, r);
                if s > best.1 {
                    best = (q, s, false);
                }
            }
        }
        // A far position chosen while pincered is kept for a moment unless another is clearly better.
        if pin {
            self.pin_goal = self.pin_goal.filter(|g| o.t < g.1 && d2(&g.0, &pos) > 2.5 && self.straight(&pos, &g.0));
            if let Some((g, _)) = self.pin_goal {
                let s = score(self, g, d2(&g, &pos)) + 10.0;
                if s > best.1 {
                    best = (g, s, false);
                }
            }
            if !best.2 && d2(&best.0, &pos) > 9.0 && self.pin_goal.map_or(true, |g| d2(&g.0, &best.0) > 1.0) {
                self.pin_goal = Some((best.0, o.t + 1.5));
            }
        } else {
            self.pin_goal = None;
        }
        if pin && std::env::var_os("TPIN").is_some() {
            eprintln!(
                "PIN fight t {:.1} best {:.0},{:.0} score {:.1} (pin {:.1}) stay {:.1} (pin {:.1}) pref {:.0}",
                o.t,
                best.0[0],
                best.0[1],
                best.1,
                self.pin_score(o, here, best.0, t.pos, ex_here),
                score(self, here, 0.0),
                self.pin_score(o, here, here, t.pos, ex_here),
                pref
            );
        }
        if !best.2 {
            return norm2(best.0[0] - pos[0], best.0[1] - pos[1]);
        }
        // Staying put wins: slow strafe across the line to the target.
        let (lo, span) = if self.params.ext_fpos > 0.0 { (self.params.fp_flip_lo, (self.params.fp_flip_hi - self.params.fp_flip_lo).max(0.0)) } else { (3.0, 2.0) };
        if let Some(f) = flank {
            self.fp_side = self.flank_sign(o, t, f);
            self.fp_until = o.t + lo;
        } else if o.t >= self.fp_until {
            self.fp_side = -self.fp_side;
            self.fp_until = o.t + lo + span * self.rnd();
        }
        let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
        let mut side = [-to[1] * self.fp_side, to[0] * self.fp_side];
        if !self.walkable(pos[0] + side[0] * 5.0, pos[1] + side[1] * 5.0) {
            self.fp_side = -self.fp_side;
            self.fp_until = o.t + lo + span * self.rnd();
            side = [-side[0], -side[1]];
            if !self.walkable(pos[0] + side[0] * 5.0, pos[1] + side[1] * 5.0) {
                return [0.0, 0.0];
            }
        }
        let sp = self.params.fp_strafe;
        [side[0] * sp, side[1] * sp]
    }

    /// Late-game home: the best of about 40 points within 0.7 of the next circle's radius, scored mainly by how
    /// little ground can see it (mean clear line over 64 directions), then by distance and known enemies near it.
    /// Re-chosen every `home_s` seconds or when it falls outside the next circle; with `ext_late` also after a
    /// fresh hit or loud sound, preferring points away from its direction.
    pub(super) fn home_spot(&mut self, o: &Obs) -> [f32; 2] {
        let pos = o.me.pos;
        let z = &o.zone;
        let ext = self.params.ext_late > 0.0;
        // A fresh hit or loud sound since the spot was chosen: choose again, away from it.
        let alert = if ext { self.alert_bearing(o) } else { None };
        let alert_new = ext
            && alert.is_some()
            && self.home.map_or(false, |(_, t0)| {
                let ht = self.hit_bearing.map_or(-1.0, |h| h.1);
                let st = self.sound_bearing.map_or(-1.0, |h| h.1);
                ht.max(st) > t0
            });
        let keep = self.home.filter(|(h, t0)| o.t - t0 < self.params.home_s && self.in_next(o, h, 10.0) && !alert_new);
        if let Some((h, _)) = keep {
            return h;
        }
        let foes: Vec<[f32; 3]> = self.known.values().filter(|k| o.t - k.seen_t < 8.0).map(|k| k.pos).collect();
        let r = (z.next_radius * 0.7).max(5.0);
        let old = self.home.map(|h| h.0);
        let mut cands: Vec<[f32; 2]> = Vec::with_capacity(42);
        if let Some(h) = old {
            cands.push(h);
        }
        cands.push([pos[0], pos[1]]);
        for _ in 0..40 {
            let a = self.rnd() * TAU;
            let rr = r * self.rnd().sqrt();
            cands.push([z.next_center[0] + a.cos() * rr, z.next_center[1] + a.sin() * rr]);
        }
        let mut best: Option<([f32; 2], f32)> = None;
        for q in cands {
            if !self.walkable(q[0], q[1]) || !self.in_next(o, &q, 10.0) {
                continue;
            }
            let expo = match self.sight.as_ref() {
                Some(st) => match st.cell_at(q[0], q[1]) {
                    Some(c) if st.reach[c] => st.mean_range(c),
                    _ => continue,
                },
                None => 100.0,
            };
            let cr = if ext { self.params.late_crowd_r } else { 120.0 };
            let crowd: f32 = foes.iter().map(|f| (1.0 - d2(f, &q) / cr).max(0.0)).sum();
            let mut s = expo + 0.12 * d2(&q, &pos) + 25.0 * crowd - if Some(q) == old { 5.0 } else { 0.0 };
            if let Some(b) = alert {
                let dq = d2(&q, &pos);
                if dq > 3.0 {
                    let toward = ((q[0] - pos[0]) * b.cos() + (q[1] - pos[1]) * b.sin()) / dq;
                    s += 30.0 * toward.max(0.0);
                }
            }
            if best.map_or(true, |b| s < b.1) {
                best = Some((q, s));
            }
        }
        let h = best.map_or([pos[0], pos[1]], |b| b.0);
        self.home = Some((h, o.t));
        h
    }

    /// Bearing to turn toward with nothing to shoot late in the match: the last hit taken, else the last sound.
    pub(super) fn alert_bearing(&self, o: &Obs) -> Option<f32> {
        match (self.hit_bearing, self.sound_bearing) {
            (Some((b, t)), _) if o.t - t < 4.0 => Some(b),
            (_, Some((b, t))) if o.t - t < 4.0 => Some(b),
            _ => None,
        }
    }

    /// Start a sidestep if a visible bullet is about to hit us, or (with no bullet in view) an enemy in sight is
    /// aimed at us and a gunshot came from its direction. Returns whether a sidestep is in progress.
    pub(super) fn dodge_check(&mut self, o: &Obs, target: Option<&Actor>, mv: [f32; 2], threats: &[[f32; 3]]) -> bool {
        let me = &o.me;
        let pos = me.pos;
        if o.t < self.dodge_until {
            return true;
        }
        if me.energy < self.params.dodge_energy || !me.grounded {
            return false;
        }
        let rad = 2.5 * (1.0 + 0.05 * me.cubes) + 0.8;
        let height = 9.0 * (1.0 + 0.05 * me.cubes);
        // Incoming direction (horizontal unit vector of travel) of the most urgent threat.
        let mut incoming: Option<[f32; 2]> = None;
        let mut soonest = f32::MAX;
        for pr in o.projectiles.iter().filter(|p| p.kind == "gun" || p.kind == "npc") {
            let r = [pr.pos[0] - pos[0], pr.pos[1] - pos[1]];
            let v = [pr.vel[0] - me.vel[0], pr.vel[1] - me.vel[1]];
            let vv = v[0] * v[0] + v[1] * v[1];
            if vv < 1.0 {
                continue;
            }
            let tc = -(r[0] * v[0] + r[1] * v[1]) / vv;
            if !(0.0..=self.params.dodge_look).contains(&tc) {
                continue;
            }
            let miss = ((r[0] + v[0] * tc).powi(2) + (r[1] + v[1] * tc).powi(2)).sqrt();
            let zc = pr.pos[2] + pr.vel[2] * tc;
            if miss < rad && zc > pos[2] - 0.5 && zc < pos[2] + height + 0.5 && tc < soonest {
                soonest = tc;
                incoming = Some(norm2(pr.vel[0], pr.vel[1]));
            }
        }
        if incoming.is_none() && o.projectiles.is_empty() {
            // No bullets listed: an enemy in sight with a line to us, aimed at us, and a gunshot from its side.
            let chest = [pos[0], pos[1], pos[2] + 5.0];
            for m in o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0) && m.weapons.iter().any(|w| w == "gun")) {
                let b = yaw_to(&pos, &m.pos);
                let shot = o.sounds.iter().any(|s| s.kind == "gun" && s.age <= 0.15 && wrap(s.bearing - b).abs() < 0.3);
                if shot && aimed_at(m, &pos) && self.clear([m.pos[0], m.pos[1], m.pos[2] + 7.0], chest) {
                    incoming = Some(norm2(pos[0] - m.pos[0], pos[1] - m.pos[1]));
                    break;
                }
            }
        }
        let Some(u) = incoming else { return false };
        let z = &o.zone;
        let dist = self.params.dodge_reach;
        let vel = [me.vel[0], me.vel[1]];
        let tgt = target.map(|t| [t.pos[0], t.pos[1], t.pos[2] + 4.5]);
        let muzzle_h = me.stat("muzzle_height", 7.0);
        let mut best: Option<([f32; 2], f32)> = None;
        for side in [1.0f32, -1.0] {
            let d = [-u[1] * side, u[0] * side];
            let q = [pos[0] + d[0] * dist, pos[1] + d[1] * dist];
            if !self.straight(&pos, &q) {
                continue;
            }
            let qz = self.ground(q[0], q[1], pos[2]);
            // Momentum matters most: reversing takes longer than the sidestep lasts.
            let mut s = (d[0] * vel[0] + d[1] * vel[1]) * 0.3 + (d[0] * mv[0] + d[1] * mv[1]) * 1.0;
            let seen = threats.iter().take(4).filter(|t| self.clear([t[0], t[1], t[2] + 8.0], [q[0], q[1], qz + 5.0])).count();
            s -= seen as f32 * 1.0;
            if let Some(tc) = tgt {
                if self.clear([q[0], q[1], qz + muzzle_h], tc) {
                    s += 0.8;
                }
            }
            if d2(&q, &z.center) > z.radius - 3.0 {
                s -= 5.0;
            }
            if best.map_or(true, |b| s > b.1) {
                best = Some((d, s));
            }
        }
        let Some((d, _)) = best else { return false };
        self.dodge_dir = d;
        self.dodge_until = o.t + self.params.dodge_hold;
        true
    }

    /// A cube within `fight_cube_r` worth grabbing mid-fight: on the ground near our height, no known enemy
    /// within 25 m of it, the target at least 25 m from it, and no enemy other than the target with a line to the
    /// cube or to the halfway point.
    pub(super) fn fight_cube(&self, o: &Obs, t: &Actor) -> Option<[f32; 3]> {
        let pos = o.me.pos;
        let low = self.params.ext_finish > 0.0 && o.me.max_hp > 0.0 && o.me.hp < 0.5 * o.me.max_hp;
        let r = if low { self.params.cube_r_low.min(self.params.fight_cube_r) } else { self.params.fight_cube_r };
        let others: Vec<[f32; 3]> =
            self.known.iter().filter(|(id, k)| **id != t.id && o.t - k.seen_t < 4.0).map(|(_, k)| [k.pos[0], k.pos[1], k.pos[2] + 8.0]).collect();
        let mut cubes: Vec<[f32; 3]> = self
            .cubes_mem
            .values()
            .filter(|q| d2(*q, &pos) < r && (q[2] - self.ground(q[0], q[1], q[2])).abs() < 3.0 && (q[2] - pos[2]).abs() < 6.0)
            .copied()
            .collect();
        cubes.sort_by(|x, y| d2(x, &pos).partial_cmp(&d2(y, &pos)).unwrap_or(std::cmp::Ordering::Equal));
        cubes.into_iter().find(|q| {
            if d2(q, &t.pos) < 25.0 || others.iter().any(|e| d2(e, q) < 25.0) {
                return false;
            }
            let mid = [(q[0] + pos[0]) * 0.5, (q[1] + pos[1]) * 0.5];
            let pts = [[q[0], q[1]], mid];
            !pts.iter().any(|c| {
                let cz = self.ground(c[0], c[1], pos[2]);
                others.iter().any(|e| self.clear(*e, [c[0], c[1], cz + 5.0]))
            }) && self.straight(&pos, q)
        })
    }
}

/// Confidence from which a believed enemy counts in a practice match.
const SURE: f32 = 0.4;

/// Distance from `q` to the segment `a`..`b`.
fn seg_dist(q: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let l2 = ab[0] * ab[0] + ab[1] * ab[1];
    let f = if l2 > 0.0 { (((q[0] - a[0]) * ab[0] + (q[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
    (q[0] - a[0] - ab[0] * f).hypot(q[1] - a[1] - ab[1] * f)
}

/// The enemy picture in a practice match (see `Params::map_preaim`).
impl Brain {
    /// The enemy picture is kept and no free-for-all is settled.
    fn map_practice(&self) -> bool {
        self.params.ffa_detect > 0.0 && !self.ffa_on()
    }

    /// With no enemy in view, where to face: the surest believed enemy out of sight within 150 m (nearer first
    /// on a tie), with `map_preaim`.
    pub(super) fn preaim(&self, pos: &[f32; 3]) -> Option<[f32; 2]> {
        if self.params.map_preaim <= 0.0 || !self.map_practice() || self.beliefs.iter().any(|b| b.listed && d2(&b.pos, pos) < 200.0) {
            return None;
        }
        self.beliefs
            .iter()
            .filter(|b| !b.listed && b.conf >= SURE && d2(&b.pos, pos) <= 150.0)
            .max_by(|x, y| (x.conf - d2(&x.pos, pos) * 1e-4).partial_cmp(&(y.conf - d2(&y.pos, pos) * 1e-4)).unwrap_or(std::cmp::Ordering::Equal))
            .map(|b| b.pos)
    }

    /// What a fight position at `q` (ground height `qz`) loses to believed enemies out of sight with a line to it
    /// (with `map_expose`): those sure enough, within 170 m of it, other than the target and those already
    /// counted (`skip`).
    pub(super) fn hidden_watch_cost(&self, q: [f32; 2], qz: f32, target: u32, skip: &[u32]) -> f32 {
        let w = self.params.map_expose;
        if w <= 0.0 || !self.map_practice() {
            return 0.0;
        }
        let mut s = 0.0;
        for b in self.beliefs.iter().filter(|b| !b.listed && b.conf >= SURE && b.id.map_or(true, |id| id != target && !skip.contains(&id))) {
            if d2(&b.pos, &q) >= 170.0 {
                continue;
            }
            let sees = match self.sight.as_ref() {
                Some(st) if st.cell_at(q[0], q[1]).is_some() => st.visible(q, b.pos),
                _ => {
                    let bz = self.ground(b.pos[0], b.pos[1], qz);
                    self.clear([b.pos[0], b.pos[1], bz + 8.0], [q[0], q[1], qz + 5.0])
                }
            };
            if sees {
                s += w;
            }
        }
        s
    }

    /// What walking from `from` to `to` costs for the sure believed enemies within 60 m of the way, once two or
    /// more are (with `map_crowd`). Never counted: the current target, and an enemy within 60 m of us that hurt
    /// us in the last 3 s.
    pub(super) fn crowd_cost(&self, now: f32, from: &[f32; 3], to: [f32; 2]) -> f32 {
        let w = self.params.map_crowd;
        if w <= 0.0 || !self.map_practice() {
            return 0.0;
        }
        let here = [from[0], from[1]];
        let n = self
            .beliefs
            .iter()
            .filter(|b| b.conf >= SURE && seg_dist(b.pos, here, to) < 60.0)
            .filter(|b| match b.id {
                Some(id) => self.focus != Some(id) && !(d2(&b.pos, from) < 60.0 && self.attackers.get(&id).map_or(false, |t| now - t < 3.0)),
                None => true,
            })
            .count();
        if n >= 2 {
            w * n as f32
        } else {
            0.0
        }
    }

    /// How much worse a candidate target scores for the other sure believed enemies within 80 m of it (at most
    /// three), with `map_reinf`.
    pub(super) fn reinforcements(&self, m: &Actor) -> f32 {
        let w = self.params.map_reinf;
        if w <= 0.0 || !self.map_practice() {
            return 0.0;
        }
        let n = self.beliefs.iter().filter(|b| b.conf >= SURE && b.id != Some(m.id) && d2(&b.pos, &m.pos) < 80.0).count();
        w * n.min(3) as f32
    }
}
