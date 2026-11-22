//! Optional fight tactics (each off by default, switched on by its parameter): sidestepping incoming shots,
//! breaking off to hidden ground when outnumbered, choosing a fight position, holding a quiet home spot late in
//! the match, choosing whom to engage, and grabbing cubes mid-fight.
use super::{d2, norm2, wrap, yaw_to, Brain};
use crate::proto::{Actor, Obs};
use std::f32::consts::TAU;

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

impl Brain {
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
            o.alive <= 3
                || d < 60.0
                || hurt_s(4.0)
                || (m.charging && aimed_at(m, &pos) && d < 120.0)
                || (m.max_hp > 0.0 && m.hp < 350.0 && d < 120.0)
        } else {
            let to_me = norm2(pos[0] - m.pos[0], pos[1] - m.pos[1]);
            let closing = m.vel[0] * to_me[0] + m.vel[1] * to_me[1];
            d < 60.0 || hurt_s(6.0) || (closing > 3.0 && d < 180.0) || (hp_frac(m) < 0.4 && d < 160.0) || (aimed_at(m, &pos) && d < 190.0)
        }
    }

    /// Target score for `engage_filter` (lower is better): distance plus 0.18 per HP point, much lower for a
    /// nearly dead mech and for the one we have been hitting, a little lower for a recent attacker.
    pub(super) fn engage_score(&self, o: &Obs, m: &Actor) -> f32 {
        let d = d2(&m.pos, &o.me.pos);
        let mut s = d + 0.18 * hp_abs(m);
        if hp_frac(m) < 0.3 {
            s -= 200.0;
        }
        if self.focus == Some(m.id) && self.hit_t.get(&m.id).map_or(false, |t| o.t - t < 2.0) {
            s -= 200.0;
        }
        if self.attackers.get(&m.id).map_or(false, |t| o.t - t < 6.0) {
            s -= 40.0;
        }
        s
    }

    /// Whether a mech standing at `c` (ground height `z`) is hidden from every eye at head and mid-body height.
    fn hidden_at(&self, c: [f32; 2], z: f32, eyes: &[[f32; 3]]) -> bool {
        eyes.iter().all(|e| !self.clear(*e, [c[0], c[1], z + 4.5]) && !self.clear(*e, [c[0], c[1], z + 8.5]))
    }

    /// Nearest walkable cell, searching the walk grid outward up to `cr_steps` steps, that is hidden from
    /// `threats` (at most four), inside the zone and not right next to an enemy. `Some(here)` when we are
    /// hidden already.
    pub(super) fn retreat_cell(&self, o: &Obs, threats: &[[f32; 3]]) -> Option<[f32; 2]> {
        let pos = o.me.pos;
        let eyes: Vec<[f32; 3]> = threats.iter().take(4).map(|t| [t[0], t[1], t[2] + 8.0]).collect();
        if eyes.is_empty() {
            return None;
        }
        if self.hidden_at([pos[0], pos[1]], pos[2], &eyes) {
            return Some([pos[0], pos[1]]);
        }
        let nav = self.nav.as_ref()?;
        let start = nav.nearest_walkable(pos[0], pos[1])?;
        let z = &o.zone;
        let foes: Vec<[f32; 2]> = threats.iter().take(4).map(|t| [t[0], t[1]]).collect();
        let mut seen = vec![false; nav.nx * nav.ny];
        seen[start] = true;
        let mut frontier = vec![start];
        let mut checks = 0;
        let max_steps = self.params.cr_steps.max(1.0) as usize;
        for _ in 0..max_steps {
            let mut next = Vec::new();
            for &c in &frontier {
                for n in nav.neighbors(c) {
                    if !seen[n] {
                        seen[n] = true;
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
                // The sight table errs toward "seen": a cell it calls hidden from nobody is skipped cheaply.
                if let Some(st) = self.sight.as_ref() {
                    if foes.iter().any(|f| st.visible(q, *f) && st.cell_at(q[0], q[1]).is_some()) {
                        continue;
                    }
                }
                checks += 1;
                if checks > 600 {
                    return best.map(|b| b.0);
                }
                let qz = self.ground(q[0], q[1], pos[2]);
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
        None
    }

    /// Move toward the best of: staying put, or one of 16 points 6 m away. Score: closeness to the preferred
    /// distance, little movement, a line to the target while a weapon is ready, few other enemies within
    /// 170 m with a line to it, and distance kept from lightning and mortar holders. When staying put wins, strafe
    /// slowly across the target, reversing every 3 to 5 s.
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
        let score = |b: &Brain, q: [f32; 2], moved: f32| -> f32 {
            let qz = b.ground(q[0], q[1], pos[2]);
            let mut s = -1.1 * (d2(&q, &t.pos) - pref).abs() - 0.4 * moved;
            if ready && b.clear([q[0], q[1], qz + muzzle_h], tc) {
                s += 45.0;
            }
            for m in &others {
                if d2(&m.pos, &q) >= 170.0 {
                    continue;
                }
                let sees = match b.sight.as_ref() {
                    Some(st) if st.cell_at(q[0], q[1]).is_some() => st.visible(q, [m.pos[0], m.pos[1]]),
                    _ => b.clear([m.pos[0], m.pos[1], m.pos[2] + 8.0], [q[0], q[1], qz + 5.0]),
                };
                if sees {
                    s -= 30.0;
                }
            }
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
            s
        };
        let here = [pos[0], pos[1]];
        let mut best = (here, score(self, here, 0.0), true);
        for k in 0..16 {
            let a = k as f32 * TAU / 16.0;
            let q = [pos[0] + a.cos() * 6.0, pos[1] + a.sin() * 6.0];
            if !self.straight(&pos, &q) {
                continue;
            }
            let s = score(self, q, 6.0);
            if s > best.1 {
                best = (q, s, false);
            }
        }
        if !best.2 {
            return norm2(best.0[0] - pos[0], best.0[1] - pos[1]);
        }
        // Staying put wins: slow strafe across the line to the target.
        if o.t >= self.fp_until {
            self.fp_side = -self.fp_side;
            self.fp_until = o.t + 3.0 + 2.0 * self.rnd();
        }
        let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
        let mut side = [-to[1] * self.fp_side, to[0] * self.fp_side];
        if !self.walkable(pos[0] + side[0] * 5.0, pos[1] + side[1] * 5.0) {
            self.fp_side = -self.fp_side;
            self.fp_until = o.t + 3.0 + 2.0 * self.rnd();
            side = [-side[0], -side[1]];
            if !self.walkable(pos[0] + side[0] * 5.0, pos[1] + side[1] * 5.0) {
                return [0.0, 0.0];
            }
        }
        [side[0] * 0.5, side[1] * 0.5]
    }

    /// Late-game home: the best of about 40 points within 0.7 of the next circle's radius, scored mainly by how
    /// little ground can see it (mean clear line over 64 directions), then by distance and known enemies near it.
    /// Re-chosen every `home_s` seconds or when it falls outside the next circle.
    pub(super) fn home_spot(&mut self, o: &Obs) -> [f32; 2] {
        let pos = o.me.pos;
        let z = &o.zone;
        let keep = self.home.filter(|(h, t0)| o.t - t0 < self.params.home_s && self.in_next(o, h, 10.0));
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
            let crowd: f32 = foes.iter().map(|f| (1.0 - d2(f, &q) / 120.0).max(0.0)).sum();
            let s = expo + 0.12 * d2(&q, &pos) + 25.0 * crowd - if Some(q) == old { 5.0 } else { 0.0 };
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
            if !(0.0..=0.8).contains(&tc) {
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
        let dist = 3.6;
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
        let r = self.params.fight_cube_r;
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
