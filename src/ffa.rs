//! Playing a free-for-all. Runs alongside the usual decision logic: the enemy picture and the match-type
//! detector are fed every observation, but nothing here changes a decision until the detector has settled on a
//! free-for-all (and `ffa_play` is on).
//!
//! Modes, chosen every decision from what we believe about each enemy:
//! - `Fight`: a fight we cannot walk away from (an enemy coming for us within 40 m with no one else about, or a
//!   lone one coming for us with under half our hit points): the usual fighting takes over.
//! - `Avoid`: an enemy coming for us within `ffa_avoid_r` (150 m), or two or more believed enemies within 120 m:
//!   break line of sight, away from them, toward the place the fewest of them can see; boost only while seen,
//!   walk quietly once out of sight; shoot back only at whoever is hurting us; no new laser charge.
//! - `Third`: two other mechs fighting each other within `ffa_third_r` (220 m) of us, neither coming for us, the
//!   weaker with at most 700 hit points or 0.85 of ours (and clearly the weaker of the two above 700), and we at
//!   half health or more: go where the weaker can be seen
//!   and the stronger cannot, and shoot the weaker. Ends with its death or when either turns on us.
//! - `Late`: the next circle at most `ffa_late_r` (140 m) across or four or fewer mechs left: hold a quiet,
//!   little-seen spot inside the next circle; fight only sure wins (a target under 350 hit points or a quarter
//!   of its health with less than 0.4 of ours while we are above 0.6) or what comes within 40 m.
//! - `Hide`: otherwise, sit at a sheltered spot few believed enemies can see and far from them, healing; take
//!   cubes and weapons only when no believed enemy is within 100 m of them. A spot outside the next circle is
//!   allowed only while the zone hurts 35/s or less and there is time to walk back in before the shrink starts.
use super::{d2, norm2, wrap, yaw_to, Brain};
use crate::contacts::{Belief, W_GUN, W_LASER, W_LIGHTNING, W_MORTAR};
use crate::proto::{Actor, Obs};
use crate::sight::SECTORS;
use std::f32::consts::{PI, TAU};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FfaMode {
    /// Not (yet) known to be a free-for-all: the usual logic, untouched.
    #[default]
    Practice,
    Fight,
    Third,
    Avoid,
    Hide,
    Late,
}

impl FfaMode {
    pub fn name(self) -> &'static str {
        match self {
            FfaMode::Practice => "practice",
            FfaMode::Fight => "fight",
            FfaMode::Third => "third",
            FfaMode::Avoid => "avoid",
            FfaMode::Hide => "hide",
            FfaMode::Late => "late",
        }
    }
    pub fn index(self) -> usize {
        self as usize
    }
}

/// What we believe about one enemy, for a viewer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EnemyInfo {
    pub id: u32,
    pub pos: [f32; 2],
    /// 0..1.
    pub conf: f32,
    /// Seconds since it was listed (or heard).
    pub age: f32,
    pub est_hp: f32,
    /// "gun", "mortar", "lightning", "laser" or "".
    pub weapon: &'static str,
    pub hunting: bool,
    pub fighting: Option<u32>,
    pub listed: bool,
}

/// A movement decision from a free-for-all mode.
pub(super) struct FfaMove {
    pub mv: [f32; 2],
    pub boost: bool,
    pub label: &'static str,
    /// Yaw to face when there is nothing to shoot (None: look around).
    pub face: Option<f32>,
    /// Do not start a laser charge.
    pub no_charge: bool,
}

fn weapon_name(w: u8) -> &'static str {
    match w {
        W_GUN => "gun",
        W_MORTAR => "mortar",
        W_LIGHTNING => "lightning",
        W_LASER => "laser",
        _ => "",
    }
}

/// Body seen from: an enemy somewhere within `r` of `pos`, with confidence `conf`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Eye {
    pub pos: [f32; 2],
    pub r: f32,
    pub conf: f32,
}

impl Brain {
    /// Read an observation for the free-for-all side: the enemy picture, the detector, every enemy's profile,
    /// and (once a free-for-all is settled) the mode.
    pub(super) fn ffa_observe(&mut self, o: &Obs) {
        let was = self.detector.latched;
        self.detector.observe(o);
        if self.detector.latched && !was && self.trace_ffa {
            eprintln!("FFA latch slot {} seed {} t {:.1} alive {} why {}", self.ffa_slot, self.ffa_seed, o.t, o.alive, self.detector.reason);
        }
        let pos = o.me.pos;
        self.update_contacts(o, &pos);
        self.profile(o, &pos);
        if self.detector.latched && self.params.ffa_play > 0.0 {
            self.choose_mode(o);
        }
        self.ffa_ticks[self.ffa_mode.index()] += 1;
        self.dbg_enemies = self
            .beliefs
            .iter()
            .filter(|b| b.id.is_some())
            .take(16)
            .map(|b| EnemyInfo {
                id: b.id.unwrap_or(0),
                pos: b.pos,
                conf: b.conf,
                age: b.age,
                est_hp: b.est_hp,
                weapon: weapon_name(b.weapon),
                hunting: b.hunting,
                fighting: b.id.and_then(|id| self.detector.fighting(id, o.t, 6.0)),
                listed: b.listed,
            })
            .collect();
        let mut kills: Vec<(f32, Option<u32>, Option<u32>)> = self.kill_log.iter().copied().chain(self.detector.deaths.iter().map(|d| (d.0, None, d.1))).collect();
        kills.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let n = kills.len();
        self.dbg_kills = kills.split_off(n.saturating_sub(8));
        let mut s = if self.ffa_mode == FfaMode::Practice {
            format!("practice clues {:.0}", self.detector.total())
        } else {
            format!("{} {}", self.ffa_mode.name(), self.ffa_why)
        };
        s.truncate(80);
        self.dbg_mode = s;
    }

    /// Feed this observation to the enemy picture and take this tick's beliefs from it.
    fn update_contacts(&mut self, o: &Obs, pos: &[f32; 3]) {
        let me = &o.me;
        let here = [pos[0], pos[1]];
        self.contacts.begin();
        for e in o.events.iter().filter(|e| e.kind == "hit") {
            if let Some(id) = e.target.filter(|t| *t < 100) {
                self.contacts.note_dealt(id, e.amount);
            }
        }
        for m in o.mechs.iter().filter(|m| m.kind == "mech" && m.id < 100) {
            if m.max_hp > 0.0 && m.hp <= 0.0 {
                self.contacts.remove(m.id);
                continue;
            }
            let aimed = wrap(m.aim[0] - yaw_to(&m.pos, pos)).abs() < 0.2;
            let hunter = aimed || self.attackers.get(&m.id).map_or(false, |t| o.t - t < 4.0);
            let hpf = if m.max_hp > 0.0 { m.hp / m.max_hp } else { 1.0 };
            self.contacts.see(o.t, m.id, m.pos, m.vel, &m.weapons, hpf, hunter);
            if m.max_hp > 0.0 {
                self.contacts.note_hp(m.id, m.hp);
            }
        }
        for e in &o.events {
            match (e.kind.as_str(), e.target, e.by, e.bearing) {
                ("kill", Some(id), _, _) => {
                    self.contacts.remove(id);
                    self.kill_log.push_back((o.t, Some(self.ffa_slot), Some(id)));
                }
                ("damaged", _, Some(by), Some(b)) if by < 100 => self.contacts.hit_by(o.t, here, Some(by), b, e.weapon.as_deref().unwrap_or("")),
                _ => {}
            }
        }
        for s in &o.sounds {
            self.contacts.hear(o.t, here, &s.kind, s.bearing, &s.dist, s.age);
        }
        let eye = [pos[0], pos[1], pos[2] + me.stat("eye_height", 8.0)];
        let aim = me.aim[0];
        let w = self.world.clone();
        let in_view = |q: [f32; 2]| -> bool {
            let Some(w) = w.as_ref() else { return false };
            let d = d2(&q, pos);
            if d > 200.0 || (d > 45.0 && wrap(yaw_to(pos, &q) - aim).abs() > PI / 3.0) {
                return false;
            }
            let g = w.ground(q[0], q[1]);
            w.sees(eye, [q[0], q[1], g + 4.5]) || w.sees(eye, [q[0], q[1], g + 9.0])
        };
        self.contacts.settle(o.t, here, &in_view);
        self.beliefs = self.contacts.beliefs(o.t);
        self.dbg_contacts = self.beliefs.iter().map(|b| [b.pos[0], b.pos[1], b.r, b.conf]).collect();
        while self.kill_log.len() > 8 {
            self.kill_log.pop_front();
        }
    }

    /// Fill in every believed enemy's profile: coming for us, alone, behind us, cornered.
    fn profile(&mut self, o: &Obs, pos: &[f32; 3]) {
        let here = [pos[0], pos[1]];
        if self.me_hist.back().map_or(true, |h| o.t - h.0 >= 0.5 - 1e-3) {
            self.me_hist.push_back((o.t, here));
        }
        while self.me_hist.front().map_or(false, |h| o.t - h.0 > 5.0) {
            self.me_hist.pop_front();
        }
        let aim = o.me.aim[0];
        for i in 0..self.beliefs.len() {
            let b = self.beliefs[i];
            let d = d2(&b.pos, &here);
            let mut hunting = b.hunter;
            if let Some(k) = b.id.and_then(|id| self.contacts.tracks.get(&id)) {
                // Closer to us than 3 to 5 s ago by at least 8 m (counting how we moved meanwhile).
                if let Some(&(t0, e0)) = k.hist.iter().find(|h| o.t - h.0 >= 3.0 - 1e-3) {
                    let me0 = self.me_hist.iter().min_by(|x, y| (x.0 - t0).abs().partial_cmp(&(y.0 - t0).abs()).unwrap_or(std::cmp::Ordering::Equal)).map_or(here, |h| h.1);
                    if d2(&e0, &me0) - d >= 8.0 {
                        hunting = true;
                    }
                }
                if k.listed && d < 200.0 {
                    let to = norm2(here[0] - b.pos[0], here[1] - b.pos[1]);
                    if k.vel[0] * to[0] + k.vel[1] * to[1] > 2.5 {
                        hunting = true;
                    }
                }
            }
            // Busy with someone else and not pointed at us: not coming for us.
            if let Some(id) = b.id {
                if hunting && !b.hunter && self.detector.fighting(id, o.t, 3.0).is_some() {
                    hunting = false;
                }
            }
            let alone = !self.beliefs.iter().enumerate().any(|(j, c)| j != i && c.conf >= 0.3 && d2(&c.pos, &b.pos) < 80.0);
            let behind = b.conf >= 0.3 && d < 60.0 && wrap(yaw_to(&here, &b.pos) - aim).abs() > PI / 3.0;
            let cornered = b.r < 15.0 && self.cornered_at(b.pos);
            let e = &mut self.beliefs[i];
            e.hunting = hunting;
            e.alone = alone;
            e.behind = behind;
            e.cornered = cornered;
        }
    }

    /// Whether a mech at `p` has few ways out: at most a quarter of the directions open, or a single lane
    /// covering under half of them.
    pub(super) fn cornered_at(&self, p: [f32; 2]) -> bool {
        match self.sight.as_ref().and_then(|st| st.cell_at(p[0], p[1]).map(|c| (st.open[c] as usize, st.lanes[c]))) {
            Some((open, lanes)) => open * 4 <= SECTORS || (lanes <= 1 && open * 2 < SECTORS),
            None => false,
        }
    }

    /// Settle this decision's mode.
    fn choose_mode(&mut self, o: &Obs) {
        let p = self.params.clone();
        let pos = o.me.pos;
        let here = [pos[0], pos[1]];
        let my_hp = o.me.hp;
        let hp_frac = my_hp / o.me.max_hp.max(1.0);
        let late = o.zone.next_radius <= p.ffa_late_r || o.alive <= 4;
        let near = |b: &Belief, r: f32| d2(&b.pos, &here) - 0.5 * b.r < r;
        let hunters: Vec<Belief> = self.beliefs.iter().filter(|b| b.hunting && b.conf >= 0.3 && near(b, p.ffa_avoid_r)).copied().collect();
        // Two mechs fighting each other near us: the weaker as the target, the stronger to keep out of sight of.
        let mut third: Option<(f32, u32, u32, String)> = None;
        if hp_frac >= 0.5 && hunters.is_empty() {
            for (_, a, b) in self.detector.pairs(o.t, 5.0) {
                let (Some(ba), Some(bb)) = (self.beliefs.iter().find(|x| x.id == Some(a)), self.beliefs.iter().find(|x| x.id == Some(b))) else { continue };
                if ba.conf < 0.3 || bb.conf < 0.3 {
                    continue;
                }
                let mid = [(ba.pos[0] + bb.pos[0]) * 0.5, (ba.pos[1] + bb.pos[1]) * 0.5];
                let dm = d2(&mid, &here);
                if dm > p.ffa_third_r {
                    continue;
                }
                let (weak, strong) = if ba.est_hp <= bb.est_hp { (ba, bb) } else { (bb, ba) };
                if weak.est_hp > (0.85 * my_hp).max(700.0) || weak.est_hp > 0.95 * strong.est_hp.max(1.0) && weak.est_hp > 700.0 {
                    continue;
                }
                let u = 3.0 + (1.0 - weak.est_hp / 1400.0).clamp(0.0, 1.0) * 2.0 - dm / p.ffa_third_r;
                if third.as_ref().map_or(true, |t| u > t.0) {
                    let why = format!("{}({:.0}) vs {}({:.0}) {:.0}m", weak.id.unwrap_or(0), weak.est_hp, strong.id.unwrap_or(0), strong.est_hp, dm);
                    third = Some((u, weak.id.unwrap_or(0), strong.id.unwrap_or(0), why));
                }
            }
        }
        // Believed enemies within 120 m, leaving out a pair busy with each other that we are about to use.
        let crowd = self
            .beliefs
            .iter()
            .filter(|b| b.conf >= 0.5 && near(b, 120.0))
            .filter(|b| third.as_ref().map_or(true, |t| b.id != Some(t.1) && b.id != Some(t.2)))
            .count();
        let crowd_all = self.beliefs.iter().filter(|b| b.conf >= 0.5 && near(b, 120.0)).count();
        // A fight we cannot walk away from.
        let lone = hunters.len() == 1 && crowd_all <= 1;
        let close = hunters.iter().filter(|b| b.listed && d2(&b.pos, &here) < 40.0).count();
        let sure = hunters.first().map_or(false, |b| lone && b.listed && b.est_hp < 0.5 * my_hp && hp_frac >= 0.5 && d2(&b.pos, &here) < 120.0);
        let pinned = close >= 1 && (lone || self.cornered_at(here));
        let (mode, why) = if pinned || sure {
            let b = hunters.iter().min_by(|x, y| d2(&x.pos, &here).partial_cmp(&d2(&y.pos, &here)).unwrap_or(std::cmp::Ordering::Equal));
            (FfaMode::Fight, format!("{} {} {:.0}hp", if sure { "sure" } else { "pinned" }, b.and_then(|b| b.id).map_or(-1, |i| i as i64), b.map_or(0.0, |b| b.est_hp)))
        } else if !hunters.is_empty() || crowd >= 2 {
            (FfaMode::Avoid, format!("hunters {} near {}", hunters.len(), crowd))
        } else if let Some((_, w, s, why)) = third.clone() {
            self.ffa_third = Some((w, s));
            (FfaMode::Third, why)
        } else if late {
            (FfaMode::Late, format!("next r {:.0} alive {}", o.zone.next_radius, o.alive))
        } else {
            (FfaMode::Hide, format!("near {} hp {:.0}", crowd_all, my_hp))
        };
        // Leave an escape early only for a fight; otherwise give it 2 s.
        let mode = if self.ffa_mode == FfaMode::Avoid && o.t - self.ffa_since < 2.0 && !matches!(mode, FfaMode::Fight | FfaMode::Avoid) { FfaMode::Avoid } else { mode };
        if mode != FfaMode::Third {
            self.ffa_third = None;
        }
        if mode != self.ffa_mode {
            if self.trace_ffa {
                eprintln!("FFA mode slot {} seed {} t {:.1} {} -> {} {}", self.ffa_slot, self.ffa_seed, o.t, self.ffa_mode.name(), mode.name(), why);
            }
            self.ffa_mode = mode;
            self.ffa_since = o.t;
            self.ffa_goal = None;
        }
        self.ffa_why = why;
    }

    /// Whether the free-for-all modes are steering this decision.
    pub(super) fn ffa_on(&self) -> bool {
        self.ffa_mode != FfaMode::Practice
    }

    /// Mechs worth shooting at in the current mode.
    pub(super) fn ffa_engage(&self, o: &Obs, m: &Actor, hp_frac: f32) -> bool {
        let pos = o.me.pos;
        let d = d2(&m.pos, &pos);
        let hurt = self.attackers.get(&m.id).map_or(false, |t| o.t - t < 3.0);
        let sure = m.max_hp > 0.0 && (m.hp <= 350.0 || m.hp / m.max_hp <= 0.25) && m.hp < 0.4 * o.me.hp && hp_frac >= 0.6 && d < 150.0;
        match self.ffa_mode {
            FfaMode::Practice | FfaMode::Fight => true,
            FfaMode::Third => self.ffa_third.map_or(false, |t| t.0 == m.id) || hurt || sure,
            FfaMode::Avoid | FfaMode::Hide => hurt || d < 40.0 || sure,
            FfaMode::Late => sure || d < 40.0 || (hurt && self.cornered_at([pos[0], pos[1]])),
        }
    }

    /// Believed enemies as watchers.
    fn eyes(&self, pos: &[f32; 3], skip: Option<u32>) -> Vec<Eye> {
        self.beliefs
            .iter()
            .filter(|b| b.conf >= 0.1 && d2(&b.pos, pos) < 260.0 + b.r && (skip.is_none() || b.id != skip))
            .map(|b| Eye { pos: b.pos, r: b.r, conf: b.conf })
            .collect()
    }

    /// Chance that a watcher has a line to `at`: its confidence times the share of its possible positions (the
    /// centre and six points round it) that have one.
    fn watch(&self, at: [f32; 2], f: &Eye) -> f32 {
        let Some(st) = self.sight.as_ref() else { return f.conf };
        if f.r < 3.0 {
            return if st.visible(at, f.pos) { f.conf } else { 0.0 };
        }
        if d2(&at, &f.pos) > crate::sight::RANGE + f.r {
            return 0.0;
        }
        let rr = f.r * 0.7;
        let mut n = st.visible(at, f.pos) as u32;
        for k in 0..6 {
            let a = k as f32 * TAU / 6.0;
            n += st.visible(at, [f.pos[0] + a.cos() * rr, f.pos[1] + a.sin() * rr]) as u32;
        }
        f.conf * n as f32 / 7.0
    }

    fn expected(&self, at: [f32; 2], eyes: &[Eye]) -> f32 {
        eyes.iter().map(|f| self.watch(at, f)).sum()
    }

    /// Standing points of the sight table on rings round `c` out to `r`, each once.
    fn ring_cells(&self, c: [f32; 2], r: f32) -> Vec<(usize, [f32; 2])> {
        let Some(st) = self.sight.as_ref() else { return Vec::new() };
        let mut v: Vec<(usize, [f32; 2])> = Vec::new();
        let mut ring = 8.0;
        while ring <= r + 0.1 {
            let n = ((ring * TAU / 9.0) as usize).clamp(12, 64);
            for k in 0..n {
                let a = k as f32 * TAU / n as f32;
                let q = [c[0] + a.cos() * ring, c[1] + a.sin() * ring];
                if let Some(cell) = st.cell_at(q[0], q[1]) {
                    if st.reach[cell] && !v.iter().any(|x| x.0 == cell) {
                        if let Some(pt) = st.point(cell) {
                            v.push((cell, pt));
                        }
                    }
                }
            }
            ring += 8.0;
        }
        v
    }

    /// Whether a place is fine as far as the zone goes: inside the next circle, or (while the zone hurts 35/s or
    /// less) inside the current one with time to walk into the next before the shrink starts.
    fn zone_ok(&self, o: &Obs, q: [f32; 2], walk: f32) -> bool {
        let z = &o.zone;
        let dc = d2(&q, &z.center);
        if dc > z.radius - 12.0 {
            return false;
        }
        let out = d2(&q, &z.next_center) - (z.next_radius - 12.0);
        if out <= 0.0 {
            return true;
        }
        z.dps <= 35.0 && z.shrink_start - o.t > out / walk.max(1.0) * 1.3 + 20.0
    }

    /// Where to break off to: within 100 m, reachable, fine for the zone, not toward the enemies' centre and not
    /// near any of them, scored by how many are expected to see it, how exposed it is, cover toward their centre,
    /// the walk, and how many are expected to see us on the way. `None` when nothing is clearly less watched
    /// than here.
    fn avoid_pick(&self, o: &Obs, here: [f32; 2], eyes: &[Eye], walk: f32, keep: Option<[f32; 2]>) -> Option<[f32; 2]> {
        let (Some(st), Some(nav)) = (self.sight.as_ref(), self.nav.as_ref()) else { return None };
        let wsum: f32 = eyes.iter().map(|f| f.conf).sum();
        if wsum <= 0.0 {
            return None;
        }
        let cen = [eyes.iter().map(|f| f.pos[0] * f.conf).sum::<f32>() / wsum, eyes.iter().map(|f| f.pos[1] * f.conf).sum::<f32>() / wsum];
        let now = self.expected(here, eyes);
        let dc_me = d2(&here, &cen);
        let away = [here[0] - cen[0], here[1] - cen[1]];
        let base = |p: [f32; 2], c: usize| -> Option<f32> {
            if !self.zone_ok(o, p, walk) {
                return None;
            }
            let dm = d2(&p, &here);
            if d2(&p, &cen) < dc_me + 0.3 * dm.min(20.0) || (dm > 1.0 && (p[0] - here[0]) * away[0] + (p[1] - here[1]) * away[1] < -0.2 * dm * dc_me) {
                return None;
            }
            if eyes.iter().any(|f| f.conf >= 0.3 && d2(&f.pos, &p) < 35.0 + f.r * 0.5) {
                return None;
            }
            let seen = self.expected(p, eyes);
            let covered = st.range_at(c, (cen[1] - p[1]).atan2(cen[0] - p[0])) + 3.0 < d2(&p, &cen).min(crate::sight::RANGE);
            Some(seen * 12.0 + st.exposure[c] * 6.0 + 0.5 * (st.lanes[c].max(1) - 1) as f32 - if covered { 3.0 } else { 0.0 })
        };
        let mut cands: Vec<(f32, usize, [f32; 2])> = self.ring_cells(here, 100.0).into_iter().filter_map(|(c, p)| base(p, c).map(|s| (s, c, p))).collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        cands.truncate(14);
        let mut nav = nav.clone();
        nav.set_goal(here[0], here[1]);
        let full = |nav: &crate::nav::Nav, p: [f32; 2], s0: f32| -> Option<f32> {
            let travel = nav.path_len(p[0], p[1])?;
            if travel > 170.0 {
                return None;
            }
            let (mut q, mut on_way, mut n) = (p, 0.0f32, 0);
            for _ in 0..12 {
                let Some(nq) = nav.steer(q[0], q[1]) else { break };
                if d2(&nq, &here) < 6.0 || d2(&nq, &q) < 0.5 {
                    break;
                }
                q = nq;
                on_way += (self.expected(q, eyes) - 0.5).max(0.0);
                n += 1;
            }
            let path = if n > 0 { on_way / n as f32 } else { 0.0 };
            Some(s0 + travel / walk.max(1.0) * 0.8 + path * 6.0)
        };
        let mut best: Option<([f32; 2], f32)> = None;
        for &(s0, _, p) in &cands {
            if let Some(sc) = full(&nav, p, s0) {
                if best.map_or(true, |b| sc < b.1) {
                    best = Some((p, sc));
                }
            }
        }
        let b = best?;
        if self.expected(b.0, eyes) > now - 0.4 {
            return None;
        }
        if let Some(q) = keep {
            if let Some(c) = st.cell_at(q[0], q[1]) {
                if let Some(sc) = base(q, c).and_then(|s0| full(&nav, q, s0)) {
                    if sc <= b.1 + 2.0 {
                        return Some(q);
                    }
                }
            }
        }
        Some(b.0)
    }

    /// A sheltered spot to sit at: among the sight table's spots within 250 m, fine for the zone, away from
    /// drones and believed enemies, scored by exposure, expected watchers, closeness of believed enemies and the
    /// walk there. `keep` stays unless another is clearly better.
    fn hide_pick(&self, o: &Obs, here: [f32; 2], eyes: &[Eye], walk: f32, keep: Option<[f32; 2]>) -> Option<[f32; 2]> {
        let (Some(st), Some(nav)) = (self.sight.as_ref(), self.nav.as_ref()) else { return None };
        let drones: Vec<[f32; 2]> = self.detector.npcs.values().map(|v| v.1).collect();
        let base = |p: [f32; 2], c: usize| -> Option<f32> {
            if !st.reach[c] || !self.zone_ok(o, p, walk) || drones.iter().any(|d| d2(d, &p) < 55.0) {
                return None;
            }
            let near: f32 = eyes.iter().map(|f| f.conf * (1.0 - (d2(&f.pos, &p) - 0.5 * f.r).max(0.0) / 200.0).max(0.0)).sum();
            if eyes.iter().any(|f| f.conf >= 0.3 && d2(&f.pos, &p) < 40.0 + 0.5 * f.r) {
                return None;
            }
            Some(st.exposure[c] * 40.0 + self.expected(p, eyes) * 15.0 + near * 12.0)
        };
        let mut cands: Vec<(f32, [f32; 2])> = st
            .spots
            .iter()
            .filter(|s| d2(&s.pos, &here) < 250.0)
            .filter_map(|s| base(s.pos, s.cell).map(|v| (v + d2(&s.pos, &here) / walk.max(1.0) * 0.3, s.pos)))
            .collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(12);
        let mut nav = nav.clone();
        nav.set_goal(here[0], here[1]);
        let score = |p: [f32; 2], s0: f32| -> Option<f32> {
            let travel = nav.path_len(p[0], p[1])?;
            Some(s0 + travel / walk.max(1.0) * 0.3)
        };
        let mut best: Option<([f32; 2], f32)> = None;
        for &(s0, p) in &cands {
            let s0 = s0 - d2(&p, &here) / walk.max(1.0) * 0.3;
            if let Some(sc) = score(p, s0) {
                if best.map_or(true, |b| sc < b.1) {
                    best = Some((p, sc));
                }
            }
        }
        let b = best?;
        if let Some(q) = keep {
            if let Some(c) = st.cell_at(q[0], q[1]) {
                if let Some(sc) = base(q, c).and_then(|s0| score(q, s0)) {
                    if sc <= b.1 + 3.0 {
                        return Some(q);
                    }
                }
            }
        }
        Some(b.0)
    }

    /// Where to shoot the weaker of two fighting mechs from: within 140 m of us, 50 to 180 m from it, with a line
    /// to it and none to the stronger, few other believed enemies with a line, a short walk and little exposure.
    fn third_pick(&self, o: &Obs, here: [f32; 2], weak: [f32; 2], strong: [f32; 2], eyes: &[Eye], walk: f32, keep: Option<[f32; 2]>) -> Option<[f32; 2]> {
        let (Some(st), Some(nav)) = (self.sight.as_ref(), self.nav.as_ref()) else { return None };
        let base = |p: [f32; 2], c: usize| -> Option<f32> {
            let dw = d2(&p, &weak);
            if !(50.0..=180.0).contains(&dw) || !self.zone_ok(o, p, walk) || !st.visible(p, weak) || st.visible(p, strong) {
                return None;
            }
            let others = self.expected(p, eyes);
            if others > 0.6 {
                return None;
            }
            Some(st.exposure[c] * 10.0 + others * 8.0 + (dw - 110.0).abs() / 40.0)
        };
        let mut cands: Vec<(f32, [f32; 2])> = self.ring_cells(here, 140.0).into_iter().filter_map(|(c, p)| base(p, c).map(|s| (s, p))).collect();
        if let Some(c) = st.cell_at(here[0], here[1]) {
            if let Some(s) = base(here, c) {
                cands.push((s, here));
            }
        }
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        cands.truncate(14);
        let mut nav = nav.clone();
        nav.set_goal(here[0], here[1]);
        let score = |p: [f32; 2], s0: f32| -> Option<f32> {
            let travel = if d2(&p, &here) < 3.0 { 0.0 } else { nav.path_len(p[0], p[1])? };
            (travel <= 200.0).then_some(s0 + travel / walk.max(1.0))
        };
        let mut best: Option<([f32; 2], f32)> = None;
        for &(s0, p) in &cands {
            if let Some(sc) = score(p, s0) {
                if best.map_or(true, |b| sc < b.1) {
                    best = Some((p, sc));
                }
            }
        }
        let b = best?;
        if let Some(q) = keep {
            if let Some(c) = st.cell_at(q[0], q[1]) {
                if let Some(sc) = base(q, c).and_then(|s0| score(q, s0)) {
                    if sc <= b.1 + 2.0 {
                        return Some(q);
                    }
                }
            }
        }
        Some(b.0)
    }

    /// Movement for the current mode, or None to leave it to the usual logic. `arm_goal`: a weapon worth
    /// fetching (from the usual logic).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ffa_move(&mut self, o: &Obs, pos: &[f32; 3], hp_frac: f32, shooting: bool, enemies: &[&Actor], arm_goal: Option<[f32; 3]>) -> Option<FfaMove> {
        let me = &o.me;
        let here = [pos[0], pos[1]];
        let walk = me.stat("walk_speed", 11.7);
        let quiet = (self.params.ffa_quiet / walk.max(0.1)).min(1.0);
        let chest = [pos[0], pos[1], pos[2] + 5.0];
        let seen_now = enemies.iter().any(|m| self.world.as_ref().map_or(true, |w| w.sees([m.pos[0], m.pos[1], m.pos[2] + 8.0], chest)));
        let nearest = self
            .beliefs
            .iter()
            .filter(|b| b.conf >= 0.3)
            .min_by(|x, y| d2(&x.pos, &here).partial_cmp(&d2(&y.pos, &here)).unwrap_or(std::cmp::Ordering::Equal))
            .copied();
        let face_near = nearest.filter(|b| d2(&b.pos, &here) < 220.0).map(|b| yaw_to(&here, &b.pos));
        let repick = |b: &Brain, every: f32| b.ffa_goal.is_none() || o.t - b.ffa_goal_t >= every;
        match self.ffa_mode {
            FfaMode::Practice | FfaMode::Fight => None,
            FfaMode::Avoid => {
                let eyes = self.eyes(pos, None);
                if repick(self, 1.5) {
                    let keep = self.ffa_goal;
                    self.ffa_goal = self.avoid_pick(o, here, &eyes, walk, keep);
                    self.ffa_goal_t = o.t;
                    if self.ffa_goal.is_none() && self.sight.is_none() {
                        // No sight table: straight away from them.
                        let n: f32 = eyes.iter().map(|e| e.conf).sum::<f32>().max(1e-3);
                        let c = eyes.iter().fold([0.0f32; 2], |s, e| [s[0] + e.pos[0] * e.conf / n, s[1] + e.pos[1] * e.conf / n]);
                        let a = norm2(here[0] - c[0], here[1] - c[1]);
                        self.ffa_goal = Some([here[0] + a[0] * 40.0, here[1] + a[1] * 40.0]);
                    }
                }
                let watched = self.expected(here, &eyes);
                let hunter = self
                    .beliefs
                    .iter()
                    .filter(|b| b.hunting && b.conf >= 0.3)
                    .min_by(|x, y| d2(&x.pos, &here).partial_cmp(&d2(&y.pos, &here)).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|b| yaw_to(&here, &b.pos));
                let face = self.alert_bearing(o).filter(|_| o.t - self.last_mech_hurt_t < 2.0).or(hunter).or(face_near);
                let (mv, boost, label) = match self.ffa_goal {
                    Some(g) if d2(&g, &here) >= 2.0 => {
                        let mut mv = self.go(pos, g);
                        let boost = seen_now && me.energy > 10.0 && d2(&g, &here) > 5.0;
                        if !seen_now && watched < 0.5 {
                            mv = [mv[0] * quiet, mv[1] * quiet];
                        }
                        (mv, boost, "ffa-avoid")
                    }
                    Some(_) => ([0.0, 0.0], false, "ffa-avoid"),
                    None if watched < 0.3 => ([0.0, 0.0], false, "ffa-avoid"),
                    None => {
                        let d = self.away_dir(o, pos, 4.0, None);
                        (d, seen_now && me.energy > 20.0, "ffa-avoid")
                    }
                };
                Some(FfaMove { mv, boost, label, face, no_charge: true })
            }
            FfaMode::Third => {
                let (wid, sid) = self.ffa_third?;
                let weak = self.beliefs.iter().find(|b| b.id == Some(wid)).map(|b| b.pos)?;
                let strong = self.beliefs.iter().find(|b| b.id == Some(sid)).map(|b| b.pos).unwrap_or(weak);
                let eyes: Vec<Eye> = self.eyes(pos, Some(wid)).into_iter().filter(|e| d2(&e.pos, &strong) > 10.0).collect();
                let weak_listed = enemies.iter().any(|m| m.id == wid);
                // In a good place already: a line to the weaker, none to the stronger.
                let good_here = match self.sight.as_ref() {
                    Some(st) => weak_listed && st.visible(here, weak) && !st.visible(here, strong) && d2(&weak, &here) < 200.0,
                    None => weak_listed && d2(&weak, &here) < 180.0,
                };
                if repick(self, 1.5) {
                    let keep = self.ffa_goal;
                    self.ffa_goal = self.third_pick(o, here, weak, strong, &eyes, walk, keep).or_else(|| {
                        // No better place known: close in to 120 m on our side of it.
                        let a = norm2(here[0] - weak[0], here[1] - weak[1]);
                        Some([weak[0] + a[0] * 120.0, weak[1] + a[1] * 120.0])
                    });
                    self.ffa_goal_t = o.t;
                }
                let face = Some(yaw_to(&here, &weak));
                if good_here && (shooting || d2(&weak, &here) < 200.0) {
                    return Some(FfaMove { mv: [0.0, 0.0], boost: false, label: "ffa-third-shoot", face, no_charge: false });
                }
                let g = self.ffa_goal.unwrap_or(weak);
                let mut mv = self.go(pos, g);
                if d2(&weak, &here) < 110.0 || d2(&strong, &here) < 110.0 {
                    mv = [mv[0] * quiet, mv[1] * quiet];
                }
                Some(FfaMove { mv, boost: false, label: "ffa-third", face, no_charge: false })
            }
            FfaMode::Late => {
                let h = self.home_spot(o);
                let dh = d2(&h, &here);
                let mut mv = if dh < 2.5 { [0.0, 0.0] } else { self.go(pos, h) };
                let slow = if dh > 40.0 { 1.0 } else { quiet };
                mv = [mv[0] * slow, mv[1] * slow];
                let face = self.alert_bearing(o).or(face_near);
                Some(FfaMove { mv, boost: dh > 60.0 && me.energy > 60.0 && !seen_now, label: "ffa-late", face, no_charge: !shooting })
            }
            FfaMode::Hide => {
                let eyes = self.eyes(pos, None);
                let safe = |b: &Brain, q: [f32; 2]| !b.beliefs.iter().any(|e| e.conf >= 0.3 && d2(&e.pos, &q) - 0.5 * e.r < 100.0);
                // Loot worth the walk: a cube within 45 m or a weapon we want, with no believed enemy near it.
                let cube = self
                    .cubes_mem
                    .values()
                    .filter(|q| d2(*q, &here) < 45.0 && (q[2] - pos[2]).abs() < 6.0)
                    .min_by(|x, y| d2(*x, &here).partial_cmp(&d2(*y, &here)).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|q| [q[0], q[1]])
                    .filter(|q| hp_frac > 0.3 && safe(self, *q));
                let arm = arm_goal.filter(|g| d2(g, &here) < 90.0 && (g[2] - pos[2]).abs() < 3.0 && safe(self, [g[0], g[1]])).map(|g| [g[0], g[1]]);
                if let Some(g) = cube.or(arm) {
                    let mv = self.go(pos, g);
                    return Some(FfaMove { mv, boost: false, label: "ffa-loot", face: face_near, no_charge: !shooting });
                }
                if repick(self, 3.0) {
                    let keep = self.ffa_goal;
                    self.ffa_goal = self.hide_pick(o, here, &eyes, walk, keep);
                    self.ffa_goal_t = o.t;
                }
                let (mv, label) = match self.ffa_goal {
                    Some(g) if d2(&g, &here) >= 2.5 => {
                        let mut mv = self.go(pos, g);
                        if nearest.map_or(false, |b| d2(&b.pos, &here) < 80.0) {
                            mv = [mv[0] * quiet, mv[1] * quiet];
                        }
                        (mv, "ffa-hide")
                    }
                    _ => ([0.0, 0.0], "ffa-hold"),
                };
                Some(FfaMove { mv, boost: false, label, face: face_near, no_charge: !shooting })
            }
        }
    }

    /// The detector's match verdict (true: a free-for-all).
    pub fn ffa_latched(&self) -> bool {
        self.detector.latched
    }

    /// Decisions spent in each mode: practice, fight, third, avoid, hide, late.
    pub fn ffa_ticks(&self) -> [u32; 6] {
        self.ffa_ticks
    }

    /// The weaker mech of the fight we are joining, if any.
    pub fn ffa_third_target(&self) -> Option<u32> {
        self.ffa_third.map(|t| t.0)
    }
}
