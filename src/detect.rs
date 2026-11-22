//! Telling what kind of match this is. A match starts out taken for a practice match (every other mech on one
//! side, never hurting each other); clues that two other mechs are fighting each other add up, and once they are
//! enough the match is taken for a free-for-all for good.
//!
//! Clues (one per kind and pair counts once, at its highest worth):
//! - a laser beam, lightning arc or gun bullet from one mech to another, along a line that does not come near
//!   us, with the one hit losing hit points within 0.6 s that neither our hits, the zone nor a nearby drone
//!   explain (2; never followed by a loss: beam 0.5, others 0; with hit points hidden: beam 1, others 0.5);
//! - a listed mech losing 40+ hit points between two consecutive decisions that nothing else explains, with
//!   another mech (not aimed at us) aimed at it, unless it holds a mortar (1);
//! - a mech holding a laser charge aimed at another mech, not at us, for two decisions in a row (0.5);
//! - two mechs within 60 m aimed at each other, neither at us, for a second (0.5);
//! - fewer mechs alive without a kill of ours, with no gap in our observations and no hit of ours around it (1).
//!
//! Every clue that names who shot whom is also kept as a fight between the two.
use crate::proto::{Actor, Obs};
use std::collections::{BTreeMap, VecDeque};

/// Clues older than this are forgotten, seconds.
const KEEP_S: f32 = 90.0;
/// Fights are remembered this long, seconds.
const FIGHT_S: f32 = 10.0;
/// Clue total that settles it.
const LATCH: f32 = 2.0;

/// From this moment (2026-11-22 16:00 UTC) every match is taken for a free-for-all from the start.
const OPEN_PLAY_UNIX: u64 = 1_795_363_200;

/// Whether the wall clock says matches are free-for-alls from now on (`MECH_NOW` overrides it, for tests).
fn after_open_play() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let now = std::env::var("MECH_NOW").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or_else(|| {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
        });
        now >= OPEN_PLAY_UNIX
    })
}

fn wrap(a: f32) -> f32 {
    let mut a = (a + std::f32::consts::PI) % std::f32::consts::TAU;
    if a < 0.0 {
        a += std::f32::consts::TAU;
    }
    a - std::f32::consts::PI
}

fn dist(a: &[f32], b: &[f32]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn yaw(from: &[f32], to: &[f32]) -> f32 {
    (to[1] - from[1]).atan2(to[0] - from[0])
}

/// Where along a line from `o` in unit direction `d` the point `q` lies, and how far off the line it is.
fn along(o: &[f32], d: [f32; 2], q: &[f32]) -> (f32, f32) {
    let e = [q[0] - o[0], q[1] - o[1]];
    (e[0] * d[0] + e[1] * d[1], (e[0] * d[1] - e[1] * d[0]).abs())
}

fn unit(x: f32, y: f32) -> Option<[f32; 2]> {
    let l = x.hypot(y);
    (l > 1e-3 && l.is_finite()).then(|| [x / l, y / l])
}

/// Body radius of a mech (it grows with cubes).
fn size(m: &Actor) -> f32 {
    2.5 * (1.0 + 0.05 * m.cubes.max(0.0))
}

/// One clue.
#[derive(Clone, Debug)]
pub struct Clue {
    pub t: f32,
    pub key: String,
    pub pts: f32,
}

/// Two mechs seen fighting: when, who shot (if known), who was hit.
#[derive(Clone, Copy, Debug)]
pub struct Fight {
    pub t: f32,
    pub by: Option<u32>,
    pub on: u32,
}

#[derive(Clone, Debug, Default)]
pub struct Detector {
    pub latched: bool,
    pub latch_t: f32,
    /// The clues that settled it.
    pub reason: String,
    pub clues: Vec<Clue>,
    pub fights: VecDeque<Fight>,
    /// Deaths of other mechs we did not cause: time, and who died when known.
    pub deaths: VecDeque<(f32, Option<u32>)>,
    /// Per listed mech: tick and hit points at the last decision.
    hp: BTreeMap<u32, (u64, f32)>,
    /// Our hits: time, target, amount.
    dealt: VecDeque<(f32, u32, f32)>,
    /// Our kills not yet matched with a drop in the alive count, and drops not yet matched with a kill.
    kills: VecDeque<f32>,
    drops: VecDeque<f32>,
    alive: Option<u32>,
    last_tick: Option<u64>,
    /// Last time a decision was skipped (a gap in the ticks we saw).
    gap_t: f32,
    /// Drones: last seen time and place.
    pub npcs: BTreeMap<u32, (f32, [f32; 2])>,
    /// Laser charges aimed at another mech: shooter -> (target, decisions in a row).
    chg: BTreeMap<u32, (u32, u32)>,
    /// Pairs aimed at each other: (low id, high id) -> since when.
    face: BTreeMap<(u32, u32), f32>,
    /// Hit points lost by a mech that nothing but another mech explains: time, who.
    lost: VecDeque<(f32, u32)>,
    /// Shots seen between two mechs, waiting for the one hit to lose hit points: time, clue key, who was hit,
    /// points if it never does.
    shots: VecDeque<(f32, String, u32, f32)>,
}

impl Detector {
    /// Total of the clues still remembered.
    pub fn total(&self) -> f32 {
        self.clues.iter().map(|c| c.pts).sum()
    }

    fn clue(&mut self, t: f32, key: String, pts: f32) {
        match self.clues.iter_mut().find(|c| c.key == key) {
            Some(c) => {
                c.t = t;
                c.pts = c.pts.max(pts);
            }
            None => self.clues.push(Clue { t, key, pts }),
        }
    }

    /// Read one observation.
    pub fn observe(&mut self, o: &Obs) {
        let t = o.t;
        let me = [o.me.pos[0], o.me.pos[1]];
        if self.last_tick.map_or(false, |k| o.tick > k + 2) {
            self.gap_t = t;
        }
        self.last_tick = Some(o.tick);
        for e in &o.events {
            match e.kind.as_str() {
                "hit" => {
                    if let Some(id) = e.target.filter(|id| *id < 100) {
                        self.dealt.push_back((t, id, e.amount.max(0.0)));
                    }
                }
                "kill" => self.kills.push_back(t),
                _ => {}
            }
        }
        while self.dealt.front().map_or(false, |d| t - d.0 > 1.0) {
            self.dealt.pop_front();
        }
        for m in o.mechs.iter().filter(|m| m.kind == "npc" || m.id >= 100) {
            self.npcs.insert(m.id, (t, [m.pos[0], m.pos[1]]));
        }
        self.npcs.retain(|_, n| t - n.0 < 30.0);
        let mechs: Vec<&Actor> = o.mechs.iter().filter(|m| m.kind == "mech" && m.id < 100 && !(m.max_hp > 0.0 && m.hp <= 0.0)).collect();
        // A line from `from` along `d` that comes within 15 m of us ahead of `from` may have been meant for us.
        let toward_us = |from: &[f32], d: [f32; 2]| -> bool {
            let (a, off) = along(from, d, &me);
            a > -5.0 && off < 15.0
        };
        let aims_at_us = |m: &Actor, tol: f32| wrap(m.aim[0] - yaw(&m.pos, &me)).abs() < tol;

        // Hit points lost that our hits do not account for.
        for m in &mechs {
            if m.max_hp <= 0.0 {
                continue;
            }
            if let Some(&(k0, h0)) = self.hp.get(&m.id) {
                if o.tick > k0 && o.tick <= k0 + 2 {
                    let ours: f32 = self.dealt.iter().filter(|d| d.1 == m.id && t - d.0 <= 0.35).map(|d| d.2).sum();
                    let lost = h0 - m.hp - ours;
                    let inside = dist(&m.pos, &o.zone.center) < o.zone.radius - 5.0;
                    let drone = self.npcs.values().any(|n| dist(&n.1, &m.pos) < 60.0);
                    if lost >= 20.0 && inside && !drone {
                        self.lost.push_back((t, m.id));
                    }
                    if lost >= 40.0 && inside && !drone {
                        let by = mechs
                            .iter()
                            .filter(|a| a.id != m.id && dist(&a.pos, &m.pos) < 280.0 && wrap(a.aim[0] - yaw(&a.pos, &m.pos)).abs() < 0.2 && !aims_at_us(a, 0.3))
                            .min_by(|x, y| dist(&x.pos, &m.pos).partial_cmp(&dist(&y.pos, &m.pos)).unwrap_or(std::cmp::Ordering::Equal))
                            .map(|a| a.id);
                        self.fights.push_back(Fight { t, by, on: m.id });
                        // A mortar holder may have caught its own blast.
                        let own = m.weapons.iter().any(|w| w == "mortar");
                        if by.is_some() && !own {
                            self.clue(t, format!("hp:{}", m.id), 1.0);
                        }
                    }
                }
            }
            self.hp.insert(m.id, (o.tick, m.hp));
        }
        self.hp.retain(|id, _| mechs.iter().any(|m| m.id == *id));

        // Beams and arcs between two other mechs.
        for b in &o.beams {
            if b.path.len() < 2 || b.by >= 100 {
                continue;
            }
            if mechs.iter().any(|m| m.id == b.by && aims_at_us(m, 0.3)) {
                continue;
            }
            let s = b.path[0];
            let e = match b.kind.as_str() {
                "laser" => *b.path.last().unwrap(),
                "lightning" => b.path[1],
                _ => continue,
            };
            if dist(&e, &me) < 15.0 {
                continue;
            }
            let Some(d) = unit(e[0] - s[0], e[1] - s[1]) else { continue };
            if toward_us(&s, d) {
                continue;
            }
            let hit = mechs
                .iter()
                .filter(|m| m.id != b.by && dist(&m.pos, &e) < size(m) + 2.0 && e[2] > m.pos[2] - 1.5 && e[2] < m.pos[2] + 9.0 * (1.0 + 0.05 * m.cubes) + 1.5)
                .min_by(|x, y| dist(&x.pos, &e).partial_cmp(&dist(&y.pos, &e)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|m| (m.id, m.max_hp > 0.0));
            if let Some((v, hp_known)) = hit {
                let laser = b.kind == "laser";
                let alone = match (laser, hp_known) {
                    (true, true) => 0.5,
                    (true, false) => 1.0,
                    (false, true) => 0.0,
                    (false, false) => 0.5,
                };
                self.fights.push_back(Fight { t, by: Some(b.by), on: v });
                self.shots.push_back((t, format!("{}:{}>{v}", if laser { "beam" } else { "arc" }, b.by), v, alone));
            }
        }

        // Gun bullets from one mech toward another.
        for pr in o.projectiles.iter().filter(|p| p.kind == "gun") {
            let Some(d) = unit(pr.vel[0], pr.vel[1]) else { continue };
            if pr.vel[0].hypot(pr.vel[1]) < 50.0 {
                continue;
            }
            let heading = d[1].atan2(d[0]);
            let shooter = mechs
                .iter()
                .filter_map(|m| {
                    let (a, off) = along(&pr.pos, d, &m.pos);
                    (a < -2.0 && a > -300.0 && off < 3.0 && wrap(m.aim[0] - heading).abs() < 0.12).then_some((m, -a))
                })
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|x| *x.0);
            let Some(sh) = shooter else { continue };
            if toward_us(&sh.pos, d) || aims_at_us(sh, 0.3) {
                continue;
            }
            let tg = mechs
                .iter()
                .filter_map(|m| {
                    let (a, off) = along(&pr.pos, d, &m.pos);
                    (m.id != sh.id && a > 0.0 && a < 250.0 && off < size(m) + 1.0).then_some((m, a))
                })
                .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|x| (x.0.id, x.0.max_hp > 0.0));
            if let Some((v, hp_known)) = tg {
                self.fights.push_back(Fight { t, by: Some(sh.id), on: v });
                self.shots.push_back((t, format!("shot:{}>{v}", sh.id), v, if hp_known { 0.0 } else { 0.5 }));
            }
        }

        // Laser charges held on another mech.
        let mut chg: BTreeMap<u32, (u32, u32)> = BTreeMap::new();
        for a in mechs.iter().filter(|m| m.charging) {
            if aims_at_us(a, 0.35) {
                continue;
            }
            let Some(d) = unit(a.aim[0].cos(), a.aim[0].sin()) else { continue };
            if toward_us(&a.pos, d) {
                continue;
            }
            let on = mechs
                .iter()
                .filter(|b| b.id != a.id && dist(&a.pos, &b.pos) < 280.0 && wrap(a.aim[0] - yaw(&a.pos, &b.pos)).abs() < 0.06)
                .min_by(|x, y| dist(&x.pos, &a.pos).partial_cmp(&dist(&y.pos, &a.pos)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|b| b.id);
            if let Some(v) = on {
                let n = match self.chg.get(&a.id) {
                    Some(&(w, n)) if w == v => n + 1,
                    _ => 1,
                };
                chg.insert(a.id, (v, n));
                if n >= 2 {
                    self.fights.push_back(Fight { t, by: Some(a.id), on: v });
                    self.clue(t, format!("chg:{}>{v}", a.id), 0.5);
                }
            }
        }
        self.chg = chg;

        // Two mechs squared up to each other.
        let mut face: BTreeMap<(u32, u32), f32> = BTreeMap::new();
        for (i, a) in mechs.iter().enumerate() {
            for b in mechs.iter().skip(i + 1) {
                if dist(&a.pos, &b.pos) > 60.0 || aims_at_us(a, 0.4) || aims_at_us(b, 0.4) {
                    continue;
                }
                if wrap(a.aim[0] - yaw(&a.pos, &b.pos)).abs() < 0.15 && wrap(b.aim[0] - yaw(&b.pos, &a.pos)).abs() < 0.15 {
                    let k = (a.id.min(b.id), a.id.max(b.id));
                    let since = self.face.get(&k).copied().unwrap_or(t);
                    face.insert(k, since);
                    if t - since >= 1.0 {
                        self.fights.push_back(Fight { t, by: Some(a.id), on: b.id });
                        self.clue(t, format!("face:{}-{}", k.0, k.1), 0.5);
                    }
                }
            }
        }
        self.face = face;

        // Fewer alive without a kill of ours.
        if let Some(prev) = self.alive {
            for _ in o.alive..prev {
                self.drops.push_back(t);
            }
        }
        self.alive = Some(o.alive);
        while !self.drops.is_empty() && !self.kills.is_empty() {
            self.drops.pop_front();
            self.kills.pop_front();
        }
        while self.kills.front().map_or(false, |k| t - k > 1.5) {
            self.kills.pop_front();
        }
        while self.drops.front().map_or(false, |d| t - d > 0.6) {
            let d = self.drops.pop_front().unwrap();
            let ours_lately = self.dealt.iter().any(|x| d - x.0 <= 1.0 && x.0 - d <= 0.6);
            if t - self.gap_t > 1.5 && !ours_lately {
                let n = self.deaths.len();
                self.deaths.push_back((d, None));
                let pts = 1.0;
                if std::env::var("TFFA").is_ok() {
                    eprintln!("FFA death t {:.2} drop {:.2} dealt {:?} gap {:.2} zone r {:.0} dps {:.0}", t, d, self.dealt, self.gap_t, o.zone.radius, o.zone.dps);
                }
                // One shared key: deaths nobody we saw explains (the zone, say) can't settle it on their own.
                let _ = n;
                self.clue(t, "death".into(), pts);
            }
        }
        while self.deaths.len() > 16 {
            self.deaths.pop_front();
        }

        // A shot between two mechs followed (or preceded) within 0.6 s by the one hit losing hit points that
        // nothing else explains settles it; one that never is worth little.
        let mut k = 0;
        while k < self.shots.len() {
            let (st, key, on, alone) = self.shots[k].clone();
            if self.lost.iter().any(|l| l.1 == on && (l.0 - st).abs() <= 0.6) {
                self.clue(t, key, 2.0);
                self.shots.remove(k);
            } else if t - st > 0.7 {
                if alone > 0.0 {
                    self.clue(t, key, alone);
                }
                self.shots.remove(k);
            } else {
                k += 1;
            }
        }
        while self.lost.front().map_or(false, |l| t - l.0 > 1.5) {
            self.lost.pop_front();
        }
        self.clues.retain(|c| t - c.t < KEEP_S);
        while self.fights.front().map_or(false, |f| t - f.t > FIGHT_S) {
            self.fights.pop_front();
        }
        if !self.latched && after_open_play() {
            self.latched = true;
            self.latch_t = t;
            self.reason = "clock".into();
        }
        if !self.latched && self.total() >= LATCH - 1e-3 {
            self.latched = true;
            self.latch_t = t;
            self.reason = self.clues.iter().map(|c| c.key.as_str()).collect::<Vec<_>>().join(" ");
        }
    }

    /// The latest fight between two named mechs within `s` seconds, per pair: (time, shooter, hit).
    pub fn pairs(&self, now: f32, s: f32) -> Vec<(f32, u32, u32)> {
        let mut v: Vec<(f32, u32, u32)> = Vec::new();
        for f in self.fights.iter().rev().filter(|f| now - f.t <= s) {
            let Some(by) = f.by else { continue };
            if by == f.on || v.iter().any(|x| (x.1 == by && x.2 == f.on) || (x.1 == f.on && x.2 == by)) {
                continue;
            }
            v.push((f.t, by, f.on));
        }
        v
    }

    /// Whom `id` was last seen fighting within `s` seconds.
    pub fn fighting(&self, id: u32, now: f32, s: f32) -> Option<u32> {
        self.fights.iter().rev().filter(|f| now - f.t <= s).find_map(|f| match f.by {
            Some(b) if b == id => Some(f.on),
            Some(b) if f.on == id => Some(b),
            _ => None,
        })
    }
}
