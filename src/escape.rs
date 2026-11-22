//! Enemy profiles, escaping a crowd and dueling a lone enemy (each behind its own parameter).
use super::{d2, norm2, wrap, yaw_to, Brain};
use crate::contacts::{Belief, WS_CHARGING};
use crate::proto::Obs;
use crate::sight::{Foe, Zone};
use std::f32::consts::PI;

impl Brain {
    /// Fill in the brain's part of every believed enemy's profile: coming for us, alone, behind us, cornered.
    pub(super) fn profile(&mut self, o: &Obs, pos: &[f32; 3]) {
        let here = [pos[0], pos[1]];
        if self.me_hist.back().map_or(true, |h| o.t - h.0 >= 0.5 - 1e-3) {
            self.me_hist.push_back((o.t, here));
        }
        while self.me_hist.front().map_or(false, |h| o.t - h.0 > 5.0) {
            self.me_hist.pop_front();
        }
        let aim = o.me.aim[0];
        let n = self.beliefs.len();
        for i in 0..n {
            let b = self.beliefs[i];
            let d = d2(&b.pos, &here);
            let mut hunting = b.hunter;
            let by_path = self.params.inf_hunt > 0.0;
            if let Some(k) = b.id.and_then(|id| self.contacts.tracks.get(&id)) {
                // Its path over the last few seconds: most of it (at least 6 m over 2 s or more) spent closing
                // on where we were at each step.
                if by_path {
                    let near_me = |t: f32| self.me_hist.iter().min_by(|x, y| (x.0 - t).abs().partial_cmp(&(y.0 - t).abs()).unwrap_or(std::cmp::Ordering::Equal)).map_or(here, |h| h.1);
                    let (mut path, mut closing) = (0.0f32, 0.0f32);
                    for w in k.hist.iter().collect::<Vec<_>>().windows(2) {
                        let me0 = near_me(w[0].0);
                        path += d2(&w[0].1, &w[1].1);
                        closing += d2(&w[0].1, &me0) - d2(&w[1].1, &me0);
                    }
                    let span = k.hist.back().map_or(0.0, |l| l.0) - k.hist.front().map_or(0.0, |f| f.0);
                    if span >= 2.0 - 1e-3 && path >= 6.0 && closing >= 0.6 * path {
                        hunting = true;
                    }
                }
                // Closer to us than 3 to 5 s ago by at least 8 m (counting how we moved meanwhile).
                if let Some(&(t0, e0)) = k.hist.iter().find(|h| !by_path && o.t - h.0 >= 3.0 - 1e-3) {
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
            let alone = !self.beliefs.iter().enumerate().any(|(j, c)| j != i && c.conf >= 0.3 && d2(&c.pos, &b.pos) < 80.0);
            let behind = b.conf >= 0.3 && d < 60.0 && wrap(yaw_to(&here, &b.pos) - aim).abs() > PI / 3.0;
            let cornered = b.r < 15.0 && self.sight.as_ref().map_or(false, |st| st.cornered_at(b.pos));
            let e = &mut self.beliefs[i];
            e.hunting = hunting;
            e.alone = alone;
            e.behind = behind;
            e.cornered = cornered;
        }
    }

    /// Start, keep or end a duel. Returns the enemy dueled (its id once it has been listed; an enemy known only
    /// from what we heard has none) and where it is believed to be.
    pub(super) fn update_duel(&mut self, o: &Obs, pos: &[f32; 3], hp_frac: f32, late: bool) -> Option<(Option<u32>, [f32; 2])> {
        let p = &self.params;
        let here = [pos[0], pos[1]];
        let my_hp = o.me.hp;
        if let Some(at) = self.duel_at {
            // Follow it: its track by id, else the believed enemy nearest to where it was; an enemy listed close
            // to where an unnamed one was believed is taken to be it.
            let cur = match self.duel {
                Some(id) => self.beliefs.iter().find(|b| b.id == Some(id) && b.conf >= 0.2).copied(),
                None => self
                    .beliefs
                    .iter()
                    .filter(|b| b.conf >= 0.2 && d2(&b.pos, &at) < 40.0 + b.r)
                    .min_by(|x, y| d2(&x.pos, &at).partial_cmp(&d2(&y.pos, &at)).unwrap_or(std::cmp::Ordering::Equal))
                    .copied(),
            };
            if let Some(b) = cur {
                self.duel_at = Some(b.pos);
                if self.duel.is_none() && b.listed {
                    self.duel = b.id;
                }
                if b.listed {
                    self.duel_seen_t = o.t;
                }
            }
            let me_id = self.duel;
            let at = self.duel_at.unwrap_or(at);
            let crowd = self.beliefs.iter().any(|b| (me_id.is_none() || b.id != me_id) && d2(&b.pos, &at) > 25.0 && b.conf >= 0.5 && d2(&b.pos, &here) < p.duel_r - 20.0);
            // Help on its way: another enemy could be shooting at us before this one is dead, and it is not
            // nearly dead.
            let help = p.duel_eta > 0.0
                && cur.map_or(false, |b| {
                    let rem = self.duel_ttk(o, &b, pos);
                    let nearly = rem < 2.0 || b.est_hp < 0.25 * b.max_hp.max(1.0);
                    !nearly && self.beliefs.iter().any(|c| c.conf >= 0.3 && (me_id.is_none() || c.id != me_id) && d2(&c.pos, &at) > 25.0 && c.eta < rem)
                });
            let keep = !help
                && cur.map_or(false, |b| d2(&b.pos, &here) < p.duel_r + 40.0)
                && hp_frac >= 0.45
                && !crowd
                && !late
                && !self.esc_on
                && o.t - self.duel_seen_t < 12.0
                && o.t - self.duel_start < 45.0;
            if !keep {
                self.duel = None;
                self.duel_at = None;
                if help {
                    self.duel_abort_n += 1;
                    if self.trace_modes {
                        eprintln!("MODE duel_abort t {:.1} hp {:.0}", o.t, o.me.hp);
                    }
                }
            }
        }
        if self.duel_at.is_none() && hp_frac >= p.duel_hp && !late && !self.esc_on {
            // Named enemies, or unnamed ones heard often enough to be sure of.
            // With `inf_hp` weighed on the estimate plus half its uncertainty; with `inf_weapon` an enemy whose
            // weapons are down may have a little more, and one charging its laser is left for now.
            let hp_of = |b: &Belief| b.est_hp + if p.inf_hp > 0.0 { 0.5 * b.hp_sd } else { 0.0 };
            let ratio = |b: &Belief| p.duel_ratio * if p.inf_weapon > 0.0 && b.open_s > 0.3 { 1.15 } else { 1.0 };
            let hunters: Vec<[f32; 2]> = self.beliefs.iter().filter(|c| c.hunting && c.conf >= 0.3).map(|c| c.pos).collect();
            let score = |b: &Belief| {
                let mut s = d2(&b.pos, &here) + 0.1 * b.est_hp;
                if p.inf_danger > 0.0 {
                    s += 150.0 * b.danger;
                }
                if p.inf_hunt > 0.0 {
                    let friends = hunters.iter().any(|h| d2(h, &b.pos) > 1.0 && d2(h, &b.pos) < 150.0);
                    s += if b.hunting && friends { 100.0 } else if !b.hunting && !friends { -40.0 } else { 0.0 };
                }
                s
            };
            let best = self
                .beliefs
                .iter()
                .filter(|b| b.alone && b.conf >= if b.id.is_some() { 0.5 } else { 0.6 } && d2(&b.pos, &here) < p.duel_r && (hp_of(b) <= my_hp * ratio(b) || b.cornered))
                .filter(|b| !self.beliefs.iter().any(|c| c != *b && c.conf >= 0.3 && d2(&c.pos, &here) < p.duel_r))
                .filter(|b| p.inf_danger <= 0.0 || b.danger < p.duel_danger || hp_of(b) <= my_hp * 0.5)
                .filter(|b| p.inf_weapon <= 0.0 || b.wstate != WS_CHARGING)
                .filter(|b| {
                    p.duel_eta <= 0.0 || {
                        let eta = self.beliefs.iter().filter(|c| c != b && c.conf >= 0.3).map(|c| c.eta).fold(f32::INFINITY, f32::min);
                        self.duel_ttk(o, b, pos) + p.duel_eta_margin < eta
                    }
                })
                .min_by(|x, y| score(x).partial_cmp(&score(y)).unwrap_or(std::cmp::Ordering::Equal))
                .copied();
            if let Some(b) = best {
                self.duel = b.id;
                self.duel_at = Some(b.pos);
                self.duel_start = o.t;
                self.duel_seen_t = o.t;
                self.duel_n += 1;
                if self.trace_modes {
                    eprintln!("MODE duel t {:.1} id {:?} hp {:.0}", o.t, b.id, o.me.hp);
                }
            }
        }
        self.duel_at.map(|at| (self.duel, at))
    }

    /// Believed enemies as foes for choosing a place.
    pub(super) fn belief_foes(&self, pos: &[f32; 3]) -> Vec<Foe> {
        self.beliefs
            .iter()
            .filter(|b| b.conf >= 0.1 && d2(&b.pos, pos) < 260.0 + b.r)
            .map(|b| Foe { pos: b.pos, hp: b.hp, r: b.r, conf: b.conf, sure: b.listed })
            .collect()
    }

    /// Start, keep or end an escape; returns where to go (None: no escape now, or nowhere better than here).
    /// `veto`: a fight about to be won or the late game.
    pub(super) fn update_escape(&mut self, o: &Obs, pos: &[f32; 3], hp_frac: f32, veto: bool) -> Option<[f32; 2]> {
        let p = self.params.clone();
        let here = [pos[0], pos[1]];
        let hunters: f32 = self.beliefs.iter().filter(|b| b.hunting && b.conf >= 0.3 && d2(&b.pos, &here) - 0.5 * b.r < p.esc_r).map(|b| b.conf).sum();
        let ambush = self.hit_hist.iter().any(|x| self.hit_hist.iter().any(|y| x.1 != y.1 && wrap(x.2 - y.2).abs() > 0.6));
        let trig = hunters >= p.esc_hunters - 1e-3 || (hunters >= p.esc_hunters - 1.0 - 1e-3 && hp_frac < p.esc_low_hp) || ambush;
        let stop = |b: &mut Brain| {
            b.esc_on = false;
            b.esc_pick = None;
            b.esc_cool_until = o.t + p.esc_cool;
        };
        if veto {
            if self.esc_on {
                stop(self);
            }
            return None;
        }
        if self.esc_on {
            let quiet_s = o.t - self.last_mech_hurt_t.max(self.esc_start);
            if (hunters < 2.0 - 1e-3 && !ambush) || (hp_frac >= p.esc_heal && !trig) || quiet_s > p.esc_cap || o.t - self.esc_start > 30.0 {
                stop(self);
                return None;
            }
        } else if trig && o.t >= self.esc_cool_until {
            self.esc_on = true;
            self.esc_start = o.t;
            self.esc_pick = None;
            self.esc_pick_t = -100.0;
            self.esc_n += 1;
            if self.trace_modes {
                eprintln!("MODE esc t {:.1} hunters {hunters:.1} ambush {ambush} hp {:.0}", o.t, o.me.hp);
            }
        }
        if !self.esc_on {
            return None;
        }
        if o.t - self.esc_pick_t >= 1.5 {
            let (Some(st), Some(nav)) = (self.sight.clone(), self.nav.as_ref()) else { return None };
            let z = &o.zone;
            let zone = Zone { center: z.center, radius: z.radius, next_center: z.next_center, next_radius: z.next_radius, soon: o.t >= z.shrink_start - 15.0 };
            let foes = self.belief_foes(pos);
            let keep = self.esc_pick.map(|k| k.pos);
            self.esc_pick = st.escape_pick(nav, here, &foes, &zone, o.me.stat("walk_speed", 11.7), keep);
            self.esc_pick_t = o.t;
        }
        self.esc_pick.map(|k| k.pos)
    }

    /// Expected number of believed enemies with a line to where we stand.
    pub(super) fn watched_here(&self, pos: &[f32; 3]) -> f32 {
        match self.sight.as_ref() {
            Some(st) => st.expected([pos[0], pos[1]], &self.belief_foes(pos)),
            None => 0.0,
        }
    }
}
