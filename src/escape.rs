//! Enemy profiles, escaping a crowd and dueling a lone enemy (each behind its own parameter).
use super::{d2, norm2, wrap, yaw_to, Brain};
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

    /// Start, keep or end a duel. Returns the enemy dueled.
    pub(super) fn update_duel(&mut self, o: &Obs, pos: &[f32; 3], hp_frac: f32, late: bool) -> Option<u32> {
        let p = &self.params;
        let here = [pos[0], pos[1]];
        let my_hp = o.me.hp;
        if o.mechs.iter().any(|m| Some(m.id) == self.duel) {
            self.duel_seen_t = o.t;
        }
        if let Some(id) = self.duel {
            let cur = self.beliefs.iter().find(|b| b.id == Some(id) && b.conf >= 0.2);
            let crowd = self.beliefs.iter().any(|b| b.id != Some(id) && b.conf >= 0.5 && d2(&b.pos, &here) < p.duel_r - 20.0);
            let keep = cur.map_or(false, |b| d2(&b.pos, &here) < p.duel_r + 40.0)
                && hp_frac >= 0.45
                && !crowd
                && !late
                && !self.esc_on
                && o.t - self.duel_seen_t < 10.0
                && o.t - self.duel_start < 45.0;
            if !keep {
                self.duel = None;
            }
        }
        if self.duel.is_none() && hp_frac >= p.duel_hp && !late && !self.esc_on {
            let best = self
                .beliefs
                .iter()
                .filter(|b| b.id.is_some() && b.alone && b.conf >= 0.5 && d2(&b.pos, &here) < p.duel_r && (b.est_hp <= my_hp || b.cornered))
                .filter(|b| !self.beliefs.iter().any(|c| c.id != b.id && c.conf >= 0.3 && d2(&c.pos, &here) < p.duel_r))
                .min_by(|x, y| (d2(&x.pos, &here) + 0.1 * x.est_hp).partial_cmp(&(d2(&y.pos, &here) + 0.1 * y.est_hp)).unwrap_or(std::cmp::Ordering::Equal))
                .and_then(|b| b.id);
            if let Some(id) = best {
                self.duel = Some(id);
                self.duel_start = o.t;
                self.duel_seen_t = o.t;
                self.duel_n += 1;
                if self.trace_modes {
                    eprintln!("MODE duel t {:.1} id {id} hp {:.0}", o.t, o.me.hp);
                }
            }
        }
        self.duel
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
