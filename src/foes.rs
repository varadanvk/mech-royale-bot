//! What each enemy is like and what it can do next: how dangerous it has shown itself to be, the state of its
//! weapons, how soon it could be shooting at us, and how long we need to kill one we duel.
use super::{d2, wrap, yaw_to, Brain};
use crate::contacts::{Belief, W_LASER, W_LIGHTNING, W_MORTAR, WS_CHARGING, WS_COOLDOWN, WS_READY, WS_RELOAD};
use crate::proto::Obs;

/// Most believed enemies described for a viewer.
const DBG_MAX: usize = 16;

#[derive(Clone, Debug)]
pub(super) struct FoeStat {
    /// Last update while listed, and when it was first listed (None: not yet).
    last_t: f32,
    first_t: Option<f32>,
    /// Seconds listed within its reach of us with its torso on us; its hits on us.
    aimed_s: f32,
    hits: f32,
    /// Seconds from when we first listed it to its first hit on us.
    react: Option<f32>,
    /// While fighting us: aim error (time-weighted sum, seconds), speed across the line to us (same).
    err_sum: f32,
    err_s: f32,
    lat_sum: f32,
    lat_s: f32,
    /// Gun shots since its last reload; until when it reloads; until when its laser cools down; last time it was
    /// seen or heard charging; last laser shot already counted.
    shots: f32,
    reload_until: f32,
    laser_cd_until: f32,
    charge_t: f32,
    laser_noted: f32,
}

impl FoeStat {
    fn new() -> FoeStat {
        FoeStat {
            last_t: f32::NEG_INFINITY,
            first_t: None,
            aimed_s: 0.0,
            hits: 0.0,
            react: None,
            err_sum: 0.0,
            err_s: 0.0,
            lat_sum: 0.0,
            lat_s: 0.0,
            shots: 0.0,
            reload_until: f32::NEG_INFINITY,
            laser_cd_until: f32::NEG_INFINITY,
            charge_t: f32::NEG_INFINITY,
            laser_noted: f32::NEG_INFINITY,
        }
    }

    /// 0..1 from easy to expert: hits on us against what its weapon could land while aimed at us, how fast it
    /// started hitting us after we first saw it, how closely its aim follows us and how much it moves across our
    /// line while fighting. Each piece is weighed by how much was seen of it, against a prior of 0.5 worth two
    /// pieces.
    pub(super) fn danger(&self, weapon: u8) -> f32 {
        let (mut num, mut den) = (1.0f32, 2.0f32);
        let ideal = match weapon {
            W_LASER => 0.42,
            W_LIGHTNING => 4.0,
            W_MORTAR => 0.4,
            _ => 2.27,
        };
        if self.aimed_s >= 1.0 {
            let acc = (self.hits / (self.aimed_s * ideal) / 0.5).clamp(0.0, 1.0);
            let w = (self.aimed_s / 3.0).min(2.0);
            num += acc * w;
            den += w;
        }
        if let Some(r) = self.react {
            num += (1.0 - (r - 0.4) / 1.6).clamp(0.0, 1.0);
            den += 1.0;
        }
        if self.err_s >= 0.5 {
            let e = self.err_sum / self.err_s;
            let w = (self.err_s / 2.0).min(1.5);
            num += (1.0 - e / 0.3).clamp(0.0, 1.0) * w;
            den += w;
        }
        if self.lat_s >= 0.5 {
            let l = self.lat_sum / self.lat_s;
            let w = (self.lat_s / 2.0).min(1.0);
            num += (l / 7.0).clamp(0.0, 1.0) * w;
            den += w;
        }
        num / den
    }

    /// Weapon state now, and for how many more seconds every weapon it holds that reaches far is down (0 when
    /// one is ready). Mortars are taken to be ready; holding nothing known means the gun.
    fn weapon_state(&self, weapons: &[String], t: f32) -> (u8, f32) {
        if t - self.charge_t < 0.2 {
            return (WS_CHARGING, 0.0);
        }
        let has = |k: &str| weapons.iter().any(|w| w == k);
        let gun = weapons.is_empty() || has("gun");
        let gun_down = if gun { (self.reload_until - t).max(0.0) } else { f32::INFINITY };
        let laser_down = if has("laser") { (self.laser_cd_until - t).max(0.0) } else { f32::INFINITY };
        let mut open = gun_down.min(laser_down);
        if has("mortar") || !open.is_finite() {
            open = 0.0;
        }
        let state = if gun && gun_down > 0.0 {
            WS_RELOAD
        } else if laser_down > 0.0 && laser_down.is_finite() {
            WS_COOLDOWN
        } else {
            WS_READY
        };
        (state, open)
    }

    fn shot(&mut self, t: f32, mag: f32, reload: f32) {
        if t < self.reload_until {
            return;
        }
        self.shots += 1.0;
        if self.shots >= mag - 0.5 {
            self.reload_until = t + reload;
            self.shots = 0.0;
        }
    }
}

/// How far an enemy's weapons shoot.
fn reach_of(weapons: &[String]) -> f32 {
    weapons
        .iter()
        .map(|w| match w.as_str() {
            "laser" => 280.0,
            "mortar" => 235.0,
            "lightning" => 45.0,
            _ => 230.0,
        })
        .fold(if weapons.is_empty() { 230.0 } else { 0.0 }, f32::max)
}

impl Brain {
    /// Learn from this observation how each enemy fights and what its weapons are doing, then fill in each
    /// believed enemy's danger, weapon state and how soon it could be shooting at us (`Belief::eta`).
    pub(super) fn judge(&mut self, o: &Obs, pos: &[f32; 3]) {
        let here = [pos[0], pos[1]];
        let (mag, reload) = (self.foe_gun[0], self.foe_gun[1]);
        let cd = self.foe_laser_cd;
        for m in o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0)) {
            let hurt = self.attackers.get(&m.id).map_or(false, |t| o.t - t < 3.0);
            let st = self.foes.entry(m.id).or_insert_with(FoeStat::new);
            let dt = (o.t - st.last_t).clamp(0.0, 0.2);
            st.last_t = o.t;
            if st.first_t.is_none() {
                st.first_t = Some(o.t);
            }
            let d = d2(&m.pos, pos);
            let to_me = yaw_to(&m.pos, pos);
            let err = wrap(m.aim[0] - to_me).abs();
            if d < reach_of(&m.weapons) + 10.0 {
                if err < 0.2 {
                    st.aimed_s += dt;
                }
                if hurt || err < 0.5 {
                    st.err_sum += err.min(0.6) * dt;
                    st.err_s += dt;
                    let perp = [-to_me.sin(), to_me.cos()];
                    st.lat_sum += (m.vel[0] * perp[0] + m.vel[1] * perp[1]).abs().min(15.0) * dt;
                    st.lat_s += dt;
                }
            }
            if m.charging {
                st.charge_t = o.t;
            }
        }
        for e in o.events.iter().filter(|e| e.kind == "damaged") {
            let Some(by) = e.by.filter(|b| *b < 100) else { continue };
            let listed = o.mechs.iter().any(|m| m.id == by);
            let st = self.foes.entry(by).or_insert_with(FoeStat::new);
            st.hits += 1.0;
            if st.react.is_none() {
                if let Some(f) = st.first_t {
                    st.react = Some((o.t - f).min(4.0));
                }
            }
            match e.weapon.as_deref() {
                Some("laser") => st.laser_cd_until = st.laser_cd_until.max(o.t + cd),
                // A listed shooter's shots are counted from what we hear.
                Some("gun") if !listed => st.shot(o.t, mag, reload),
                _ => {}
            }
        }
        for b in o.beams.iter().filter(|b| b.kind == "laser" && b.by < 100) {
            let st = self.foes.entry(b.by).or_insert_with(FoeStat::new);
            st.laser_cd_until = st.laser_cd_until.max(o.t - b.age + cd);
        }
        for (id, t) in self.laser_fired.iter() {
            if let Some(st) = self.foes.get_mut(id) {
                if *t > st.laser_noted {
                    st.laser_noted = *t;
                    st.laser_cd_until = st.laser_cd_until.max(*t + cd);
                }
            }
        }
        // Weapon sounds put on the listed enemy they came from (bearing and distance band).
        for s in &o.sounds {
            if !matches!(s.kind.as_str(), "gun" | "reload" | "laser" | "laser_charge") {
                continue;
            }
            let (lo, hi) = match s.dist.as_str() {
                "near" => (0.0, 60.0),
                "mid" => (60.0, 160.0),
                _ => (160.0, 1000.0),
            };
            let from = o
                .mechs
                .iter()
                .filter(|m| m.kind == "mech")
                .map(|m| (m.id, d2(&m.pos, pos), wrap(yaw_to(pos, &m.pos) - s.bearing).abs()))
                .filter(|x| x.1 > lo - 10.0 && x.1 < hi + 10.0 && x.2 < 0.3)
                .min_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal))
                .map(|x| x.0);
            let Some(id) = from else { continue };
            let t = o.t - s.age.max(0.0);
            let st = self.foes.entry(id).or_insert_with(FoeStat::new);
            match s.kind.as_str() {
                "gun" => st.shot(t, mag, reload),
                "reload" => {
                    st.reload_until = st.reload_until.max(t + reload);
                    st.shots = 0.0;
                }
                "laser" => st.laser_cd_until = st.laser_cd_until.max(t + cd),
                _ => st.charge_t = st.charge_t.max(t),
            }
        }
        let alive: Vec<u32> = self.contacts.tracks.keys().copied().collect();
        self.foes.retain(|id, _| alive.contains(id));

        let p = &self.params;
        let speed = p.duel_eta_speed.max(1.0);
        let reach_cap = p.duel_eta_reach;
        for b in self.beliefs.iter_mut() {
            let k = b.id.and_then(|id| self.contacts.tracks.get(&id));
            let weapons: &[String] = k.map_or(&[], |k| k.weapons.as_slice());
            let st = b.id.and_then(|id| self.foes.get(&id));
            b.danger = st.map_or(0.5, |s| s.danger(b.weapon));
            let (ws, open) = st.map_or((WS_READY, 0.0), |s| s.weapon_state(weapons, o.t));
            b.wstate = ws;
            b.open_s = open;
            let reach = reach_of(weapons).min(reach_cap);
            let gap = (d2(&b.pos, &here) - 0.5 * b.r - reach).max(0.0);
            b.eta = gap / speed * if b.hunting { 1.0 } else { 2.0 } / b.conf.clamp(0.3, 1.0);
        }
        self.dbg_enemies = self
            .beliefs
            .iter()
            .take(DBG_MAX)
            .map(|b| [b.id.map_or(-1.0, |i| i as f32), b.est_hp, b.hp_sd, b.danger, b.eta.min(999.0), b.wstate as f32, b.hunting as u8 as f32, b.open_s])
            .collect();
    }

    /// Our damage per second against an enemy `d` m away once in range (weapon rates with reloads and charge
    /// times, cubes, times `duel_acc`), the longest reach among our weapons, and how long until one can fire.
    pub(super) fn our_dps(&self, o: &Obs, d: f32) -> (f32, f32, f32) {
        let (mut dps, mut range, mut wait) = (0.0f32, 0.0f32, f32::INFINITY);
        for w in o.me.weapons.iter().flatten() {
            let (rate, r, down) = match w.kind.as_str() {
                "gun" => {
                    let (dmg, cd, mag, rl) = (w.stat("damage", 45.0), w.stat("cooldown", 0.24), w.stat("mag", 12.0), w.stat("reload", 2.4));
                    let down = if w.reload > 0.0 { w.reload } else if w.ammo <= 0.0 { rl } else { w.cooldown };
                    (dmg * mag / (mag * cd + rl).max(0.1), w.stat("range", 230.0), down)
                }
                "laser" => (w.stat("damage", 315.0) / (w.stat("charge", 0.75) + w.stat("cooldown", 1.65)).max(0.1), w.stat("range", 280.0), w.cooldown + w.stat("charge", 0.75)),
                "mortar" => (0.6 * w.stat("damage", 200.0) / w.stat("cooldown", 2.0).max(0.3), w.stat("range", 235.0).min(235.0), w.cooldown),
                "lightning" if d < 60.0 => (w.stat("damage", 60.0) / w.stat("cooldown", 0.25).max(0.05), 0.0, w.cooldown),
                _ => continue,
            };
            dps += rate;
            range = range.max(r);
            wait = wait.min(down.max(0.0));
        }
        let cubes = 1.0 + self.dmg_cube[0] * o.me.cubes.min(self.dmg_cube[1]);
        ((dps * cubes * self.params.duel_acc).max(10.0), range.max(45.0), if wait.is_finite() { wait } else { 0.0 })
    }

    /// Seconds we need to kill believed enemy `b`: closing to within 80 % of our reach, waiting for a weapon,
    /// then its hit points (with `inf_hp`, plus half their uncertainty) at our damage per second.
    pub(super) fn duel_ttk(&self, o: &Obs, b: &Belief, pos: &[f32; 3]) -> f32 {
        let d = d2(&b.pos, pos);
        let (dps, range, wait) = self.our_dps(o, d);
        let hp = b.est_hp + if self.params.inf_hp > 0.0 { 0.5 * b.hp_sd } else { 0.0 };
        (d - range * 0.8).max(0.0) / o.me.stat("walk_speed", 11.7) + wait + hp / dps
    }
}
