//! Decision logic: one `act` per observation.
use crate::nav::Nav;
use crate::proto::{Action, Actor, Hello, Obs, Weapon};
use crate::world::World;
use std::collections::{BTreeMap, VecDeque};
use std::f32::consts::{PI, TAU};
use std::sync::Arc;

fn wrap(a: f32) -> f32 {
    let mut a = (a + PI) % TAU;
    if a < 0.0 {
        a += TAU;
    }
    a - PI
}
fn d2(a: &[f32], b: &[f32]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}
fn d3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}
fn norm2(x: f32, y: f32) -> [f32; 2] {
    let l = (x * x + y * y).sqrt();
    if l < 1e-6 || !l.is_finite() {
        [0.0, 0.0]
    } else {
        [x / l, y / l]
    }
}
fn yaw_to(from: &[f32], to: &[f32]) -> f32 {
    (to[1] - from[1]).atan2(to[0] - from[0])
}

/// Tunable thresholds. Every field has a default; `Params::from_json` overrides any subset.
#[derive(Clone, Debug)]
pub struct Params {
    /// Start breaking contact below this HP fraction while a threat is around.
    pub retreat_hp: f32,
    /// Stay in hiding until HP is back above this fraction.
    pub reengage_hp: f32,
    /// Preferred distance when fighting with the gun.
    pub gun_range: f32,
    /// Half-width of the distance band we accept around `gun_range` before moving radially.
    pub gun_band: f32,
    /// Preferred distance when closing in with lightning.
    pub arc_range: f32,
    /// Only close in with lightning above this HP fraction.
    pub arc_min_hp: f32,
    /// Enemies within this distance count toward "outnumbered".
    pub crowd_radius: f32,
    /// Disengage when at least this many enemies are within `crowd_radius`.
    pub crowd_count: usize,
    /// Do not start a fight beyond this distance unless shot at.
    pub open_fire_range: f32,
    /// Strafe direction dwell, random in [min, max] seconds.
    pub strafe_min: f32,
    pub strafe_max: f32,
    /// Chance that each new strafe dwell reverses direction (0.5 = random direction, 1 = always alternate).
    pub strafe_alt: f32,
    /// Seconds after an aimed laser charge is first seen at which we reverse direction (large = never).
    pub dodge_after: f32,
    /// Weight of the sideways weave (relative to the travel direction) when an enemy is in sight outside a fight.
    pub evade_weight: f32,
    /// Target choice: metres of distance one full HP bar is worth (higher = focus weak targets).
    pub focus_hp: f32,
    /// Keep fighting (and close in) when the target is below this HP fraction.
    pub finish_hp: f32,
    /// Distance to close to when finishing a target.
    pub finish_range: f32,
    /// When retreating with an enemy in sight: weight of backing away while weaving (0 = run to cover instead).
    pub kite: f32,
    /// Our own speed scale while fighting (0..1).
    pub fight_speed: f32,
    /// Farming (crates, drones, cubes) only above this HP fraction.
    pub farm_hp: f32,
    /// Shoot drones within this distance when farming.
    pub drone_range: f32,
    /// Go for supply drops within this distance.
    pub drop_range: f32,
    /// Skip a supply drop if a known enemy is within this distance of it.
    pub drop_enemy_clear: f32,
    /// Extra seconds of margin when rotating into the next circle.
    pub zone_lead: f32,
    /// Multiplier on the walking time to the next circle when deciding to rotate.
    pub zone_lead_mult: f32,
    /// Radius of idle wandering around the chosen post.
    pub wander_radius: f32,
    /// Seconds a known enemy position is kept for hiding decisions.
    pub memory_s: f32,
    /// Boost while retreating if energy is above this.
    pub retreat_boost_energy: f32,
    /// Keep shooting the current target unless another scores better by this margin (0 = no stickiness).
    pub focus_stick: f32,
    /// Before this match time, farm crates, drones and loot from further away.
    pub farm_until: f32,
    /// Reach for crates, cubes and drones while farming early.
    pub farm_reach: f32,
    /// With no enemy in sight, hide and heal below this HP fraction (0 = off).
    pub heal_lo: f32,
    /// Stay hidden until HP is back above this fraction.
    pub heal_hi: f32,
    /// Weight against fight positions that other enemies can see (0 = off).
    pub expose_weight: f32,
    /// Walk over to cubes within this distance even in a fight (0 = off).
    pub cube_grab: f32,
    /// Step out of sight while the gun reloads, up to this far (0 = off).
    pub reload_hide: f32,
    /// Step out of sight of a laser charging at us, up to this far (0 = off).
    pub laser_hide: f32,
    /// Seconds between sending an action and it taking effect, assumed when planning (0 = immediate).
    pub delay_s: f32,
    /// Estimate the delay from our own motion and use the estimate once it is clear (0 = off).
    pub delay_auto: f32,
    /// Gun lead: fraction of a grounded target's horizontal velocity to lead by.
    pub lead_ground: f32,
    /// Gun lead: fraction of an airborne target's horizontal velocity to lead by.
    pub lead_air: f32,
    /// Seconds added to the bullet flight time when leading.
    pub lead_latency: f32,
    /// Predict an airborne target's height with gravity or jet thrust, stopping at the surface below it
    /// (0 = lead by half its vertical speed).
    pub lead_ballistic: f32,
    /// Pull the gun trigger when the aim is within this many ticks of torso turn of the solution
    /// (the torso turns before the shot leaves; 0 = only when already on target).
    pub aim_slack: f32,
    /// Weapon handling: 0 = aim with the highest-ranked ready weapon; 1 = aim serves the gun while the
    /// second weapon fires whenever its own solution lines up (laser takes the aim only once fully charged).
    pub dual: f32,
    /// With a full laser charge, shift the aim from the laser solution toward the gun solution by up to
    /// this fraction of the laser tolerance, so both can hit.
    pub laser_mix: f32,
    /// Close in with lightning only while at most this many enemies are within `crowd_radius`.
    pub arc_crowd: f32,
    /// Close in with lightning only above this energy.
    pub arc_energy: f32,
    /// Rush a target with lightning (0 = off): close in and hold the arc and the gun on it.
    pub rush: f32,
    /// Start a rush on targets within this distance.
    pub rush_start: f32,
    /// Distance to hold during a rush.
    pub rush_range: f32,
    /// Rush only above this HP fraction (or when the target has less HP than us).
    pub rush_hp: f32,
    /// Rush only while at most this many other enemies are within `crowd_radius`.
    pub rush_crowd: f32,
    /// Start a rush above this energy; keep going while above `arc_energy`/2.
    pub rush_energy: f32,
    /// Boost toward the target during a rush while further than the arc range and energy is above this.
    pub rush_boost: f32,
    /// Prefer the laser over lightning as the second weapon.
    pub laser_first: f32,
    /// With an empty slot, go for the best known weapon pickup before anything but the zone (0 = off).
    pub arm: f32,
    /// Go for weapon pickups within this distance.
    pub arm_reach: f32,
    /// Only above this HP fraction.
    pub arm_hp: f32,
    /// Metres of extra distance one rank of weapon preference is worth when choosing a pickup.
    pub arm_rank: f32,
    /// Holding a weaker second weapon: go for a better one within this distance (0 = off).
    pub arm_better: f32,
    /// Upgrade order: 0 = survival first, 1 = damage first.
    pub upg_dmg: f32,
    /// Holding lightning: outside a rush, the zone or getting unstuck, boost only above this energy (0 = no limit).
    pub arc_save: f32,
    /// Preferred fight distance while holding a laser (0 = `gun_range`).
    pub laser_range: f32,
    /// Holding a laser or lightning: swap the gun for the other of the two (0 = keep the gun).
    pub drop_gun: f32,
    /// Charge the laser ahead of time when an enemy seen within the last 2 s is within this distance
    /// but not in view (0 = off).
    pub precharge: f32,
    /// Let the mortar take the aim whenever it is ready and the target is beyond this distance (0 = only when the gun cannot fire).
    pub mortar_take: f32,
    /// Do not start a laser charge (half speed) while running for cover or backing off (0 = off).
    pub flee_nocharge: f32,
    /// With nothing else to do, healthy and holding a laser, walk toward the freshest seen or heard
    /// enemy within this distance (0 = off).
    pub hunt: f32,
    /// Hunt only above this HP fraction.
    pub hunt_hp: f32,
    /// Never pick up a mortar (keep the slot for a laser or lightning).
    pub no_mortar: f32,
    /// Without a laser: go for a known laser on the ground within this distance even with both slots full (0 = off).
    pub laser_seek: f32,
    /// Without a laser and nothing else to do: walk toward the nearest part of the map not looked at yet (0 = off).
    pub explore: f32,
    /// Map cells within this distance of us count as looked at.
    pub explore_r: f32,
    /// Also go for weapons on roofs, jumping and jetting up the wall next to them (0 = ground only).
    pub roof: f32,
    /// Extra metres a roof pickup counts as when choosing one.
    pub roof_cost: f32,
    /// Start a climb only above this energy.
    pub roof_energy: f32,
    /// Without a laser: metres of target preference for an enemy holding one (its laser drops when it dies).
    pub loot_target: f32,
    /// Rank the mortar above lightning (with `laser_first`: laser, mortar, lightning).
    pub mortar_over_arc: f32,
    /// Release a partial laser charge once it is enough to kill the target (0 = off).
    pub laser_finish: f32,
    /// Seconds of target motion to lead the laser by (the beam leaves on a later tick).
    pub laser_lead: f32,
    /// Reload once at most this many rounds are left while the laser is charging (0 = only when empty).
    pub reload_early: f32,
    /// Step out of sight to reload even when the laser could fire meanwhile (0 = keep the laser on target).
    pub reload_laser: f32,
    /// Retreat: 0 = back away while shooting; 1 = break line of sight from every known enemy and heal out of sight.
    pub retreat_mode: f32,
    /// Also break line of sight (instead of backing away) when outnumbered at full health.
    pub break_crowd: f32,
    /// While breaking line of sight, start laser charges (half speed) only if the cover is closer than this many metres (large = always).
    pub break_laser: f32,
    /// Start retreating when at least this many enemies are in view and HP is below `break_los_hp` (0 = off).
    pub break_los: f32,
    /// HP fraction for `break_los`.
    pub break_los_hp: f32,
    /// After a kill, go and pick up the cubes it dropped (0 = off).
    pub collect: f32,
    /// For this many seconds after the kill.
    pub collect_s: f32,
    /// Only above this HP fraction.
    pub collect_hp: f32,
    /// Only while fewer than this many enemies are in view.
    pub collect_seen: f32,
    /// Metres of preference for the weapon pickup we are already walking to.
    pub arm_keep: f32,
    /// From this match time, stop roaming and fights: hold a spot that few approaches can see (0 = off).
    pub wall_t: f32,
    /// Or once the zone radius is at most this.
    pub wall_r: f32,
    /// 1 = hold a low-exposure spot inside the next circle; 2 = keep away from the known enemies inside the zone.
    pub wall_mode: f32,
    /// Keep every enemy in sight at least this far away, backing off from all of them while shooting (0 = off).
    pub standoff: f32,
    /// Weight of the sideways weave while backing off.
    pub stand_weave: f32,
    /// Boost while backing off above this energy.
    pub stand_boost: f32,
    /// Let enemies below this HP fraction come closer.
    pub stand_finish: f32,
    /// Only back off from enemies seen within this many seconds.
    pub stand_mem: f32,
    /// While backing off, start laser charges (half speed) only when the nearest enemy is at least this far (0 = always).
    pub stand_charge: f32,
    /// Backing off comes before fetching weapons and collecting cubes (0 = after).
    pub stand_first: f32,
    /// Keep backing off while moving into the next circle (if still this many metres inside the current one),
    /// with this weight pulling toward the next circle (0 = off: the zone comes first).
    pub stand_zone: f32,
    pub stand_zone_m: f32,
    /// When backing away from a fight we are losing, move away from every enemy seen recently rather than from the target alone.
    pub kite_away: f32,
    /// After the laser fires, step out of sight up to this far, charge it there and step back out (0 = off).
    pub peek: f32,
    /// Give up waiting out of sight after this many seconds.
    pub peek_s: f32,
    /// Only while at most this many enemies are in sight.
    pub peek_crowd: f32,
    /// When the line to the target's centre is blocked, aim at its upper body if that line is clear (0 = off).
    pub aim_high: f32,
    /// Start a laser charge on any enemy in sight even without a clear line yet (0 = off).
    pub charge_seen: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            retreat_hp: 0.55,
            reengage_hp: 0.92,
            gun_range: 40.0,
            gun_band: 10.0,
            arc_range: 30.0,
            arc_min_hp: 0.6,
            crowd_radius: 150.0,
            crowd_count: 2,
            open_fire_range: 200.0,
            strafe_min: 0.3,
            strafe_max: 0.6,
            strafe_alt: 1.0,
            dodge_after: 9.0,
            evade_weight: 0.8,
            focus_hp: 120.0,
            kite: 1.0,
            finish_hp: 0.3,
            finish_range: 45.0,
            fight_speed: 1.0,
            farm_hp: 0.7,
            drone_range: 80.0,
            drop_range: 170.0,
            drop_enemy_clear: 70.0,
            zone_lead: 10.0,
            zone_lead_mult: 1.6,
            wander_radius: 14.0,
            memory_s: 12.0,
            retreat_boost_energy: 15.0,
            focus_stick: 50.0,
            farm_until: 0.0,
            farm_reach: 160.0,
            heal_lo: 0.8,
            heal_hi: 0.97,
            expose_weight: 1.0,
            cube_grab: 45.0,
            reload_hide: 16.0,
            laser_hide: 8.0,
            delay_s: 0.0,
            delay_auto: 1.0,
            lead_ground: 0.9,
            lead_air: 1.0,
            lead_latency: 0.025,
            lead_ballistic: 1.0,
            aim_slack: 1.0,
            dual: 1.0,
            laser_mix: 0.5,
            arc_crowd: 1.0,
            arc_energy: 20.0,
            rush: 1.0,
            rush_start: 90.0,
            rush_range: 22.0,
            rush_hp: 0.45,
            rush_crowd: 1.0,
            rush_energy: 40.0,
            rush_boost: 50.0,
            laser_first: 1.0,
            arm: 1.0,
            arm_reach: 160.0,
            arm_hp: 0.35,
            arm_rank: 25.0,
            arm_better: 60.0,
            upg_dmg: 1.0,
            arc_save: 0.0,
            laser_range: 0.0,
            drop_gun: 0.0,
            precharge: 0.0,
            mortar_take: 0.0,
            flee_nocharge: 0.0,
            hunt: 0.0,
            hunt_hp: 0.8,
            no_mortar: 0.0,
            laser_seek: 0.0,
            explore: 0.0,
            explore_r: 110.0,
            roof: 0.0,
            roof_cost: 40.0,
            roof_energy: 55.0,
            loot_target: 0.0,
            mortar_over_arc: 0.0,
            laser_finish: 1.0,
            laser_lead: 0.0,
            reload_early: 0.0,
            reload_laser: 1.0,
            retreat_mode: 0.0,
            break_crowd: 0.0,
            break_laser: 1000.0,
            break_los: 0.0,
            break_los_hp: 0.8,
            collect: 0.0,
            collect_s: 15.0,
            collect_hp: 0.3,
            collect_seen: 2.0,
            arm_keep: 0.0,
            wall_t: 0.0,
            wall_r: 290.0,
            wall_mode: 1.0,
            standoff: 0.0,
            stand_weave: 0.5,
            stand_boost: 30.0,
            stand_finish: 0.25,
            stand_mem: 1.0,
            stand_charge: 0.0,
            stand_first: 0.0,
            stand_zone: 0.0,
            stand_zone_m: 25.0,
            kite_away: 0.0,
            peek: 0.0,
            peek_s: 5.0,
            peek_crowd: 2.0,
            aim_high: 1.0,
            charge_seen: 1.0,
        }
    }
}

impl Params {
    /// Override defaults from a JSON object such as `{"gun_range": 120}`; unknown keys are ignored.
    pub fn from_json(v: &serde_json::Value) -> Params {
        let mut p = Params::default();
        let g = |k: &str, d: f32| v.get(k).and_then(|x| x.as_f64()).map(|x| x as f32).filter(|x| x.is_finite()).unwrap_or(d);
        p.retreat_hp = g("retreat_hp", p.retreat_hp);
        p.reengage_hp = g("reengage_hp", p.reengage_hp);
        p.gun_range = g("gun_range", p.gun_range);
        p.gun_band = g("gun_band", p.gun_band);
        p.arc_range = g("arc_range", p.arc_range);
        p.arc_min_hp = g("arc_min_hp", p.arc_min_hp);
        p.crowd_radius = g("crowd_radius", p.crowd_radius);
        p.crowd_count = g("crowd_count", p.crowd_count as f32).max(1.0) as usize;
        p.open_fire_range = g("open_fire_range", p.open_fire_range);
        p.strafe_min = g("strafe_min", p.strafe_min);
        p.strafe_max = g("strafe_max", p.strafe_max);
        p.strafe_alt = g("strafe_alt", p.strafe_alt);
        p.dodge_after = g("dodge_after", p.dodge_after);
        p.evade_weight = g("evade_weight", p.evade_weight);
        p.focus_hp = g("focus_hp", p.focus_hp);
        p.kite = g("kite", p.kite);
        p.finish_hp = g("finish_hp", p.finish_hp);
        p.finish_range = g("finish_range", p.finish_range);
        p.fight_speed = g("fight_speed", p.fight_speed);
        p.farm_hp = g("farm_hp", p.farm_hp);
        p.drone_range = g("drone_range", p.drone_range);
        p.drop_range = g("drop_range", p.drop_range);
        p.drop_enemy_clear = g("drop_enemy_clear", p.drop_enemy_clear);
        p.zone_lead = g("zone_lead", p.zone_lead);
        p.zone_lead_mult = g("zone_lead_mult", p.zone_lead_mult);
        p.wander_radius = g("wander_radius", p.wander_radius);
        p.memory_s = g("memory_s", p.memory_s);
        p.retreat_boost_energy = g("retreat_boost_energy", p.retreat_boost_energy);
        p.focus_stick = g("focus_stick", p.focus_stick);
        p.farm_until = g("farm_until", p.farm_until);
        p.farm_reach = g("farm_reach", p.farm_reach);
        p.heal_lo = g("heal_lo", p.heal_lo);
        p.heal_hi = g("heal_hi", p.heal_hi);
        p.expose_weight = g("expose_weight", p.expose_weight);
        p.cube_grab = g("cube_grab", p.cube_grab);
        p.reload_hide = g("reload_hide", p.reload_hide);
        p.laser_hide = g("laser_hide", p.laser_hide);
        p.delay_s = g("delay_s", p.delay_s);
        p.delay_auto = g("delay_auto", p.delay_auto);
        p.lead_ground = g("lead_ground", p.lead_ground);
        p.lead_air = g("lead_air", p.lead_air);
        p.lead_latency = g("lead_latency", p.lead_latency);
        p.lead_ballistic = g("lead_ballistic", p.lead_ballistic);
        p.aim_slack = g("aim_slack", p.aim_slack);
        p.dual = g("dual", p.dual);
        p.laser_mix = g("laser_mix", p.laser_mix);
        p.arc_crowd = g("arc_crowd", p.arc_crowd);
        p.arc_energy = g("arc_energy", p.arc_energy);
        p.rush = g("rush", p.rush);
        p.rush_start = g("rush_start", p.rush_start);
        p.rush_range = g("rush_range", p.rush_range);
        p.rush_hp = g("rush_hp", p.rush_hp);
        p.rush_crowd = g("rush_crowd", p.rush_crowd);
        p.rush_energy = g("rush_energy", p.rush_energy);
        p.rush_boost = g("rush_boost", p.rush_boost);
        p.laser_first = g("laser_first", p.laser_first);
        p.arm = g("arm", p.arm);
        p.arm_reach = g("arm_reach", p.arm_reach);
        p.arm_hp = g("arm_hp", p.arm_hp);
        p.arm_rank = g("arm_rank", p.arm_rank);
        p.arm_better = g("arm_better", p.arm_better);
        p.upg_dmg = g("upg_dmg", p.upg_dmg);
        p.arc_save = g("arc_save", p.arc_save);
        p.laser_range = g("laser_range", p.laser_range);
        p.drop_gun = g("drop_gun", p.drop_gun);
        p.precharge = g("precharge", p.precharge);
        p.mortar_take = g("mortar_take", p.mortar_take);
        p.flee_nocharge = g("flee_nocharge", p.flee_nocharge);
        p.hunt = g("hunt", p.hunt);
        p.hunt_hp = g("hunt_hp", p.hunt_hp);
        p.no_mortar = g("no_mortar", p.no_mortar);
        p.laser_seek = g("laser_seek", p.laser_seek);
        p.explore = g("explore", p.explore);
        p.explore_r = g("explore_r", p.explore_r);
        p.roof = g("roof", p.roof);
        p.roof_cost = g("roof_cost", p.roof_cost);
        p.roof_energy = g("roof_energy", p.roof_energy);
        p.loot_target = g("loot_target", p.loot_target);
        p.mortar_over_arc = g("mortar_over_arc", p.mortar_over_arc);
        p.laser_finish = g("laser_finish", p.laser_finish);
        p.laser_lead = g("laser_lead", p.laser_lead);
        p.reload_early = g("reload_early", p.reload_early);
        p.reload_laser = g("reload_laser", p.reload_laser);
        p.retreat_mode = g("retreat_mode", p.retreat_mode);
        p.break_crowd = g("break_crowd", p.break_crowd);
        p.break_laser = g("break_laser", p.break_laser);
        p.break_los = g("break_los", p.break_los);
        p.break_los_hp = g("break_los_hp", p.break_los_hp);
        p.collect = g("collect", p.collect);
        p.collect_s = g("collect_s", p.collect_s);
        p.collect_hp = g("collect_hp", p.collect_hp);
        p.collect_seen = g("collect_seen", p.collect_seen);
        p.arm_keep = g("arm_keep", p.arm_keep);
        p.wall_t = g("wall_t", p.wall_t);
        p.wall_r = g("wall_r", p.wall_r);
        p.wall_mode = g("wall_mode", p.wall_mode);
        p.standoff = g("standoff", p.standoff);
        p.stand_weave = g("stand_weave", p.stand_weave);
        p.stand_boost = g("stand_boost", p.stand_boost);
        p.stand_finish = g("stand_finish", p.stand_finish);
        p.stand_mem = g("stand_mem", p.stand_mem);
        p.stand_charge = g("stand_charge", p.stand_charge);
        p.stand_first = g("stand_first", p.stand_first);
        p.stand_zone = g("stand_zone", p.stand_zone);
        p.stand_zone_m = g("stand_zone_m", p.stand_zone_m);
        p.kite_away = g("kite_away", p.kite_away);
        p.peek = g("peek", p.peek);
        p.peek_s = g("peek_s", p.peek_s);
        p.peek_crowd = g("peek_crowd", p.peek_crowd);
        p.aim_high = g("aim_high", p.aim_high);
        p.charge_seen = g("charge_seen", p.charge_seen);
        p
    }

    /// Defaults, overridden by the JSON in the `TINMAN_PARAMS` environment variable if set.
    pub fn from_env() -> Params {
        std::env::var("TINMAN_PARAMS")
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|v| Params::from_json(&v))
            .unwrap_or_default()
    }
}

/// Upgrade preference with damage first.
const UPGRADES_DMG: &[&str] = &[
    "laser_damage", "gun_mag", "laser_charge", "plating", "gun_damage", "armor", "lightning_efficiency", "gun_rate",
    "nanites", "lightning_chain", "servos", "reactor", "capacitor", "gyros", "lightning_range", "jets",
    "gun_velocity", "mortar_reload", "mortar_splash", "sensors", "mortar_velocity", "laser_pierce",
];

/// Upgrade preference, best first. Survival first: placement is the score.
const UPGRADES: &[&str] = &[
    "plating", "armor", "nanites", "gun_damage", "lightning_efficiency", "servos", "gun_rate", "laser_damage",
    "lightning_chain", "reactor", "laser_charge", "capacitor", "gun_mag", "gyros", "lightning_range", "jets",
    "gun_velocity", "mortar_reload", "mortar_splash", "sensors", "mortar_velocity", "laser_pierce",
];

fn weapon_rank(kind: &str, p: &Params) -> i32 {
    let laser_first = p.laser_first > 0.0;
    let mortar_second = p.mortar_over_arc > 0.0;
    match kind {
        "lightning" => 3 - laser_first as i32 - mortar_second as i32,
        "laser" => 2 + laser_first as i32,
        "mortar" => (p.no_mortar <= 0.0) as i32 * (1 + mortar_second as i32),
        _ => 0,
    }
}

struct Shot {
    aim: [f32; 2],
    /// Allowed yaw and pitch error for pulling the trigger.
    tol: [f32; 2],
    ok: bool,
}

/// What we remember about another mech.
#[derive(Clone)]
struct Known {
    pos: [f32; 3],
    vel: [f32; 3],
    seen_t: f32,
    /// Velocity and time of the observation before `seen_t`.
    prev_vel: Option<([f32; 3], f32)>,
    /// When we first saw its current laser charge (None if not charging).
    charge_since: Option<f32>,
    aim_yaw: f32,
}

/// An action as sent, as far as motion and aim are concerned.
#[derive(Clone, Copy)]
struct Sent {
    mv: [f32; 2],
    boost: bool,
    aim: [f32; 2],
    fire: [bool; 2],
}

/// Our own motion state at some decision.
#[derive(Clone, Copy)]
struct Motion {
    pos: [f32; 3],
    vel: [f32; 3],
    aim: [f32; 2],
    grounded: bool,
    energy: f32,
    charging: bool,
    tick: u64,
}

/// Motion model constants.
#[derive(Clone, Copy)]
struct Kin {
    walk: f32,
    boost: f32,
    ground_accel: f32,
    air_accel: f32,
    turn: f32,
    pitch: f32,
    min_pitch: f32,
    max_pitch: f32,
    charge_mult: f32,
}

impl Kin {
    /// Advance `m` by one decision (two ticks) under action `s`.
    fn step(&self, m: &mut Motion, s: &Sent) {
        let dt = 0.05;
        for _ in 0..2 {
            let dy = wrap(s.aim[0] - m.aim[0]);
            m.aim[0] = wrap(m.aim[0] + dy.clamp(-self.turn * dt, self.turn * dt));
            let pt = s.aim[1].clamp(self.min_pitch, self.max_pitch);
            m.aim[1] += (pt - m.aim[1]).clamp(-self.pitch * dt, self.pitch * dt);
            let mut mv = s.mv;
            let l = (mv[0] * mv[0] + mv[1] * mv[1]).sqrt();
            if l > 1.0 {
                mv = [mv[0] / l, mv[1] / l];
            }
            let mut speed = self.walk;
            if s.boost && m.energy > 0.0 && (!m.grounded || l > 0.05) {
                speed = self.boost;
            }
            if m.charging {
                speed *= self.charge_mult;
            }
            let tv = [mv[0] * speed, mv[1] * speed];
            let acc = if m.grounded { self.ground_accel } else { self.air_accel } * dt;
            let dv = [tv[0] - m.vel[0], tv[1] - m.vel[1]];
            let dl = (dv[0] * dv[0] + dv[1] * dv[1]).sqrt();
            if dl <= acc {
                m.vel[0] = tv[0];
                m.vel[1] = tv[1];
            } else {
                m.vel[0] += dv[0] / dl * acc;
                m.vel[1] += dv[1] / dl * acc;
            }
            m.pos[0] += m.vel[0] * dt;
            m.pos[1] += m.vel[1] * dt;
        }
    }
}

pub struct Brain {
    pub world: Option<Arc<World>>,
    pub nav: Option<Nav>,
    pub params: Params,
    rng: u64,
    pickup_radius: f32,
    /// Seconds per simulation tick.
    tick_s: f32,
    strafe: f32,
    strafe_until: f32,
    last_pos: [f32; 3],
    stuck: u32,
    unstick_until: f32,
    last_hurt_t: f32,
    /// Last time a mech (not a drone or the zone) damaged us.
    last_mech_hurt_t: f32,
    threat: Option<([f32; 3], f32)>,
    fleeing: bool,
    cover: Option<([f32; 2], f32)>,
    away: [f32; 2],
    /// Out of sight between laser shots: where we fired from, at whom, since when.
    peek: Option<([f32; 2], u32, f32)>,
    scan_yaw: f32,
    known: BTreeMap<u32, Known>,
    /// Mechs that shot us recently: id -> time.
    attackers: BTreeMap<u32, f32>,
    /// Drones that shot us recently: id -> time.
    drone_attackers: BTreeMap<u32, f32>,
    post: Option<([f32; 2], f32)>,
    wander: Option<[f32; 2]>,
    dodged_charge: BTreeMap<u32, f32>,
    nav_goal: Option<[f32; 2]>,
    /// Mech we are concentrating fire on.
    focus: Option<u32>,
    /// Loot seen earlier and not known to be gone: id -> position (crates, cubes) or kind and position (weapons).
    crates_mem: BTreeMap<u32, [f32; 3]>,
    cubes_mem: BTreeMap<u32, [f32; 3]>,
    picks_mem: BTreeMap<u32, (String, [f32; 3])>,
    healing: bool,
    kin: Kin,
    /// Recent actions sent, oldest first.
    sent: VecDeque<Sent>,
    /// Our motion at the previous decision.
    prev_motion: Option<Motion>,
    /// Accumulated prediction error for each candidate delay (in decisions), and how informative the samples were.
    delay_err: [f32; 4],
    delay_info: f32,
    /// Delay in seconds currently assumed.
    pub delay_used: f32,
    /// Short name of the current behaviour, for debugging.
    pub label: &'static str,
    trace_dmg: bool,
    /// Target of the lightning rush in progress.
    rushing: Option<u32>,
    /// When our laser charge became full (None while not full).
    laser_full_t: Option<f32>,
    /// Coarse map cells we have been near (for weapon search): flag per cell.
    looked: Vec<bool>,
    /// Weapon pickup we are walking to.
    arm_pick: Option<u32>,
    /// Where and when we last killed a mech (its cubes drop there).
    kill_spot: Option<([f32; 3], f32)>,
    /// Spot held late in the match, and when it was chosen.
    hold: Option<([f32; 2], f32)>,
}

impl Brain {
    pub fn new(hello: &Hello) -> Brain {
        let cfg = &hello.config;
        let f = |a: &str, b: &str, d: f32| cfg.get(a).and_then(|v| v.get(b)).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(d);
        let world = World::load(std::path::Path::new(&hello.map.dir)).map_err(|e| eprintln!("map load failed: {e}")).ok();
        let nav = world.as_ref().map(|w| {
            Nav::build(w, f("mech", "radius", 2.5), f("mech", "height", 9.0), f("mech", "step_up", 1.2), f("mech", "max_slope_deg", 50.0))
        });
        Brain::with_map(hello, world.map(Arc::new), nav)
    }

    /// Build from an already loaded map and nav grid (both can be shared between brains).
    pub fn with_map(hello: &Hello, world: Option<Arc<World>>, nav: Option<Nav>) -> Brain {
        Brain::with_params(hello, world, nav, Params::from_env())
    }

    /// Like `with_map`, with explicit parameters.
    pub fn with_params(hello: &Hello, world: Option<Arc<World>>, nav: Option<Nav>, params: Params) -> Brain {
        let cfg = &hello.config;
        let f = |a: &str, b: &str, d: f32| cfg.get(a).and_then(|v| v.get(b)).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(d);
        Brain {
            world,
            nav,
            params,
            rng: hello.seed | 1,
            pickup_radius: f("loot", "pickup_radius", 4.0),
            tick_s: 1.0 / f("match", "tick_hz", 20.0).max(1.0),
            strafe: 1.0,
            strafe_until: 0.0,
            last_pos: [0.0; 3],
            stuck: 0,
            unstick_until: 0.0,
            last_hurt_t: -100.0,
            last_mech_hurt_t: -100.0,
            threat: None,
            fleeing: false,
            cover: None,
            away: [0.0, 0.0],
            peek: None,
            scan_yaw: 0.0,
            known: BTreeMap::new(),
            attackers: BTreeMap::new(),
            drone_attackers: BTreeMap::new(),
            post: None,
            wander: None,
            dodged_charge: BTreeMap::new(),
            nav_goal: None,
            focus: None,
            crates_mem: BTreeMap::new(),
            cubes_mem: BTreeMap::new(),
            picks_mem: BTreeMap::new(),
            healing: false,
            kin: Kin {
                walk: f("mech", "walk_speed", 11.7),
                boost: f("mech", "boost_speed", 19.5),
                ground_accel: f("mech", "ground_accel", 58.5),
                air_accel: f("mech", "air_accel", 12.0),
                turn: f("mech", "turn_rate", 4.0),
                pitch: f("mech", "pitch_rate", 3.0),
                min_pitch: f("mech", "min_pitch", -0.6),
                max_pitch: f("mech", "max_pitch", 1.3),
                charge_mult: f("laser", "move_mult", 0.5),
            },
            sent: VecDeque::new(),
            prev_motion: None,
            delay_err: [0.0; 4],
            delay_info: 0.0,
            delay_used: 0.0,
            label: "",
            trace_dmg: std::env::var("TDMG").is_ok(),
            rushing: None,
            laser_full_t: None,
            looked: Vec::new(),
            arm_pick: None,
            kill_spot: None,
            hold: None,
        }
    }

    fn rnd(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    fn clear(&self, a: [f32; 3], b: [f32; 3]) -> bool {
        self.world.as_ref().map_or(true, |w| w.clear(a, b))
    }

    fn ground(&self, x: f32, y: f32, fallback: f32) -> f32 {
        self.world.as_ref().map_or(fallback, |w| w.ground(x, y))
    }

    fn walkable(&self, x: f32, y: f32) -> bool {
        self.nav.as_ref().map_or(true, |n| n.walkable(x, y))
    }

    /// Straight ground line between two points is walkable.
    fn straight(&self, a: &[f32], b: &[f32]) -> bool {
        let d = d2(a, b);
        let n = (d / 2.0).ceil().max(1.0) as usize;
        (1..=n).all(|k| {
            let f = k as f32 / n as f32;
            self.walkable(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
        })
    }

    /// Unit move direction toward a goal along the ground path.
    fn go(&mut self, pos: &[f32; 3], goal: [f32; 2]) -> [f32; 2] {
        let dist = d2(pos, &goal);
        if dist < 1.5 {
            return [0.0, 0.0];
        }
        // Short hops in the open need no path search.
        if dist < 40.0 && self.straight(pos, &goal) {
            return norm2(goal[0] - pos[0], goal[1] - pos[1]);
        }
        let mut target = goal;
        if let Some(nav) = self.nav.as_mut() {
            // Avoid recomputing the field for goals that moved only a little.
            let g = match self.nav_goal {
                Some(g) if d2(&g, &goal) < 4.0 => g,
                _ => goal,
            };
            self.nav_goal = Some(g);
            nav.set_goal(g[0], g[1]);
            if let Some(p) = nav.steer(pos[0], pos[1]) {
                if dist > 6.0 {
                    target = p;
                }
            }
        }
        norm2(target[0] - pos[0], target[1] - pos[1])
    }

    /// Fire every slot whose own solution lines up. The aim serves the gun while it can shoot (lightning's
    /// cone is wide enough to come along); a fully charged laser takes the aim, nudged toward the gun's lead.
    /// Returns whether any weapon has a firing solution.
    fn fire_both(&mut self, o: &Obs, t: &Actor, a: &mut Action) -> bool {
        let me = &o.me;
        let p = &self.params;
        let slack = [me.stat("turn_rate", 4.0) * self.tick_s * p.aim_slack, me.stat("pitch_rate", 3.0) * self.tick_s * p.aim_slack];
        let lined = |s: &Shot, sl: [f32; 2]| wrap(s.aim[0] - me.aim[0]).abs() <= s.tol[0] + sl[0] && (s.aim[1] - me.aim[1]).abs() <= s.tol[1] + sl[1];
        let mut sols: Vec<(usize, &Weapon, Shot)> = Vec::new();
        for (i, slot) in me.weapons.iter().enumerate().take(2) {
            if let Some(w) = slot {
                sols.push((i, w, self.solve(o, w, t)));
            }
        }
        let find = |k: &str| sols.iter().find(|(_, w, _)| w.kind == k);
        let gun = find("gun").filter(|(_, _, s)| s.ok);
        let mut aim: Option<[f32; 2]> = gun.map(|(_, _, s)| s.aim);
        let mut engaged = gun.is_some();
        if let Some((i, w, s)) = gun {
            a.fire[*i] = lined(s, slack) && w.cooldown <= 0.12;
        }
        if let Some((i, w, s)) = find("laser") {
            if s.ok {
                engaged = true;
                let full_c = w.stat("charge", 0.75);
                let mut full = w.charge >= full_c - 0.01;
                if !full && p.laser_finish > 0.0 && t.max_hp > 0.0 && w.charge >= w.stat("min_charge", 0.15) {
                    let frac = w.stat("min_frac", 0.35);
                    let dmg = w.stat("damage", 315.0) * (frac + (1.0 - frac) * (w.charge / full_c).min(1.0)) * (1.0 + 0.04 * me.cubes.min(10.0));
                    full = t.hp < dmg * 0.95;
                }
                if full {
                    // Aim for the beam, pulled toward the gun's lead as far as the beam still hits.
                    let mut la = s.aim;
                    if let Some(g) = aim {
                        let m = p.laser_mix;
                        la[0] += wrap(g[0] - s.aim[0]).clamp(-s.tol[0] * m, s.tol[0] * m);
                        la[1] += (g[1] - s.aim[1]).clamp(-s.tol[1] * m, s.tol[1] * m);
                    }
                    aim = Some(la);
                    a.fire[*i] = !lined(s, [0.0, 0.0]);
                } else {
                    a.fire[*i] = w.cooldown <= 0.0 || w.charge > 0.0;
                }
                if aim.is_none() {
                    aim = Some(s.aim);
                }
            } else if w.charge > 0.0 || (p.charge_seen > 0.0 && w.cooldown <= 0.0) {
                a.fire[*i] = true;
            }
        }
        if let Some((i, w, s)) = find("lightning") {
            if s.ok {
                engaged = true;
                let cur = aim.unwrap_or(s.aim);
                // Keep the arc inside the cone of where the torso is pointing now and where it is going.
                let inside = wrap(s.aim[0] - me.aim[0]).abs() <= s.tol[0] && wrap(s.aim[0] - cur[0]).abs() <= s.tol[0];
                a.fire[*i] = inside && w.cooldown <= 0.12;
                if aim.is_none() || wrap(s.aim[0] - cur[0]).abs() > s.tol[0] * 0.8 {
                    aim = Some(s.aim);
                }
            }
        }
        if let Some((i, w, s)) = find("mortar") {
            if s.ok {
                engaged = true;
                let far = p.mortar_take > 0.0 && d2(&t.pos, &me.pos) > p.mortar_take && w.cooldown <= 0.1;
                if aim.is_none() || far {
                    aim = Some(s.aim);
                }
                a.fire[*i] = lined(s, [0.0, 0.0]) && w.cooldown <= 0.12;
            }
        }
        let fallback = sols.first().map(|x| x.2.aim);
        if let Some(x) = aim.or(fallback) {
            a.aim = x;
        }
        engaged
    }

    const LOOK_CELL: f32 = 40.0;

    fn look_dims(&self) -> Option<(f32, f32, usize, usize)> {
        let w = self.world.as_ref()?;
        let nx = (((w.max_x - w.min_x) / Self::LOOK_CELL).ceil() as usize).max(1);
        let ny = (((w.max_y - w.min_y) / Self::LOOK_CELL).ceil() as usize).max(1);
        Some((w.min_x, w.min_y, nx, ny))
    }

    /// Mark the map cells around us as looked at.
    fn mark_looked(&mut self, pos: &[f32; 3]) {
        let Some((x0, y0, nx, ny)) = self.look_dims() else { return };
        if self.looked.len() != nx * ny {
            self.looked = vec![false; nx * ny];
        }
        let r = self.params.explore_r;
        let c = Self::LOOK_CELL;
        let (ix0, ix1) = (((pos[0] - r - x0) / c).floor().max(0.0) as usize, (((pos[0] + r - x0) / c).floor().max(0.0) as usize).min(nx - 1));
        let (iy0, iy1) = (((pos[1] - r - y0) / c).floor().max(0.0) as usize, (((pos[1] + r - y0) / c).floor().max(0.0) as usize).min(ny - 1));
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                let cx = x0 + (ix as f32 + 0.5) * c;
                let cy = y0 + (iy as f32 + 0.5) * c;
                if (cx - pos[0]).hypot(cy - pos[1]) < r {
                    self.looked[iy * nx + ix] = true;
                }
            }
        }
    }

    /// Nearest walkable cell centre not looked at yet, inside the zone (and the next circle once it matters).
    fn unexplored(&self, o: &Obs, pos: &[f32; 3]) -> Option<[f32; 2]> {
        let (x0, y0, nx, ny) = self.look_dims()?;
        if self.looked.len() != nx * ny {
            return None;
        }
        let z = &o.zone;
        let c = Self::LOOK_CELL;
        let mut best: Option<([f32; 2], f32)> = None;
        for iy in 0..ny {
            for ix in 0..nx {
                if self.looked[iy * nx + ix] {
                    continue;
                }
                let q = [x0 + (ix as f32 + 0.5) * c, y0 + (iy as f32 + 0.5) * c];
                let inz = (q[0] - z.center[0]).hypot(q[1] - z.center[1]) < z.radius - 20.0
                    && (o.t < z.shrink_start - 30.0 || (q[0] - z.next_center[0]).hypot(q[1] - z.next_center[1]) < z.next_radius - 10.0);
                if !inz || !self.walkable(q[0], q[1]) {
                    continue;
                }
                let d = (q[0] - pos[0]).hypot(q[1] - pos[1]);
                if best.map_or(true, |b| d < b.1) {
                    best = Some((q, d));
                }
            }
        }
        best.map(|b| b.0)
    }

    /// Aim solution for one weapon against a target, from our current position.
    fn solve(&self, o: &Obs, w: &Weapon, t: &Actor) -> Shot {
        let me = &o.me;
        let scale = 1.0 + 0.05 * t.cubes;
        let muzzle = [me.pos[0], me.pos[1], me.pos[2] + me.stat("muzzle_height", 7.0)];
        let mut tc = [t.pos[0], t.pos[1], t.pos[2] + 4.5 * scale];
        if self.params.aim_high > 0.0 && !self.clear(muzzle, tc) {
            let hi = [t.pos[0], t.pos[1], t.pos[2] + 7.6 * scale];
            if self.clear(muzzle, hi) {
                tc = hi;
            }
        }
        let dist = d3(&muzzle, &tc);
        let flat = |p: &[f32; 3], from: &[f32; 3]| [(p[1] - from[1]).atan2(p[0] - from[0]), (p[2] - from[2]).atan2(d2(p, from).max(0.01))];
        let body = [(2.5 * scale * 0.85 / dist.max(1.0)).atan(), (4.5 * scale * 0.8 / dist.max(1.0)).atan()];
        match w.kind.as_str() {
            "gun" => {
                let speed = w.stat("speed", 252.0);
                let p = &self.params;
                // Shots start a little ahead of the mech along the aim.
                let lead_out = 2.5 * (1.0 + 0.05 * me.cubes) + 0.5;
                // Where it would land if it is in the air: the highest surface under it.
                let floor = self.world.as_ref().filter(|_| p.lead_ballistic > 0.0).map(|w| w.surface_below(t.pos[0], t.pos[1], t.pos[2] + 0.3));
                let airborne = floor.map_or(false, |f| t.pos[2] - f > 0.4);
                // Jetting shows as vertical speed that is not falling at gravity.
                let jet = airborne
                    && (self.known.get(&t.id).and_then(|k| k.prev_vel).map_or(false, |(pv, pt)| {
                        let dt = (o.t - pt).max(0.05);
                        (t.vel[2] - pv[2]) / dt > -10.0
                    }) || (t.vel[2] > 8.3 && t.vel[2] < 9.5));
                let lead = if airborne { p.lead_air } else { p.lead_ground };
                let rise = |tt: f32| -> f32 {
                    match floor {
                        None => t.vel[2] * tt * 0.5,
                        Some(_) if !airborne => 0.0,
                        Some(f) => {
                            let vz = t.vel[2];
                            let dz = if jet {
                                // thrust +12 m/s^2 net up to a climb of 8.9 m/s
                                let t1 = tt.min(((8.9 - vz) / 12.0).max(0.0));
                                vz * t1 + 6.0 * t1 * t1 + 8.9 * (tt - t1)
                            } else {
                                vz * tt - 11.0 * tt * tt
                            };
                            (t.pos[2] + dz).max(f) - t.pos[2]
                        }
                    }
                };
                let mut pt = tc;
                for _ in 0..3 {
                    let tt = ((d3(&muzzle, &pt) - lead_out).max(0.0)) / speed + p.lead_latency;
                    pt = [tc[0] + t.vel[0] * tt * lead, tc[1] + t.vel[1] * tt * lead, tc[2] + rise(tt)];
                }
                let ok = w.ammo > 0.0 && w.reload <= 0.0 && d3(&muzzle, &pt) < w.stat("range", 230.0) && self.clear(muzzle, pt);
                Shot { aim: flat(&pt, &muzzle), tol: body, ok }
            }
            "laser" => {
                let ok = dist < w.stat("range", 280.0) && self.clear(muzzle, tc);
                let ll = self.params.laser_lead;
                let lc = [tc[0] + t.vel[0] * ll, tc[1] + t.vel[1] * ll, tc[2] + t.vel[2] * ll];
                Shot { aim: flat(&lc, &muzzle), tol: [body[0] * 0.8, body[1] * 0.8], ok }
            }
            "lightning" => {
                let cone = w.stat("cone_deg", 30.0).to_radians() * 0.5;
                let ok = dist < w.stat("range", 45.0) - 1.0 && me.energy >= w.stat("energy", 3.0) && self.clear(muzzle, tc);
                Shot { aim: flat(&tc, &muzzle), tol: [cone * 0.9, 10.0], ok }
            }
            "mortar" => {
                let (v, g) = (w.stat("speed", 72.0), w.stat("gravity", 22.0));
                let tube = [me.pos[0], me.pos[1], me.pos[2] + me.stat("height", 9.0) + 0.5];
                let mut p = t.pos;
                let mut pitch = 0.0f32;
                let mut ok = false;
                for _ in 0..2 {
                    let d = d2(&p, &tube).max(1.0);
                    let h = p[2] - tube[2];
                    let disc = v.powi(4) - g * (g * d * d + 2.0 * h * v * v);
                    if disc < 0.0 {
                        ok = false;
                        break;
                    }
                    pitch = ((v * v - disc.sqrt()) / (g * d)).atan();
                    ok = true;
                    let tt = d / (v * pitch.cos()).max(1.0);
                    p = [t.pos[0] + t.vel[0] * tt, t.pos[1] + t.vel[1] * tt, t.pos[2]];
                }
                let d = d2(&p, &tube);
                let safe = d > w.stat("splash_radius", 7.0) + 8.0;
                let along = [tube[0] + (p[0] - tube[0]) * 0.3, tube[1] + (p[1] - tube[1]) * 0.3, tube[2] + pitch.tan() * d * 0.3];
                let ok = ok && safe && pitch < me.stat("max_pitch", 1.3) && self.clear(tube, along);
                Shot { aim: [(p[1] - tube[1]).atan2(p[0] - tube[0]), pitch], tol: [(3.0 / d.max(1.0)).atan(), 0.03], ok }
            }
            _ => Shot { aim: me.aim, tol: [0.0, 0.0], ok: false },
        }
    }

    /// Positions of enemies we should hide from (recently known), nearest first, at most four.
    fn threats(&self, o: &Obs) -> Vec<[f32; 3]> {
        let me = o.me.pos;
        let mut v: Vec<[f32; 3]> = self
            .known
            .values()
            .filter(|k| o.t - k.seen_t < self.params.memory_s && d2(&k.pos, &me) < 260.0)
            .map(|k| {
                let dt = (o.t - k.seen_t).min(3.0);
                [k.pos[0] + k.vel[0] * dt, k.pos[1] + k.vel[1] * dt, k.pos[2]]
            })
            .collect();
        if let Some((p, t)) = self.threat {
            if o.t - t < 6.0 && !v.iter().any(|q| d2(q, &p) < 30.0) {
                v.push(p);
            }
        }
        v.sort_by(|a, b| d2(a, &me).partial_cmp(&d2(b, &me)).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(4);
        v
    }

    fn hidden_from(&self, c: [f32; 2], z: f32, threats: &[[f32; 3]]) -> usize {
        threats
            .iter()
            .filter(|t| {
                let eye = [t[0], t[1], t[2] + 8.0];
                !self.clear(eye, [c[0], c[1], z + 4.5]) && !self.clear(eye, [c[0], c[1], z + 8.5])
            })
            .count()
    }

    fn zone_penalty(&self, o: &Obs, c: &[f32; 2]) -> f32 {
        let z = &o.zone;
        let soon = o.t > z.shrink_start - 25.0;
        (d2(c, &z.center) - z.radius + 15.0).max(0.0) * 4.0 + if soon { (d2(c, &z.next_center) - z.next_radius + 10.0).max(0.0) * 2.0 } else { 0.0 }
    }

    /// A reachable spot that none of the threats can see, reasonably close and inside the zone.
    fn find_cover(&mut self, o: &Obs, threats: &[[f32; 3]]) -> [f32; 2] {
        let me = &o.me;
        let (cx, cy) = threats.iter().fold((0.0, 0.0), |a, t| (a.0 + t[0], a.1 + t[1]));
        let n = threats.len().max(1) as f32;
        let centroid = [cx / n, cy / n];
        let away = norm2(me.pos[0] - centroid[0], me.pos[1] - centroid[1]);
        let mut best = ([me.pos[0] + away[0] * 50.0, me.pos[1] + away[1] * 50.0], f32::MIN);
        for ring in [12.0f32, 25.0, 40.0, 60.0, 85.0] {
            for k in 0..16 {
                let a = k as f32 * TAU / 16.0 + ring * 0.01;
                let c = [me.pos[0] + a.cos() * ring, me.pos[1] + a.sin() * ring];
                if !self.walkable(c[0], c[1]) {
                    continue;
                }
                let z = self.ground(c[0], c[1], me.pos[2]);
                let hid = self.hidden_from(c, z, threats) as f32;
                let near = threats.iter().map(|t| d2(&c, t)).fold(f32::MAX, f32::min);
                let s = hid * 100.0 + if hid as usize == threats.len() { 80.0 } else { 0.0 } - ring * 0.8 - self.zone_penalty(o, &c)
                    + (a.cos() * away[0] + a.sin() * away[1]) * 20.0
                    - (60.0 - near).max(0.0) * 2.0;
                if s > best.1 {
                    best = (c, s);
                }
            }
        }
        best.0
    }

    /// Every enemy position we know of (recently seen, extrapolated a little), nearest first, at most eight.
    fn threats_all(&self, o: &Obs) -> Vec<[f32; 3]> {
        let me = o.me.pos;
        let mut v: Vec<[f32; 3]> = self
            .known
            .values()
            .filter(|k| o.t - k.seen_t < self.params.memory_s && d2(&k.pos, &me) < 260.0)
            .map(|k| {
                let dt = (o.t - k.seen_t).min(2.0);
                [k.pos[0] + k.vel[0] * dt, k.pos[1] + k.vel[1] * dt, k.pos[2]]
            })
            .collect();
        if let Some((p, t)) = self.threat {
            if o.t - t < 6.0 && !v.iter().any(|q| d2(q, &p) < 30.0) {
                v.push(p);
            }
        }
        v.sort_by(|a, b| d2(a, &me).partial_cmp(&d2(b, &me)).unwrap_or(std::cmp::Ordering::Equal));
        v.truncate(8);
        v
    }

    /// The quickest spot to reach that every threat's line of sight is blocked from, preferring
    /// spots away from them.
    fn find_hide(&mut self, o: &Obs, threats: &[[f32; 3]]) -> [f32; 2] {
        let me = &o.me;
        let (cx, cy) = threats.iter().fold((0.0, 0.0), |a, t| (a.0 + t[0], a.1 + t[1]));
        let n = threats.len().max(1) as f32;
        let away = norm2(me.pos[0] - cx / n, me.pos[1] - cy / n);
        let mut best = ([me.pos[0] + away[0] * 40.0, me.pos[1] + away[1] * 40.0], f32::MIN);
        for ring in [6.0f32, 12.0, 20.0, 30.0, 45.0, 65.0, 90.0] {
            for k in 0..24 {
                let a = k as f32 * TAU / 24.0 + ring * 0.01;
                let c = [me.pos[0] + a.cos() * ring, me.pos[1] + a.sin() * ring];
                if !self.walkable(c[0], c[1]) {
                    continue;
                }
                let z = self.ground(c[0], c[1], me.pos[2]);
                let hid = self.hidden_from(c, z, threats);
                let near = threats.iter().map(|t| d2(&c, t)).fold(f32::MAX, f32::min);
                let direct = self.straight(&me.pos, &c);
                let path = if direct { ring } else { ring * 1.6 };
                let s = hid as f32 * 100.0 + if hid == threats.len() { 300.0 } else { 0.0 } - path * 2.0 - self.zone_penalty(o, &c)
                    + (a.cos() * away[0] + a.sin() * away[1]) * 15.0
                    - (40.0 - near).max(0.0) * 3.0;
                if s > best.1 {
                    best = (c, s);
                }
            }
        }
        best.0
    }

    fn in_next(&self, o: &Obs, c: &[f32; 2], margin: f32) -> bool {
        let z = &o.zone;
        (c[0] - z.next_center[0]).hypot(c[1] - z.next_center[1]) < z.next_radius - margin && (c[0] - z.center[0]).hypot(c[1] - z.center[1]) < z.radius - margin
    }

    /// A walkable spot near us inside the next circle that as little of the surrounding ground as
    /// possible can see (attackers come into view one or two at a time), but that sees out a little.
    fn hold_spot(&self, o: &Obs, pos: &[f32; 3]) -> [f32; 2] {
        let mut best = ([pos[0], pos[1]], f32::MAX);
        for ring in [0.0f32, 20.0, 45.0, 75.0, 110.0] {
            let nk = if ring == 0.0 { 1 } else { 12 };
            for k in 0..nk {
                let a = k as f32 * TAU / nk as f32 + ring * 0.013;
                let c = [pos[0] + a.cos() * ring, pos[1] + a.sin() * ring];
                if !self.walkable(c[0], c[1]) || !self.in_next(o, &c, 15.0) {
                    continue;
                }
                let z = self.ground(c[0], c[1], pos[2]);
                let eye = [c[0], c[1], z + 7.0];
                let mut seen = 0;
                let mut tot = 0;
                for j in 0..16 {
                    let b = j as f32 * TAU / 16.0;
                    for r in [35.0f32, 80.0, 140.0] {
                        let q = [c[0] + b.cos() * r, c[1] + b.sin() * r];
                        if !self.walkable(q[0], q[1]) {
                            continue;
                        }
                        tot += 1;
                        let qz = self.ground(q[0], q[1], z);
                        if self.clear(eye, [q[0], q[1], qz + 7.0]) {
                            seen += 1;
                        }
                    }
                }
                if tot == 0 || seen == 0 {
                    continue;
                }
                let s = seen as f32 / tot as f32 * 100.0 + ring * 0.15;
                if s < best.1 {
                    best = (c, s);
                }
            }
        }
        best.0
    }

    /// The spot inside the next circle furthest from `from` (or the circle's centre when nothing is known).
    fn far_spot(&self, o: &Obs, pos: &[f32; 3], from: Option<[f32; 2]>) -> [f32; 2] {
        let z = &o.zone;
        let Some(f) = from else { return z.next_center };
        let mut best = ([pos[0], pos[1]], f32::MIN);
        for k in 0..24 {
            let a = k as f32 * TAU / 24.0;
            for fr in [0.3f32, 0.6, 0.85] {
                let r = (z.next_radius - 15.0).max(0.0) * fr;
                let c = [z.next_center[0] + a.cos() * r, z.next_center[1] + a.sin() * r];
                if !self.walkable(c[0], c[1]) {
                    continue;
                }
                let s = (c[0] - f[0]).hypot(c[1] - f[1]) - 0.3 * (c[0] - pos[0]).hypot(c[1] - pos[1]);
                if s > best.1 {
                    best = (c, s);
                }
            }
        }
        best.0
    }

    /// How enclosed a spot is: the number of eight directions blocked by buildings within 40 m.
    fn enclosure(&self, c: [f32; 2], z: f32) -> usize {
        (0..8)
            .filter(|k| {
                let a = *k as f32 * TAU / 8.0;
                !self.clear([c[0], c[1], z + 6.0], [c[0] + a.cos() * 40.0, c[1] + a.sin() * 40.0, z + 6.0])
            })
            .count()
    }

    /// An idle post: a sheltered walkable spot inside the safe area, near us.
    fn choose_post(&mut self, o: &Obs) -> [f32; 2] {
        let me = &o.me;
        let z = &o.zone;
        // Aim for the part of the next circle nearest to us.
        let r_next = (z.next_radius * 0.7).max(0.0);
        let to_me = norm2(me.pos[0] - z.next_center[0], me.pos[1] - z.next_center[1]);
        let d_me = d2(&me.pos, &z.next_center);
        let anchor = if d_me > r_next {
            [z.next_center[0] + to_me[0] * r_next, z.next_center[1] + to_me[1] * r_next]
        } else {
            [me.pos[0], me.pos[1]]
        };
        let mut best = (anchor, f32::MIN);
        for ring in [0.0f32, 20.0, 40.0, 65.0] {
            let nk = if ring == 0.0 { 1 } else { 12 };
            for k in 0..nk {
                let a = k as f32 * TAU / nk as f32 + self.rnd() * 0.3;
                let c = [anchor[0] + a.cos() * ring, anchor[1] + a.sin() * ring];
                if !self.walkable(c[0], c[1]) || d2(&c, &z.next_center) > r_next.max(15.0) + 5.0 {
                    continue;
                }
                let zz = self.ground(c[0], c[1], me.pos[2]);
                let s = self.enclosure(c, zz) as f32 * 10.0 - d2(&c, &me.pos) * 0.15;
                if s > best.1 {
                    best = (c, s);
                }
            }
        }
        best.0
    }

    pub fn act(&mut self, o: &Obs) -> Action {
        let raw = o.me.pos;
        let mut kin = self.kin;
        kin.walk = o.me.stat("walk_speed", kin.walk);
        kin.boost = o.me.stat("boost_speed", kin.boost);
        kin.turn = o.me.stat("turn_rate", kin.turn);
        kin.pitch = o.me.stat("pitch_rate", kin.pitch);
        kin.min_pitch = o.me.stat("min_pitch", kin.min_pitch);
        kin.max_pitch = o.me.stat("max_pitch", kin.max_pitch);
        kin.charge_mult = o.me.stat("charge_move_mult", kin.charge_mult);
        self.kin = kin;
        let now = Motion {
            pos: o.me.pos,
            vel: o.me.vel,
            aim: o.me.aim,
            grounded: o.me.grounded,
            energy: o.me.energy,
            charging: o.me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.charge > 0.0),
            tick: o.tick,
        };
        self.estimate_delay(&now);
        self.prev_motion = Some(now);
        let delay = self.assumed_delay();
        self.delay_used = delay;
        let n = ((delay / 0.1).round() as usize).min(self.sent.len());
        let a = if n == 0 {
            self.decide(o, raw)
        } else {
            let q = self.predicted(o, now, n, n as f32 * 0.1);
            self.decide(&q, raw)
        };
        self.sent.push_back(Sent { mv: a.mv, boost: a.boost, aim: a.aim, fire: a.fire });
        while self.sent.len() > 6 {
            self.sent.pop_front();
        }
        a
    }

    fn assumed_delay(&self) -> f32 {
        let p = &self.params;
        let mut d = p.delay_s;
        if p.delay_auto > 0.0 && self.delay_info >= 8.0 {
            let (mut bk, mut be) = (0usize, f32::MAX);
            for (k, e) in self.delay_err.iter().enumerate() {
                if *e < be {
                    be = *e;
                    bk = k;
                }
            }
            d = bk as f32 * 0.1;
        }
        d.clamp(0.0, 0.3)
    }

    /// Compare how we actually moved since the last decision with what each candidate delay predicts.
    fn estimate_delay(&mut self, now: &Motion) {
        let Some(prev) = self.prev_motion else { return };
        if now.tick != prev.tick + 2 || self.sent.is_empty() {
            return;
        }
        let kin = self.kin;
        let hold = Sent { mv: [0.0, 0.0], boost: false, aim: prev.aim, fire: [false; 2] };
        let use_vel = prev.grounded && now.grounded && !prev.charging && !now.charging;
        let mut errs = [0.0f32; 4];
        for (k, e) in errs.iter_mut().enumerate() {
            let s = if k < self.sent.len() { self.sent[self.sent.len() - 1 - k] } else { hold };
            let mut m = prev;
            kin.step(&mut m, &s);
            let ya = wrap(m.aim[0] - now.aim[0]).abs() + (m.aim[1] - now.aim[1]).abs();
            let va = if use_vel { ((m.vel[0] - now.vel[0]).powi(2) + (m.vel[1] - now.vel[1]).powi(2)).sqrt() / 10.0 } else { 0.0 };
            *e = ya.min(1.0) + va.min(1.0);
        }
        let lo = errs.iter().cloned().fold(f32::MAX, f32::min);
        let hi = errs.iter().cloned().fold(0.0f32, f32::max);
        // Only samples that tell the candidates apart count.
        if hi - lo > 0.05 {
            for k in 0..4 {
                self.delay_err[k] = self.delay_err[k] * 0.995 + errs[k];
            }
            self.delay_info = self.delay_info * 0.995 + 1.0;
        }
    }

    /// The observation as it will be when an action sent now takes effect: our own state advanced
    /// through the `n` actions still pending, other mechs moved on, timers run down.
    fn predicted(&self, o: &Obs, now: Motion, n: usize, dt: f32) -> Obs {
        let mut q = o.clone();
        let mut m = now;
        let pending: Vec<Sent> = self.sent.iter().rev().take(n).rev().copied().collect();
        for s in &pending {
            self.kin.step(&mut m, s);
        }
        q.me.pos = m.pos;
        q.me.vel = m.vel;
        q.me.aim = m.aim;
        q.t = o.t + dt;
        for a in q.mechs.iter_mut() {
            for k in 0..3 {
                a.pos[k] += a.vel[k] * dt * if k == 2 { 0.5 } else { 1.0 };
            }
        }
        for d in q.drops.iter_mut() {
            d.lands_in -= dt;
        }
        for (i, w) in q.me.weapons.iter_mut().enumerate() {
            let Some(w) = w else { continue };
            let held = pending.iter().all(|s| s.fire.get(i).copied().unwrap_or(false));
            w.cooldown = (w.cooldown - dt).max(0.0);
            w.reload = (w.reload - dt).max(0.0);
            if w.kind == "laser" && w.charge > 0.0 {
                if held {
                    w.charge = (w.charge + dt).min(w.stat("charge", 0.75));
                } else {
                    w.charge = 0.0;
                    w.cooldown = w.stat("cooldown", 1.65);
                }
            }
        }
        q
    }

    fn decide(&mut self, o: &Obs, raw: [f32; 3]) -> Action {
        let me = o.me.clone();
        let pos = me.pos;
        let p = self.params.clone();
        let mut a = Action { tick: o.tick, aim: me.aim, ..Default::default() };
        let hp_frac = me.hp / me.max_hp.max(1.0);
        let walk_speed = me.stat("walk_speed", 11.7);

        // --- memory of mechs
        for m in o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0)) {
            let prev_charge = self.known.get(&m.id).and_then(|k| k.charge_since);
            let prev_vel = self.known.get(&m.id).filter(|k| o.t - k.seen_t < 0.35 && o.t > k.seen_t).map(|k| (k.vel, k.seen_t));
            self.known.insert(
                m.id,
                Known {
                    pos: m.pos,
                    vel: m.vel,
                    seen_t: o.t,
                    prev_vel,
                    charge_since: if m.charging { Some(prev_charge.unwrap_or(o.t)) } else { None },
                    aim_yaw: m.aim[0],
                },
            );
        }
        let mem = p.memory_s;
        self.known.retain(|_, k| o.t - k.seen_t < mem);
        self.attackers.retain(|_, t| o.t - *t < 10.0);
        self.drone_attackers.retain(|_, t| o.t - *t < 6.0);
        self.dodged_charge.retain(|_, t| o.t - *t < 3.0);

        // --- loot memory: remember what we saw, forget what should be in view but is not
        for c in &o.crates {
            self.crates_mem.insert(c.id, c.pos);
        }
        for c in &o.cubes {
            self.cubes_mem.insert(c.id, c.pos);
        }
        for pk in &o.pickups {
            self.picks_mem.insert(pk.id, (pk.kind.clone(), pk.pos));
        }
        {
            let eye = [pos[0], pos[1], pos[2] + me.stat("eye_height", 8.0)];
            let w = self.world.clone();
            let gone = |q: &[f32; 3], listed: bool| -> bool {
                let d = d2(q, &pos);
                d > 400.0 || (!listed && d < 40.0 && w.as_ref().map_or(true, |w| w.clear(eye, [q[0], q[1], q[2] + 1.0])))
            };
            self.crates_mem.retain(|id, q| !gone(q, o.crates.iter().any(|c| c.id == *id)));
            self.cubes_mem.retain(|id, q| !gone(q, o.cubes.iter().any(|c| c.id == *id)));
            self.picks_mem.retain(|id, q| !gone(&q.1, o.pickups.iter().any(|c| c.id == *id)));
        }

        // --- what hurt us
        for e in &o.events {
            if e.kind == "kill" {
                if let Some(k) = e.target.and_then(|id| self.known.get(&id)) {
                    self.kill_spot = Some((k.pos, o.t));
                }
            }
            if e.kind == "damaged" {
                if let Some(by) = e.by {
                    self.last_hurt_t = o.t;
                    if by < 100 {
                        self.last_mech_hurt_t = o.t;
                        self.attackers.insert(by, o.t);
                        if let Some(b) = e.bearing {
                            if !o.mechs.iter().any(|m| m.id == by) {
                                self.threat = Some(([pos[0] + b.cos() * 70.0, pos[1] + b.sin() * 70.0, pos[2]], o.t));
                            }
                        }
                    } else {
                        self.drone_attackers.insert(by, o.t);
                    }
                }
            }
        }

        // --- upgrades
        if let Some(offer) = &o.offer {
            let list = if p.upg_dmg > 0.0 { UPGRADES_DMG } else { UPGRADES };
            let rank = |id: &str| list.iter().position(|u| *u == id).unwrap_or(list.len());
            a.pick = offer.choices.iter().enumerate().min_by_key(|(_, c)| rank(&c.id)).map(|(i, _)| i as u32);
        }

        // --- enemies and target
        let enemies: Vec<&Actor> = o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0)).collect();
        let near_enemies = enemies.iter().filter(|m| d2(&m.pos, &pos) < p.crowd_radius).count();
        let no_laser = !me.weapons.iter().flatten().any(|w| w.kind == "laser");
        let score = |m: &Actor| {
            let d = d2(&m.pos, &pos);
            let hpf = if m.max_hp > 0.0 { m.hp / m.max_hp } else { 1.0 };
            let attacker = self.attackers.get(&m.id).map_or(false, |t| o.t - t < 4.0);
            let loot = if no_laser && m.weapons.iter().any(|w| w == "laser") { p.loot_target } else { 0.0 };
            d + hpf * p.focus_hp - if attacker { 40.0 } else { 0.0 } - loot
        };
        let best = enemies
            .iter()
            .map(|m| (*m, score(*m)))
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
        let target = match (best, self.focus.and_then(|f| enemies.iter().find(|m| m.id == f))) {
            (Some((_, bs)), Some(f)) if p.focus_stick > 0.0 && score(*f) - p.focus_stick < bs && d2(&f.pos, &pos) < p.open_fire_range => Some(*f),
            (Some((b, _)), _) => Some(b),
            _ => None,
        };
        self.focus = target.map(|t| t.id);
        if let Some(t) = target {
            self.threat = Some((t.pos, o.t));
        }
        if self.threat.map_or(false, |(_, t)| o.t - t > 8.0) {
            self.threat = None;
        }
        let threats = self.threats(o);

        // Drone to deal with: one shooting us, or (when calm) one worth farming.
        let calm = enemies.is_empty() && o.t - self.last_mech_hurt_t > 5.0;
        let drones: Vec<&Actor> = o.mechs.iter().filter(|m| m.kind == "npc" && m.hp > 0.0).collect();
        let hostile_drone = drones
            .iter()
            .filter(|d| self.drone_attackers.contains_key(&d.id))
            .min_by(|x, y| d2(&x.pos, &pos).partial_cmp(&d2(&y.pos, &pos)).unwrap_or(std::cmp::Ordering::Equal))
            .copied();
        let early = o.t < p.farm_until;
        let drone_reach = if early { p.drone_range.max(p.farm_reach.min(120.0)) } else { p.drone_range };
        let farm_drone = if calm && hp_frac > p.farm_hp && !self.healing {
            drones
                .iter()
                .filter(|d| d2(&d.pos, &pos) < drone_reach)
                .min_by(|x, y| d2(&x.pos, &pos).partial_cmp(&d2(&y.pos, &pos)).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
        } else {
            None
        };
        let drone_target = if target.is_none() {
            match hostile_drone {
                Some(d) if hp_frac > 0.35 => Some(d),
                _ => farm_drone,
            }
        } else {
            None
        };
        let shoot_at = target.filter(|t| d2(&t.pos, &pos) < p.open_fire_range || self.attackers.contains_key(&t.id)).or(drone_target);

        // --- weapons: solve every slot, aim with the best ready one, fire all that line up
        let mut engaged = false;
        if let (Some(t), true) = (shoot_at, p.dual > 0.0) {
            engaged = self.fire_both(o, t, &mut a);
        } else if let Some(t) = shoot_at {
            let mut best: Option<(i32, [f32; 2])> = None;
            let mut fallback = None;
            for (i, slot) in me.weapons.iter().enumerate().take(2) {
                let Some(w) = slot else { continue };
                let s = self.solve(o, w, t);
                fallback.get_or_insert(s.aim);
                if !s.ok {
                    if w.kind == "laser" && w.charge > 0.0 {
                        a.fire[i] = true;
                    }
                    continue;
                }
                engaged = true;
                // The torso turns toward the new aim before a gun shot leaves, so allow that much.
                let slack = if w.kind == "gun" {
                    [me.stat("turn_rate", 4.0) * self.tick_s * p.aim_slack, me.stat("pitch_rate", 3.0) * self.tick_s * p.aim_slack]
                } else {
                    [0.0, 0.0]
                };
                let lined = wrap(s.aim[0] - me.aim[0]).abs() <= s.tol[0] + slack[0] && (s.aim[1] - me.aim[1]).abs() <= s.tol[1] + slack[1];
                let pri = match w.kind.as_str() {
                    "lightning" => 4,
                    "laser" => 3,
                    "gun" => 2,
                    _ => 1,
                };
                if w.kind == "laser" {
                    let full = w.charge >= w.stat("charge", 0.75) - 0.01;
                    a.fire[i] = w.cooldown <= 0.0 && !(full && lined);
                } else {
                    a.fire[i] = lined && w.cooldown <= 0.12;
                }
                let usable = w.cooldown <= 0.6;
                if usable && best.map_or(true, |b| pri > b.0) {
                    best = Some((pri, s.aim));
                }
            }
            if let Some(aim) = best.map(|b| b.1).or(fallback) {
                a.aim = aim;
            }
        }

        // --- laser: track how long a full charge has been held; charge ahead for an enemy about to show
        if let Some((li, lw)) = me.weapons.iter().enumerate().take(2).find_map(|(i, w)| w.as_ref().filter(|w| w.kind == "laser").map(|w| (i, w))) {
            let full = lw.charge >= lw.stat("charge", 0.75) - 0.01;
            if !full {
                self.laser_full_t = None;
            } else if self.laser_full_t.is_none() {
                self.laser_full_t = Some(o.t);
            }
            let held_full = self.laser_full_t.map_or(0.0, |t0| o.t - t0);
            let soon = p.precharge > 0.0
                && !engaged
                && lw.cooldown <= 0.0
                && self.known.iter().any(|(id, k)| o.t - k.seen_t < 2.0 && d2(&k.pos, &pos) < p.precharge && !enemies.iter().any(|m| m.id == *id));
            if soon && held_full < lw.stat("hold_s", 2.0) - 0.4 {
                a.fire[li] = true;
            }
        }
        // --- gun housekeeping
        if let Some(Some(g)) = me.weapons.iter().find(|w| w.as_ref().map_or(false, |w| w.kind == "gun")) {
            let mag = g.stat("mag", 12.0);
            // With a laser shot on the way, reload a low magazine now so the gun is back when the beam is spent.
            let laser_busy = me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.charge > 0.0);
            let early = p.reload_early > 0.0 && laser_busy && g.ammo <= p.reload_early;
            if g.reload <= 0.0 && (g.ammo <= 0.0 || early || (!engaged && g.ammo < mag * 0.6)) {
                a.reload = true;
            }
        }

        // --- loot within reach
        let held: Vec<&str> = me.weapons.iter().flatten().map(|w| w.kind.as_str()).collect();
        let empty_slot = me.weapons.len() < 2 || me.weapons.iter().take(2).any(|w| w.is_none());
        let worst_slot = me.weapons.iter().enumerate().take(2).filter_map(|(i, w)| w.as_ref().map(|w| (i, weapon_rank(&w.kind, &p)))).filter(|(_, r)| *r > 0).min_by_key(|(_, r)| *r);
        // Optionally the gun is the slot to give up once the other slot holds a laser or lightning.
        let gun_i = me.weapons.iter().take(2).position(|w| w.as_ref().map_or(false, |w| w.kind == "gun"));
        let worst_slot = match (gun_i, worst_slot) {
            (Some(gi), Some((_, r))) if p.drop_gun > 0.0 && !empty_slot && r >= 2 => Some((gi, 1)),
            _ => worst_slot,
        };
        let wanted = |kind: &str| -> Option<Option<u32>> {
            if held.contains(&kind) || weapon_rank(kind, &p) == 0 {
                None
            } else if empty_slot {
                Some(None)
            } else {
                worst_slot.filter(|(_, r)| weapon_rank(kind, &p) > *r).map(|(i, _)| Some(i as u32))
            }
        };
        for pk in &o.pickups {
            if d2(&pk.pos, &pos) < self.pickup_radius - 0.3 && (pk.pos[2] - pos[2]).abs() < 3.7 {
                if let Some(drop) = wanted(&pk.kind) {
                    let charging = me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.charge > 0.0);
                    if !(drop.is_some() && charging) {
                        a.take = Some(pk.id);
                        a.drop = drop;
                        break;
                    }
                }
            }
        }

        // --- zone: where we must be, and how urgently
        let z = &o.zone;
        let margin = (z.next_radius * 0.35).clamp(0.0, 30.0);
        let out_next = d2(&pos, &z.next_center) - (z.next_radius - margin);
        let lead = out_next.max(0.0) / walk_speed * p.zone_lead_mult + p.zone_lead;
        let must_rotate = !me.in_zone || (out_next > 0.0 && o.t >= z.shrink_start - lead) || z.next_radius <= 1.0;
        let zone_goal = {
            let d = norm2(pos[0] - z.next_center[0], pos[1] - z.next_center[1]);
            let r = (z.next_radius - margin - 5.0).max(0.0);
            [z.next_center[0] + d[0] * r, z.next_center[1] + d[1] * r]
        };

        // --- mode
        let threat_live = !threats.is_empty();
        let recently_hurt_by_mech = o.t - self.last_mech_hurt_t < 6.0;
        if hp_frac < p.retreat_hp && (threat_live || recently_hurt_by_mech) {
            self.fleeing = true;
        }
        if p.break_los > 0.0 && enemies.len() as f32 >= p.break_los && hp_frac < p.break_los_hp {
            self.fleeing = true;
        }
        if hp_frac >= p.reengage_hp {
            self.fleeing = false;
        }
        // A nearly dead target is worth finishing rather than running from.
        let finishing = target.map_or(false, |t| {
            let tf = if t.max_hp > 0.0 { t.hp / t.max_hp } else { 1.0 };
            tf < p.finish_hp && tf * t.max_hp < me.hp * 0.8 && near_enemies <= 2
        });
        if finishing {
            self.fleeing = false;
        }
        let crowded = near_enemies >= p.crowd_count && !finishing;
        if p.heal_lo > 0.0 && enemies.is_empty() && hp_frac < p.heal_lo && o.t - self.last_mech_hurt_t > 1.0 {
            self.healing = true;
        }
        if hp_frac >= p.heal_hi || p.heal_lo <= 0.0 {
            self.healing = false;
        }

        // A laser charging and pointed at us: keep moving across its line until it fires.
        // Several charging at once: track the nearest (ties broken by id).
        let mut laser_on: Option<[f32; 3]> = None;
        let mut flip_now = false;
        let mut laser_d = f32::MAX;
        let mut flips: Vec<u32> = Vec::new();
        for (id, k) in self.known.iter() {
            if let Some(since) = k.charge_since {
                let d = d2(&k.pos, &pos);
                if o.t - k.seen_t < 0.25 && wrap(k.aim_yaw - yaw_to(&k.pos, &pos)).abs() < 0.2 && d < 300.0 {
                    if d < laser_d {
                        laser_d = d;
                        laser_on = Some(k.pos);
                    }
                    if o.t - since >= p.dodge_after && !self.dodged_charge.contains_key(id) {
                        flip_now = true;
                        flips.push(*id);
                    }
                }
            }
        }
        for id in flips {
            self.dodged_charge.insert(id, o.t);
        }
        if laser_on.is_some() {
            // Do not let the random dwell reverse us mid-charge.
            self.strafe_until = self.strafe_until.max(o.t + 0.2);
            if flip_now {
                self.strafe = -self.strafe;
            }
        }
        // The nearest enemy we can see, to move across its line of fire.
        let shooter = laser_on.or_else(|| enemies.iter().filter(|m| d2(&m.pos, &pos) < 200.0).min_by(|x, y| d2(&x.pos, &pos).partial_cmp(&d2(&y.pos, &pos)).unwrap_or(std::cmp::Ordering::Equal)).map(|m| m.pos));

        // Nearest cube on the ground close by: worth stepping over to even mid-fight.
        let near_cube = if p.cube_grab > 0.0 {
            self.cubes_mem
                .values()
                .filter(|q| d2(*q, &pos) < p.cube_grab && (q[2] - self.ground(q[0], q[1], q[2])).abs() < 3.0 && (q[2] - pos[2]).abs() < 6.0)
                .min_by(|x, y| d2(*x, &pos).partial_cmp(&d2(*y, &pos)).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
        } else {
            None
        };
        if p.explore > 0.0 {
            self.mark_looked(&pos);
        }
        // Lightning rush: a target close enough and few others around, close in and burn it down.
        let arc_w = me.weapons.iter().flatten().find(|w| w.kind == "lightning");
        let rush_t = match (target, arc_w) {
            (Some(t), Some(_)) if p.rush > 0.0 => {
                let d = d2(&t.pos, &pos);
                let others = enemies.iter().filter(|m| m.id != t.id && d2(&m.pos, &pos) < p.crowd_radius).count() as f32;
                let tf = if t.max_hp > 0.0 { t.hp } else { 1400.0 };
                let cont = self.rushing == Some(t.id);
                let e_ok = if cont { me.energy > p.arc_energy * 0.5 } else { me.energy > p.rush_energy };
                let hp_ok = hp_frac > p.rush_hp || tf < me.hp * 0.7;
                let reach = if cont { p.rush_start + 20.0 } else { p.rush_start };
                if d < reach && e_ok && hp_ok && others <= p.rush_crowd {
                    Some(t)
                } else {
                    None
                }
            }
            _ => None,
        };
        self.rushing = rush_t.map(|t| t.id);
        if rush_t.is_some() {
            self.fleeing = false;
        }
        // An empty slot: the nearest good weapon on the ground comes first.
        let worst_rank = worst_slot.map_or(0, |(_, r)| r);
        let arm_goal = if p.arm > 0.0 && (empty_slot || p.arm_better > 0.0 || p.laser_seek > 0.0) && hp_frac > p.arm_hp {
            let reach = if empty_slot { p.arm_reach } else { p.arm_better };
            let cands: Vec<(u32, [f32; 3], f32)> = self
                .picks_mem
                .iter()
                .filter(|(_, (k, q))| {
                    let r = weapon_rank(k, &p);
                    let d = d2(q, &pos);
                    let laser = k == "laser" && p.laser_seek > 0.0 && d < p.laser_seek;
                    let up = q[2] - self.ground(q[0], q[1], q[2]);
                    let reachable = up.abs() < 3.0 || (p.roof > 0.0 && up < 22.0);
                    r > 0 && (laser || ((empty_slot || r > worst_rank) && d < reach)) && !held.contains(&k.as_str()) && reachable
                })
                .map(|(id, (k, q))| {
                    let up = q[2] - self.ground(q[0], q[1], q[2]);
                    let extra = if up.abs() < 3.0 || (q[2] - pos[2]).abs() < 3.0 { 0.0 } else { p.roof_cost };
                    let keep = if self.arm_pick == Some(*id) { p.arm_keep } else { 0.0 };
                    (*id, *q, d2(q, &pos) + extra - keep - p.arm_rank * weapon_rank(k, &p) as f32)
                })
                .collect();
            let best = cands.iter().min_by(|x, y| x.2.partial_cmp(&y.2).unwrap_or(std::cmp::Ordering::Equal).then(x.0.cmp(&y.0)));
            self.arm_pick = best.map(|b| b.0);
            best.map(|b| b.1)
        } else {
            self.arm_pick = None;
            None
        };
        // Cubes dropped by a kill: the nearest one known around the spot, or the spot itself until we have looked.
        let collect_goal = match self.kill_spot {
            Some((ks, kt)) if p.collect > 0.0 && o.t - kt < p.collect_s && hp_frac > p.collect_hp && (enemies.len() as f32) < p.collect_seen => {
                let c = self
                    .cubes_mem
                    .values()
                    .filter(|q| d2(*q, &ks) < 14.0 && (q[2] - pos[2]).abs() < 6.0)
                    .min_by(|x, y| d2(*x, &pos).partial_cmp(&d2(*y, &pos)).unwrap_or(std::cmp::Ordering::Equal))
                    .copied();
                match c {
                    Some(c) => Some(c),
                    None if d2(&ks, &pos) > 6.0 && o.t - kt < 3.0 => Some(ks),
                    _ => None,
                }
            }
            _ => None,
        };
        if self.peek.map_or(false, |(_, id, t0)| o.t - t0 > p.peek_s || !self.known.contains_key(&id)) {
            self.peek = None;
        }
        let peek_wait = self.peek.filter(|_| hp_frac > p.retreat_hp && !must_rotate && enemies.is_empty()).and_then(|(spot, id, _)| self.known.get(&id).map(|k| (spot, k.pos)));
        // Stand-off: the nearest enemy in sight that is closer than we want (nearly dead ones may come in).
        let stand_from = if p.standoff > 0.0 && !finishing {
            enemies
                .iter()
                .filter(|m| {
                    let f = if m.max_hp > 0.0 { m.hp / m.max_hp } else { 1.0 };
                    d2(&m.pos, &pos) < p.standoff && f >= p.stand_finish
                })
                .min_by(|x, y| d2(&x.pos, &pos).partial_cmp(&d2(&y.pos, &pos)).unwrap_or(std::cmp::Ordering::Equal))
                .map(|m| m.pos)
        } else {
            None
        };
        let late = (p.wall_t > 0.0 && o.t >= p.wall_t) || (p.wall_t > 0.0 && o.zone.radius <= p.wall_r);
        let mut boost = false;
        let mut mv: [f32; 2];
        let mut label = "idle";
        let zone_stand = p.stand_zone > 0.0 && stand_from.is_some() && me.in_zone && d2(&pos, &z.center) < z.radius - p.stand_zone_m;
        if !me.in_zone || (must_rotate && out_next > 25.0 && !zone_stand) {
            label = "zone";
            mv = self.go(&pos, zone_goal);
            boost = !me.in_zone && me.energy > 5.0;
        } else if let Some(g) = arm_goal.filter(|_| !(p.stand_first > 0.0 && stand_from.is_some())) {
            label = "arm";
            mv = self.go(&pos, [g[0], g[1]]);
            boost = me.energy > 30.0 && d2(&g, &pos) > 12.0;
            if p.roof > 0.0 && (g[2] - pos[2] > 2.5 || (self.label == "climb" && !me.grounded)) {
                // Up on a roof: walk to the wall below it, then jump and jet up and over the edge.
                let hd = d2(&g, &pos);
                let dir = norm2(g[0] - pos[0], g[1] - pos[1]);
                let at_wall = !self.walkable(pos[0] + dir[0] * 4.0, pos[1] + dir[1] * 4.0) || hd < 4.0;
                let climbing = !me.grounded && self.label == "climb";
                boost = me.energy > 70.0 && hd > 20.0;
                if climbing || (hd < 30.0 && at_wall && me.energy > p.roof_energy) {
                    label = "climb";
                    mv = dir;
                    a.jump = me.grounded;
                    boost = pos[2] < g[2] + 2.0;
                }
            }
        } else if let Some(c) = collect_goal.filter(|_| !(p.stand_first > 0.0 && stand_from.is_some())) {
            label = "collect";
            mv = self.go(&pos, [c[0], c[1]]);
            boost = me.energy > 30.0 && d2(&c, &pos) > 10.0;
        } else if let Some(t) = rush_t {
            label = "rush";
            let dist = d2(&t.pos, &pos);
            let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
            let arc_r = arc_w.map_or(45.0, |w| w.stat("range", 45.0));
            let radial = ((dist - p.rush_range) / 8.0).clamp(-1.0, 1.0);
            mv = if dist > p.rush_range + 10.0 && !self.straight(&pos, &t.pos) {
                self.go(&pos, [t.pos[0], t.pos[1]])
            } else {
                self.strafe_dir(o, &pos, to, radial * 2.0, false)
            };
            boost = dist > arc_r - 6.0 && me.energy > p.rush_boost;
            if laser_on.is_some() && me.energy > 10.0 {
                boost = true;
            }
        } else if let Some(near) = stand_from.filter(|_| !must_rotate || out_next < 0.0 || zone_stand) {
            // Too close: back away from everyone in sight while the weapons keep working.
            label = "stand";
            let pull = if must_rotate && out_next > 0.0 && zone_stand {
                Some((norm2(zone_goal[0] - pos[0], zone_goal[1] - pos[1]), p.stand_zone))
            } else {
                None
            };
            let dir = self.away_dir(o, &pos, p.stand_mem, pull);
            let to = norm2(near[0] - pos[0], near[1] - pos[1]);
            if o.t >= self.strafe_until {
                self.next_strafe(o.t);
            }
            let perp = [-to[1] * self.strafe, to[0] * self.strafe];
            let w = if laser_on.is_some() { 1.5 } else { p.stand_weave };
            let cand = norm2(dir[0] + perp[0] * w, dir[1] + perp[1] * w);
            mv = if self.walkable(pos[0] + cand[0] * 5.0, pos[1] + cand[1] * 5.0) { cand } else { dir };
            if d2(&near, &pos) < p.stand_charge {
                for (i, w) in me.weapons.iter().enumerate().take(2) {
                    if w.as_ref().map_or(false, |w| w.kind == "laser" && w.charge <= 0.0) {
                        a.fire[i] = false;
                    }
                }
            }
            let charging = me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.charge > 0.0);
            boost = me.energy > p.stand_boost && !charging;
        } else if (self.fleeing || (crowded && p.break_crowd > 0.0)) && threat_live && p.retreat_mode > 0.0 {
            // Break line of sight from everyone we know of, then heal out of sight.
            label = "break";
            let all = self.threats_all(o);
            let expired = self.cover.map_or(true, |(_, until)| o.t > until);
            let exposed = self.cover.map_or(false, |(c, _)| {
                let z = self.ground(c[0], c[1], pos[2]);
                self.hidden_from(c, z, &all) < all.len()
            });
            if expired || exposed {
                let c = self.find_hide(o, &all);
                self.cover = Some((c, o.t + 1.0));
            }
            let c = self.cover.unwrap().0;
            let dc = d2(&pos, &c);
            mv = self.go(&pos, c);
            if dc < 2.0 {
                mv = [0.0, 0.0];
            }
            boost = me.energy > p.retreat_boost_energy && dc > 5.0;
            if dc > p.break_laser {
                for (i, w) in me.weapons.iter().enumerate().take(2) {
                    if w.as_ref().map_or(false, |w| w.kind == "laser" && w.charge <= 0.0) {
                        a.fire[i] = false;
                    }
                }
            }
            if shoot_at.is_none() {
                if let Some(th) = all.first() {
                    a.aim = [yaw_to(&pos, th), 0.0];
                }
            }
        } else if late && !self.fleeing {
            if p.wall_mode >= 2.0 {
                // Keep away from the enemies we know of, inside the zone.
                label = "evade";
                let all = self.threats_all(o);
                let refp = if all.is_empty() { None } else {
                    let n = all.len() as f32;
                    Some(all.iter().fold([0.0f32, 0.0f32], |a, t| [a[0] + t[0] / n, a[1] + t[1] / n]))
                };
                let g = self.far_spot(o, &pos, refp);
                mv = self.go(&pos, g);
                if d2(&pos, &g) < 3.0 {
                    mv = [0.0, 0.0];
                }
                boost = me.energy > 60.0 && d2(&pos, &g) > 20.0;
            } else {
                label = "hold";
                let stale = self.hold.map_or(true, |(h, t0)| o.t - t0 > 20.0 || !self.in_next(o, &h, 10.0));
                if stale {
                    let h = self.hold_spot(o, &pos);
                    self.hold = Some((h, o.t));
                }
                let h = self.hold.unwrap().0;
                let dh = d2(&pos, &h);
                mv = self.go(&pos, h);
                if dh < 2.5 {
                    mv = [0.0, 0.0];
                }
                boost = me.energy > 60.0 && dh > 25.0;
            }
            if shoot_at.is_none() {
                if let Some(th) = threats.first() {
                    a.aim = [yaw_to(&pos, th), 0.0];
                } else {
                    self.scan(&mut a);
                }
            }
        } else if let (Some(t), true) = (target, (self.fleeing || crowded) && p.kite > 0.0 && threat_live) {
            // Back away from the nearest enemy while weaving and shooting.
            label = if self.fleeing { "kite" } else { "kitec" };
            let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
            let radial = if laser_on.is_some() { -0.2 } else { -p.kite };
            mv = self.strafe_dir(o, &pos, to, radial, false);
            if p.kite_away > 0.0 && laser_on.is_none() {
                let dir = self.away_dir(o, &pos, 3.0, None);
                let perp = [-to[1] * self.strafe, to[0] * self.strafe];
                let cand = norm2(dir[0] + perp[0] * p.stand_weave, dir[1] + perp[1] * p.stand_weave);
                mv = if self.walkable(pos[0] + cand[0] * 5.0, pos[1] + cand[1] * 5.0) { cand } else { dir };
            }
            boost = me.energy > p.retreat_boost_energy;
        } else if (self.fleeing || crowded) && threat_live {
            label = if self.fleeing { "heal" } else { "crowd" };
            if self.cover.map_or(true, |(_, until)| o.t > until) {
                let c = self.find_cover(o, &threats);
                self.cover = Some((c, o.t + 1.0));
            }
            let c = self.cover.unwrap().0;
            mv = self.go(&pos, c);
            if d2(&pos, &c) < 3.0 {
                // At cover: small side steps so we are not a still target.
                mv = self.jitter(o, &pos, c);
            }
            boost = me.energy > p.retreat_boost_energy && d2(&pos, &c) > 8.0;
            if shoot_at.is_none() {
                let th = threats[0];
                a.aim = [yaw_to(&pos, &th), 0.0];
            }
        } else if self.healing && enemies.is_empty() && hostile_drone.is_none() && !must_rotate && near_cube.is_none() {
            // Out of sight and hurt: get out of the lines of anyone we know of, then hold still and heal.
            label = "mend";
            if !threats.is_empty() {
                if self.cover.map_or(true, |(_, until)| o.t > until) {
                    let c = self.find_cover(o, &threats);
                    self.cover = Some((c, o.t + 1.5));
                }
                let c = self.cover.unwrap().0;
                mv = self.go(&pos, c);
                if d2(&pos, &c) < 3.0 {
                    mv = [0.0, 0.0];
                }
                a.aim = [yaw_to(&pos, &threats[0]), 0.0];
            } else {
                mv = self.idle_move(o, &pos);
                mv = [mv[0] * 0.6, mv[1] * 0.6];
                self.scan(&mut a);
            }
        } else if self.fleeing {
            // Healing with no known threat: stay near shelter.
            label = "rest";
            mv = if p.retreat_mode > 0.0 { [0.0, 0.0] } else { self.idle_move(o, &pos) };
            if shoot_at.is_none() {
                self.scan(&mut a);
            }
        } else if let Some((spot, tk)) = peek_wait {
            // Out of sight: charge the laser, then step back to where we fired from.
            let lw = me.weapons.iter().enumerate().take(2).find_map(|(i, w)| w.as_ref().filter(|w| w.kind == "laser").map(|w| (i, w.charge >= w.stat("charge", 0.75) - 0.01, w.cooldown)));
            a.aim = [yaw_to(&pos, &tk), 0.0];
            match lw {
                Some((li, full, cd)) if !full => {
                    label = "hide";
                    a.fire[li] = cd <= 0.0 || me.weapons[li].as_ref().map_or(false, |w| w.charge > 0.0);
                    mv = [0.0, 0.0];
                }
                Some((li, _, _)) => {
                    label = "repeek";
                    a.fire[li] = true;
                    mv = self.go(&pos, spot);
                    if d2(&pos, &spot) < 1.5 {
                        mv = norm2(tk[0] - pos[0], tk[1] - pos[1]);
                    }
                }
                None => {
                    label = "repeek";
                    mv = self.go(&pos, spot);
                }
            }
        } else if let (Some(t), true) = (target, engaged || recently_hurt_by_mech || target.map_or(false, |t| d2(&t.pos, &pos) < p.open_fire_range)) {
            label = "fight";
            let dist = d2(&t.pos, &pos);
            let has_arc = me.weapons.iter().flatten().any(|w| w.kind == "lightning") && me.energy > p.arc_energy;
            let they_arc = t.weapons.iter().any(|w| w == "lightning");
            let has_laser = me.weapons.iter().flatten().any(|w| w.kind == "laser");
            let pref = if finishing {
                p.finish_range
            } else if has_arc && hp_frac > p.arc_min_hp && near_enemies as f32 <= p.arc_crowd {
                p.arc_range
            } else if they_arc {
                p.gun_range.max(60.0).max(p.laser_range * has_laser as i32 as f32)
            } else if has_laser && p.laser_range > 0.0 {
                p.laser_range
            } else {
                p.gun_range
            };
            let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
            let off = dist - pref;
            let radial = if off.abs() < p.gun_band && pref > 50.0 { 0.0 } else { (off / 15.0).clamp(-1.0, 1.0) };
            let radial = if laser_on.is_some() { radial * 0.2 } else { radial };
            mv = self.strafe_dir(o, &pos, to, radial, false);
            let reloading = me.weapons.iter().flatten().any(|w| w.kind == "gun" && (w.reload > 0.4 || w.ammo <= 0.0));
            let eyes: Vec<[f32; 3]> = self
                .known
                .values()
                .filter(|k| o.t - k.seen_t < 4.0 && d2(&k.pos, &pos) < 200.0)
                .map(|k| [k.pos[0], k.pos[1], k.pos[2] + 8.0])
                .collect();
            let laser_up = has_laser && p.reload_laser <= 0.0 && me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.cooldown < 0.8);
            let hide = if p.reload_hide > 0.0 && reloading && !(has_arc && dist < 45.0) && !finishing && !laser_up {
                self.hide_step(&pos, &eyes, p.reload_hide)
            } else {
                None
            };
            let just_fired = me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.cooldown > 0.6 && w.charge <= 0.0);
            let peek = if p.peek > 0.0 && hide.is_none() && just_fired && !finishing && (enemies.len() as f32) <= p.peek_crowd {
                self.hide_step(&pos, &eyes, p.peek)
            } else {
                None
            };
            if let Some(h) = peek {
                label = "peek";
                mv = h;
                boost = me.energy > 25.0;
                self.peek = Some(([pos[0], pos[1]], t.id, o.t));
            } else if let Some(h) = hide {
                label = "reload";
                mv = h;
                boost = me.energy > 25.0;
            } else if let Some(c) = near_cube.filter(|_| !finishing || dist < 20.0) {
                label = "fcube";
                let g = self.go(&pos, [c[0], c[1]]);
                mv = norm2(g[0] + mv[0] * 0.35, g[1] + mv[1] * 0.35);
                boost = me.energy > 30.0;
            } else if p.expose_weight > 0.0 {
                mv = self.less_exposed(o, &pos, t, mv, p.expose_weight);
            }
            mv = [mv[0] * p.fight_speed, mv[1] * p.fight_speed];
            boost = boost || (laser_on.is_some() && me.energy > 10.0);
        } else if let Some(d) = drone_target {
            label = "drone";
            let dist = d2(&d.pos, &pos);
            let to = norm2(d.pos[0] - pos[0], d.pos[1] - pos[1]);
            let radial = ((dist - 50.0) / 15.0).clamp(-1.0, 1.0);
            mv = self.strafe_dir(o, &pos, to, radial, false);
        } else if let Some(d) = hostile_drone {
            // Too hurt to fight it: walk away from it.
            label = "leave";
            let away = norm2(pos[0] - d.pos[0], pos[1] - d.pos[1]);
            let goal = [pos[0] + away[0] * 30.0, pos[1] + away[1] * 30.0];
            mv = self.go(&pos, goal);
        } else {
            mv = self.quiet(o, &mut a, &pos, hp_frac, must_rotate, out_next, zone_goal, &threats, &mut label);
            if label == "idle" && p.explore > 0.0 && !held.contains(&"laser") && hp_frac > p.arm_hp && !self.healing && !must_rotate {
                if let Some(g) = self.unexplored(o, &pos) {
                    label = "seek";
                    mv = self.go(&pos, g);
                    self.scan(&mut a);
                }
            }
            let armed = me.weapons.iter().flatten().any(|w| w.kind == "laser");
            if label == "idle" && p.hunt > 0.0 && armed && hp_frac > p.hunt_hp && !self.healing && !must_rotate {
                let seen = self
                    .known
                    .values()
                    .filter(|k| d2(&k.pos, &pos) < p.hunt)
                    .max_by(|x, y| x.seen_t.partial_cmp(&y.seen_t).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|k| k.pos);
                let heard = o
                    .sounds
                    .iter()
                    .filter(|sd| matches!(sd.kind.as_str(), "gun" | "laser" | "lightning" | "mortar" | "step" | "boost"))
                    .min_by(|x, y| x.age.partial_cmp(&y.age).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|sd| {
                        let r = match sd.dist.as_str() {
                            "near" => 40.0,
                            "mid" => 90.0,
                            _ => 150.0,
                        };
                        [pos[0] + sd.bearing.cos() * r, pos[1] + sd.bearing.sin() * r, pos[2]]
                    });
                if let Some(g) = seen.or(heard) {
                    let inz = d2(&g, &z.center) < z.radius - 10.0;
                    if inz {
                        label = "hunt";
                        mv = self.go(&pos, [g[0], g[1]]);
                    }
                }
            }
        }

        // A laser about to fire at us: duck out of its line if a step away does it.
        let mut ducked = false;
        if let (Some(src), true) = (laser_on, p.laser_hide > 0.0 && label != "zone") {
            if let Some(h) = self.hide_step(&pos, &[[src[0], src[1], src[2] + 7.0]], p.laser_hide) {
                mv = h;
                boost = me.energy > 5.0;
                ducked = true;
            }
        }
        // Outside a fight, still weave across the line of the nearest shooter.
        if let (Some(src), false) = (shooter, ducked || label == "zone" || label == "fight" || label == "rush" || label == "climb" || label == "hold" || label == "drone" || label == "reload" || label == "stand" || label == "peek" || label == "hide" || label == "repeek") {
            if o.t >= self.strafe_until {
                self.next_strafe(o.t);
            }
            let to = norm2(src[0] - pos[0], src[1] - pos[1]);
            let perp = [-to[1] * self.strafe, to[0] * self.strafe];
            let w = if laser_on.is_some() { 3.0 } else { p.evade_weight };
            let cand = norm2(mv[0] + perp[0] * w, mv[1] + perp[1] * w);
            if self.walkable(pos[0] + cand[0] * 5.0, pos[1] + cand[1] * 5.0) {
                mv = cand;
            } else {
                self.strafe = -self.strafe;
            }
            if laser_on.is_some() && me.energy > 10.0 {
                boost = true;
            }
        }

        // --- stuck: wanting to move but not moving means a ledge or a wall; hop and jet over it
        let wants = mv[0].abs() + mv[1].abs() > 0.1;
        if wants && d2(&raw, &self.last_pos) < 0.15 {
            self.stuck += 1;
        } else {
            self.stuck = 0;
        }
        if self.stuck >= 5 {
            self.unstick_until = o.t + 0.8;
            self.strafe = -self.strafe;
            self.wander = None;
            self.stuck = 0;
        }
        if o.t < self.unstick_until {
            a.jump = me.grounded;
            boost = true;
        }
        self.label = label;
        if std::env::var("TTRACE_DELAY").is_ok() && o.tick % 200 == 0 {
            eprintln!("DL t {:.0} used {:.2} info {:.1} err {:?}", o.t, self.delay_used, self.delay_info, self.delay_err);
        }
        if self.trace_dmg {
            let taken: f32 = o.events.iter().filter(|e| e.kind == "damaged" && e.by.map_or(false, |b| b < 100)).map(|e| e.amount).sum();
            let nd = target.map_or(-1.0, |t| d2(&t.pos, &pos));
            let dealt: f32 = o.events.iter().filter(|e| e.kind == "hit" && e.target.map_or(false, |b| b < 100)).map(|e| e.amount).sum();
            let lz: f32 = o.events.iter().filter(|e| e.kind == "hit" && e.weapon.as_deref() == Some("laser") && e.target.map_or(false, |b| b < 100)).map(|e| e.amount).sum();
            let lst = me.weapons.iter().flatten().find(|w| w.kind == "laser").map_or(-1.0, |w| if w.charge > 0.0 { 2.0 } else if w.cooldown > 0.0 { 1.0 } else { 0.0 });
            eprintln!("TD {:.1} {} {:.0} {:.2} {} {} {:.0} {} {:.0} {:.0} {} {} {}", o.t, label, taken, hp_frac, enemies.len(), near_enemies, nd, me.cubes, dealt, lz, lst, held.join("+"), self.picks_mem.values().filter(|q| q.0 == "laser").map(|q| d2(&q.1, &pos)).fold(9999.0f32, f32::min) as i32);
        }
        if std::env::var("TTRACE").is_ok() && o.tick % 20 == 0 {
            eprintln!("TR {:.1} {} hp {:.0} cubes {} enemies {} near {} e {:.0} w {}", o.t, label, me.hp, me.cubes, enemies.len(), near_enemies, me.energy, held.join("+"));
        }
        self.last_pos = raw;
        a.mv = if mv.iter().all(|v| v.is_finite()) { mv } else { [0.0, 0.0] };
        if p.flee_nocharge > 0.0 && matches!(label, "kite" | "kitec" | "heal" | "crowd") {
            for (i, w) in me.weapons.iter().enumerate().take(2) {
                if w.as_ref().map_or(false, |w| w.kind == "laser" && w.charge <= 0.0) {
                    a.fire[i] = false;
                }
            }
        }
        if p.arc_save > 0.0 && arc_w.is_some() && me.energy < p.arc_save && !(label == "rush" || label == "zone" || o.t < self.unstick_until) {
            boost = false;
        }
        a.boost = boost;
        if !a.aim.iter().all(|v| v.is_finite()) {
            a.aim = me.aim;
        }
        a
    }

    /// Direction that best increases the distance to every enemy seen recently, staying on walkable ground
    /// and inside the zone.
    fn away_dir(&mut self, o: &Obs, pos: &[f32; 3], mem: f32, pull: Option<([f32; 2], f32)>) -> [f32; 2] {
        let foes: Vec<([f32; 2], f32)> = self
            .known
            .values()
            .filter(|k| o.t - k.seen_t <= mem && d2(&k.pos, pos) < 220.0)
            .map(|k| {
                let d = d2(&k.pos, pos).max(10.0);
                (norm2(pos[0] - k.pos[0], pos[1] - k.pos[1]), 1.0 / d)
            })
            .collect();
        let z = &o.zone;
        let mut best = (f32::MIN, [0.0f32, 0.0f32]);
        for i in 0..24 {
            let ang = i as f32 * std::f32::consts::TAU / 24.0;
            let dir = [ang.cos(), ang.sin()];
            let mut open = 0.0;
            for step in [3.0f32, 6.0, 10.0, 15.0, 21.0, 28.0] {
                if !self.walkable(pos[0] + dir[0] * step, pos[1] + dir[1] * step) {
                    break;
                }
                open = step;
            }
            if open < 3.0 {
                continue;
            }
            let mut sc: f32 = foes.iter().map(|(u, w)| (u[0] * dir[0] + u[1] * dir[1]) * w).sum::<f32>();
            let wsum: f32 = foes.iter().map(|(_, w)| *w).sum::<f32>().max(1e-3);
            sc /= wsum;
            sc -= (28.0 - open) / 28.0 * 0.8;
            let ahead = [pos[0] + dir[0] * 25.0, pos[1] + dir[1] * 25.0];
            let out = d2(&ahead, &z.center) - (z.radius - 15.0);
            if out > 0.0 {
                sc -= out / 25.0 * 1.5;
            }
            sc += 0.25 * (dir[0] * self.away[0] + dir[1] * self.away[1]);
            if let Some((g, w)) = pull {
                sc += w * (dir[0] * g[0] + dir[1] * g[1]);
            }
            if sc > best.0 {
                best = (sc, dir);
            }
        }
        if best.0 == f32::MIN {
            best.1 = self.away;
        }
        self.away = best.1;
        best.1
    }

    /// Prefer a nearby step from which the target is still in sight but fewer other enemies can see us.
    fn less_exposed(&self, o: &Obs, pos: &[f32; 3], t: &Actor, mv: [f32; 2], w: f32) -> [f32; 2] {
        let others: Vec<[f32; 3]> = self
            .known
            .iter()
            .filter(|(id, k)| **id != t.id && o.t - k.seen_t < 6.0 && d2(&k.pos, pos) < 220.0)
            .map(|(_, k)| [k.pos[0], k.pos[1], k.pos[2] + 8.0])
            .collect();
        if others.is_empty() {
            return mv;
        }
        let tc = [t.pos[0], t.pos[1], t.pos[2] + 4.5];
        let eval = |d: [f32; 2]| -> Option<f32> {
            let c = [pos[0] + d[0] * 6.0, pos[1] + d[1] * 6.0];
            if !self.walkable(c[0], c[1]) {
                return None;
            }
            let z = self.ground(c[0], c[1], pos[2]);
            let seen = others.iter().filter(|e| self.clear(**e, [c[0], c[1], z + 5.0])).count() as f32;
            let tv = self.clear([c[0], c[1], z + 7.0], tc);
            Some(-seen * 2.0 * w + if tv { 1.0 } else { 0.0 } + (d[0] * mv[0] + d[1] * mv[1]) * 0.8)
        };
        let mut best = (mv, eval(mv).unwrap_or(f32::MIN));
        for k in 0..8 {
            let a = k as f32 * TAU / 8.0;
            let d = [a.cos(), a.sin()];
            if let Some(s) = eval(d) {
                if s > best.1 + 0.3 {
                    best = (d, s);
                }
            }
        }
        best.0
    }

    /// Direction to the nearest spot within `r` (straight walk) that none of `eyes` can see.
    fn hide_step(&self, pos: &[f32; 3], eyes: &[[f32; 3]], r: f32) -> Option<[f32; 2]> {
        if eyes.is_empty() {
            return None;
        }
        let hidden = |c: [f32; 2], z: f32| eyes.iter().all(|e| !self.clear(*e, [c[0], c[1], z + 4.5]) && !self.clear(*e, [c[0], c[1], z + 8.5]));
        if hidden([pos[0], pos[1]], pos[2]) {
            return Some([0.0, 0.0]);
        }
        let mut ring = 3.5f32;
        while ring <= r + 0.01 {
            let mut best: Option<([f32; 2], f32)> = None;
            for k in 0..12 {
                let a = k as f32 * TAU / 12.0 + ring * 0.1;
                let c = [pos[0] + a.cos() * ring, pos[1] + a.sin() * ring];
                if !self.straight(pos, &c) {
                    continue;
                }
                let z = self.ground(c[0], c[1], pos[2]);
                if (z - pos[2]).abs() > 1.5 || !hidden(c, z) {
                    continue;
                }
                // Prefer spots further from the nearest eye.
                let near = eyes.iter().map(|e| d2(e, &c)).fold(f32::MAX, f32::min);
                if best.map_or(true, |b| near > b.1) {
                    best = Some((c, near));
                }
            }
            if let Some((c, _)) = best {
                return Some(norm2(c[0] - pos[0], c[1] - pos[1]));
            }
            ring *= 1.6;
        }
        None
    }

    fn scan(&mut self, a: &mut Action) {
        self.scan_yaw = wrap(self.scan_yaw + 0.16);
        a.aim = [self.scan_yaw, 0.0];
    }

    /// Sideways movement relative to `to` (unit vector toward the opponent) with random dwell.
    /// Start a new strafe dwell: reverse with probability `strafe_alt`, for a random time in [strafe_min, strafe_max].
    fn next_strafe(&mut self, now: f32) {
        let (smin, smax, alt) = (self.params.strafe_min, self.params.strafe_max.max(self.params.strafe_min), self.params.strafe_alt);
        if self.rnd() < alt {
            self.strafe = -self.strafe;
        }
        self.strafe_until = now + smin + self.rnd() * (smax - smin);
    }

    fn strafe_dir(&mut self, o: &Obs, pos: &[f32; 3], to: [f32; 2], radial: f32, flip: bool) -> [f32; 2] {
        let smin = self.params.strafe_min;
        if flip {
            self.strafe = -self.strafe;
            self.strafe_until = o.t + smin;
        } else if o.t >= self.strafe_until {
            self.next_strafe(o.t);
        }
        let mut dir = norm2(to[0] * radial - to[1] * self.strafe, to[1] * radial + to[0] * self.strafe);
        if !self.walkable(pos[0] + dir[0] * 6.0, pos[1] + dir[1] * 6.0) {
            self.strafe = -self.strafe;
            dir = norm2(to[0] * radial - to[1] * self.strafe, to[1] * radial + to[0] * self.strafe);
            if !self.walkable(pos[0] + dir[0] * 6.0, pos[1] + dir[1] * 6.0) {
                dir = [-to[0], -to[1]];
            }
        }
        dir
    }

    /// Small random steps around a point.
    fn jitter(&mut self, o: &Obs, pos: &[f32; 3], c: [f32; 2]) -> [f32; 2] {
        if o.t >= self.strafe_until {
            self.strafe = if self.rnd() < 0.5 { 1.0 } else { -1.0 };
            self.strafe_until = o.t + 0.5 + self.rnd() * 0.6;
        }
        let ang = self.strafe * 1.3 + (o.t * 0.7).sin();
        let back = [c[0] - pos[0], c[1] - pos[1]];
        let d = norm2(ang.cos() + back[0] * 0.3, ang.sin() + back[1] * 0.3);
        if self.walkable(pos[0] + d[0] * 4.0, pos[1] + d[1] * 4.0) {
            [d[0] * 0.6, d[1] * 0.6]
        } else {
            [0.0, 0.0]
        }
    }

    /// Wander around a sheltered post inside the safe area.
    fn idle_move(&mut self, o: &Obs, pos: &[f32; 3]) -> [f32; 2] {
        let z = &o.zone;
        let stale = self.post.map_or(true, |(c, t)| {
            o.t - t > 25.0 || d2(&c, &z.next_center) > z.next_radius.max(15.0) || d2(&c, &z.center) > z.radius - 10.0
        });
        if stale {
            let c = self.choose_post(o);
            self.post = Some((c, o.t));
            self.wander = None;
        }
        let post = self.post.unwrap().0;
        if self.wander.map_or(true, |w| d2(&w, pos) < 2.5) {
            let r = self.params.wander_radius;
            let mut w = post;
            for _ in 0..6 {
                let a = self.rnd() * TAU;
                let rr = r * (0.3 + 0.7 * self.rnd());
                let c = [post[0] + a.cos() * rr, post[1] + a.sin() * rr];
                if self.walkable(c[0], c[1]) {
                    w = c;
                    break;
                }
            }
            self.wander = Some(w);
        }
        let w = self.wander.unwrap();
        let mv = self.go(pos, w);
        // Walk slowly near the post: quiet and still hard to hit.
        if d2(pos, &post) < self.params.wander_radius * 1.5 {
            [mv[0] * 0.45, mv[1] * 0.45]
        } else {
            mv
        }
    }

    /// No fight: loot, drops and crates, otherwise drift around a sheltered post.
    #[allow(clippy::too_many_arguments)]
    fn quiet(
        &mut self, o: &Obs, a: &mut Action, pos: &[f32; 3], hp_frac: f32, must_rotate: bool, out_next: f32, zone_goal: [f32; 2],
        threats: &[[f32; 3]], label: &mut &'static str,
    ) -> [f32; 2] {
        let me = &o.me;
        let p = self.params.clone();
        let z = &o.zone;
        let on_ground = |w: &Brain, q: &[f32; 3]| (q[2] - w.ground(q[0], q[1], q[2])).abs() < 3.0;
        let in_zone = |q: &[f32; 3]| d2(q, &z.center) < z.radius - 10.0 && (o.t < z.shrink_start - 20.0 || d2(q, &z.next_center) < z.next_radius);
        let mut goal: Option<([f32; 2], f32)> = None;
        let consider = |g: &mut Option<([f32; 2], f32)>, q: &[f32; 3], value: f32, reach: f32, ok_ground: bool| {
            let d = d2(q, pos);
            if d < reach && ok_ground && in_zone(q) {
                let s = value - d;
                if g.map_or(true, |x| s > x.1) {
                    *g = Some(([q[0], q[1]], s));
                }
            }
        };
        let held: Vec<&str> = me.weapons.iter().flatten().map(|w| w.kind.as_str()).collect();
        let empty_slot = me.weapons.len() < 2 || me.weapons.iter().take(2).any(|w| w.is_none());
        let worst = me.weapons.iter().take(2).flatten().map(|w| weapon_rank(&w.kind, &p)).filter(|r| *r > 0).min().unwrap_or(0);
        let early = o.t < p.farm_until;
        let reach = if early { p.farm_reach.max(100.0) } else { 100.0 };
        let picks: Vec<(String, [f32; 3])> = self.picks_mem.values().cloned().collect();
        for (kind, q) in &picks {
            let r = weapon_rank(kind, &p);
            if r > 0 && !held.contains(&kind.as_str()) && (empty_slot || r > worst) {
                let v = if empty_slot && early { 120.0 + 30.0 * r as f32 } else { 60.0 + 30.0 * r as f32 };
                consider(&mut goal, q, v, reach.max(140.0), on_ground(self, q));
            }
        }
        let cubes: Vec<[f32; 3]> = self.cubes_mem.values().copied().collect();
        for q in &cubes {
            consider(&mut goal, q, 100.0, reach, on_ground(self, q));
        }
        // Supply drop: full heal and cubes, if nobody we know of is near it.
        for dr in &o.drops {
            let d = d2(&dr.pos, pos);
            let contested = threats.iter().any(|t| d2(t, &dr.pos) < p.drop_enemy_clear);
            let arrive = d / me.stat("walk_speed", 11.7);
            if d < p.drop_range && !contested && arrive + 4.0 > dr.lands_in - 3.0 && in_zone(&dr.pos) {
                let v = 150.0 + (1.0 - hp_frac) * 100.0;
                if goal.map_or(true, |g| v - d * 0.5 > g.1) {
                    goal = Some(([dr.pos[0], dr.pos[1]], v - d * 0.5));
                }
            }
        }
        let crates: Vec<[f32; 3]> = self.crates_mem.values().copied().filter(|q| on_ground(self, q) && in_zone(q)).collect();
        let crate_target = crates
            .iter()
            .filter(|c| d3(c, pos) < reach && hp_frac > p.farm_hp && !self.healing)
            .min_by(|x, y| d3(x, pos).partial_cmp(&d3(y, pos)).unwrap_or(std::cmp::Ordering::Equal));
        let mut aimed = false;
        if let Some(c) = crate_target {
            let muzzle = [pos[0], pos[1], pos[2] + me.stat("muzzle_height", 7.0)];
            let cc = [c[0], c[1], c[2] + 1.2];
            if !self.clear(muzzle, cc) || d2(c, pos) > 90.0 {
                consider(&mut goal, c, 70.0, reach, true);
            }
            if self.clear(muzzle, cc) {
                let want = [yaw_to(&muzzle, &cc), (cc[2] - muzzle[2]).atan2(d2(&cc, &muzzle).max(0.1))];
                a.aim = want;
                aimed = true;
                let dd = d3(&cc, &muzzle).max(1.0);
                let lined = wrap(want[0] - me.aim[0]).abs() < (1.0 / dd).atan() && (want[1] - me.aim[1]).abs() < (0.9 / dd).atan();
                for (i, w) in me.weapons.iter().enumerate().take(2) {
                    if w.as_ref().map_or(false, |w| w.kind == "gun") {
                        a.fire[i] = lined;
                        a.reload = false;
                    }
                }
                if d2(c, pos) > 35.0 {
                    consider(&mut goal, c, 70.0, reach, true);
                }
            }
        }
        let g = match goal {
            Some((g, _)) if !must_rotate => Some(g),
            _ if must_rotate || out_next > 0.0 => Some(zone_goal),
            _ => None,
        };
        let mv = match g {
            Some(g) => {
                *label = "loot";
                self.go(pos, g)
            }
            None => self.idle_move(o, pos),
        };
        if !aimed {
            if let Some(th) = threats.first() {
                a.aim = [yaw_to(pos, th), 0.0];
            } else {
                self.scan(a);
            }
        }
        mv
    }
}
