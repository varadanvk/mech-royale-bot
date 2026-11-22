//! Decision logic: one `act` per observation.
use crate::contacts::{Belief, Contacts, MapCfg};
use crate::detect::Detector;
use crate::nav::Nav;
use crate::proto::{Action, Actor, Hello, Obs, Weapon};
use crate::sight::{Foe, Pick, SightTable, Zone};
use crate::world::{PrismKind, World};
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
    /// Upgrade order: 0 = survival first, 1 = damage first, 2 = hit points and armour, then laser, then the rest.
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
    /// Boost while moving in a fight whenever energy is above this (0 = off).
    pub fight_boost: f32,
    /// From this match time on, no detours for weapons, cubes, drones or crates (0 = never stop).
    pub farm_stop: f32,
    /// Hunt also without a laser.
    pub hunt_any: f32,
    /// With at most one enemy within `crowd_radius`, retreat only below this HP fraction (0 = use `retreat_hp`).
    pub solo_hp: f32,
    /// Boost toward the target in a fight while further than the preferred distance by 15 m and energy is above this (0 = off).
    pub close_boost: f32,
    /// Take on drones by choice (farming, chasing). 0: only shoot back at a drone that hurt us in the last
    /// 3 s while no mech is in view, and never move toward one.
    pub drone_fight: f32,
    /// Late in the match, a live fight (a target in sight while engaged or recently hurt by a mech) is fought
    /// with normal movement instead of holding a spot, and holding weaves across the nearest shooter.
    pub hold_fight: f32,
    /// Hurt by a mech in the last 3 s: no looting or crate shooting, and face the unseen attacker.
    pub threat_face: f32,
    /// Against a target holding lightning within 60 m, back away (boosting) while kiting or holding.
    pub ltg_space: f32,
    /// Laser discipline: let a charge go only when the beam lines up on a mech where it will be; never dump it
    /// unaimed. With the target out of view, keep holding toward where it should be until the charge fires itself.
    pub laser_hold: f32,
    /// Shoot back at drones near enough to attack us (or that hurt us recently) while no mech is in view or
    /// shooting at us, finishing the most damaged first; never walk toward them.
    pub drone_threat: f32,
    /// Target choice: metres of preference for an enemy whose torso is aimed at us (0 = off).
    pub focus_aimer: f32,
    /// While walking to a weapon, shoot crates in view within 100 m when no mech is around (0 = off).
    pub arm_crates: f32,
    /// No lightning rush on a laser holder from beyond the arc range + 15 m unless its laser fired within the
    /// last 1.5 s, and none from beyond 60 m with an empty magazine (0 = off).
    pub rush_guard: f32,
    /// Late game (from `wall_t`, or while at least three enemies are close or hurting us): keep moving on a loop
    /// around a building inside the safe area, on the side away from the known enemies, shooting whoever has a
    /// line on us (0 = off).
    pub ring: f32,
    /// Before the late game, take on only enemies aimed at us, that hurt us in the last 4 s, or within
    /// `counter_range`; leave the others alone (0 = off).
    pub counter: f32,
    pub counter_range: f32,
    /// When pressed (as for `ring`: from `wall_t`, or while `spot_crowd` or more enemies are close or hurting us),
    /// fight from sheltered ground (0 = off). 1: hold the most sheltered spot nearby that is inside the safe
    /// area and not past the enemies, facing its open side; move on to the next when the zone closes in, two or
    /// more enemies keep a line on us, or we lose health fast without a kill coming. 2: as 1, but first look for
    /// a position nearby where exactly one known enemy has a line to us and the others do not.
    pub spot: f32,
    pub spot_crowd: f32,
    /// Sidestep an incoming bullet (or a shot from an enemy aimed at us), boosting perpendicular to its path
    /// for `dodge_hold` seconds (0 = off). A new sidestep starts only above `dodge_energy`.
    pub dodge: f32,
    pub dodge_hold: f32,
    pub dodge_energy: f32,
    /// Bullet sidestep, tighter: react only to shots due within 0.7 s that would pass within body radius + 1.1 m,
    /// step to the side the shot already misses on, and blend the step over the planned move at two to one
    /// (practice matches only; 0 = as before).
    pub dodge_tight: f32,
    /// Outnumbered or losing: break off in bursts of `cr_burst` seconds to the nearest walkable cell hidden
    /// from the closest enemies, or fight in place when there is none (0 = off; replaces the low-health
    /// retreat and backing-away rules).
    pub crowd_retreat: f32,
    pub cr_radius: f32,
    pub cr_burst: f32,
    pub cr_steps: f32,
    /// In a fight, step to whichever nearby point best combines the preferred distance, a line to the target
    /// and few other enemies with a line to it (0 = off).
    pub fight_pos: f32,
    pub pref_gun: f32,
    pub pref_laser: f32,
    /// Distances kept from enemies holding lightning and mortars.
    pub keep_ltg: f32,
    pub keep_mortar: f32,
    /// Late game (next circle radius at most `late_r`): hold a sheltered home spot inside the next circle,
    /// walking there quietly, and take on only enemies that are close or come for us (0 = off).
    pub late_hold: f32,
    pub late_r: f32,
    pub home_s: f32,
    pub home_speed: f32,
    /// Before the late game, take on only enemies that are close, come for us or are weak (0 = off).
    pub engage_filter: f32,
    /// Laser: count the cook-off on mortar holders when releasing early for a kill, and never charge outside
    /// the zone (0 = off).
    pub laser_fix: f32,
    /// Pick up cubes within `fight_cube_r` during a fight when no other enemy watches the way (0 = off).
    pub fight_cubes: f32,
    pub fight_cube_r: f32,
    /// Retreat rule refinements (0 = off): the low-health clause compares health fractions (target at least
    /// `lose_margin` healthier), "finishing" means a target within 120 m under 200 HP or under half our HP, and
    /// the cover search weighs walking steps against distance from the enemies over up to 350 cells.
    pub ext_lose: f32,
    pub lose_margin: f32,
    /// Engagement refinements (0 = off): engage everyone once at most three mechs remain; target score keeps the
    /// current target for 2 s, favours an attacker of the last 4 s and a mech closing on us.
    pub ext_target: f32,
    /// Fight-position refinements (0 = off): count every enemy seen in the last few seconds, strafe reversals
    /// every `fp_flip_lo`..`fp_flip_hi` seconds, keep `pref_ltg` with lightning.
    pub ext_fpos: f32,
    pub fp_flip_lo: f32,
    pub fp_flip_hi: f32,
    pub pref_ltg: f32,
    /// Late-game refinements (0 = off): engage anyone only with one opponent left, crowding counted within
    /// `late_crowd_r`, and the home spot re-chosen away from the direction of a fresh hit or sound.
    pub ext_late: f32,
    pub late_crowd_r: f32,
    /// Finishing refinements (0 = off): close in on a weak target only while healthy and not crowded; grab cubes
    /// only within `cube_r_low` when below half health.
    pub ext_finish: f32,
    pub cube_r_low: f32,
    /// Sidestep tuning: how far ahead a bullet is tracked and how far we step.
    pub dodge_look: f32,
    pub dodge_reach: f32,
    /// Tell a free-for-all from a practice match (see `detect`) and keep a picture of every enemy alongside the
    /// usual logic; nothing it learns changes a decision unless `ffa_play` is on and a free-for-all is settled
    /// (0 = off).
    pub ffa_detect: f32,
    /// Once a free-for-all is settled, play it in the modes of `ffa` (0 = off).
    pub ffa_play: f32,
    /// Free-for-all mode distances: an enemy coming for us within `ffa_avoid_r` is avoided; a fight between two
    /// others within `ffa_third_r` is joined; the late game starts once the next circle is at most
    /// `ffa_late_r` across. `ffa_quiet`: walking speed when keeping quiet (below the step-sound speed).
    pub ffa_avoid_r: f32,
    pub ffa_third_r: f32,
    pub ffa_late_r: f32,
    pub ffa_quiet: f32,
    /// Enemy-picture clean-ups (0 = off; see `contacts::MapCfg`): sounds move only the one tightly placed unseen
    /// track they fit; faded beliefs are left out; contacts seen empty are dropped; a listed enemy takes over
    /// the contacts it explains; a hit from an unknown enemy lands on the contact along its line; a sound joins
    /// only a contact it mostly overlaps, and joined sounds narrow it only so far; an
    /// estimate pushed out of a place seen empty keeps its radius; a coasting track's radius grows from the moment
    /// it left the list.
    pub map_strict_pull: f32,
    pub map_prune: f32,
    pub map_clear_seen: f32,
    pub map_merge_seen: f32,
    pub map_hit_line: f32,
    pub map_tight_merge: f32,
    pub map_keep_spread: f32,
    pub map_hold_r: f32,
    pub map_coast_grow: f32,
    /// The enemy picture in a practice match (each 0 = off; never once a free-for-all is settled). `map_preaim`:
    /// with no enemy in view, face the surest believed enemy out of sight within 150 m. `map_expose`: a fight
    /// position loses this much for each believed enemy out of sight (confidence 0.4 or more, within 170 m) with
    /// a line to it. `map_crowd`: a loot goal or post costs this much for each sure believed enemy within 60 m of
    /// the way there, once two or more are (never the target, nor an enemy within 60 m hurting us).
    /// `map_reinf`: a candidate target scores this much worse for each other sure believed enemy within 80 m of
    /// it (at most three).
    pub map_preaim: f32,
    pub map_expose: f32,
    pub map_crowd: f32,
    pub map_reinf: f32,
    /// Enemy motion (0 = off): each enemy within reach is carried `pred_s` seconds ahead along its velocity over
    /// walkable ground; one coming for us is also walked toward us (stopping `pred_stop` m short) and the two
    /// guesses blended with weight `pred_hunt` on the walk.
    pub pred: f32,
    pub pred_s: f32,
    pub pred_hunt: f32,
    pub pred_stop: f32,
    /// Keeping enemies on one side while fighting (0 = off). Enemies within `pin_r` (in view, sure believed ones,
    /// and where they are expected to be) count. Two or more spread over more than 144 degrees round us,
    /// or one expected to come round more than 90 degrees off the target, is a pincer. With two or more about,
    /// a fight position loses `pin_rear` for each enemy (now and expected) behind it as we face the target,
    /// `pin_spread` per quarter circle the enemies spread over beyond the first, and `pin_close` per metre it
    /// brings us closer to an enemy off to the side of the target or behind us. While pincered it also loses
    /// `pin_open` per tenth more open ground than here and `pin_zone` per metre outside the next circle, and
    /// gains up to `pin_cover` for a wall close behind it; points up to `pin_ring` m away are weighed too, and
    /// a far one chosen is kept for a moment.
    pub pincer: f32,
    pub pin_r: f32,
    pub pin_rear: f32,
    pub pin_spread: f32,
    pub pin_cover: f32,
    pub pin_open: f32,
    pub pin_zone: f32,
    pub pin_ring: f32,
    pub pin_close: f32,
    /// Break-off goal choice (each 0 = off). `cb_away`: the goal must be farther than we are from the enemies'
    /// centre and from the nearest enemy, and not in a direction within 60 degrees of an enemy within 80 m
    /// (other directions toward enemies cost more). `cb_path`: the way there (straight line and the searched
    /// cells) keeps 25 m from every enemy we know of, seen, remembered or believed. `cb_known`: hide from up to
    /// six enemies, remembered ones included. `cb_corner`: with no such goal, stay and fight without boosting
    /// toward an enemy. `cb_turn`: a new burst does not turn back toward the enemies.
    pub cb_away: f32,
    pub cb_path: f32,
    pub cb_known: f32,
    pub cb_corner: f32,
    pub cb_turn: f32,
    /// Speed of the sideways strafe across the target when the fight position is held (fraction of walking speed).
    pub fp_strafe: f32,
    /// Target choice: metres of preference for an enemy holding a laser, counted twice while it charges at us (0 = off).
    pub focus_laser: f32,
    /// Limit on what the keep-enemies-in-front terms may add to or take from a fight position (0 = no limit).
    pub pin_cap: f32,
    /// With nothing else to do and healthy, walk toward the nearest enemy in view, remembered or believed within this distance (0 = off).
    pub seek: f32,
    /// Seek only above this HP fraction.
    pub seek_hp: f32,
    /// An enemy we are walking toward is taken on once it is within this distance.
    pub seek_fire: f32,
    /// Seek only from this match time, or earlier once the second weapon slot is filled.
    pub seek_t: f32,
    /// With a target within this distance, fight it before walking on to a weapon pickup more than 20 m away (0 = off).
    pub arm_yield: f32,
    /// Gun lead learned per enemy (0 = off): every look at an enemy is checked later as a shot led by none, a third, two thirds or all of its speed, and the lead that would have hit most often lately is used once it is clearly ahead of the usual one.
    pub lead_adapt: f32,
    /// Break off only to hidden ground within this distance; with none that near, stay and fight (0 = any distance).
    pub cr_max_d: f32,
    /// Break-off health fractions: below cr_hp1 while losing to the target, below cr_hp2 with two enemies near, below cr_hp3 with three or more.
    pub cr_hp1: f32,
    /// See cr_hp1.
    pub cr_hp2: f32,
    /// See cr_hp1.
    pub cr_hp3: f32,
    /// Break off from a lone enemy we are losing to (0 = only when two or more are near).
    pub cr_solo: f32,
    /// Keep starting laser charges while breaking off (0 = only within 6 m of the hidden ground).
    pub cr_laser: f32,
    /// Stop breaking off once no enemy in view is aimed at us and nothing has hit us for this many seconds (0 = off).
    pub cr_unaimed: f32,
    /// Outside a free-for-all: after a kill, or after 3 s idle at 70% health or more, with no enemy in view, walk toward the nearest enemy we know of until one is in view (0 = off).
    pub seek_idle: f32,
    /// Share of an enemy's speed across our line of sight that its predicted motion keeps (its speed toward or away from us is kept whole): a side-to-side dodge reverses within the second, so less than 1 guesses closer two seconds out.
    pub pred_tan: f32,
    /// Having broken off to ground that turns out to be in an enemy's view, sidestep across its line while shooting back instead of standing still (0 = stand).
    pub cr_strafe: f32,
    /// In a free-for-all, break off, reload and sidestep laser shots as before: cr_max_d, cr_laser, cr_strafe, reload_laser and dodge_after take their earlier values there (0 = the same in every kind of match).
    pub ffa_plain: f32,
    /// In a live fight (a target in view while engaged, or hurt by a mech in the last 6 s), keep fighting instead of breaking off, backing away or hiding to heal; a break-off to hidden ground within stay_cover m still goes ahead (practice matches only; 0 = off).
    pub stay_fight: f32,
    /// See stay_fight.
    pub stay_cover: f32,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            retreat_hp: 0.3,
            reengage_hp: 0.6,
            gun_range: 40.0,
            gun_band: 10.0,
            arc_range: 30.0,
            arc_min_hp: 0.6,
            crowd_radius: 150.0,
            crowd_count: 4,
            open_fire_range: 200.0,
            strafe_min: 0.3,
            strafe_max: 0.6,
            strafe_alt: 1.0,
            dodge_after: 0.6,
            evade_weight: 0.8,
            focus_hp: 120.0,
            kite: 1.0,
            finish_hp: 0.3,
            finish_range: 45.0,
            fight_speed: 1.0,
            farm_hp: 0.7,
            drone_range: 80.0,
            drop_range: 250.0,
            drop_enemy_clear: 90.0,
            zone_lead: 10.0,
            zone_lead_mult: 1.6,
            wander_radius: 14.0,
            memory_s: 12.0,
            retreat_boost_energy: 15.0,
            focus_stick: 50.0,
            farm_until: 0.0,
            farm_reach: 160.0,
            heal_lo: 0.5,
            heal_hi: 0.85,
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
            upg_dmg: 2.0,
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
            reload_laser: 0.0,
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
            wall_t: 170.0,
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
            fight_boost: 0.0,
            farm_stop: 0.0,
            hunt_any: 0.0,
            solo_hp: 0.0,
            close_boost: 30.0,
            drone_fight: 0.0,
            hold_fight: 1.0,
            threat_face: 1.0,
            ltg_space: 1.0,
            laser_hold: 0.0,
            drone_threat: 0.0,
            focus_aimer: 0.0,
            arm_crates: 0.0,
            rush_guard: 0.0,
            ring: 0.0,
            counter: 1.0,
            counter_range: 60.0,
            spot: 0.0,
            spot_crowd: 3.0,
            dodge: 1.0,
            dodge_hold: 0.35,
            dodge_energy: 20.0,
            dodge_tight: 1.0,
            crowd_retreat: 1.0,
            cr_radius: 160.0,
            cr_burst: 1.5,
            cr_steps: 50.0,
            fight_pos: 1.0,
            pref_gun: 85.0,
            pref_laser: 120.0,
            keep_ltg: 64.0,
            keep_mortar: 80.0,
            late_hold: 1.0,
            late_r: 140.0,
            home_s: 12.0,
            home_speed: 4.6,
            engage_filter: 1.0,
            laser_fix: 1.0,
            fight_cubes: 1.0,
            fight_cube_r: 70.0,
            ext_lose: 1.0,
            lose_margin: 0.10,
            ext_target: 1.0,
            ext_fpos: 1.0,
            fp_flip_lo: 0.3,
            fp_flip_hi: 1.0,
            pref_ltg: 28.0,
            ext_late: 1.0,
            late_crowd_r: 80.0,
            ext_finish: 1.0,
            cube_r_low: 40.0,
            dodge_look: 1.2,
            dodge_reach: 3.5,
            ffa_detect: 1.0,
            ffa_play: 1.0,
            ffa_avoid_r: 150.0,
            ffa_third_r: 220.0,
            ffa_late_r: 140.0,
            ffa_quiet: 4.6,
            map_strict_pull: 1.0,
            map_prune: 1.0,
            map_clear_seen: 1.0,
            map_merge_seen: 1.0,
            map_hit_line: 1.0,
            map_tight_merge: 1.0,
            map_keep_spread: 1.0,
            map_hold_r: 1.0,
            map_coast_grow: 1.0,
            map_preaim: 0.0,
            map_expose: 0.0,
            map_crowd: 0.0,
            map_reinf: 0.0,
            pred: 1.0,
            pred_s: 2.0,
            pred_hunt: 0.25,
            pred_stop: 30.0,
            pincer: 1.0,
            pin_r: 120.0,
            pin_rear: 25.0,
            pin_spread: 20.0,
            pin_cover: 15.0,
            pin_open: 15.0,
            pin_zone: 3.0,
            pin_ring: 24.0,
            pin_close: 3.0,
            cb_away: 1.0,
            cb_path: 1.0,
            cb_known: 1.0,
            cb_corner: 1.0,
            cb_turn: 1.0,
            fp_strafe: 0.5,
            focus_laser: 0.0,
            pin_cap: 0.0,
            seek: 0.0,
            seek_hp: 0.8,
            seek_fire: 90.0,
            seek_t: 0.0,
            arm_yield: 0.0,
            lead_adapt: 1.0,
            cr_max_d: 15.0,
            cr_hp1: 0.35,
            cr_hp2: 0.55,
            cr_hp3: 0.8,
            cr_solo: 1.0,
            cr_laser: 1.0,
            cr_unaimed: 0.0,
            seek_idle: 1.0,
            pred_tan: 1.0,
            cr_strafe: 1.0,
            ffa_plain: 1.0,
            stay_fight: 1.0,
            stay_cover: 15.0,
        }
    }
}

impl Params {
    /// The enemy-picture clean-ups these parameters turn on.
    pub fn map_cfg(&self) -> MapCfg {
        MapCfg {
            strict_pull: self.map_strict_pull > 0.0,
            prune: self.map_prune > 0.0,
            clear_seen: self.map_clear_seen > 0.0,
            merge_seen: self.map_merge_seen > 0.0,
            hit_line: self.map_hit_line > 0.0,
            tight_merge: self.map_tight_merge > 0.0,
            keep_spread: self.map_keep_spread > 0.0,
            hold_r: self.map_hold_r > 0.0,
            coast_grow: self.map_coast_grow > 0.0,
        }
    }

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
        p.fight_boost = g("fight_boost", p.fight_boost);
        p.farm_stop = g("farm_stop", p.farm_stop);
        p.hunt_any = g("hunt_any", p.hunt_any);
        p.solo_hp = g("solo_hp", p.solo_hp);
        p.close_boost = g("close_boost", p.close_boost);
        p.drone_fight = g("drone_fight", p.drone_fight);
        p.hold_fight = g("hold_fight", p.hold_fight);
        p.threat_face = g("threat_face", p.threat_face);
        p.ltg_space = g("ltg_space", p.ltg_space);
        p.laser_hold = g("laser_hold", p.laser_hold);
        p.drone_threat = g("drone_threat", p.drone_threat);
        p.focus_aimer = g("focus_aimer", p.focus_aimer);
        p.arm_crates = g("arm_crates", p.arm_crates);
        p.rush_guard = g("rush_guard", p.rush_guard);
        p.ring = g("ring", p.ring);
        p.counter = g("counter", p.counter);
        p.counter_range = g("counter_range", p.counter_range);
        p.spot = g("spot", p.spot);
        p.spot_crowd = g("spot_crowd", p.spot_crowd);
        p.dodge = g("dodge", p.dodge);
        p.dodge_hold = g("dodge_hold", p.dodge_hold);
        p.dodge_energy = g("dodge_energy", p.dodge_energy);
        p.dodge_tight = g("dodge_tight", p.dodge_tight);
        p.crowd_retreat = g("crowd_retreat", p.crowd_retreat);
        p.cr_radius = g("cr_radius", p.cr_radius);
        p.cr_burst = g("cr_burst", p.cr_burst);
        p.cr_steps = g("cr_steps", p.cr_steps);
        p.fight_pos = g("fight_pos", p.fight_pos);
        p.pref_gun = g("pref_gun", p.pref_gun);
        p.pref_laser = g("pref_laser", p.pref_laser);
        p.keep_ltg = g("keep_ltg", p.keep_ltg);
        p.keep_mortar = g("keep_mortar", p.keep_mortar);
        p.late_hold = g("late_hold", p.late_hold);
        p.late_r = g("late_r", p.late_r);
        p.home_s = g("home_s", p.home_s);
        p.home_speed = g("home_speed", p.home_speed);
        p.engage_filter = g("engage_filter", p.engage_filter);
        p.laser_fix = g("laser_fix", p.laser_fix);
        p.fight_cubes = g("fight_cubes", p.fight_cubes);
        p.fight_cube_r = g("fight_cube_r", p.fight_cube_r);
        p.ext_lose = g("ext_lose", p.ext_lose);
        p.lose_margin = g("lose_margin", p.lose_margin);
        p.ext_target = g("ext_target", p.ext_target);
        p.ext_fpos = g("ext_fpos", p.ext_fpos);
        p.fp_flip_lo = g("fp_flip_lo", p.fp_flip_lo);
        p.fp_flip_hi = g("fp_flip_hi", p.fp_flip_hi);
        p.pref_ltg = g("pref_ltg", p.pref_ltg);
        p.ext_late = g("ext_late", p.ext_late);
        p.late_crowd_r = g("late_crowd_r", p.late_crowd_r);
        p.ext_finish = g("ext_finish", p.ext_finish);
        p.cube_r_low = g("cube_r_low", p.cube_r_low);
        p.dodge_look = g("dodge_look", p.dodge_look);
        p.dodge_reach = g("dodge_reach", p.dodge_reach);
        p.ffa_detect = g("ffa_detect", p.ffa_detect);
        p.ffa_play = g("ffa_play", p.ffa_play);
        p.ffa_avoid_r = g("ffa_avoid_r", p.ffa_avoid_r);
        p.ffa_third_r = g("ffa_third_r", p.ffa_third_r);
        p.ffa_late_r = g("ffa_late_r", p.ffa_late_r);
        p.ffa_quiet = g("ffa_quiet", p.ffa_quiet);
        p.map_strict_pull = g("map_strict_pull", p.map_strict_pull);
        p.map_prune = g("map_prune", p.map_prune);
        p.map_clear_seen = g("map_clear_seen", p.map_clear_seen);
        p.map_merge_seen = g("map_merge_seen", p.map_merge_seen);
        p.map_hit_line = g("map_hit_line", p.map_hit_line);
        p.map_tight_merge = g("map_tight_merge", p.map_tight_merge);
        p.map_keep_spread = g("map_keep_spread", p.map_keep_spread);
        p.map_hold_r = g("map_hold_r", p.map_hold_r);
        p.map_coast_grow = g("map_coast_grow", p.map_coast_grow);
        p.map_preaim = g("map_preaim", p.map_preaim);
        p.map_expose = g("map_expose", p.map_expose);
        p.map_crowd = g("map_crowd", p.map_crowd);
        p.map_reinf = g("map_reinf", p.map_reinf);
        p.pred = g("pred", p.pred);
        p.pred_s = g("pred_s", p.pred_s);
        p.pred_hunt = g("pred_hunt", p.pred_hunt);
        p.pred_stop = g("pred_stop", p.pred_stop);
        p.pincer = g("pincer", p.pincer);
        p.pin_r = g("pin_r", p.pin_r);
        p.pin_rear = g("pin_rear", p.pin_rear);
        p.pin_spread = g("pin_spread", p.pin_spread);
        p.pin_cover = g("pin_cover", p.pin_cover);
        p.pin_open = g("pin_open", p.pin_open);
        p.pin_zone = g("pin_zone", p.pin_zone);
        p.pin_ring = g("pin_ring", p.pin_ring);
        p.pin_close = g("pin_close", p.pin_close);
        p.cb_away = g("cb_away", p.cb_away);
        p.cb_path = g("cb_path", p.cb_path);
        p.cb_known = g("cb_known", p.cb_known);
        p.cb_corner = g("cb_corner", p.cb_corner);
        p.cb_turn = g("cb_turn", p.cb_turn);
        p.fp_strafe = g("fp_strafe", p.fp_strafe);
        p.focus_laser = g("focus_laser", p.focus_laser);
        p.pin_cap = g("pin_cap", p.pin_cap);
        p.seek = g("seek", p.seek);
        p.seek_hp = g("seek_hp", p.seek_hp);
        p.seek_fire = g("seek_fire", p.seek_fire);
        p.seek_t = g("seek_t", p.seek_t);
        p.arm_yield = g("arm_yield", p.arm_yield);
        p.lead_adapt = g("lead_adapt", p.lead_adapt);
        p.cr_max_d = g("cr_max_d", p.cr_max_d);
        p.cr_hp1 = g("cr_hp1", p.cr_hp1);
        p.cr_hp2 = g("cr_hp2", p.cr_hp2);
        p.cr_hp3 = g("cr_hp3", p.cr_hp3);
        p.cr_solo = g("cr_solo", p.cr_solo);
        p.cr_laser = g("cr_laser", p.cr_laser);
        p.cr_unaimed = g("cr_unaimed", p.cr_unaimed);
        p.seek_idle = g("seek_idle", p.seek_idle);
        p.pred_tan = g("pred_tan", p.pred_tan);
        p.cr_strafe = g("cr_strafe", p.cr_strafe);
        p.ffa_plain = g("ffa_plain", p.ffa_plain);
        p.stay_fight = g("stay_fight", p.stay_fight);
        p.stay_cover = g("stay_cover", p.stay_cover);
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

/// Upgrade preference with hit points and armour first, then the laser, regeneration and the gun.
const UPGRADES_TANK: &[&str] = &[
    "plating", "armor", "laser_damage", "laser_charge", "nanites", "gun_damage", "gun_rate", "gun_mag",
    "lightning_efficiency", "servos", "lightning_chain", "reactor", "capacitor", "gyros", "lightning_range", "jets",
    "gun_velocity", "mortar_reload", "mortar_splash", "sensors", "mortar_velocity", "laser_pierce",
];

/// Upgrade preference, best first. Survival first: hit points, armour, regeneration and speed before damage.
const UPGRADES: &[&str] = &[
    "plating", "armor", "nanites", "servos", "gun_damage", "lightning_efficiency", "gun_rate", "laser_damage",
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
    /// Health fraction when last seen.
    hp_frac: f32,
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

/// Leads tried for each enemy (fractions of its speed), the usual one last.
const LEADS: [f32; 4] = [0.0, 0.3, 0.6, 0.9];

/// How shots at one enemy would have done with each lead in `LEADS`.
#[derive(Default)]
pub(crate) struct LeadStat {
    /// Looks waiting to be checked: when the shot would land, where the enemy was, how far it would be led at full lead,
    /// and the sideways direction of the shot.
    pending: VecDeque<(f32, [f32; 2], [f32; 2], [f32; 2])>,
    /// Running share of hits per lead.
    ema: [f32; 4],
    n: u32,
}

impl LeadStat {
    /// Note a look at the enemy at time `t`: a shot from `from` landing `tt` seconds later.
    pub(crate) fn look(&mut self, t: f32, from: [f32; 2], pos: [f32; 2], vel: [f32; 2], tt: f32) {
        let d = norm2(pos[0] - from[0], pos[1] - from[1]);
        self.pending.push_back((t + tt, pos, [vel[0] * tt, vel[1] * tt], [-d[1], d[0]]));
        if self.pending.len() > 16 {
            self.pending.pop_front();
        }
    }

    /// Check the looks that have come due against where the enemy is now (`pos`, moving at `vel`, at time `t`).
    pub(crate) fn settle(&mut self, t: f32, pos: [f32; 2], vel: [f32; 2], radius: f32) {
        while self.pending.front().map_or(false, |x| x.0 <= t) {
            let (due, p0, lead, side) = self.pending.pop_front().unwrap();
            if t - due > 0.25 {
                continue;
            }
            // Where it was when the shot would have landed.
            let at = [pos[0] - vel[0] * (t - due), pos[1] - vel[1] * (t - due)];
            for (k, f) in LEADS.iter().enumerate() {
                let aim = [p0[0] + lead[0] * f, p0[1] + lead[1] * f];
                let miss = ((at[0] - aim[0]) * side[0] + (at[1] - aim[1]) * side[1]).abs();
                let hit = if miss < radius { 1.0 } else { 0.0 };
                self.ema[k] += 0.04 * (hit - self.ema[k]);
            }
            self.n += 1;
        }
    }

    /// The lead to use instead of the usual one, once another has clearly done better.
    pub(crate) fn choice(&self) -> Option<f32> {
        if self.n < 15 {
            return None;
        }
        let usual = LEADS.len() - 1;
        let best = (0..usual).max_by(|a, b| self.ema[*a].partial_cmp(&self.ema[*b]).unwrap_or(std::cmp::Ordering::Equal))?;
        (self.ema[best] > self.ema[usual] + 0.08).then(|| LEADS[best])
    }
}

pub struct Brain {
    /// `seek_idle`: going after the nearest known enemy, and since when we have been idle with no enemy in view.
    seeking: bool,
    idle_since: Option<f32>,
    lead_stats: BTreeMap<u32, LeadStat>,
    /// The enemy we are walking toward with nothing else to do (`seek`).
    seek_id: Option<u32>,
    pub world: Option<Arc<World>>,
    pub nav: Option<Nav>,
    pub params: Params,
    rng: u64,
    pickup_radius: f32,
    /// Drone body: radius and height.
    npc_size: [f32; 2],
    /// Distance within which a drone opens fire on a mech it can see.
    npc_aggro: f32,
    /// Seconds per simulation tick.
    tick_s: f32,
    strafe: f32,
    strafe_until: f32,
    last_pos: [f32; 3],
    stuck: u32,
    unstick_until: f32,
    /// Match time of the decision in progress.
    now_t: f32,
    /// Whether `go` steered us this decision.
    nav_used: bool,
    /// Recent path-following directions, one per decision: (time, direction).
    nav_hist: VecDeque<(f32, [f32; 2])>,
    /// Path-following direction held after it kept reversing: (direction, goal, until).
    nav_hold: Option<([f32; 2], [f32; 2], f32)>,
    /// Recent decisions: (time, true position, steered by `go`, move sent).
    trail: VecDeque<(f32, [f32; 2], bool, [f32; 2])>,
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
    /// Mechs whose laser we saw fire recently: id -> time.
    laser_fired: BTreeMap<u32, f32>,
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
    /// Buildings worth looping around: footprint centre and loop radius (built on first use).
    ring_spots: Option<Vec<([f32; 2], f32)>>,
    /// Building we are looping around, and which way round (+1 counter-clockwise).
    ring_b: Option<([f32; 2], f32)>,
    ring_dir: f32,
    /// Loop mode stays on until this time.
    ring_until: f32,
    /// Enemies that showed they are after us (aimed at us, hurt us or came close): id -> last time.
    hunters: BTreeMap<u32, f32>,
    /// Lines of sight over the map and the sheltered spots (built when `spot` is on).
    pub sight: Option<Arc<SightTable>>,
    /// Sheltered-ground mode stays on until this time.
    spot_until: f32,
    /// Sheltered spot being held or walked to.
    spot_pick: Option<Pick>,
    /// Spots left behind: index -> until when they are not chosen again.
    spot_ban: BTreeMap<usize, f32>,
    /// Since when two or more enemies have had a line to us.
    spot_seen_since: Option<f32>,
    /// Position where only one enemy has a line to us, and when it was last chosen.
    iso_pick: Option<Pick>,
    iso_t: f32,
    /// Our health over the last few seconds: (time, hp).
    hp_hist: VecDeque<(f32, f32)>,
    /// Sidestep in progress: direction and end time; when the last one started.
    dodge_dir: [f32; 2],
    dodge_until: f32,
    /// Retreat burst in progress until this time, toward this cell (None: none found, fight in place).
    cr_until: f32,
    cr_goal: Option<[f32; 2]>,
    /// Fight-position strafe: direction sign and when it next reverses.
    fp_side: f32,
    fp_until: f32,
    /// When the current target was last locked in.
    focus_t: f32,
    /// Late-game home spot and when it was chosen.
    home: Option<([f32; 2], f32)>,
    /// Mechs we hit recently: id -> time.
    hit_t: BTreeMap<u32, f32>,
    /// Bearing and time of the last hit taken from a mech, and of the last weapon or step sound heard.
    hit_bearing: Option<(f32, f32)>,
    sound_bearing: Option<(f32, f32)>,
    /// Free-for-all side (`ffa_detect`): match-type detector, enemy picture, this tick's beliefs, our position
    /// every half second over the last five seconds, and recent kills (time, killer, victim).
    detector: Detector,
    contacts: Contacts,
    beliefs: Vec<Belief>,
    me_hist: VecDeque<(f32, [f32; 2])>,
    kill_log: VecDeque<(f32, Option<u32>, Option<u32>)>,
    /// Current free-for-all mode, since when, why; the pair being third-partied (weaker, stronger); the place
    /// the mode is heading for and when it was chosen; decisions spent per mode.
    ffa_mode: ffa::FfaMode,
    ffa_since: f32,
    ffa_why: String,
    ffa_third: Option<(u32, u32)>,
    ffa_goal: Option<[f32; 2]>,
    ffa_goal_t: f32,
    ffa_ticks: [u32; 6],
    ffa_slot: u32,
    ffa_seed: u64,
    trace_ffa: bool,
    /// Slowest decision so far in ms (only measured while tracing).
    ffa_slowest: f32,
    /// For a viewer only, never read by decisions: believed enemies as x, y, radius, confidence; the current
    /// mode and why; what we believe about each enemy; the last kills seen (time, killer, victim; our own kills
    /// name our slot as the killer, deaths we only counted name neither).
    pub dbg_contacts: Vec<[f32; 4]>,
    /// Debug only, alongside `dbg_contacts`: each belief's enemy id (if a track), where it came from and its age.
    pub dbg_contact_src: Vec<(Option<u32>, &'static str, f32)>,
    pub dbg_kills: Vec<(f32, Option<u32>, Option<u32>)>,
    pub dbg_mode: String,
    pub dbg_enemies: Vec<ffa::EnemyInfo>,
    /// Debug only: whether we are pincered and how widely the enemies round us are spread (degrees).
    pub dbg_pincer: (bool, f32),
    /// Enemies round us now and where they are expected to be (`pred`, `pincer`).
    pin: pincer::PinPicture,
    /// A far fight position chosen while pincered, kept until the given time.
    pin_goal: Option<([f32; 2], f32)>,
    /// Debug only (`TPRED`): motion predictions waiting to be checked, and their errors.
    pred_trace: Option<pincer::PredTrace>,
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
        let map_cfg = params.map_cfg();
        let cfg = &hello.config;
        let f = |a: &str, b: &str, d: f32| cfg.get(a).and_then(|v| v.get(b)).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(d);
        let sight = match (&world, &nav) {
            (Some(w), Some(n)) if params.spot > 0.0 || params.crowd_retreat > 0.0 || params.fight_pos > 0.0 || params.late_hold > 0.0 || params.fight_cubes > 0.0 => {
                Some(Arc::new(SightTable::build(w, n)))
            }
            _ => None,
        };
        Brain {
            seeking: false,
            idle_since: None,
            lead_stats: BTreeMap::new(),
            seek_id: None,
            world,
            nav,
            params,
            rng: (hello.seed ^ 0x9e3779b97f4a7c19) | 1,
            pickup_radius: f("loot", "pickup_radius", 4.0),
            npc_size: [f("npc", "radius", 1.8), f("npc", "height", 3.5)],
            npc_aggro: f("npc", "aggro_range", 45.0),
            tick_s: 1.0 / f("match", "tick_hz", 20.0).max(1.0),
            strafe: 1.0,
            strafe_until: 0.0,
            last_pos: [0.0; 3],
            stuck: 0,
            unstick_until: 0.0,
            now_t: 0.0,
            nav_used: false,
            nav_hist: VecDeque::new(),
            nav_hold: None,
            trail: VecDeque::new(),
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
            laser_fired: BTreeMap::new(),
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
            ring_spots: None,
            ring_b: None,
            ring_dir: 1.0,
            ring_until: -1.0,
            hunters: BTreeMap::new(),
            sight,
            spot_until: -1.0,
            spot_pick: None,
            spot_ban: BTreeMap::new(),
            spot_seen_since: None,
            iso_pick: None,
            iso_t: -1.0,
            hp_hist: VecDeque::new(),
            dodge_dir: [0.0, 0.0],
            dodge_until: -1.0,
            cr_until: -1.0,
            cr_goal: None,
            fp_side: 1.0,
            fp_until: -1.0,
            focus_t: -1.0,
            home: None,
            hit_t: BTreeMap::new(),
            hit_bearing: None,
            sound_bearing: None,
            detector: Detector::default(),
            contacts: Contacts { cfg: map_cfg, ..Contacts::default() },
            beliefs: Vec::new(),
            me_hist: VecDeque::new(),
            kill_log: VecDeque::new(),
            ffa_mode: ffa::FfaMode::Practice,
            ffa_since: 0.0,
            ffa_why: String::new(),
            ffa_third: None,
            ffa_goal: None,
            ffa_goal_t: -100.0,
            ffa_ticks: [0; 6],
            ffa_slot: hello.slot,
            ffa_seed: hello.seed,
            trace_ffa: std::env::var("TFFA").is_ok(),
            ffa_slowest: 0.0,
            dbg_contacts: Vec::new(),
            dbg_contact_src: Vec::new(),
            dbg_kills: Vec::new(),
            dbg_mode: String::new(),
            dbg_enemies: Vec::new(),
            dbg_pincer: (false, 0.0),
            pin: pincer::PinPicture::default(),
            pin_goal: None,
            pred_trace: std::env::var("TPRED").is_ok().then(pincer::PredTrace::default),
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
        let dir = self.path_dir(pos, goal);
        if dir == [0.0, 0.0] {
            return dir;
        }
        self.nav_used = true;
        // Close to the goal, turning around is just arriving.
        if d2(pos, &goal) <= 6.0 {
            return dir;
        }
        let t = self.now_t;
        if let Some((hd, hg, until)) = self.nav_hold {
            if t < until && d2(&hg, &goal) < 10.0 {
                return hd;
            }
            self.nav_hold = None;
        }
        // One entry per decision; keep the last second.
        if self.nav_hist.back().map_or(false, |e| e.0 == t) {
            self.nav_hist.pop_back();
        }
        self.nav_hist.push_back((t, dir));
        while self.nav_hist.front().map_or(false, |e| t - e.0 > 1.0) {
            self.nav_hist.pop_front();
        }
        // The path direction keeps reversing: commit to one way for a second to break the cycle.
        let flips = self.nav_hist.iter().zip(self.nav_hist.iter().skip(1)).filter(|(x, y)| x.1[0] * y.1[0] + x.1[1] * y.1[1] < -0.5).count();
        if flips >= 3 {
            let prev = self.nav_hist.iter().rev().nth(1).map_or(dir, |e| e.1);
            let open = |d: [f32; 2]| self.walkable(pos[0] + d[0] * 4.0, pos[1] + d[1] * 4.0);
            let hd = if !open(dir) && open(prev) { prev } else { dir };
            self.nav_hold = Some((hd, goal, t + 1.0));
            self.nav_hist.clear();
            return hd;
        }
        dir
    }

    fn path_dir(&mut self, pos: &[f32; 3], goal: [f32; 2]) -> [f32; 2] {
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
        let full_now = |w: &Weapon| w.charge >= w.stat("charge", 0.75) - 0.01;
        let gun = find("gun").filter(|(_, _, s)| s.ok);
        let mut aim: Option<[f32; 2]> = gun.map(|(_, _, s)| s.aim);
        let mut engaged = gun.is_some();
        if let Some((i, w, s)) = gun {
            a.fire[*i] = lined(s, slack) && w.cooldown <= 0.12;
        }
        // The laser is for mechs; a charge that has waited nearly as long as it can without a line is let go
        // rather than holding the aim away from the gun.
        // With `laser_hold` a charge is never let go unaimed: it is held on the solution until it lines up or
        // fires by itself at the end of the hold.
        let hold = p.laser_hold > 0.0;
        let near_auto = |w: &Weapon| !hold && self.laser_full_t.map_or(false, |t0| o.t - t0 >= w.stat("hold_s", 2.0) - 0.25);
        if let Some((i, w, s)) = find("laser").filter(|_| t.kind == "mech") {
            if hold && !s.ok && w.charge > 0.0 {
                // No clear line right now: keep the charge and the aim on where the mech is.
                a.fire[*i] = true;
                if aim.is_none() {
                    aim = Some(s.aim);
                }
            } else if s.ok && full_now(w) && near_auto(w) && !lined(s, [0.0, 0.0]) {
                engaged = true;
                a.fire[*i] = false;
            } else if s.ok {
                engaged = true;
                let full_c = w.stat("charge", 0.75);
                let mut full = w.charge >= full_c - 0.01;
                if !full && p.laser_finish > 0.0 && t.max_hp > 0.0 && w.charge >= w.stat("min_charge", 0.15) {
                    let frac = w.stat("min_frac", 0.35);
                    let mut dmg = w.stat("damage", 315.0) * (frac + (1.0 - frac) * (w.charge / full_c).min(1.0)) * (1.0 + 0.04 * me.cubes.min(10.0));
                    if p.laser_fix > 0.0 && t.weapons.iter().any(|k| k == "mortar") {
                        dmg *= 1.0 + w.stat("cookoff", 0.6);
                    }
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
            } else if (w.charge > 0.0 && !(full_now(w) && near_auto(w))) || (p.charge_seen > 0.0 && w.cooldown <= 0.0 && w.charge <= 0.0) {
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
        // Body radius and half height: a mech grows with its cubes, a drone is much smaller.
        let drone = t.kind == "npc";
        let (radius, half) = if drone { (self.npc_size[0], self.npc_size[1] * 0.5) } else { (2.5 * scale, 4.5 * scale) };
        let mut tc = [t.pos[0], t.pos[1], t.pos[2] + half];
        if self.params.aim_high > 0.0 && !drone && !self.clear(muzzle, tc) {
            let hi = [t.pos[0], t.pos[1], t.pos[2] + 7.6 * scale];
            if self.clear(muzzle, hi) {
                tc = hi;
            }
        }
        let dist = d3(&muzzle, &tc);
        let flat = |p: &[f32; 3], from: &[f32; 3]| [(p[1] - from[1]).atan2(p[0] - from[0]), (p[2] - from[2]).atan2(d2(p, from).max(0.01))];
        let body = [(radius * 0.85 / dist.max(1.0)).atan(), (half * 0.8 / dist.max(1.0)).atan()];
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
                let learned = if p.lead_adapt > 0.0 && !airborne { self.lead_stats.get(&t.id).and_then(|st| st.choice()) } else { None };
                let lead = if airborne { p.lead_air } else { learned.unwrap_or(p.lead_ground) };
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
                // Holding for a lined-up release: lead by at least the tick the beam takes to leave.
                let ll = if self.params.laser_hold > 0.0 { self.params.laser_lead.max(self.tick_s) } else { self.params.laser_lead };
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
                let s = self.enclosure(c, zz) as f32 * 10.0 - d2(&c, &me.pos) * 0.15 - self.crowd_cost(o.t, &me.pos, c);
                if s > best.1 {
                    best = (c, s);
                }
            }
        }
        best.0
    }

    pub fn act(&mut self, o: &Obs) -> Action {
        let t0 = self.trace_ffa.then(std::time::Instant::now);
        if self.params.ffa_detect > 0.0 {
            self.ffa_observe(o);
        }
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
        if let Some(t0) = t0 {
            self.ffa_slowest = self.ffa_slowest.max(t0.elapsed().as_secs_f32() * 1000.0);
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
        self.now_t = o.t;
        self.nav_used = false;

        // --- memory of mechs
        for m in o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0)) {
            let prev_charge = self.known.get(&m.id).and_then(|k| k.charge_since);
            // A charge we were watching that is gone: the beam went out.
            if prev_charge.is_some() && !m.charging {
                self.laser_fired.insert(m.id, o.t);
            }
            if p.lead_adapt > 0.0 {
                let st = self.lead_stats.entry(m.id).or_default();
                let (mp, mvl) = ([m.pos[0], m.pos[1]], [m.vel[0], m.vel[1]]);
                st.settle(o.t, mp, mvl, 2.5 * (1.0 + 0.05 * m.cubes) * 0.85);
                let d = d2(&m.pos, &pos);
                if d < 230.0 {
                    st.look(o.t, [pos[0], pos[1]], mp, mvl, (d - 3.0).max(0.0) / 252.0 + p.lead_latency);
                }
            }
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
                    hp_frac: if m.max_hp > 0.0 { m.hp / m.max_hp } else { 1.0 },
                },
            );
        }
        let mem = p.memory_s;
        self.known.retain(|_, k| o.t - k.seen_t < mem);
        self.attackers.retain(|_, t| o.t - *t < 10.0);
        self.drone_attackers.retain(|_, t| o.t - *t < 6.0);
        for b in o.beams.iter().filter(|b| b.kind == "laser") {
            let t = o.t - b.age;
            if self.laser_fired.get(&b.by).map_or(true, |x| *x < t) {
                self.laser_fired.insert(b.by, t);
            }
        }
        self.laser_fired.retain(|_, t| o.t - *t < 10.0);
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
                self.seeking = true;
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

        self.track_contacts(o);

        // --- upgrades
        if let Some(offer) = &o.offer {
            let list = if p.upg_dmg >= 2.0 { UPGRADES_TANK } else if p.upg_dmg > 0.0 { UPGRADES_DMG } else { UPGRADES };
            let rank = |id: &str| list.iter().position(|u| *u == id).unwrap_or(list.len());
            a.pick = offer.choices.iter().enumerate().min_by_key(|(_, c)| rank(&c.id)).map(|(i, _)| i as u32);
        }

        // --- enemies and target
        let enemies: Vec<&Actor> = o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0)).collect();
        if !enemies.is_empty() {
            self.seeking = false;
            self.idle_since = None;
        }
        let near_enemies = enemies.iter().filter(|m| d2(&m.pos, &pos) < p.crowd_radius).count();
        let no_laser = !me.weapons.iter().flatten().any(|w| w.kind == "laser");
        // With `late_hold`, the late game starts once the next circle is small.
        let late = if p.late_hold > 0.0 {
            o.zone.next_radius <= p.late_r
        } else {
            (p.wall_t > 0.0 && o.t >= p.wall_t) || (p.wall_t > 0.0 && o.zone.radius <= p.wall_r)
        };
        let hold_late = p.late_hold > 0.0 && late;
        let filter_early = p.engage_filter > 0.0 && !late;
        // Loop mode: late, or pressed by three or more enemies (close by, or hurting us).
        if p.ring > 0.0 {
            let mut pressing: Vec<u32> = self.known.iter().filter(|(_, k)| o.t - k.seen_t < 4.0 && d2(&k.pos, &pos) < 150.0).map(|(id, _)| *id).collect();
            pressing.extend(self.attackers.iter().filter(|(_, t)| o.t - **t < 3.0).map(|(id, _)| *id));
            pressing.sort_unstable();
            pressing.dedup();
            if late || pressing.len() >= 3 {
                self.ring_until = o.t + 5.0;
            }
        }
        let ring_on = p.ring > 0.0 && o.t < self.ring_until;
        // Sheltered-ground mode: same trigger as the loop, with its own crowd size.
        if p.spot > 0.0 {
            let mut pressing: Vec<u32> = self.known.iter().filter(|(_, k)| o.t - k.seen_t < 4.0 && d2(&k.pos, &pos) < 150.0).map(|(id, _)| *id).collect();
            pressing.extend(self.attackers.iter().filter(|(_, t)| o.t - **t < 3.0).map(|(id, _)| *id));
            pressing.sort_unstable();
            pressing.dedup();
            if late || pressing.len() as f32 >= p.spot_crowd {
                self.spot_until = o.t + 5.0;
            }
            self.hp_hist.push_back((o.t, me.hp));
            while self.hp_hist.front().map_or(false, |e| o.t - e.0 > 3.0) {
                self.hp_hist.pop_front();
            }
        }
        let spot_on = p.spot > 0.0 && o.t < self.spot_until && self.sight.is_some() && !ring_on;
        if p.spot > 0.0 && !spot_on {
            // Choose afresh next time.
            self.spot_pick = None;
            self.iso_pick = None;
        }
        // Counter-punching: before the late game only enemies that came for us are targets.
        let counter = p.counter > 0.0 && !late && !ring_on;
        if counter {
            for m in &enemies {
                let aimed = wrap(m.aim[0] - yaw_to(&m.pos, &pos)).abs() < 0.2;
                let hurt = self.attackers.get(&m.id).map_or(false, |t| o.t - t < 4.0);
                let sought = p.seek > 0.0 && self.seek_id == Some(m.id) && d2(&m.pos, &pos) < p.seek_fire;
                if aimed || hurt || sought || d2(&m.pos, &pos) < p.counter_range {
                    self.hunters.insert(m.id, o.t);
                }
            }
        }
        self.hunters.retain(|_, t| o.t - *t < 10.0);
        // Free-for-all modes (only once a free-for-all is settled): who is worth shooting, and the fight we join.
        let ffa_on = self.ffa_on();
        // In a free-for-all placing counts for more than kills: the earlier, more careful settings (`ffa_plain`).
        let (cr_max_d, cr_laser, cr_strafe, reload_laser, dodge_after) =
            if ffa_on && p.ffa_plain > 0.0 { (0.0, 0.0, 0.0, 1.0, 9.0) } else { (p.cr_max_d, p.cr_laser, p.cr_strafe, p.reload_laser, p.dodge_after) };
        let cands: Vec<&Actor> = enemies
            .iter()
            .filter(|m| !counter || self.hunters.get(&m.id).map_or(false, |t| o.t - t < 4.0))
            .filter(|m| !(hold_late || filter_early) || self.engage_ok(o, m, hold_late) || (p.seek > 0.0 && !late && self.seek_id == Some(m.id) && d2(&m.pos, &pos) < p.seek_fire))
            .filter(|m| !ffa_on || self.ffa_engage(o, m, hp_frac))
            .copied()
            .collect();
        let muzzle = [pos[0], pos[1], pos[2] + me.stat("muzzle_height", 7.0)];
        let score = |m: &Actor| {
            let d = d2(&m.pos, &pos);
            let hpf = if m.max_hp > 0.0 { m.hp / m.max_hp } else { 1.0 };
            let attacker = self.attackers.get(&m.id).map_or(false, |t| o.t - t < 4.0);
            let loot = if no_laser && m.weapons.iter().any(|w| w == "laser") { p.loot_target } else { 0.0 };
            // Its torso (and so its next shot) is pointed at us.
            let aimer = if p.focus_aimer > 0.0 && wrap(m.aim[0] - yaw_to(&m.pos, &pos)).abs() < 0.2 { p.focus_aimer } else { 0.0 };
            // Looping: whoever has a line on us now comes first.
            let unseen = if (ring_on || spot_on) && !self.clear(muzzle, [m.pos[0], m.pos[1], m.pos[2] + 4.5]) { 150.0 } else { 0.0 };
            d + hpf * p.focus_hp - if attacker { 40.0 } else { 0.0 } - loot - aimer + unseen + self.reinforcements(m)
        };
        let best = cands
            .iter()
            .map(|m| (*m, score(*m)))
            .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
        let target = match (best, self.focus.and_then(|f| cands.iter().find(|m| m.id == f))) {
            _ if p.engage_filter > 0.0 => {
                cands.iter().map(|m| (*m, self.engage_score(o, m))).min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal)).map(|x| x.0)
            }
            (Some((_, bs)), Some(f)) if p.focus_stick > 0.0 && score(*f) - p.focus_stick < bs && d2(&f.pos, &pos) < p.open_fire_range => Some(*f),
            (Some((b, _)), _) => Some(b),
            _ => None,
        };
        let target = match self.ffa_third.filter(|_| ffa_on) {
            Some((w, _)) => enemies.iter().find(|m| m.id == w).copied().or(target),
            None => target,
        };
        let new_focus = target.map(|t| t.id);
        if new_focus != self.focus || o.t - self.focus_t >= 2.0 {
            self.focus_t = o.t;
        }
        self.focus = new_focus;
        if let Some(t) = target {
            self.threat = Some((t.pos, o.t));
        }
        if self.threat.map_or(false, |(_, t)| o.t - t > 8.0) {
            self.threat = None;
        }
        let threats = self.threats(o);
        self.update_pincer(o, &pos, target);

        // Drone to deal with: one shooting us, or (when calm) one worth farming.
        let calm = enemies.is_empty() && o.t - self.last_mech_hurt_t > 5.0;
        let drones: Vec<&Actor> = o.mechs.iter().filter(|m| m.kind == "npc" && m.hp > 0.0).collect();
        // Without `drone_fight`, only a drone that hurt us in the last 3 s counts.
        let drone_mem = if p.drone_fight > 0.0 { f32::MAX } else { 3.0 };
        let hostile_drone = drones
            .iter()
            .filter(|d| self.drone_attackers.get(&d.id).map_or(false, |t| o.t - t < drone_mem))
            .min_by(|x, y| d2(&x.pos, &pos).partial_cmp(&d2(&y.pos, &pos)).unwrap_or(std::cmp::Ordering::Equal))
            .copied();
        let early = o.t < p.farm_until;
        let drone_reach = if early { p.drone_range.max(p.farm_reach.min(120.0)) } else { p.drone_range };
        let farm_on = p.farm_stop <= 0.0 || o.t < p.farm_stop;
        // In a free-for-all, drones are farmed only with no believed enemy within 150 m.
        let ffa_busy = ffa_on && self.beliefs.iter().any(|b| b.conf >= 0.3 && d2(&b.pos, &pos) < 150.0);
        let farm_drone = if p.drone_fight > 0.0 && calm && farm_on && hp_frac > p.farm_hp && !self.healing && !ffa_busy {
            drones
                .iter()
                .filter(|d| d2(&d.pos, &pos) < drone_reach)
                .min_by(|x, y| d2(&x.pos, &pos).partial_cmp(&d2(&y.pos, &pos)).unwrap_or(std::cmp::Ordering::Equal))
                .copied()
        } else {
            None
        };
        // A drone close enough to open fire on us, or one that hurt us lately: shoot it while no mech is in view
        // or shooting at us, the most damaged first.
        let threat_drone = if p.drone_threat > 0.0 && target.is_none() && o.t - self.last_mech_hurt_t > 3.0 {
            let sc = |d: &Actor| d2(&d.pos, &pos) + 60.0 * if d.max_hp > 0.0 { d.hp / d.max_hp } else { 1.0 };
            drones
                .iter()
                .filter(|d| d2(&d.pos, &pos) < self.npc_aggro + 5.0 || self.drone_attackers.get(&d.id).map_or(false, |t| o.t - t < 4.0))
                .min_by(|x, y| sc(x).partial_cmp(&sc(y)).unwrap_or(std::cmp::Ordering::Equal).then(x.id.cmp(&y.id)))
                .copied()
        } else {
            None
        };
        let drone_target = if threat_drone.is_some() {
            threat_drone
        } else if target.is_none() {
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
        // Aim that a held charge needs kept whatever the movement mode does with the torso.
        let mut laser_aim: Option<[f32; 2]> = None;
        if let Some((li, lw)) = me.weapons.iter().enumerate().take(2).find_map(|(i, w)| w.as_ref().filter(|w| w.kind == "laser").map(|w| (i, w))) {
            let full = lw.charge >= lw.stat("charge", 0.75) - 0.01;
            if !full {
                self.laser_full_t = None;
            } else if self.laser_full_t.is_none() {
                self.laser_full_t = Some(o.t);
            }
            let held_full = self.laser_full_t.map_or(0.0, |t0| o.t - t0);
            // Without a mech to shoot at, never start a charge; hold one only for a mech seen within the last
            // second, aimed where it should be now, and let it go before it would fire on its own.
            if !shoot_at.map_or(false, |t| t.kind == "mech") {
                // With `laser_hold`: the mech seen last (any time we still remember it), held until the charge
                // fires by itself.
                let hold = p.laser_hold > 0.0;
                let recent = self
                    .known
                    .values()
                    .filter(|k| hold || o.t - k.seen_t < 1.0)
                    .min_by(|x, y| {
                        if hold {
                            y.seen_t.partial_cmp(&x.seen_t).unwrap_or(std::cmp::Ordering::Equal).then(d2(&x.pos, &pos).partial_cmp(&d2(&y.pos, &pos)).unwrap_or(std::cmp::Ordering::Equal))
                        } else {
                            d2(&x.pos, &pos).partial_cmp(&d2(&y.pos, &pos)).unwrap_or(std::cmp::Ordering::Equal)
                        }
                    })
                    .map(|k| {
                        let dt = (o.t - k.seen_t).min(if hold { 1.5 } else { f32::MAX });
                        [k.pos[0] + k.vel[0] * dt, k.pos[1] + k.vel[1] * dt, k.pos[2] + 4.5]
                    });
                match recent {
                    Some(q) if lw.charge > 0.0 && (hold || held_full < lw.stat("hold_s", 2.0) - 0.25) => {
                        a.fire[li] = true;
                        let muzzle = [pos[0], pos[1], pos[2] + me.stat("muzzle_height", 7.0)];
                        a.aim = [yaw_to(&muzzle, &q), (q[2] - muzzle[2]).atan2(d2(&q, &muzzle).max(0.1))];
                        if hold {
                            laser_aim = Some(a.aim);
                        }
                    }
                    _ => a.fire[li] = false,
                }
            }
            let soon = p.precharge > 0.0
                && p.laser_hold <= 0.0
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
        let flee_hp = if p.solo_hp > 0.0 && near_enemies <= 1 { p.solo_hp } else { p.retreat_hp };
        if hp_frac < flee_hp && (threat_live || recently_hurt_by_mech) {
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
        if finishing || p.crowd_retreat > 0.0 {
            self.fleeing = false;
        }
        let crowded = near_enemies >= p.crowd_count && !finishing && p.crowd_retreat <= 0.0;
        if p.heal_lo > 0.0 && enemies.is_empty() && hp_frac < p.heal_lo && o.t - self.last_mech_hurt_t > 1.0 {
            self.healing = true;
        }
        if hp_frac >= p.heal_hi || p.heal_lo <= 0.0 {
            self.healing = false;
        }
        // Stay in a live fight rather than break off (`stay_fight`).
        let stay = p.stay_fight > 0.0 && !ffa_on && target.is_some() && (engaged || recently_hurt_by_mech);
        if stay {
            self.fleeing = false;
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
                    if o.t - since >= dodge_after && !self.dodged_charge.contains_key(id) {
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
                // A laser holder is closed on from afar only while its laser is cooling down; an empty
                // magazine leaves nothing to cover a long approach with.
                let guard_ok = p.rush_guard <= 0.0 || {
                    let arc_r = arc_w.map_or(45.0, |w| w.stat("range", 45.0));
                    let laser_ready = t.weapons.iter().any(|w| w == "laser") && self.laser_fired.get(&t.id).map_or(true, |x| o.t - x > 1.5);
                    let empty = me.weapons.iter().flatten().any(|w| w.kind == "gun" && w.ammo <= 0.0);
                    !(laser_ready && d > arc_r + 15.0) && !(empty && d > 60.0)
                };
                if d < reach && e_ok && hp_ok && others <= p.rush_crowd && guard_ok {
                    Some(t)
                } else {
                    None
                }
            }
            _ => None,
        };
        let rush_t = rush_t.filter(|_| !hold_late);
        self.rushing = rush_t.map(|t| t.id);
        if rush_t.is_some() {
            self.fleeing = false;
        }
        // An empty slot: the nearest good weapon on the ground comes first.
        let worst_rank = worst_slot.map_or(0, |(_, r)| r);
        let arm_goal = if p.arm > 0.0 && farm_on && !hold_late && (empty_slot || p.arm_better > 0.0 || p.laser_seek > 0.0) && hp_frac > p.arm_hp {
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
            Some((ks, kt)) if p.collect > 0.0 && !hold_late && o.t - kt < p.collect_s && hp_frac > p.collect_hp && (enemies.len() as f32) < p.collect_seen => {
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
        let live_fight = p.hold_fight > 0.0 && target.is_some() && (engaged || recently_hurt_by_mech);
        // Finishing a weak target up close (`engage_filter`).
        let close_kill = p.engage_filter > 0.0
            && target.map_or(false, |t| t.max_hp > 0.0 && t.hp / t.max_hp < 0.38)
            && (p.ext_finish <= 0.0 || (hp_frac > 0.42 && near_enemies <= 2));
        // Outnumbered or losing: break off in bursts to the nearest hidden cell (`crowd_retreat`).
        let mut cr_goal: Option<[f32; 2]> = None;
        // Breaking off was called for but there is nowhere to go (`cb_corner`), and every enemy we know of.
        let mut cb_cornered = false;
        let mut cb_all: Vec<[f32; 2]> = Vec::new();
        if p.crowd_retreat > 0.0 {
            let near = enemies.iter().filter(|m| d2(&m.pos, &pos) < p.cr_radius).count();
            let tf = target.map_or(1.0, |t| if t.max_hp > 0.0 { t.hp / t.max_hp } else { 1.0 });
            let t_hp = target.map_or(0.0, |t| if t.max_hp > 0.0 { t.hp } else { 1400.0 });
            let losing = if p.ext_lose > 0.0 { tf > hp_frac + p.lose_margin } else { t_hp > me.hp };
            let trigger = (near >= 2 && hp_frac < p.cr_hp2)
                || (near >= 3 && hp_frac < p.cr_hp3)
                || (hp_frac < p.cr_hp1 && target.is_some() && losing && (p.cr_solo > 0.0 || near >= 2));
            // About to win: a target close by that is nearly dead or much weaker than us.
            let winning = p.ext_lose > 0.0 && target.map_or(false, |t| d2(&t.pos, &pos) < 120.0 && (t_hp < 200.0 || t_hp < 0.5 * me.hp));
            let veto = late || finishing || winning || close_kill || (target.is_some() && tf < 0.25 && (p.ext_lose <= 0.0 || hp_frac > 0.2) && near <= 2) || threats.is_empty();
            let unaimed = p.cr_unaimed > 0.0
                && o.t - self.last_mech_hurt_t > p.cr_unaimed
                && !enemies.iter().any(|m| wrap(m.aim[0] - yaw_to(&m.pos, &pos)).abs() < 0.3);
            if veto || unaimed {
                self.cr_until = -1.0;
            } else if o.t < self.cr_until || trigger {
                let cb_on = p.cb_away > 0.0 || p.cb_path > 0.0 || p.cb_known > 0.0 || p.cb_corner > 0.0 || p.cb_turn > 0.0;
                let foes = if cb_on { self.cb_foes(o, &threats) } else { Vec::new() };
                let all: Vec<[f32; 2]> = foes.iter().map(|f| [f[0], f[1]]).collect();
                let eyes: Vec<[f32; 3]> = if p.cb_known > 0.0 {
                    foes.iter().filter(|f| d2(*f, &pos) < 260.0).take(6).copied().collect()
                } else {
                    threats[..threats.len().min(4)].to_vec()
                };
                let at_exposed = self.cr_goal.map_or(false, |g| d2(&g, &pos) < 2.5 && self.hidden_from(g, pos[2], &eyes) < eyes.len());
                // The way to the goal now runs past an enemy: choose again.
                let here2 = [pos[0], pos[1]];
                let cut_off = p.cb_path > 0.0 && self.cr_goal.map_or(false, |g| d2(&g, &pos) > 2.5 && !Self::cb_line_clear(here2, g, &all));
                if o.t >= self.cr_until || at_exposed || cut_off {
                    if o.t >= self.cr_until {
                        self.cr_until = o.t + p.cr_burst;
                    }
                    let prev = self.cr_goal;
                    let mut g = self.retreat_cell(o, &eyes, if p.cb_away > 0.0 || p.cb_path > 0.0 { &all } else { &[] });
                    if cr_max_d > 0.0 && g.map_or(false, |g| d2(&g, &pos) > cr_max_d) {
                        g = None;
                    }
                    if p.cb_turn > 0.0 {
                        // A new goal back the way we came and toward the enemies: keep the old one while it is
                        // still away from them and clear, else stay and fight.
                        if let (Some(ng), Some(pg)) = (g, prev) {
                            let dn = d2(&ng, &pos);
                            let dp = d2(&pg, &pos);
                            if dn > 3.0 && dp > 3.0 {
                                let un = norm2(ng[0] - pos[0], ng[1] - pos[1]);
                                let up = norm2(pg[0] - pos[0], pg[1] - pos[1]);
                                let near: Vec<&[f32; 2]> = all.iter().filter(|f| d2(*f, &pos) < 200.0).collect();
                                let toward = !near.is_empty() && {
                                    let n = near.len() as f32;
                                    let c = [near.iter().map(|f| f[0]).sum::<f32>() / n, near.iter().map(|f| f[1]).sum::<f32>() / n];
                                    let uc = norm2(c[0] - pos[0], c[1] - pos[1]);
                                    un[0] * uc[0] + un[1] * uc[1] > 0.0
                                };
                                if un[0] * up[0] + un[1] * up[1] < -0.3 && toward {
                                    let keep = Self::cb_away_cost(here2, pg, &all).is_some() && Self::cb_line_clear(here2, pg, &all);
                                    g = if keep { Some(pg) } else { None };
                                }
                            }
                        }
                    }
                    self.cr_goal = g;
                }
                cr_goal = self.cr_goal;
                cb_cornered = p.cb_corner > 0.0 && cr_goal.is_none();
                cb_all = all;
            }
        }
        if stay && (cr_goal.is_some() || cb_cornered) && cr_goal.map_or(true, |g| d2(&g, &pos) > p.stay_cover) {
            cr_goal = None;
            cb_cornered = false;
            self.cr_until = -1.0;
        }
        let mut boost = false;
        let mut mv: [f32; 2];
        let mut label = "idle";
        let zone_stand = p.stand_zone > 0.0 && stand_from.is_some() && me.in_zone && d2(&pos, &z.center) < z.radius - p.stand_zone_m;
        let zone_now = !me.in_zone || (must_rotate && out_next > 25.0 && !zone_stand);
        let ffa_mv = if ffa_on && !zone_now { self.ffa_move(o, &pos, hp_frac, shoot_at.is_some(), &enemies, arm_goal) } else { None };
        if zone_now {
            label = "zone";
            mv = self.go(&pos, zone_goal);
            boost = !me.in_zone && me.energy > 5.0;
        } else if let Some(f) = ffa_mv {
            label = f.label;
            mv = f.mv;
            boost = f.boost;
            if f.no_charge {
                for (i, w) in me.weapons.iter().enumerate().take(2) {
                    if w.as_ref().map_or(false, |w| w.kind == "laser" && w.charge <= 0.0) {
                        a.fire[i] = false;
                    }
                }
            }
            if shoot_at.is_none() {
                match f.face {
                    Some(y) => a.aim = [y, 0.0],
                    None => self.scan(&mut a),
                }
            }
        } else if let Some(g) = cr_goal {
            // Break off to hidden ground; stand still once there.
            label = "cbreak";
            let dg = d2(&pos, &g);
            let seen_by = shoot_at.filter(|t| cr_strafe > 0.0 && t.kind == "mech");
            mv = if dg >= 2.0 {
                self.go(&pos, g)
            } else if let Some(t) = seen_by {
                if o.t >= self.fp_until {
                    self.fp_side = -self.fp_side;
                    self.fp_until = o.t + 0.4 + 0.5 * self.rnd();
                }
                let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
                let side = [-to[1] * self.fp_side, to[0] * self.fp_side];
                if self.walkable(pos[0] + side[0] * 4.0, pos[1] + side[1] * 4.0) {
                    side
                } else {
                    [-side[0], -side[1]]
                }
            } else {
                [0.0, 0.0]
            };
            boost = me.energy > p.retreat_boost_energy && dg > 5.0;
            if dg > 6.0 && cr_laser <= 0.0 {
                for (i, w) in me.weapons.iter().enumerate().take(2) {
                    if w.as_ref().map_or(false, |w| w.kind == "laser" && w.charge <= 0.0) {
                        a.fire[i] = false;
                    }
                }
            }
            if shoot_at.is_none() {
                if let Some(th) = threats.first() {
                    a.aim = [yaw_to(&pos, th), 0.0];
                }
            }
        } else if ring_on {
            // Keep going round a building, on the side away from the enemies; boost only to get out of the
            // sight of two or more.
            label = "ring";
            mv = self.ring_move(o, &pos);
            let chest = [pos[0], pos[1], pos[2] + 5.0];
            let sighted = enemies.iter().filter(|m| self.world.as_ref().map_or(true, |w| w.sees([m.pos[0], m.pos[1], m.pos[2] + 8.0], chest))).count();
            boost = sighted >= 2 && me.energy > 10.0;
            if shoot_at.is_none() {
                if let Some(th) = threats.first() {
                    a.aim = [yaw_to(&pos, th), 0.0];
                } else {
                    self.scan(&mut a);
                }
            }
        } else if spot_on {
            // Fight from sheltered ground; boost only to get out of the sight of two or more.
            let killing = target.map_or(false, |t| engaged && t.max_hp > 0.0 && t.hp / t.max_hp < 0.35);
            let (m, b, l) = self.spot_move(o, &pos, &mut a, shoot_at.is_some(), &threats, killing);
            mv = m;
            boost = b;
            label = l;
        } else if let Some(g) = arm_goal
            .filter(|_| !(p.stand_first > 0.0 && stand_from.is_some()))
            .filter(|g| !(p.arm_yield > 0.0 && d2(g, &pos) > 20.0 && target.map_or(false, |t| d2(&t.pos, &pos) < p.arm_yield)))
        {
            label = "arm";
            mv = self.go(&pos, [g[0], g[1]]);
            boost = me.energy > 30.0 && d2(&g, &pos) > 12.0;
            // On the way, break crates in view with the gun while no mech is around.
            let gun_ready = me.weapons.iter().flatten().any(|w| w.kind == "gun" && w.ammo > 0.0 && w.reload <= 0.0);
            if p.arm_crates > 0.0 && gun_ready && shoot_at.is_none() && enemies.is_empty() && o.t - self.last_mech_hurt_t > 3.0 {
                let mut near: Vec<[f32; 3]> = o.crates.iter().map(|c| c.pos).filter(|q| d3(q, &pos) < 100.0).collect();
                near.sort_by(|x, y| d3(x, &pos).partial_cmp(&d3(y, &pos)).unwrap_or(std::cmp::Ordering::Equal));
                for c in &near {
                    if self.aim_crate(o, &mut a, &pos, c) {
                        break;
                    }
                }
            }
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
        } else if hold_late && !live_fight {
            // Hold a quiet, sheltered home spot inside the next circle.
            label = "home";
            let h = self.home_spot(o);
            let dh = d2(&pos, &h);
            mv = if dh < 2.5 { [0.0, 0.0] } else { self.go(&pos, h) };
            let slow = (p.home_speed / walk_speed.max(0.1)).min(1.0);
            mv = [mv[0] * slow, mv[1] * slow];
            if shoot_at.is_none() {
                if let Some(b) = self.alert_bearing(o) {
                    a.aim = [b, 0.0];
                } else if let Some(th) = threats.first() {
                    a.aim = [yaw_to(&pos, th), 0.0];
                } else {
                    self.scan(&mut a);
                }
            }
        } else if late && !self.fleeing && !live_fight {
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
                if let Some(t) = target.filter(|t| p.ltg_space > 0.0 && t.weapons.iter().any(|w| w == "lightning") && d2(&t.pos, &pos) < 60.0) {
                    let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
                    mv = self.strafe_dir(o, &pos, to, -1.0, false);
                    boost = me.energy > 10.0;
                }
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
            let ltg = p.ltg_space > 0.0 && t.weapons.iter().any(|w| w == "lightning") && d2(&t.pos, &pos) < 60.0;
            let radial = if ltg { -1.0 } else if laser_on.is_some() { -0.2 } else { -p.kite };
            mv = self.strafe_dir(o, &pos, to, radial, false);
            if p.kite_away > 0.0 && laser_on.is_none() {
                let dir = self.away_dir(o, &pos, 3.0, None);
                let perp = [-to[1] * self.strafe, to[0] * self.strafe];
                let cand = norm2(dir[0] + perp[0] * p.stand_weave, dir[1] + perp[1] * p.stand_weave);
                mv = if self.walkable(pos[0] + cand[0] * 5.0, pos[1] + cand[1] * 5.0) { cand } else { dir };
            }
            boost = me.energy > p.retreat_boost_energy || (ltg && me.energy > 10.0);
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
                    a.fire[li] = (cd <= 0.0 && p.laser_hold <= 0.0) || me.weapons[li].as_ref().map_or(false, |w| w.charge > 0.0);
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
            let pref = if close_kill {
                20.0
            } else if finishing {
                p.finish_range
            } else if p.fight_pos > 0.0 && !(has_arc && hp_frac > p.arc_min_hp && near_enemies as f32 <= p.arc_crowd) {
                let base = if has_laser { p.pref_laser } else { p.pref_gun };
                let ltg = if they_arc { p.keep_ltg } else { 0.0 };
                let mtr = if t.weapons.iter().any(|w| w == "mortar") { p.keep_mortar } else { 0.0 };
                base.max(ltg).max(mtr)
            } else if has_arc && hp_frac > p.arc_min_hp && near_enemies as f32 <= p.arc_crowd {
                if p.fight_pos > 0.0 && p.ext_fpos > 0.0 { p.pref_ltg } else { p.arc_range }
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
            mv = if p.fight_pos > 0.0 {
                let ready = me.weapons.iter().flatten().any(|w| match w.kind.as_str() {
                    "gun" => w.ammo > 0.0 && w.reload <= 0.0 && w.cooldown <= 0.3,
                    "lightning" => dist < w.stat("range", 45.0) && me.energy >= w.stat("energy", 3.0),
                    _ => w.cooldown <= 0.3,
                });
                self.fight_pos_move(o, t, pref, ready)
            } else {
                self.strafe_dir(o, &pos, to, radial, false)
            };
            let reloading = me.weapons.iter().flatten().any(|w| w.kind == "gun" && (w.reload > 0.4 || w.ammo <= 0.0));
            let eyes: Vec<[f32; 3]> = self
                .known
                .values()
                .filter(|k| o.t - k.seen_t < 4.0 && d2(&k.pos, &pos) < 200.0)
                .map(|k| [k.pos[0], k.pos[1], k.pos[2] + 8.0])
                .collect();
            let laser_up = has_laser && reload_laser <= 0.0 && me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.cooldown < 0.8);
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
            } else if let Some(c) = (if p.fight_cubes > 0.0 { self.fight_cube(o, t) } else { near_cube }).filter(|_| !finishing || dist < 20.0) {
                label = "fcube";
                let g = self.go(&pos, [c[0], c[1]]);
                mv = norm2(g[0] + mv[0] * 0.35, g[1] + mv[1] * 0.35);
                boost = me.energy > 30.0;
            } else if p.expose_weight > 0.0 && p.fight_pos <= 0.0 {
                mv = self.less_exposed(o, &pos, t, mv, p.expose_weight);
            }
            mv = [mv[0] * p.fight_speed, mv[1] * p.fight_speed];
            boost = boost || (laser_on.is_some() && me.energy > 10.0);
            if p.fight_boost > 0.0 && me.energy > p.fight_boost {
                boost = true;
            }
            if p.close_boost > 0.0 && p.fight_pos <= 0.0 && off > 15.0 && me.energy > p.close_boost && label == "fight" {
                boost = true;
            }
        } else if let (Some(d), true) = (drone_target, p.drone_fight > 0.0) {
            label = "drone";
            let dist = d2(&d.pos, &pos);
            let to = norm2(d.pos[0] - pos[0], d.pos[1] - pos[1]);
            let radial = ((dist - 50.0) / 15.0).clamp(-1.0, 1.0);
            mv = self.strafe_dir(o, &pos, to, radial, false);
        } else if let Some(d) = drone_target {
            // Shoot back but carry on with our own business, never closing in on the drone.
            let (aim, fire, reload) = (a.aim, a.fire, a.reload);
            mv = self.quiet(o, &mut a, &pos, hp_frac, must_rotate, out_next, zone_goal, &threats, &mut label);
            (a.aim, a.fire, a.reload) = (aim, fire, reload);
            label = "drone";
            let to = norm2(d.pos[0] - pos[0], d.pos[1] - pos[1]);
            let toward = mv[0] * to[0] + mv[1] * to[1];
            if toward > 0.0 && !must_rotate {
                mv = [mv[0] - to[0] * toward, mv[1] - to[1] * toward];
            }
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
            if p.seek_idle > 0.0 && !ffa_on && enemies.is_empty() {
                if label == "idle" {
                    let since = *self.idle_since.get_or_insert(o.t);
                    let go_now = self.seeking || (o.t - since >= 3.0 && hp_frac >= 0.7);
                    if go_now && !self.healing && !must_rotate {
                        // Nearest enemy we believe in, else the nearest remembered one.
                        let near = |a: &[f32; 2], b: &[f32; 2]| d2(a, &pos).partial_cmp(&d2(b, &pos)).unwrap_or(std::cmp::Ordering::Equal);
                        let goal = self
                            .beliefs
                            .iter()
                            .filter(|b| b.conf > 0.1)
                            .map(|b| b.pos)
                            .min_by(near)
                            .or_else(|| self.known.values().map(|k| [k.pos[0], k.pos[1]]).min_by(near))
                            .filter(|g| d2(g, &z.center) < z.radius - 10.0);
                        if let Some(g) = goal {
                            label = "seek";
                            mv = self.go(&pos, g);
                            a.aim = [yaw_to(&pos, &g), 0.0];
                        }
                    }
                } else {
                    self.idle_since = None;
                }
            }
            let mut sought: Option<u32> = None;
            let seek_ready = o.t >= p.seek_t || !empty_slot;
            if p.seek > 0.0 && label == "idle" && seek_ready && !late && hp_frac > p.seek_hp && !self.healing && !must_rotate {
                let inz = |q: &[f32]| d2(q, &z.center) < z.radius - 10.0 && (o.t < z.shrink_start - 20.0 || d2(q, &z.next_center) < z.next_radius);
                // Nearest of: enemies in view, those seen lately, and those believed to be about.
                let mut best: Option<(f32, [f32; 2], Option<u32>)> = None;
                for m in &enemies {
                    let d = d2(&m.pos, &pos);
                    if d < p.seek && inz(&m.pos) && best.map_or(true, |b| d < b.0) {
                        best = Some((d, [m.pos[0], m.pos[1]], Some(m.id)));
                    }
                }
                if best.is_none() {
                    for (id, k) in self.known.iter() {
                        let d = d2(&k.pos, &pos) + 5.0 * (o.t - k.seen_t);
                        if d < p.seek && inz(&k.pos) && best.map_or(true, |b| d < b.0) {
                            best = Some((d, [k.pos[0], k.pos[1]], Some(*id)));
                        }
                    }
                    for b in self.beliefs.iter().filter(|b| b.conf >= 0.4 && !b.listed) {
                        let d = d2(&b.pos, &pos) + b.r * 0.5 + 20.0;
                        if d < p.seek && inz(&b.pos) && best.map_or(true, |x| d < x.0) {
                            best = Some((d, b.pos, b.id));
                        }
                    }
                }
                if let Some((_, g, id)) = best {
                    if d2(&g, &pos) > 8.0 {
                        label = "seek";
                        mv = self.go(&pos, g);
                        sought = id;
                        if enemies.is_empty() {
                            a.aim = [yaw_to(&pos, &g), 0.0];
                        }
                    }
                }
            }
            self.seek_id = sought;
            let armed = p.hunt_any > 0.0 || me.weapons.iter().flatten().any(|w| w.kind == "laser");
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

        // A held laser charge keeps its aim whatever the mode did with the torso.
        if let Some(la) = laser_aim {
            a.aim = la;
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
        if let (Some(src), false) = (shooter, ducked || label == "zone" || label == "ring" || label == "spot" || label == "iso" || label == "fight" || label == "rush" || label == "climb" || (label == "hold" && p.hold_fight <= 0.0) || label == "drone" || label == "reload" || label == "stand" || label == "peek" || label == "hide" || label == "repeek" || label == "home" || label == "cbreak" || label.starts_with("ffa")) {
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

        // Sidestep an incoming shot.
        if p.dodge > 0.0 && label != "zone" && label != "climb" && o.t >= self.unstick_until && self.dodge_check(o, target, mv, &threats) {
            // `dodge_tight` keeps part of the planned move (and so the strafe while firing) under the step.
            mv = if p.dodge_tight > 0.0 && !self.ffa_on() {
                norm2(self.dodge_dir[0] * 2.0 + mv[0], self.dodge_dir[1] * 2.0 + mv[1])
            } else {
                self.dodge_dir
            };
            boost = me.energy > 3.0;
        }

        // --- stuck: wanting to move but not moving means a ledge or a wall; hop and jet over it
        let wants = mv[0].abs() + mv[1].abs() > 0.1;
        if wants && d2(&raw, &self.last_pos) < 0.15 {
            self.stuck += 1;
        } else {
            self.stuck = 0;
        }
        // Progress watchdog: over the last 2 s we meant to get somewhere (following a path, or a move that
        // adds up to real distance) yet moved less than 2 m.
        self.trail.push_back((o.t, [raw[0], raw[1]], self.nav_used && wants, mv));
        while self.trail.front().map_or(false, |e| o.t - e.0 > 2.05) {
            self.trail.pop_front();
        }
        let mut no_progress = false;
        if let Some(first) = self.trail.front().filter(|e| o.t - e.0 >= 1.95) {
            let n = self.trail.len() as f32;
            let nav_frac = self.trail.iter().filter(|e| e.2).count() as f32 / n;
            let intent = self.trail.iter().fold([0.0f32; 2], |s, e| [s[0] + e.3[0], s[1] + e.3[1]]);
            let intent = (intent[0] * intent[0] + intent[1] * intent[1]).sqrt() * walk_speed * 0.1;
            no_progress = wants && d2(&raw, &first.1) < 2.0 && (nav_frac >= 0.8 || intent >= 6.0);
        }
        if self.stuck >= 5 || no_progress {
            self.unstick_until = o.t + 0.8;
            self.strafe = -self.strafe;
            self.wander = None;
            self.stuck = 0;
            self.trail.clear();
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
        // Charging halves our speed: never charge outside the zone.
        if p.laser_fix > 0.0 && !me.in_zone {
            for (i, w) in me.weapons.iter().enumerate().take(2) {
                if w.as_ref().map_or(false, |w| w.kind == "laser") {
                    a.fire[i] = false;
                }
            }
        }
        // Cornered (`cb_corner`): fight where we are, never boosting toward an enemy close by.
        if cb_cornered && boost && label != "zone" && o.t >= self.unstick_until {
            let l = a.mv[0].hypot(a.mv[1]);
            if l > 1e-3 {
                let u = [a.mv[0] / l, a.mv[1] / l];
                if cb_all.iter().any(|f| d2(f, &pos) < 80.0 && {
                    let t = norm2(f[0] - pos[0], f[1] - pos[1]);
                    u[0] * t[0] + u[1] * t[1] > 0.5
                }) {
                    boost = false;
                }
            }
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

    /// Loop around the nearest suitable building inside the safe area, sweeping back and forth across the side
    /// away from the enemies we know of. Always returns a move.
    fn ring_move(&mut self, o: &Obs, pos: &[f32; 3]) -> [f32; 2] {
        if self.ring_spots.is_none() {
            let spots = self.world.as_ref().map_or(Vec::new(), |w| {
                w.prisms
                    .iter()
                    .filter(|p| matches!(p.kind, PrismKind::Building))
                    .filter_map(|p| {
                        let (sx, sy) = (p.max[0] - p.min[0], p.max[1] - p.min[1]);
                        let c = [(p.min[0] + p.max[0]) * 0.5, (p.min[1] + p.max[1]) * 0.5];
                        let r = 0.5 * sx.hypot(sy) + 7.0;
                        let tall = p.top - w.ground(c[0], c[1]) >= 9.0;
                        (sx * sy >= 300.0 && sx.min(sy) >= 10.0 && tall && r <= 45.0).then_some((c, r))
                    })
                    .collect()
            });
            self.ring_spots = Some(spots);
        }
        let z = &o.zone;
        let soon = o.t >= z.shrink_start - 15.0;
        let valid = |c: &[f32; 2], r: f32| d2(c, &z.center) + r < z.radius - 5.0 && (!soon || d2(c, &z.next_center) + r * 0.5 < z.next_radius);
        if self.ring_b.map_or(true, |(c, r)| !valid(&c, r)) {
            self.ring_b = self
                .ring_spots
                .as_ref()
                .and_then(|v| v.iter().filter(|(c, r)| valid(c, *r) && d2(c, pos) < 250.0).min_by(|x, y| (d2(&x.0, pos) - x.1).partial_cmp(&(d2(&y.0, pos) - y.1)).unwrap_or(std::cmp::Ordering::Equal)))
                .copied();
        }
        let Some((c, r)) = self.ring_b else {
            return self.idle_move(o, pos);
        };
        // Where the enemies are: the centre of those seen in the last 6 s.
        let foes: Vec<[f32; 3]> = self.known.values().filter(|k| o.t - k.seen_t < 6.0).map(|k| k.pos).collect();
        let th = (pos[1] - c[1]).atan2(pos[0] - c[0]);
        if !foes.is_empty() {
            let n = foes.len() as f32;
            let e = foes.iter().fold([0.0f32; 2], |s, f| [s[0] + f[0] / n, s[1] + f[1] / n]);
            let want = (c[1] - e[1]).atan2(c[0] - e[0]);
            let diff = wrap(want - th);
            if diff.abs() > 0.7 {
                self.ring_dir = diff.signum();
            }
        }
        // A point a little further round, pushed out until it is walkable; turn back if the way is shut.
        for _ in 0..2 {
            let a = th + self.ring_dir * 0.45;
            for extra in [0.0f32, 4.0, 8.0, 14.0, 20.0] {
                let g = [c[0] + (r + extra) * a.cos(), c[1] + (r + extra) * a.sin()];
                if self.walkable(g[0], g[1]) {
                    let mv = self.go(pos, g);
                    if mv != [0.0, 0.0] {
                        return mv;
                    }
                }
            }
            self.ring_dir = -self.ring_dir;
        }
        self.idle_move(o, pos)
    }

    /// Sheltered-ground movement: where to go, whether to boost, and the label ("iso" for a position that only
    /// one enemy can see, "spot" for a sheltered spot). Sets the aim when nothing is being shot at.
    fn spot_move(&mut self, o: &Obs, pos: &[f32; 3], a: &mut Action, aiming: bool, threats: &[[f32; 3]], killing: bool) -> ([f32; 2], bool, &'static str) {
        let (Some(st), Some(nav)) = (self.sight.clone(), self.nav.clone()) else {
            return (self.idle_move(o, pos), false, "spot");
        };
        let me = &o.me;
        let p = self.params.clone();
        let here = [pos[0], pos[1]];
        let walk = me.stat("walk_speed", 11.7);
        // Enemies seen in the last 6 s, carried along their last velocity for up to 2 s.
        let foes: Vec<Foe> = self
            .known
            .values()
            .filter(|k| o.t - k.seen_t < 6.0 && d2(&k.pos, pos) < 260.0)
            .map(|k| {
                let dt = (o.t - k.seen_t).min(2.0);
                Foe { pos: [k.pos[0] + k.vel[0] * dt, k.pos[1] + k.vel[1] * dt], hp: k.hp_frac }
            })
            .collect();
        // Enemies with a line to us right now.
        let chest = [pos[0], pos[1], pos[2] + 5.0];
        let seeing = self.known.values().filter(|k| o.t - k.seen_t < 1.0 && d2(&k.pos, pos) < 200.0 && self.clear([k.pos[0], k.pos[1], k.pos[2] + 8.0], chest)).count();
        if seeing >= 2 {
            self.spot_seen_since.get_or_insert(o.t);
        } else {
            self.spot_seen_since = None;
        }
        let z = &o.zone;
        let zone = Zone { center: z.center, radius: z.radius, next_center: z.next_center, next_radius: z.next_radius, soon: o.t >= z.shrink_start - 15.0 };
        self.spot_ban.retain(|_, t| o.t < *t);

        // Level 2: a position only one enemy can see, re-chosen every half second (keeping the last one unless
        // another is clearly better).
        if p.spot >= 2.0 {
            if o.t - self.iso_t >= 0.5 {
                self.iso_t = o.t;
                let keep = self.iso_pick.map(|q| q.pos);
                self.iso_pick = st.pick_isolating(&nav, here, &foes, &zone, walk, keep);
            }
            if let Some(q) = self.iso_pick.filter(|q| zone.holds(q.pos, 4.0)) {
                let d = d2(&q.pos, pos);
                let mv = if d > 3.0 { self.go(pos, q.pos) } else { self.jitter(o, pos, q.pos) };
                if !aiming {
                    if let Some(f) = q.foe {
                        a.aim = [yaw_to(pos, &f), 0.0];
                    }
                }
                return (mv, d > 3.0 && seeing >= 2 && me.energy > 10.0, "iso");
            }
        }

        // Level 1: hold a sheltered spot; move on when the zone closes in, two or more enemies keep a line on us
        // for 1.5 s, or our health drops fast with no kill coming.
        let lost = self.hp_hist.front().map_or(0.0, |e| e.1 - me.hp);
        let at = self.spot_pick.map_or(false, |q| d2(&q.pos, pos) < 5.0);
        let pinned = at && self.spot_seen_since.map_or(false, |t0| o.t - t0 > 1.5);
        let bleeding = lost > me.max_hp * 0.2 && !killing;
        let stale = self.spot_pick.map_or(true, |q| !zone.holds(q.pos, 4.0));
        if stale || pinned || bleeding {
            let mut next = None;
            if let Some(i) = self.spot_pick.and_then(|q| q.spot) {
                if !stale {
                    self.spot_ban.insert(i, o.t + 15.0);
                    // The partner spot first, if it is inside the safe area and fewer enemies can see it.
                    if let Some(j) = st.spots[i].fallback.filter(|j| !self.spot_ban.contains_key(j)) {
                        let f = &st.spots[j];
                        if zone.holds(f.pos, 8.0) && st.seen_by(f.pos, &foes) < seeing.max(1) {
                            next = Some(Pick { pos: f.pos, yaw: f.lane_yaw, spot: Some(j), foe: None, score: 0.0 });
                        }
                    }
                }
            }
            if next.is_none() {
                let banned: Vec<usize> = self.spot_ban.keys().copied().collect();
                next = st.pick_spot(&nav, here, &foes, &zone, &banned, walk);
            }
            self.spot_pick = next;
            self.spot_seen_since = None;
            self.hp_hist.clear();
        }
        let Some(q) = self.spot_pick else {
            return (self.idle_move(o, pos), false, "spot");
        };
        let d = d2(&q.pos, pos);
        let mv = if d > 3.0 { self.go(pos, q.pos) } else { self.jitter(o, pos, q.pos) };
        if !aiming {
            match threats.first() {
                Some(th) if d > 3.0 => a.aim = [yaw_to(pos, th), 0.0],
                _ => a.aim = [q.yaw, 0.0],
            }
        }
        (mv, d > 3.0 && seeing >= 2 && me.energy > 10.0, "spot")
    }

    /// Aim the gun at a crate and fire once lined up, if the line to it is clear. Returns whether it was.
    fn aim_crate(&self, o: &Obs, a: &mut Action, pos: &[f32; 3], c: &[f32; 3]) -> bool {
        let me = &o.me;
        let muzzle = [pos[0], pos[1], pos[2] + me.stat("muzzle_height", 7.0)];
        let cc = [c[0], c[1], c[2] + 1.2];
        if !self.clear(muzzle, cc) {
            return false;
        }
        let want = [yaw_to(&muzzle, &cc), (cc[2] - muzzle[2]).atan2(d2(&cc, &muzzle).max(0.1))];
        a.aim = want;
        let dd = d3(&cc, &muzzle).max(1.0);
        let lined = wrap(want[0] - me.aim[0]).abs() < (1.0 / dd).atan() && (want[1] - me.aim[1]).abs() < (0.9 / dd).atan();
        for (i, w) in me.weapons.iter().enumerate().take(2) {
            if w.as_ref().map_or(false, |w| w.kind == "gun") {
                a.fire[i] = lined;
                a.reload = false;
            }
        }
        true
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
        let consider = |b: &Brain, g: &mut Option<([f32; 2], f32)>, q: &[f32; 3], value: f32, reach: f32, ok_ground: bool| {
            let d = d2(q, pos);
            if d < reach && ok_ground && in_zone(q) {
                let s = value - d - b.crowd_cost(o.t, pos, [q[0], q[1]]);
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
        let reach = if p.farm_stop > 0.0 && o.t >= p.farm_stop { 0.0 } else { reach };
        let picks: Vec<(String, [f32; 3])> = self.picks_mem.values().cloned().collect();
        for (kind, q) in &picks {
            let r = weapon_rank(kind, &p);
            if r > 0 && !held.contains(&kind.as_str()) && (empty_slot || r > worst) {
                let v = if empty_slot && early { 120.0 + 30.0 * r as f32 } else { 60.0 + 30.0 * r as f32 };
                consider(self, &mut goal, q, v, if reach > 0.0 { reach.max(140.0) } else { 0.0 }, on_ground(self, q));
            }
        }
        let cubes: Vec<[f32; 3]> = self.cubes_mem.values().copied().collect();
        for q in &cubes {
            consider(self, &mut goal, q, 100.0, reach, on_ground(self, q));
        }
        // Supply drop: full heal and cubes, if nobody we know of is near it.
        for dr in &o.drops {
            let d = d2(&dr.pos, pos);
            let contested = threats.iter().any(|t| d2(t, &dr.pos) < p.drop_enemy_clear);
            let arrive = d / me.stat("walk_speed", 11.7);
            if d < p.drop_range && !contested && arrive + 4.0 > dr.lands_in - 3.0 && in_zone(&dr.pos) {
                let v = 150.0 + (1.0 - hp_frac) * 100.0 - self.crowd_cost(o.t, pos, [dr.pos[0], dr.pos[1]]);
                if goal.map_or(true, |g| v - d * 0.5 > g.1) {
                    goal = Some(([dr.pos[0], dr.pos[1]], v - d * 0.5));
                }
            }
        }
        let crates: Vec<[f32; 3]> = self.crates_mem.values().copied().filter(|q| on_ground(self, q) && in_zone(q)).collect();
        // Under fire from a mech we cannot see: no detours or crate shooting, face where it hit us from.
        let hurt = p.threat_face > 0.0 && o.t - self.last_mech_hurt_t <= 3.0;
        let crate_target = crates
            .iter()
            .filter(|c| d3(c, pos) < reach && hp_frac > p.farm_hp && !self.healing && !hurt)
            .min_by(|x, y| d3(x, pos).partial_cmp(&d3(y, pos)).unwrap_or(std::cmp::Ordering::Equal));
        let mut aimed = false;
        if let Some(c) = crate_target {
            let shot = self.aim_crate(o, a, pos, c);
            if !shot || d2(c, pos) > 90.0 {
                consider(self, &mut goal, c, 70.0, reach, true);
            }
            if shot {
                aimed = true;
                if d2(c, pos) > 35.0 {
                    consider(self, &mut goal, c, 70.0, reach, true);
                }
            }
        }
        let g = match goal {
            Some((g, _)) if !must_rotate && !hurt => Some(g),
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
        let fresh = self.threat.filter(|(_, t)| p.threat_face > 0.0 && o.t - t < 2.0).map(|(q, _)| q);
        // With `map_preaim` and no enemy in view, the surest believed enemy nearby comes before the last known ones.
        let pre = if fresh.is_none() { self.preaim(pos).map(|q| [q[0], q[1], pos[2]]) } else { None };
        if !aimed {
            if let Some(th) = fresh.as_ref().or(pre.as_ref()).or(threats.first()) {
                a.aim = [yaw_to(pos, th), 0.0];
            } else {
                self.scan(a);
            }
        }
        mv
    }
}

#[path = "tactics.rs"]
mod tactics;
#[path = "ffa.rs"]
pub mod ffa;
#[path = "pincer.rs"]
mod pincer;

impl Drop for Brain {
    fn drop(&mut self) {
        if let Some(tr) = &self.pred_trace {
            tr.report(self.ffa_slot, self.ffa_seed);
        }
        if self.trace_ffa {
            let t = self.ffa_ticks;
            eprintln!(
                "FFA sum slot {} seed {} latched {} at {:.1} practice {} fight {} third {} avoid {} hide {} late {} slowest_ms {:.1}",
                self.ffa_slot, self.ffa_seed, self.detector.latched as u8, self.detector.latch_t, t[0], t[1], t[2], t[3], t[4], t[5], self.ffa_slowest
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predict;

    /// Defaults with the optional tactics off: the checks below are about the behaviour underneath them.
    fn base() -> Params {
        Params {
            dodge: 0.0,
            crowd_retreat: 0.0,
            fight_pos: 0.0,
            late_hold: 0.0,
            engage_filter: 0.0,
            laser_fix: 0.0,
            fight_cubes: 0.0,
            ext_lose: 0.0,
            ext_target: 0.0,
            ext_fpos: 0.0,
            ext_late: 0.0,
            ext_finish: 0.0,
            pred: 0.0,
            pincer: 0.0,
            ..Params::default()
        }
    }

    /// `base` with the enemy picture used in practice play.
    fn map_on() -> Params {
        Params { map_preaim: 1.0, map_expose: 15.0, map_crowd: 40.0, map_reinf: 15.0, ..base() }
    }

    fn brain() -> Brain {
        Brain::with_params(&Hello::default(), None, None, base())
    }

    fn obs(other: &str, events: &str) -> Obs {
        let s = format!(
            r#"{{"tick": 100, "t": 5.0,
              "self": {{"pos": [0,0,0], "aim": [0,0], "hp": 1400, "max_hp": 1400, "energy": 100, "grounded": true, "in_zone": true,
                        "weapons": [{{"kind": "gun", "ammo": 12}}, {{"kind": "laser"}}]}},
              "mechs": [{other}], "events": [{events}],
              "zone": {{"center": [0,0], "radius": 500, "next_center": [0,0], "next_radius": 400, "shrink_start": 100}}}}"#
        );
        serde_json::from_str(&s).unwrap()
    }

    const DRONE: &str = r#"{"id": 104, "kind": "npc", "pos": [40, 0, 0], "hp": 350, "max_hp": 350}"#;
    const MECH: &str = r#"{"id": 3, "kind": "mech", "pos": [40, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;

    #[test]
    fn drones_are_aimed_at_by_their_own_size() {
        let b = brain();
        let o = obs(DRONE, "");
        let gun = o.me.weapons[0].clone().unwrap();
        let drone = b.solve(&o, &gun, &o.mechs[0]);
        let want = (1.75f32 - 7.0).atan2(40.0);
        assert!((drone.aim[1] - want).abs() < 0.01, "pitch {} want {}", drone.aim[1], want);
        let o = obs(MECH, "");
        let mech = b.solve(&o, &gun, &o.mechs[0]);
        assert!(drone.tol[0] < mech.tol[0] && drone.tol[1] < mech.tol[1]);
    }

    #[test]
    fn laser_is_kept_for_mechs() {
        let hurt = r#"{"type": "damaged", "amount": 25, "weapon": "npc", "by": 104, "bearing": 0}"#;
        let a = brain().decide(&obs(DRONE, hurt), [0.0; 3]);
        assert!(!a.fire[1], "laser charged at a drone");
        assert!(a.aim[0].abs() < 0.2, "not aimed at the drone");
        let a = brain().decide(&obs(MECH, ""), [0.0; 3]);
        assert!(a.fire[1], "laser not charged at a mech");
    }

    #[test]
    fn late_fights_are_fought_moving() {
        let mut o = obs(MECH, "");
        o.t = 200.0;
        let mut b = with(Params { ring: 0.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "fight");
        let mut b = with(Params { hold_fight: 0.0, ring: 0.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "hold");
    }

    #[test]
    fn unseen_attacker_is_faced_instead_of_looting() {
        let hit = r#"{"type": "damaged", "amount": 40, "weapon": "gun", "by": 5, "bearing": 1.5708}"#;
        let mut o = obs("", hit);
        o.mechs.clear();
        o.cubes = vec![crate::proto::Item { id: 5000, pos: [20.0, -30.0, 0.0], ..Default::default() }];
        let mut b = brain();
        let a = b.decide(&o, [0.0; 3]);
        assert!(b.label != "loot", "looting while under fire");
        assert!((a.aim[0] - 1.5708).abs() < 0.05, "aim {:?}", a.aim);
        let mut b = Brain::with_params(&Hello::default(), None, None, Params { threat_face: 0.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "loot");
    }

    #[test]
    fn lightning_holder_is_kept_at_a_distance_while_holding() {
        let arc = r#"{"id": 3, "kind": "mech", "pos": [40, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun", "lightning"]}"#;
        let mut o = obs(arc, "");
        o.t = 200.0;
        let params = Params { hold_fight: 0.0, ring: 0.0, ..base() };
        let mut b = Brain::with_params(&Hello::default(), None, None, params.clone());
        let a = b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "hold");
        assert!(a.mv[0] < -0.5 && a.boost, "mv {:?}", a.mv);
        let mut b = Brain::with_params(&Hello::default(), None, None, Params { ltg_space: 0.0, ..params });
        let a = b.decide(&o, [0.0; 3]);
        assert!(a.mv[0] > -0.5, "mv {:?}", a.mv);
    }

    #[test]
    fn drones_are_left_alone_unless_they_shoot() {
        let a = with(Params { drone_threat: 0.0, ..base() }).decide(&obs(DRONE, ""), [0.0; 3]);
        assert!(!a.fire[0] && !a.fire[1]);
    }

    fn with(params: Params) -> Brain {
        Brain::with_params(&Hello::default(), None, None, params)
    }

    #[test]
    fn held_laser_is_not_let_go_unaimed() {
        // Full charge held almost to the limit, torso pointed away from the mech.
        let mut o = obs(MECH, "");
        o.me.aim = [1.5, 0.0];
        o.me.weapons[1].as_mut().unwrap().charge = 0.75;
        let mut b = with(Params { laser_hold: 0.0, ..base() });
        b.laser_full_t = Some(3.0);
        assert!(!b.decide(&o, [0.0; 3]).fire[1], "default lets a charge go near the limit");
        let mut b = with(Params { laser_hold: 1.0, ..base() });
        b.laser_full_t = Some(3.0);
        let a = b.decide(&o, [0.0; 3]);
        assert!(a.fire[1], "unaimed charge let go");
        // Lined up: let go.
        o.me.aim = b.solve(&o, o.me.weapons[1].as_ref().unwrap(), &o.mechs[0]).aim;
        let mut b = with(Params { laser_hold: 1.0, ..base() });
        assert!(!b.decide(&o, [0.0; 3]).fire[1], "lined-up charge held");
    }

    #[test]
    fn held_laser_follows_a_mech_out_of_view() {
        let mut o = obs(MECH, "");
        let mut b = with(Params { laser_hold: 1.0, ..base() });
        b.decide(&o, [0.0; 3]);
        o.mechs.clear();
        o.t += 1.5;
        o.me.aim = [2.0, 0.0];
        o.me.weapons[1].as_mut().unwrap().charge = 0.5;
        let a = b.decide(&o, [0.0; 3]);
        assert!(a.fire[1], "charge dumped after losing the mech");
        assert!(a.aim[0].abs() < 0.2, "aim {:?}", a.aim);
        let mut b = with(Params { laser_hold: 0.0, ..base() });
        let mut o2 = obs(MECH, "");
        b.decide(&o2, [0.0; 3]);
        o2.mechs.clear();
        o2.t += 1.5;
        o2.me.weapons[1].as_mut().unwrap().charge = 0.5;
        assert!(!b.decide(&o2, [0.0; 3]).fire[1]);
    }

    #[test]
    fn drones_in_attack_range_are_shot() {
        let a = with(Params { drone_threat: 1.0, ..base() }).decide(&obs(DRONE, ""), [0.0; 3]);
        assert!(a.fire[0] && !a.fire[1], "fire {:?}", a.fire);
        let far = r#"{"id": 104, "kind": "npc", "pos": [80, 0, 0], "hp": 350, "max_hp": 350}"#;
        let a = with(Params { drone_threat: 1.0, ..base() }).decide(&obs(far, ""), [0.0; 3]);
        assert!(!a.fire[0]);
    }

    #[test]
    fn an_enemy_aimed_at_us_is_preferred() {
        let two = r#"{"id": 3, "kind": "mech", "pos": [40, 0, 0], "aim": [1.0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]},
                     {"id": 4, "kind": "mech", "pos": [-60, 0, 0], "aim": [0.05, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let o = obs(two, "");
        let mut b = with(Params { focus_aimer: 0.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.focus, Some(3));
        let mut b = with(Params { focus_aimer: 40.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.focus, Some(4));
    }

    #[test]
    fn crates_are_shot_on_the_way_to_a_weapon() {
        let mut o = obs("", "");
        o.mechs.clear();
        o.me.weapons = vec![Some(Weapon { kind: "gun".into(), ammo: 12.0, ..Default::default() }), None];
        o.pickups = vec![crate::proto::Item { id: 8000, kind: "laser".into(), pos: [100.0, 0.0, 0.0], ..Default::default() }];
        o.crates = vec![crate::proto::Item { id: 1000, pos: [0.0, 30.0, 0.0], hp: 300.0, ..Default::default() }];
        let mut b = with(Params { arm_crates: 1.0, ..base() });
        let a = b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "arm");
        assert!((a.aim[0] - 1.5708).abs() < 0.05, "aim {:?}", a.aim);
        let mut b = with(Params { arm_crates: 0.0, ..base() });
        let a = b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "arm");
        assert!(a.aim[0].abs() < 0.05, "aim {:?}", a.aim);
    }

    #[test]
    fn no_long_rush_into_a_ready_laser() {
        let lz = r#"{"id": 3, "kind": "mech", "pos": [70, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun", "laser"]}"#;
        let mut o = obs(lz, "");
        o.me.weapons[1] = Some(Weapon { kind: "lightning".into(), ..Default::default() });
        let mut b = with(Params { rush_guard: 0.0, counter: 0.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "rush");
        let mut b = with(Params { rush_guard: 1.0, counter: 0.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_ne!(b.label, "rush");
        // Its laser just fired: it is cooling down.
        o.beams = vec![crate::proto::Beam { by: 3, kind: "laser".into(), path: vec![], age: 0.1 }];
        let mut b = with(Params { rush_guard: 1.0, counter: 0.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "rush");
        // Empty magazine from beyond 60 m.
        o.me.weapons[0].as_mut().unwrap().ammo = 0.0;
        let mut b = with(Params { rush_guard: 1.0, counter: 0.0, ..base() });
        b.decide(&o, [0.0; 3]);
        assert_ne!(b.label, "rush");
    }

    #[test]
    fn late_game_keeps_moving() {
        let mut o = obs(MECH, "");
        o.t = 200.0;
        let mut b = with(Params { ring: 1.0, ..base() });
        let a = b.decide(&o, [0.0; 3]);
        assert_eq!(b.label, "ring");
        assert!(a.mv[0].abs() + a.mv[1].abs() > 0.3, "mv {:?}", a.mv);
        assert!(a.fire[0] || a.fire[1], "not shooting while looping");
        // Three enemies close by switch it on early.
        let three = r#"{"id": 3, "kind": "mech", "pos": [40, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]},
                       {"id": 4, "kind": "mech", "pos": [0, 60, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]},
                       {"id": 5, "kind": "mech", "pos": [-80, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let mut b = with(Params { ring: 1.0, ..base() });
        b.decide(&obs(three, ""), [0.0; 3]);
        assert_eq!(b.label, "ring");
        let mut b = with(Params { ring: 1.0, ..base() });
        b.decide(&obs(MECH, ""), [0.0; 3]);
        assert_ne!(b.label, "ring");
    }

    /// On the real map (if the kit is next to the repo or `MAP_DIR` is set): the loop goes round a building
    /// and keeps moving.
    #[test]
    fn late_game_loops_round_a_building() {
        let dir = std::env::var("MAP_DIR").unwrap_or_else(|_| "../mechfight-kit/arena".into());
        let Ok(w) = World::load(std::path::Path::new(&dir)) else { return };
        let nav = Nav::build(&w, 2.5, 9.0, 1.2, 50.0);
        let w = Arc::new(w);
        let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav), Params { ring: 1.0, ..base() });
        let mut o = obs(r#"{"id": 3, "kind": "mech", "pos": [140, 0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#, "");
        o.mechs[0].pos[2] = w.ground(140.0, 0.0);
        o.t = 200.0;
        let start = b.nav.as_ref().unwrap().center(b.nav.as_ref().unwrap().nearest_walkable(0.0, 0.0).unwrap());
        o.me.pos = [start[0], start[1], w.ground(start[0], start[1])];
        let first = o.me.pos;
        let mut still = 0;
        let mut slowest = 0.0f32;
        for _ in 0..300 {
            let t0 = std::time::Instant::now();
            let a = b.decide(&o, o.me.pos);
            slowest = slowest.max(t0.elapsed().as_secs_f32());
            assert_eq!(b.label, "ring");
            let l = a.mv[0].hypot(a.mv[1]);
            if l < 0.3 {
                still += 1;
            }
            let (nx, ny) = (o.me.pos[0] + a.mv[0] * 11.7 * 0.1, o.me.pos[1] + a.mv[1] * 11.7 * 0.1);
            if b.walkable(nx, ny) {
                o.me.pos = [nx, ny, w.ground(nx, ny)];
            }
            o.t += 0.1;
            o.tick += 2;
        }
        assert!(b.ring_b.is_some(), "no building to loop around");
        assert!(still < 10, "stood still {still} times");
        assert!(d2(&first, &o.me.pos) > 1.0 || still == 0);
        assert!(slowest < 0.02, "slowest decision {slowest} s");
    }

    #[test]
    fn counter_puncher_leaves_distant_mechs_alone() {
        let away = r#"{"id": 3, "kind": "mech", "pos": [100, 0, 0], "aim": [0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let at_us = r#"{"id": 3, "kind": "mech", "pos": [100, 0, 0], "aim": [3.1, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let mut b = with(Params { counter: 1.0, ..base() });
        b.decide(&obs(away, ""), [0.0; 3]);
        assert_eq!(b.focus, None);
        let mut b = with(Params { counter: 1.0, ..base() });
        b.decide(&obs(at_us, ""), [0.0; 3]);
        assert_eq!(b.focus, Some(3));
        let mut b = with(Params { counter: 0.0, ..base() });
        b.decide(&obs(away, ""), [0.0; 3]);
        assert_eq!(b.focus, Some(3));
    }

    #[test]
    fn engage_filter_leaves_distant_mechs_alone() {
        let away = r#"{"id": 3, "kind": "mech", "pos": [100, 0, 0], "aim": [0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let weak = r#"{"id": 3, "kind": "mech", "pos": [100, 0, 0], "aim": [0, 0], "hp": 400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let mut b = with(Params { engage_filter: 1.0, counter: 0.0, ..base() });
        b.decide(&obs(away, ""), [0.0; 3]);
        assert_eq!(b.focus, None);
        let mut b = with(Params { engage_filter: 1.0, counter: 0.0, ..base() });
        b.decide(&obs(weak, ""), [0.0; 3]);
        assert_eq!(b.focus, Some(3));
    }

    #[test]
    fn incoming_bullet_is_sidestepped() {
        let bullet = r#"{"id": 280, "kind": "gun", "pos": [60, 0, 5], "vel": [-252, 0, 0]}"#;
        let s = format!(
            r#"{{"tick": 100, "t": 5.0,
              "self": {{"pos": [0,0,0], "aim": [0,0], "hp": 1400, "max_hp": 1400, "energy": 100, "grounded": true, "in_zone": true,
                        "weapons": [{{"kind": "gun", "ammo": 12}}, null]}},
              "mechs": [{MECH}], "projectiles": [{bullet}],
              "zone": {{"center": [0,0], "radius": 500, "next_center": [0,0], "next_radius": 400, "shrink_start": 100}}}}"#
        );
        let o: Obs = serde_json::from_str(&s).unwrap();
        let mut b = with(Params { dodge: 1.0, ..base() });
        let a = b.decide(&o, [0.0; 3]);
        assert!(a.boost && a.mv[0].abs() < 0.1 && a.mv[1].abs() > 0.9, "move {:?} boost {}", a.mv, a.boost);
    }

    #[test]
    fn laser_is_not_charged_outside_the_zone() {
        let mut o = obs(MECH, "");
        o.me.in_zone = false;
        let mut b = with(Params { laser_fix: 1.0, ..base() });
        let a = b.decide(&o, [0.0; 3]);
        assert!(!a.fire[1]);
    }

    #[test]
    fn survival_upgrades_come_before_damage() {
        let pos = |id: &str| UPGRADES.iter().position(|u| *u == id).unwrap();
        for s in ["plating", "armor", "nanites", "servos"] {
            for d in ["gun_damage", "laser_damage", "lightning_efficiency", "gun_rate"] {
                assert!(pos(s) < pos(d), "{s} after {d}");
            }
        }
    }

    fn belief(id: Option<u32>, pos: [f32; 2], conf: f32, listed: bool) -> Belief {
        Belief { id, pos, r: 10.0, conf, listed, ..Belief::default() }
    }

    #[test]
    fn with_no_enemy_in_view_we_face_the_surest_believed_one_nearby() {
        let o = obs("", "");
        let picture = vec![belief(Some(7), [0.0, 100.0], 0.8, false), belief(None, [100.0, 0.0], 0.3, false), belief(Some(9), [-180.0, 0.0], 0.9, false)];
        let mut b = with(map_on());
        b.beliefs = picture.clone();
        let a = b.decide(&o, [0.0; 3]);
        assert!((a.aim[0] - std::f32::consts::FRAC_PI_2).abs() < 0.05, "aim {:?}", a.aim);
        // Off: the usual sweep.
        let mut b = with(Params { map_preaim: 0.0, ..map_on() });
        b.beliefs = picture.clone();
        let a = b.decide(&o, [0.0; 3]);
        assert!((a.aim[0] - std::f32::consts::FRAC_PI_2).abs() > 0.5, "aim {:?}", a.aim);
        // Only out-of-sight ones within 150 m, and only with none in view.
        let mut far = with(map_on());
        far.beliefs = vec![belief(Some(9), [-180.0, 0.0], 0.9, false)];
        assert_eq!(far.preaim(&[0.0; 3]), None);
        let mut seen = with(map_on());
        seen.beliefs = vec![belief(Some(7), [0.0, 100.0], 0.8, false), belief(Some(3), [50.0, 0.0], 1.0, true)];
        assert_eq!(seen.preaim(&[0.0; 3]), None);
    }

    #[test]
    fn fight_positions_seen_by_believed_enemies_score_lower() {
        let mut b = with(map_on());
        b.beliefs = vec![
            belief(Some(7), [0.0, 100.0], 0.6, false),
            belief(None, [100.0, 0.0], 0.3, false),
            belief(Some(3), [-50.0, 0.0], 0.9, false),
            belief(Some(4), [0.0, -60.0], 1.0, true),
            belief(Some(5), [0.0, 200.0], 0.9, false),
            belief(Some(6), [80.0, 80.0], 0.9, false),
        ];
        // 7 counts; 6 is already counted; 3 is the target; too unsure, listed or too far do not.
        assert_eq!(b.hidden_watch_cost([0.0, 0.0], 0.0, 3, &[6]), 15.0);
        assert_eq!(b.hidden_watch_cost([0.0, 0.0], 0.0, 3, &[]), 30.0);
        b.params.map_expose = 0.0;
        assert_eq!(b.hidden_watch_cost([0.0, 0.0], 0.0, 3, &[]), 0.0);
    }

    #[test]
    fn ways_past_two_or_more_believed_enemies_cost_more() {
        let mut b = with(map_on());
        let two = vec![belief(Some(7), [100.0, 30.0], 0.7, false), belief(None, [150.0, -40.0], 0.5, false), belief(Some(8), [100.0, 150.0], 0.9, false)];
        b.beliefs = two.clone();
        assert_eq!(b.crowd_cost(5.0, &[0.0; 3], [200.0, 0.0]), 80.0);
        // One alone is not a crowd.
        b.beliefs = two[..1].to_vec();
        assert_eq!(b.crowd_cost(5.0, &[0.0; 3], [200.0, 0.0]), 0.0);
        // The target never counts.
        b.beliefs = two.clone();
        b.focus = Some(7);
        assert_eq!(b.crowd_cost(5.0, &[0.0; 3], [200.0, 0.0]), 0.0);
        // Nor an enemy close by that is hurting us.
        b.focus = None;
        b.beliefs = vec![belief(Some(7), [30.0, 0.0], 0.9, true), belief(None, [150.0, -40.0], 0.5, false)];
        b.attackers.insert(7, 4.0);
        assert_eq!(b.crowd_cost(5.0, &[0.0; 3], [200.0, 0.0]), 0.0);
        b.attackers.clear();
        assert_eq!(b.crowd_cost(5.0, &[0.0; 3], [200.0, 0.0]), 80.0);
        b.params.map_crowd = 0.0;
        assert_eq!(b.crowd_cost(5.0, &[0.0; 3], [200.0, 0.0]), 0.0);
    }

    #[test]
    fn of_two_like_targets_the_one_with_fewer_enemies_near_it_comes_first() {
        let two = r#"{"id": 3, "kind": "mech", "pos": [60, 0, 0], "aim": [3.1416, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]},
                     {"id": 5, "kind": "mech", "pos": [0, 62, 0], "aim": [-1.5708, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let o = obs(two, "");
        let picture = vec![
            belief(Some(3), [60.0, 0.0], 1.0, true),
            belief(Some(5), [0.0, 62.0], 1.0, true),
            belief(None, [110.0, 10.0], 0.7, false),
            belief(Some(8), [100.0, -30.0], 0.6, false),
        ];
        let mut b = with(map_on());
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.focus, Some(3));
        let mut b = with(map_on());
        b.beliefs = picture.clone();
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.focus, Some(5));
        let mut b = with(Params { map_reinf: 0.0, ..map_on() });
        b.beliefs = picture.clone();
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.focus, Some(3));
        // The same in the engage-filter score.
        let mut b = with(Params { map_reinf: 15.0, ..Params::default() });
        let plain = b.engage_score(&o, &o.mechs[0]);
        b.beliefs = picture;
        assert_eq!(b.engage_score(&o, &o.mechs[0]) - plain, 30.0);
        assert_eq!(b.reinforcements(&o.mechs[1]), 0.0);
    }

    fn mech(id: u32, pos: [f32; 2], vel: [f32; 2], aim: f32) -> String {
        format!(r#"{{"id": {id}, "kind": "mech", "pos": [{}, {}, 0], "vel": [{}, {}, 0], "aim": [{aim}, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}}"#, pos[0], pos[1], vel[0], vel[1])
    }

    fn pin_brain(on: bool) -> Brain {
        with(Params { pred: on as i32 as f32, pincer: on as i32 as f32, fight_pos: 1.0, ..base() })
    }

    #[test]
    fn enemies_are_expected_along_their_way() {
        // One walking north past us, not looking at us; one coming at us from the east.
        let others = format!("{},{}", mech(3, [-50.0, -40.0], [0.0, 10.0], 1.0), mech(4, [80.0, 0.0], [-10.0, 0.0], 3.14159));
        let o = obs(&others, "");
        let mut b = pin_brain(true);
        b.update_pincer(&o, &[0.0; 3], None);
        let t3 = b.pin.threats.iter().find(|t| t.id == Some(3)).unwrap();
        assert!((t3.pred[1] - -20.0).abs() < 0.5 && (t3.pred[0] - -50.0).abs() < 0.5, "{:?}", t3);
        // The one coming for us stops at its fighting distance.
        let t4 = b.pin.threats.iter().find(|t| t.id == Some(4)).unwrap();
        assert!((t4.pred[0] - 60.0).abs() < 0.5, "{:?}", t4);
        // Off: where they are now, and no picture at all.
        let mut b = pin_brain(false);
        b.update_pincer(&o, &[0.0; 3], None);
        assert!(b.pin.threats.is_empty() && !b.pin.pincered);
    }

    #[test]
    fn enemies_on_both_sides_are_a_pincer() {
        let mut b = pin_brain(true);
        // East and west.
        let o = obs(&format!("{},{}", mech(3, [60.0, 0.0], [0.0, 0.0], 3.14), mech(4, [-60.0, 5.0], [0.0, 0.0], 0.0)), "");
        b.update_pincer(&o, &[0.0; 3], o.mechs.first());
        assert!(b.pin.pincered && b.dbg_pincer.1 > 144.0, "{:?}", b.pin);
        assert!(b.dbg_pincer.0 && b.dbg_pincer.1 > 150.0);
        // Three round us without a target.
        let o3 = obs(&format!("{},{},{}", mech(3, [60.0, 0.0], [0.0, 0.0], 3.14), mech(4, [-30.0, 50.0], [0.0, 0.0], 0.0), mech(5, [-30.0, -50.0], [0.0, 0.0], 0.0)), "");
        b.update_pincer(&o3, &[0.0; 3], None);
        assert!(b.pin.pincered && b.dbg_pincer.1 > 180.0);
        // Both to the east: not.
        let o = obs(&format!("{},{}", mech(3, [60.0, 0.0], [0.0, 0.0], 3.14), mech(4, [60.0, 30.0], [0.0, 0.0], 3.14)), "");
        b.update_pincer(&o, &[0.0; 3], o.mechs.first());
        assert!(!b.pin.pincered, "{:?}", b.pin);
        // Too far away to count.
        let o = obs(&format!("{},{}", mech(3, [60.0, 0.0], [0.0, 0.0], 3.14), mech(4, [-200.0, 0.0], [0.0, 0.0], 0.0)), "");
        b.update_pincer(&o, &[0.0; 3], o.mechs.first());
        assert!(!b.pin.pincered && b.pin.threats.len() == 1);
        // A flanker: north-east now, going round to the north-west within two seconds.
        let o = obs(&format!("{},{}", mech(3, [60.0, 0.0], [0.0, 0.0], 3.14), mech(4, [15.0, 30.0], [-16.0, 0.0], 0.0)), "");
        b.update_pincer(&o, &[0.0; 3], o.mechs.first());
        assert!(b.pin.pincered && b.dbg_pincer.1 < 144.0, "{:?}", b.pin);
        // The same flanker without motion prediction is not one yet.
        let mut np = with(Params { pred: 0.0, pincer: 1.0, fight_pos: 1.0, ..base() });
        np.update_pincer(&o, &[0.0; 3], o.mechs.first());
        assert!(!np.pin.pincered, "{:?}", np.pin);
    }

    #[test]
    fn pincered_fight_position_puts_the_enemies_in_front() {
        // Target to the east, another enemy behind us to the north-west.
        let others = format!("{},{}", mech(3, [40.0, 0.0], [0.0, 0.0], 3.14), mech(4, [-15.0, 30.0], [0.0, 0.0], -1.1));
        let o = obs(&others, "");
        let worst_off = |q: [f32; 2]| predict::off(yaw_to(&q, &[-15.0, 30.0]), yaw_to(&q, &[40.0, 0.0]));
        let mut b = pin_brain(true);
        b.update_pincer(&o, &[0.0; 3], o.mechs.first());
        assert!(b.pin.pincered);
        let mv = b.fight_pos_move(&o, &o.mechs[0], 40.0, true);
        assert!(mv[0].hypot(mv[1]) > 0.9, "{:?}", mv);
        let q = [mv[0] * 24.0, mv[1] * 24.0];
        assert!(worst_off(q) < worst_off([0.0, 0.0]) - 0.3, "{:?} {} {}", mv, worst_off(q), worst_off([0.0, 0.0]));
        // The far point is kept for a moment.
        assert!(b.pin_goal.is_some());
        // Off: no far points and no such pull.
        let mut b = pin_brain(false);
        b.update_pincer(&o, &[0.0; 3], o.mechs.first());
        b.fight_pos_move(&o, &o.mechs[0], 40.0, true);
        assert!(b.pin_goal.is_none());
    }

    #[test]
    fn with_the_new_switches_off_the_picture_is_left_empty() {
        let others = format!("{},{}", mech(3, [40.0, 0.0], [0.0, 0.0], 3.14), mech(4, [-15.0, 30.0], [0.0, 0.0], -1.1));
        let o = obs(&others, "");
        let mut on = with(Params { fight_pos: 1.0, ..Params::default() });
        let mut off = with(Params { fight_pos: 1.0, pred: 0.0, pincer: 0.0, ..Params::default() });
        let mut plain = with(Params { fight_pos: 1.0, pred: 0.0, pincer: 0.0, ..Params::default() });
        plain.pin = pincer::PinPicture::default();
        on.decide(&o, [0.0; 3]);
        off.decide(&o, [0.0; 3]);
        plain.decide(&o, [0.0; 3]);
        assert!(on.pin.threats.len() == 2 && off.pin.threats.is_empty());
        assert_eq!(off.dbg_mode, plain.dbg_mode);
        // The map uses of the practice picture are off by default.
        let d = Params::default();
        assert_eq!((d.map_preaim, d.map_expose, d.map_crowd, d.map_reinf), (0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn a_dithering_enemy_is_led_less_and_a_steady_one_as_usual() {
        // Steady walk across our line: the usual lead keeps hitting.
        let mut st = LeadStat::default();
        for i in 0..60 {
            let t = i as f32 * 0.1;
            let (p, v) = ([100.0, 8.0 * t], [0.0, 8.0]);
            st.settle(t, p, v, 2.1);
            st.look(t, [0.0, 0.0], p, v, 0.4);
        }
        assert_eq!(st.choice(), None);
        // Reversing every 0.4 s: where it is now beats where its speed points.
        let mut st = LeadStat::default();
        let mut y = 0.0f32;
        for i in 0..80 {
            let t = i as f32 * 0.1;
            let vy = if (i / 4) % 2 == 0 { 8.0 } else { -8.0 };
            st.settle(t, [100.0, y], [0.0, vy], 2.1);
            st.look(t, [0.0, 0.0], [100.0, y], [0.0, vy], 0.4);
            y += vy * 0.1;
        }
        assert!(st.choice().map_or(false, |l| l < 0.8), "{:?} {:?}", st.choice(), st.ema);
    }

    #[test]
    fn with_nothing_to_do_and_healthy_we_walk_to_an_enemy_and_take_it_on() {
        let far = r#"{"id": 3, "kind": "mech", "pos": [150, 0, 0], "aim": [0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let near = r#"{"id": 3, "kind": "mech", "pos": [80, 0, 0], "aim": [0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let on = Params { seek: 300.0, engage_filter: 1.0, ..base() };
        let mut b = with(on.clone());
        let a = b.decide(&obs(far, ""), [0.0; 3]);
        assert_eq!((b.label, b.focus, b.seek_id), ("seek", None, Some(3)));
        assert!(a.mv[0] > 0.5, "{:?}", a.mv);
        // Once it is within `seek_fire` it is a target although it is not coming for us.
        b.decide(&obs(near, ""), [0.0; 3]);
        assert_eq!(b.focus, Some(3));
        // Off, or hurt: left alone.
        let mut b = with(Params { seek: 0.0, ..on.clone() });
        b.decide(&obs(far, ""), [0.0; 3]);
        b.decide(&obs(near, ""), [0.0; 3]);
        assert_eq!((b.focus, b.seek_id), (None, None));
        let mut b = with(on);
        let mut o = obs(far, "");
        o.me.hp = 700.0;
        b.decide(&o, [0.0; 3]);
        assert_eq!(b.seek_id, None);
    }

    #[test]
    fn of_two_like_targets_the_laser_holder_comes_first() {
        let two = r#"{"id": 3, "kind": "mech", "pos": [40, 0, 0], "aim": [3.14, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]},
                     {"id": 4, "kind": "mech", "pos": [0, 45, 0], "aim": [-1.57, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun", "laser"]}"#;
        let mut b = with(Params { engage_filter: 1.0, focus_laser: 50.0, ..base() });
        b.decide(&obs(two, ""), [0.0; 3]);
        assert_eq!(b.focus, Some(4));
        let mut b = with(Params { engage_filter: 1.0, ..base() });
        b.decide(&obs(two, ""), [0.0; 3]);
        assert_eq!(b.focus, Some(3));
    }

    #[test]
    fn idle_goes_after_the_nearest_known_enemy() {
        let away = r#"{"id": 3, "kind": "mech", "pos": [100, 0, 0], "aim": [0, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        for (on, want) in [(1.0, true), (0.0, false)] {
            let mut b = with(Params { seek_idle: on, ..base() });
            let mut o = obs(away, "");
            b.decide(&o, [0.0; 3]);
            o.mechs.clear();
            o.t = 5.5;
            b.decide(&o, [0.0; 3]);
            o.t = 9.0;
            let a = b.decide(&o, [0.0; 3]);
            assert_eq!(b.label == "seek", want, "label {}", b.label);
            if want {
                assert!(a.mv[0] > 0.5, "mv {:?}", a.mv);
            }
        }
    }

    #[test]
    fn predicted_motion_keeps_less_of_a_sideways_dodge() {
        // An enemy ahead of us moving 6 m/s toward us and 8 m/s across.
        let m = mech(3, [60.0, 0.0], [-6.0, 8.0], 3.14);
        let o = obs(&m, "");
        let b = with(Params { pred_tan: 0.5, ..base() });
        let v = b.pred_vel(&o.mechs[0], [0.0, 0.0]);
        assert!((v[0] + 6.0).abs() < 1e-3 && (v[1] - 4.0).abs() < 1e-3, "{:?}", v);
        let b = with(base());
        assert_eq!(b.pred_vel(&o.mechs[0], [0.0, 0.0]), [-6.0, 8.0]);
    }

    #[test]
    fn broken_off_but_still_in_view_we_sidestep_instead_of_standing() {
        let foe = r#"{"id": 3, "kind": "mech", "pos": [40, 0, 0], "aim": [3.14, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]},
                     {"id": 4, "kind": "mech", "pos": [50, 20, 0], "aim": [3.5, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        for (on, moving) in [(1.0, true), (0.0, false)] {
            let mut b = with(Params { crowd_retreat: 1.0, cr_strafe: on, ..base() });
            let mut o = obs(foe, "");
            o.me.hp = 500.0;
            b.cr_until = 100.0;
            b.cr_goal = Some([0.0, 0.0]);
            let a = b.decide(&o, [0.0; 3]);
            if b.label == "cbreak" && b.cr_goal.map_or(false, |g| g[0].hypot(g[1]) < 2.0) {
                assert_eq!(a.mv[0].hypot(a.mv[1]) > 0.5, moving, "{:?}", a.mv);
                assert!(a.mv[0].abs() < 0.3, "{:?}", a.mv);
            }
        }
    }

    #[test]
    fn a_live_fight_is_not_left_for_hidden_ground_far_off() {
        let foes = r#"{"id": 3, "kind": "mech", "pos": [40, 0, 0], "aim": [3.14, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]},
                      {"id": 4, "kind": "mech", "pos": [50, 20, 0], "aim": [3.5, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}"#;
        let hit = r#"{"type": "damaged", "amount": 45, "weapon": "gun", "bearing": 0.0, "by": 3}"#;
        for (stay, goal, broke) in [(0.0, [-60.0, 0.0], true), (1.0, [-60.0, 0.0], false), (1.0, [-10.0, 0.0], true)] {
            let mut b = with(Params { crowd_retreat: 1.0, stay_fight: stay, cr_max_d: 0.0, ..base() });
            let mut o = obs(foes, hit);
            o.me.hp = 500.0;
            b.cr_until = 100.0;
            b.cr_goal = Some(goal);
            b.decide(&o, [0.0; 3]);
            assert_eq!(b.label == "cbreak", broke, "stay {stay} goal {goal:?}: {}", b.label);
        }
    }
}
