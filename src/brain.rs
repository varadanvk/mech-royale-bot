//! Decision logic: one `act` per observation.
use crate::nav::Nav;
use crate::proto::{Action, Actor, Hello, Obs, Weapon};
use crate::world::World;
use std::collections::BTreeMap;
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
}

impl Default for Params {
    fn default() -> Self {
        Params {
            retreat_hp: 0.55,
            reengage_hp: 0.92,
            gun_range: 55.0,
            gun_band: 15.0,
            arc_range: 30.0,
            arc_min_hp: 0.6,
            crowd_radius: 150.0,
            crowd_count: 2,
            open_fire_range: 200.0,
            strafe_min: 0.35,
            strafe_max: 0.9,
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

/// Upgrade preference, best first. Survival first: placement is the score.
const UPGRADES: &[&str] = &[
    "plating", "armor", "nanites", "gun_damage", "lightning_efficiency", "servos", "gun_rate", "laser_damage",
    "lightning_chain", "reactor", "laser_charge", "capacitor", "gun_mag", "gyros", "lightning_range", "jets",
    "gun_velocity", "mortar_reload", "mortar_splash", "sensors", "mortar_velocity", "laser_pierce",
];

fn weapon_rank(kind: &str) -> i32 {
    match kind {
        "lightning" => 3,
        "laser" => 2,
        "mortar" => 1,
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
    /// When we first saw its current laser charge (None if not charging).
    charge_since: Option<f32>,
    aim_yaw: f32,
}

pub struct Brain {
    pub world: Option<Arc<World>>,
    pub nav: Option<Nav>,
    pub params: Params,
    rng: u64,
    pickup_radius: f32,
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
    /// Short name of the current behaviour, for debugging.
    pub label: &'static str,
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
            label: "",
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

    /// Aim solution for one weapon against a target, from our current position.
    fn solve(&self, o: &Obs, w: &Weapon, t: &Actor) -> Shot {
        let me = &o.me;
        let scale = 1.0 + 0.05 * t.cubes;
        let tc = [t.pos[0], t.pos[1], t.pos[2] + 4.5 * scale];
        let muzzle = [me.pos[0], me.pos[1], me.pos[2] + me.stat("muzzle_height", 7.0)];
        let dist = d3(&muzzle, &tc);
        let flat = |p: &[f32; 3], from: &[f32; 3]| [(p[1] - from[1]).atan2(p[0] - from[0]), (p[2] - from[2]).atan2(d2(p, from).max(0.01))];
        let body = [(2.5 * scale * 0.85 / dist.max(1.0)).atan(), (4.5 * scale * 0.8 / dist.max(1.0)).atan()];
        match w.kind.as_str() {
            "gun" => {
                let speed = w.stat("speed", 252.0);
                // Shots start a little ahead of the mech along the aim.
                let lead_out = 2.5 * (1.0 + 0.05 * me.cubes) + 0.5;
                let mut p = tc;
                for _ in 0..3 {
                    let tt = ((d3(&muzzle, &p) - lead_out).max(0.0)) / speed + 0.05;
                    p = [tc[0] + t.vel[0] * tt, tc[1] + t.vel[1] * tt, tc[2] + t.vel[2] * tt * 0.5];
                }
                let ok = w.ammo > 0.0 && w.reload <= 0.0 && d3(&muzzle, &p) < w.stat("range", 230.0) && self.clear(muzzle, p);
                Shot { aim: flat(&p, &muzzle), tol: body, ok }
            }
            "laser" => {
                let ok = dist < w.stat("range", 280.0) && self.clear(muzzle, tc);
                Shot { aim: flat(&tc, &muzzle), tol: [body[0] * 0.8, body[1] * 0.8], ok }
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
        let me = o.me.clone();
        let pos = me.pos;
        let p = self.params.clone();
        let mut a = Action { tick: o.tick, aim: me.aim, ..Default::default() };
        let hp_frac = me.hp / me.max_hp.max(1.0);
        let walk_speed = me.stat("walk_speed", 11.7);

        // --- memory of mechs
        for m in o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0)) {
            let prev_charge = self.known.get(&m.id).and_then(|k| k.charge_since);
            self.known.insert(
                m.id,
                Known {
                    pos: m.pos,
                    vel: m.vel,
                    seen_t: o.t,
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
            let rank = |id: &str| UPGRADES.iter().position(|u| *u == id).unwrap_or(UPGRADES.len());
            a.pick = offer.choices.iter().enumerate().min_by_key(|(_, c)| rank(&c.id)).map(|(i, _)| i as u32);
        }

        // --- enemies and target
        let enemies: Vec<&Actor> = o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0)).collect();
        let near_enemies = enemies.iter().filter(|m| d2(&m.pos, &pos) < p.crowd_radius).count();
        let score = |m: &Actor| {
            let d = d2(&m.pos, &pos);
            let hpf = if m.max_hp > 0.0 { m.hp / m.max_hp } else { 1.0 };
            let attacker = self.attackers.get(&m.id).map_or(false, |t| o.t - t < 4.0);
            d + hpf * p.focus_hp - if attacker { 40.0 } else { 0.0 }
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
        if let Some(t) = shoot_at {
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
                let lined = wrap(s.aim[0] - me.aim[0]).abs() <= s.tol[0] && (s.aim[1] - me.aim[1]).abs() <= s.tol[1];
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

        // --- gun housekeeping
        if let Some(Some(g)) = me.weapons.iter().find(|w| w.as_ref().map_or(false, |w| w.kind == "gun")) {
            let mag = g.stat("mag", 12.0);
            if g.reload <= 0.0 && (g.ammo <= 0.0 || (!engaged && g.ammo < mag * 0.6)) {
                a.reload = true;
            }
        }

        // --- loot within reach
        let held: Vec<&str> = me.weapons.iter().flatten().map(|w| w.kind.as_str()).collect();
        let empty_slot = me.weapons.len() < 2 || me.weapons.iter().take(2).any(|w| w.is_none());
        let worst_slot = me.weapons.iter().enumerate().take(2).filter_map(|(i, w)| w.as_ref().map(|w| (i, weapon_rank(&w.kind)))).filter(|(_, r)| *r > 0).min_by_key(|(_, r)| *r);
        let wanted = |kind: &str| -> Option<Option<u32>> {
            if held.contains(&kind) || weapon_rank(kind) == 0 {
                None
            } else if empty_slot {
                Some(None)
            } else {
                worst_slot.filter(|(_, r)| weapon_rank(kind) > *r).map(|(i, _)| Some(i as u32))
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
        let mut boost = false;
        let mut mv: [f32; 2];
        let mut label = "idle";
        if !me.in_zone || (must_rotate && out_next > 25.0) {
            label = "zone";
            mv = self.go(&pos, zone_goal);
            boost = !me.in_zone && me.energy > 5.0;
        } else if let (Some(t), true) = (target, (self.fleeing || crowded) && p.kite > 0.0 && threat_live) {
            // Back away from the nearest enemy while weaving and shooting.
            label = if self.fleeing { "kite" } else { "kitec" };
            let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
            let radial = if laser_on.is_some() { -0.2 } else { -p.kite };
            mv = self.strafe_dir(o, &pos, to, radial, false);
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
            mv = self.idle_move(o, &pos);
            if shoot_at.is_none() {
                self.scan(&mut a);
            }
        } else if let (Some(t), true) = (target, engaged || recently_hurt_by_mech || target.map_or(false, |t| d2(&t.pos, &pos) < p.open_fire_range)) {
            label = "fight";
            let dist = d2(&t.pos, &pos);
            let has_arc = me.weapons.iter().flatten().any(|w| w.kind == "lightning") && me.energy > 20.0;
            let they_arc = t.weapons.iter().any(|w| w == "lightning");
            let pref = if finishing {
                p.finish_range
            } else if has_arc && hp_frac > p.arc_min_hp && near_enemies <= 1 {
                p.arc_range
            } else if they_arc {
                p.gun_range.max(60.0)
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
            let hide = if p.reload_hide > 0.0 && reloading && !(has_arc && dist < 45.0) && !finishing {
                self.hide_step(&pos, &eyes, p.reload_hide)
            } else {
                None
            };
            if let Some(h) = hide {
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
        if let (Some(src), false) = (shooter, ducked || label == "zone" || label == "fight" || label == "drone" || label == "reload") {
            if o.t >= self.strafe_until {
                self.strafe = if self.rnd() < 0.5 { 1.0 } else { -1.0 };
                self.strafe_until = o.t + p.strafe_min + self.rnd() * (p.strafe_max - p.strafe_min).max(0.0);
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
        if wants && d2(&pos, &self.last_pos) < 0.15 {
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
        if std::env::var("TTRACE").is_ok() && o.tick % 20 == 0 {
            eprintln!("TR {:.0} {} hp {:.0} cubes {} seen_cubes {} mem_cubes {} enemies {} heal {} flee {}", o.t, label, me.hp, me.cubes, o.cubes.len(), self.cubes_mem.len(), enemies.len(), self.healing, self.fleeing);
        }
        self.last_pos = pos;
        a.mv = if mv.iter().all(|v| v.is_finite()) { mv } else { [0.0, 0.0] };
        a.boost = boost;
        if !a.aim.iter().all(|v| v.is_finite()) {
            a.aim = me.aim;
        }
        a
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
    fn strafe_dir(&mut self, o: &Obs, pos: &[f32; 3], to: [f32; 2], radial: f32, flip: bool) -> [f32; 2] {
        let p = &self.params;
        let (smin, smax) = (p.strafe_min, p.strafe_max.max(p.strafe_min));
        if flip {
            self.strafe = -self.strafe;
            self.strafe_until = o.t + smin;
        } else if o.t >= self.strafe_until {
            self.strafe = if self.rnd() < 0.5 { 1.0 } else { -1.0 };
            self.strafe_until = o.t + smin + self.rnd() * (smax - smin);
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
        let worst = me.weapons.iter().take(2).flatten().map(|w| weapon_rank(&w.kind)).filter(|r| *r > 0).min().unwrap_or(0);
        let early = o.t < p.farm_until;
        let reach = if early { p.farm_reach.max(100.0) } else { 100.0 };
        let picks: Vec<(String, [f32; 3])> = self.picks_mem.values().cloned().collect();
        for (kind, q) in &picks {
            let r = weapon_rank(kind);
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
