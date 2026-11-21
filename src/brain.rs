//! Decision logic: one `act` per observation.
use crate::nav::Nav;
use crate::proto::{Action, Actor, Hello, Obs, Weapon};
use crate::world::World;
use std::f32::consts::{PI, TAU};

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
    if l < 1e-6 {
        [0.0, 0.0]
    } else {
        [x / l, y / l]
    }
}

/// Upgrade preference, best first. Survival first: placement is the score.
const UPGRADES: &[&str] = &[
    "plating", "armor", "nanites", "lightning_efficiency", "gun_damage", "servos", "gun_rate", "laser_damage",
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

pub struct Brain {
    pub world: Option<World>,
    pub nav: Option<Nav>,
    rng: u64,
    pickup_radius: f32,
    strafe: f32,
    strafe_until: f32,
    last_pos: [f32; 3],
    stuck: u32,
    unstick_until: f32,
    last_hurt_t: f32,
    threat: Option<([f32; 3], f32)>,
    fleeing: bool,
    cover: Option<([f32; 2], f32)>,
    scan_yaw: f32,
}

impl Brain {
    pub fn new(hello: &Hello) -> Brain {
        let cfg = &hello.config;
        let f = |a: &str, b: &str, d: f32| cfg.get(a).and_then(|v| v.get(b)).and_then(|v| v.as_f64()).map(|v| v as f32).unwrap_or(d);
        let world = World::load(std::path::Path::new(&hello.map.dir)).map_err(|e| eprintln!("map load failed: {e}")).ok();
        let nav = world.as_ref().map(|w| {
            Nav::build(w, f("mech", "radius", 2.5), f("mech", "height", 9.0), f("mech", "step_up", 1.2), f("mech", "max_slope_deg", 50.0))
        });
        Brain {
            world,
            nav,
            rng: hello.seed | 1,
            pickup_radius: f("loot", "pickup_radius", 4.0),
            strafe: 1.0,
            strafe_until: 0.0,
            last_pos: [0.0; 3],
            stuck: 0,
            unstick_until: 0.0,
            last_hurt_t: -100.0,
            threat: None,
            fleeing: false,
            cover: None,
            scan_yaw: 0.0,
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

    /// Unit move direction toward a goal along the ground path.
    fn go(&mut self, pos: &[f32; 3], goal: [f32; 2]) -> [f32; 2] {
        if d2(pos, &goal) < 1.5 {
            return [0.0, 0.0];
        }
        let mut target = goal;
        if let Some(nav) = self.nav.as_mut() {
            nav.set_goal(goal[0], goal[1]);
            if let Some(p) = nav.steer(pos[0], pos[1]) {
                // Close to the goal the grid is too coarse: walk straight in.
                if d2(pos, &goal) > 6.0 {
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
                let mut p = tc;
                for _ in 0..3 {
                    let tt = d3(&muzzle, &p) / speed;
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
                // Aim at the feet so a near miss still splashes; lead by the flight time.
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
                // Low arc only: it needs the first stretch of the path clear.
                let along = [tube[0] + (p[0] - tube[0]) * 0.3, tube[1] + (p[1] - tube[1]) * 0.3, tube[2] + pitch.tan() * d * 0.3];
                let ok = ok && safe && pitch < me.stat("max_pitch", 1.3) && self.clear(tube, along);
                Shot { aim: [(p[1] - tube[1]).atan2(p[0] - tube[0]), pitch], tol: [(3.0 / d.max(1.0)).atan(), 0.03], ok }
            }
            _ => Shot { aim: me.aim, tol: [0.0, 0.0], ok: false },
        }
    }

    /// A spot the threat cannot see, reasonably close and inside the zone.
    fn find_cover(&mut self, o: &Obs, threat: [f32; 3]) -> [f32; 2] {
        let me = &o.me;
        let eye = [threat[0], threat[1], threat[2] + 8.0];
        let away = norm2(me.pos[0] - threat[0], me.pos[1] - threat[1]);
        let mut best = ([me.pos[0] + away[0] * 60.0, me.pos[1] + away[1] * 60.0], f32::MIN);
        for ring in [25.0f32, 45.0, 70.0, 100.0] {
            for k in 0..16 {
                let a = k as f32 * TAU / 16.0;
                let c = [me.pos[0] + a.cos() * ring, me.pos[1] + a.sin() * ring];
                if !self.walkable(c[0], c[1]) {
                    continue;
                }
                let z = self.ground(c[0], c[1], me.pos[2]);
                let hidden = !self.clear(eye, [c[0], c[1], z + 4.5]) && !self.clear(eye, [c[0], c[1], z + 8.5]);
                let zone_out = (d2(&c, &o.zone.center) - o.zone.radius + 15.0).max(0.0) + (d2(&c, &o.zone.next_center) - o.zone.next_radius).max(0.0) * 0.3;
                let s = if hidden { 120.0 } else { 0.0 } + d2(&c, &threat) * 0.5 - ring * 0.6 - zone_out * 4.0
                    + (a.cos() * away[0] + a.sin() * away[1]) * 15.0;
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
        let mut a = Action { tick: o.tick, aim: me.aim, ..Default::default() };
        let hp_frac = me.hp / me.max_hp.max(1.0);
        let walk_speed = me.stat("walk_speed", 11.7);

        // --- what hurt us
        for e in &o.events {
            if e.kind == "damaged" && e.by.is_some() {
                self.last_hurt_t = o.t;
                if let (Some(b), Some(by)) = (e.bearing, e.by) {
                    if by < 100 && !o.mechs.iter().any(|m| m.id == by) {
                        self.threat = Some(([pos[0] + b.cos() * 60.0, pos[1] + b.sin() * 60.0, pos[2]], o.t));
                    }
                }
            }
        }

        // --- upgrades
        if let Some(offer) = &o.offer {
            let rank = |id: &str| UPGRADES.iter().position(|u| *u == id).unwrap_or(UPGRADES.len());
            a.pick = offer.choices.iter().enumerate().min_by_key(|(_, c)| rank(&c.id)).map(|(i, _)| i as u32);
        }

        // --- target: the nearest living mech we know about
        let enemies: Vec<&Actor> = o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0)).collect();
        let target = enemies.iter().min_by(|x, y| d3(&x.pos, &pos).partial_cmp(&d3(&y.pos, &pos)).unwrap()).copied();
        if let Some(t) = target {
            self.threat = Some((t.pos, o.t));
        }
        if self.threat.map_or(false, |(_, t)| o.t - t > 8.0) {
            self.threat = None;
        }

        // --- weapons: solve every slot, aim with the best ready one, fire all that line up
        let mut engaged = false;
        if let Some(t) = target {
            let mut best: Option<(i32, [f32; 2])> = None;
            let mut fallback = None;
            for (i, slot) in me.weapons.iter().enumerate().take(2) {
                let Some(w) = slot else { continue };
                let s = self.solve(o, w, t);
                fallback.get_or_insert(s.aim);
                if !s.ok {
                    // A laser already charging must keep the trigger held or it fires blind.
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
                    // Hold to charge; let go only when fully charged and on target.
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
        for p in &o.pickups {
            if d2(&p.pos, &pos) < self.pickup_radius - 0.3 && (p.pos[2] - pos[2]).abs() < 3.7 {
                if let Some(drop) = wanted(&p.kind) {
                    let charging = me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.charge > 0.0);
                    if !(drop.is_some() && charging) {
                        a.take = Some(p.id);
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
        let lead = out_next.max(0.0) / walk_speed * 1.6 + 8.0;
        let must_rotate = !me.in_zone || (out_next > 0.0 && o.t >= z.shrink_start - lead) || z.next_radius <= 1.0;
        let zone_goal = {
            let d = norm2(pos[0] - z.next_center[0], pos[1] - z.next_center[1]);
            let r = (z.next_radius - margin - 5.0).max(0.0);
            [z.next_center[0] + d[0] * r, z.next_center[1] + d[1] * r]
        };

        // --- mode
        let threat_live = self.threat.filter(|(_, t)| o.t - *t < 6.0).map(|(p, _)| p);
        if hp_frac < 0.4 && threat_live.is_some() {
            self.fleeing = true;
        }
        if hp_frac > 0.75 || threat_live.is_none() {
            self.fleeing = false;
        }

        let mut boost = false;
        let mv: [f32; 2];
        if !me.in_zone || (must_rotate && out_next > 25.0) {
            // Zone first: nothing else matters outside it.
            mv = self.go(&pos, zone_goal);
            boost = !me.in_zone && me.energy > 5.0;
        } else if self.fleeing {
            let th = threat_live.unwrap();
            if self.cover.map_or(true, |(_, until)| o.t > until) {
                let c = self.find_cover(o, th);
                self.cover = Some((c, o.t + 2.0));
            }
            let c = self.cover.unwrap().0;
            mv = self.go(&pos, c);
            boost = me.energy > 15.0 && d2(&pos, &c) > 8.0;
            if target.is_none() {
                // Keep our front toward where the damage comes from.
                a.aim = [(th[1] - pos[1]).atan2(th[0] - pos[0]), 0.0];
            }
        } else if let (Some(t), true) = (target, engaged || o.t - self.last_hurt_t < 3.0) {
            // Fight: hold a range that suits what we carry and never stop moving sideways.
            let dist = d2(&t.pos, &pos);
            let has_arc = me.weapons.iter().flatten().any(|w| w.kind == "lightning") && me.energy > 20.0;
            let pref = if has_arc { 30.0 } else { 55.0 };
            let to = norm2(t.pos[0] - pos[0], t.pos[1] - pos[1]);
            let radial = ((dist - pref) / 15.0).clamp(-1.0, 1.0);
            if o.t >= self.strafe_until {
                self.strafe = if self.rnd() < 0.5 { 1.0 } else { -1.0 };
                self.strafe_until = o.t + 0.5 + self.rnd() * 0.9;
            }
            let mut dir = norm2(to[0] * radial - to[1] * self.strafe, to[1] * radial + to[0] * self.strafe);
            if !self.walkable(pos[0] + dir[0] * 6.0, pos[1] + dir[1] * 6.0) {
                self.strafe = -self.strafe;
                dir = norm2(to[0] * radial - to[1] * self.strafe, to[1] * radial + to[0] * self.strafe);
                if !self.walkable(pos[0] + dir[0] * 6.0, pos[1] + dir[1] * 6.0) {
                    dir = [-to[0], -to[1]];
                }
            }
            mv = dir;
        } else {
            // Quiet time: loot nearby, otherwise sit still (standing is silent) inside the next circle.
            let on_ground = |p: &[f32; 3]| (p[2] - self.ground(p[0], p[1], p[2])).abs() < 3.0;
            let in_zone = |p: &[f32; 3]| d2(p, &z.center) < z.radius - 10.0 && (o.t < z.shrink_start - 20.0 || d2(p, &z.next_center) < z.next_radius);
            let mut goal: Option<([f32; 2], f32)> = None;
            let mut consider = |p: &[f32; 3], value: f32, reach: f32| {
                let d = d2(p, &pos);
                if d < reach && on_ground(p) && in_zone(p) {
                    let s = value - d;
                    if goal.map_or(true, |g| s > g.1) {
                        goal = Some(([p[0], p[1]], s));
                    }
                }
            };
            for p in &o.pickups {
                if wanted(&p.kind).is_some() {
                    consider(&p.pos, 60.0 + 30.0 * weapon_rank(&p.kind) as f32, 140.0);
                }
            }
            for c in &o.cubes {
                consider(&c.pos, 90.0, 90.0);
            }
            let crate_target = o.crates.iter().filter(|c| d3(&c.pos, &pos) < 90.0 && hp_frac > 0.6).min_by(|x, y| d3(&x.pos, &pos).partial_cmp(&d3(&y.pos, &pos)).unwrap());
            if let (Some(c), None) = (crate_target, target) {
                // Shoot the crate with the gun; walk closer if it is far.
                let muzzle = [pos[0], pos[1], pos[2] + me.stat("muzzle_height", 7.0)];
                let cc = [c.pos[0], c.pos[1], c.pos[2] + 1.2];
                if self.clear(muzzle, cc) {
                    let want = [(cc[1] - muzzle[1]).atan2(cc[0] - muzzle[0]), (cc[2] - muzzle[2]).atan2(d2(&cc, &muzzle).max(0.1))];
                    a.aim = want;
                    let lined = wrap(want[0] - me.aim[0]).abs() < (1.0 / d3(&cc, &muzzle).max(1.0)).atan() && (want[1] - me.aim[1]).abs() < (0.9 / d3(&cc, &muzzle).max(1.0)).atan();
                    for (i, w) in me.weapons.iter().enumerate().take(2) {
                        if w.as_ref().map_or(false, |w| w.kind == "gun") {
                            a.fire[i] = lined;
                            a.reload = false;
                        }
                    }
                    if on_ground(&c.pos) && d2(&c.pos, &pos) > 35.0 {
                        consider(&c.pos, 70.0, 90.0);
                    }
                }
            }
            let g = match goal {
                Some((g, _)) if !must_rotate => Some(g),
                _ if must_rotate || out_next > 0.0 => Some(zone_goal),
                _ => None,
            };
            mv = match g {
                Some(g) => self.go(&pos, g),
                None => [0.0, 0.0],
            };
            if target.is_none() && crate_target.is_none() {
                // Look around: toward recent damage first, otherwise a steady sweep.
                if let Some(th) = threat_live {
                    a.aim = [(th[1] - pos[1]).atan2(th[0] - pos[0]), 0.0];
                } else {
                    self.scan_yaw = wrap(self.scan_yaw + 0.16);
                    a.aim = [self.scan_yaw, 0.0];
                }
            } else if target.is_none() {
                self.scan_yaw = me.aim[0];
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
            self.stuck = 0;
        }
        if o.t < self.unstick_until {
            a.jump = me.grounded;
            boost = true;
        }
        self.last_pos = pos;
        a.mv = mv;
        a.boost = boost;
        a
    }
}
