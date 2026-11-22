//! Ammunition and energy upkeep: reload top-up, cover while the gun is empty, energy reserve, target memory.
//! The scenes need the arena map: set `MAP_DIR`, or keep the kit next to this repo (`../mechfight-kit/arena`).
use std::sync::Arc;
use tinman::brain::{Brain, Params};
use tinman::nav::Nav;
use tinman::proto::{Action, Hello, Obs};
use tinman::world::World;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

fn load() -> Option<(Arc<World>, Nav)> {
    let dir = std::env::var("MAP_DIR").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../mechfight-kit/arena").to_string());
    let Ok(world) = World::load(std::path::Path::new(&dir)) else {
        eprintln!("map not found at {dir}; skipping");
        return None;
    };
    let nav = Nav::build(&world, 2.5, 9.0, 1.2, 50.0);
    Some((Arc::new(world), nav))
}

fn params(v: &str) -> Params {
    Params::from_json(&serde_json::from_str(v).unwrap())
}

/// Every upkeep switch off.
const OFF: &str = r#"{"topup_after": 0, "empty_cover": 0, "energy_reserve": 0, "target_mem": 0}"#;

/// A scene that runs through what the upkeep switches look at: energy swinging from low to full, the gun
/// emptying and reloading, enemies closing in and shooting, and every enemy dropping out of view for a while.
/// Returns the hash of every action.
fn upkeep_scene(world: Arc<World>, nav: Nav, p: Params) -> u64 {
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":91,"map":{"dir":""},"config":{}}"#).unwrap();
    let mut b = Brain::with_params(&hello, Some(world), Some(nav.clone()), p);
    let c = nav.nearest_walkable(41.3, -72.8).unwrap();
    let mut me = nav.center(c);
    let mut aim = [0.0f32, 0.0];
    let ids = [3u32, 5, 7];
    let mut h = 0xcbf29ce484222325u64;
    let mut ammo = 12.0f32;
    let mut reload = 0.0f32;
    for k in 0..700u64 {
        let t = 2.0 + k as f32 * 0.1;
        let tick = 40 + 2 * k;
        let hidden = (k % 60) >= 40;
        let mut mechs = Vec::new();
        let mut events = Vec::new();
        let mut projectiles = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            let r = (120.0 - t * 1.0 - i as f32 * 15.0).max(30.0);
            let a = i as f32 * 2.1 + t * 0.07;
            let p = [me[0] + a.cos() * r, me[1] + a.sin() * r, 0.0];
            let yaw = (me[1] - p[1]).atan2(me[0] - p[0]);
            if !hidden {
                mechs.push(format!(
                    r#"{{"id":{id},"kind":"mech","pos":[{:.2},{:.2},0],"vel":[{:.2},{:.2},0],"aim":[{yaw:.3},0],"hp":1400,"max_hp":1400,"cubes":0,"weapons":["gun","laser"],"charging":false}}"#,
                    p[0],
                    p[1],
                    -a.sin() * r * 0.07,
                    a.cos() * r * 0.07
                ));
                if k % 5 == i as u64 {
                    let v = [yaw.cos() * 252.0, yaw.sin() * 252.0];
                    projectiles.push(format!(r#"{{"id":{},"kind":"gun","pos":[{:.2},{:.2},7],"vel":[{:.1},{:.1},0]}}"#, 500 + k, p[0] + yaw.cos() * 6.0, p[1] + yaw.sin() * 6.0, v[0], v[1]));
                }
            }
            if k % 8 == i as u64 && !hidden {
                events.push(format!(r#"{{"type":"damaged","amount":45,"weapon":"gun","bearing":{:.3},"by":{id}}}"#, yaw + std::f32::consts::PI));
            }
        }
        let energy = 5.0 + ((k * 7) % 95) as f32;
        let s = format!(
            r#"{{"type":"obs","tick":{tick},"t":{t:.2},"alive":16,
               "self":{{"id":0,"pos":[{:.2},{:.2},0],"vel":[0,0,0],"aim":[{:.3},{:.3}],"hp":900,"max_hp":1400,"energy":{energy},"max_energy":100,"grounded":true,"cubes":0,"in_zone":true,
                        "weapons":[{{"kind":"gun","cooldown":0,"ammo":{ammo},"reload":{reload:.2}}},{{"kind":"laser","cooldown":{:.2},"charge":0}}]}},
               "mechs":[{}],"projectiles":[{}],"events":[{}],
               "zone":{{"center":[41.3,-72.8],"radius":560,"next_center":[-23.7,2.2],"next_radius":380,"shrink_start":60,"shrink_end":110,"dps":20}}}}"#,
            me[0],
            me[1],
            aim[0],
            aim[1],
            ((k % 30) as f32 * 0.1).min(2.0),
            mechs.join(","),
            projectiles.join(","),
            events.join(",")
        );
        let o: Obs = serde_json::from_str(&s).unwrap();
        let a: Action = b.act(&o);
        fnv(&mut h, serde_json::to_string(&a).unwrap().as_bytes());
        // The gun as the game would run it.
        if reload > 0.0 {
            reload -= 0.1;
            if reload <= 0.0 {
                reload = 0.0;
                ammo = 12.0;
            }
        } else if a.reload && ammo < 12.0 || ammo <= 0.0 {
            reload = 2.4;
        } else if a.fire[0] && ammo > 0.0 {
            ammo -= 1.0;
        }
        let l = a.mv[0].hypot(a.mv[1]).min(1.0);
        if l > 0.0 {
            let n = [me[0] + a.mv[0] / a.mv[0].hypot(a.mv[1]) * l * 1.17, me[1] + a.mv[1] / a.mv[0].hypot(a.mv[1]) * l * 1.17];
            if nav.walkable(n[0], n[1]) {
                me = n;
            }
        }
        aim = a.aim;
    }
    h
}

/// The scene's actions as the build without the upkeep switches makes them.
const UPKEEP_HASH_BEFORE: u64 = 0x5d00df339ae5dbf5;

#[test]
fn all_upkeep_switches_off_play_as_before() {
    let Some((world, nav)) = load() else { return };
    let h = upkeep_scene(world.clone(), nav.clone(), params(OFF));
    eprintln!("upkeep scene hash (off) {h:#018x}");
    assert_eq!(h, UPKEEP_HASH_BEFORE, "actions differ from the build without the upkeep switches");
    let on = upkeep_scene(world, nav, Params::default());
    assert_ne!(on, h, "the switches change nothing in the scene");
}

fn fight_obs(me: [f32; 3], foe: [f32; 3], gun: &str, laser: &str) -> Obs {
    let yaw = (me[1] - foe[1]).atan2(me[0] - foe[0]);
    let s = format!(
        r#"{{"type":"obs","tick":200,"t":10.0,"alive":16,
           "self":{{"id":0,"pos":[{:.2},{:.2},{:.2}],"vel":[0,0,0],"aim":[{:.3},0],"hp":1400,"max_hp":1400,"energy":100,"max_energy":100,"grounded":true,"cubes":0,"in_zone":true,
                    "weapons":[{gun},{laser}]}},
           "mechs":[{{"id":3,"kind":"mech","pos":[{:.2},{:.2},{:.2}],"vel":[0,0,0],"aim":[{yaw:.3},0],"hp":1400,"max_hp":1400,"weapons":["gun"]}}],
           "zone":{{"center":[{:.2},{:.2}],"radius":600,"next_center":[{:.2},{:.2}],"next_radius":500,"shrink_start":100,"shrink_end":150,"dps":20}}}}"#,
        me[0],
        me[1],
        me[2],
        yaw + std::f32::consts::PI,
        foe[0],
        foe[1],
        foe[2],
        me[0],
        me[1],
        me[0],
        me[1]
    );
    serde_json::from_str(&s).unwrap()
}

const EMPTY: &str = r#"{"kind":"gun","cooldown":0,"ammo":0,"reload":1.8}"#;
const LOADED: &str = r#"{"kind":"gun","cooldown":0,"ammo":12,"reload":0}"#;
const LASER_DOWN: &str = r#"{"kind":"laser","cooldown":3.0,"charge":0}"#;
const LASER_UP: &str = r#"{"kind":"laser","cooldown":0,"charge":0}"#;

fn label_of(world: &Arc<World>, nav: &Nav, p: &Params, o: &Obs) -> (&'static str, Action) {
    let mut b = Brain::with_params(&Hello::default(), Some(world.clone()), Some(nav.clone()), p.clone());
    let a = b.act(o);
    (b.label, a)
}

#[test]
fn empty_gun_steps_into_cover_or_fights_on_with_the_laser() {
    let Some((world, nav)) = load() else { return };
    // Without the sheltered-ground tactics (they only slow the brain's start here).
    let on = params(r#"{"spot": 0, "crowd_retreat": 0, "fight_pos": 0, "late_hold": 0, "fight_cubes": 0}"#);
    let no_w2 = params(r#"{"spot": 0, "crowd_retreat": 0, "fight_pos": 0, "late_hold": 0, "fight_cubes": 0, "empty_w2": 0}"#);
    // Find a spot in a fight where an empty gun with the laser down sends us into cover.
    let mut found = 0;
    'grid: for gx in -12..12 {
        for gy in -12..12 {
            let Some(c) = nav.nearest_walkable(gx as f32 * 25.0, gy as f32 * 25.0) else { continue };
            let m = nav.center(c);
            for d in 0..8 {
                let a = d as f32 * std::f32::consts::TAU / 8.0;
                let Some(fc) = nav.nearest_walkable(m[0] + a.cos() * 40.0, m[1] + a.sin() * 40.0) else { continue };
                let f = nav.center(fc);
                let me = [m[0], m[1], world.ground(m[0], m[1])];
                let foe = [f[0], f[1], world.ground(f[0], f[1])];
                if !world.clear([me[0], me[1], me[2] + 6.0], [foe[0], foe[1], foe[2] + 6.0]) {
                    continue;
                }
                // Cheap check first: some ground within 12 m out of the enemy's sight.
                let eye = [foe[0], foe[1], foe[2] + 8.0];
                let cover = (0..12).any(|k| {
                    let b = k as f32 * std::f32::consts::TAU / 12.0;
                    (1..4).any(|r| {
                        let c = [me[0] + b.cos() * 4.0 * r as f32, me[1] + b.sin() * 4.0 * r as f32];
                        let z = world.ground(c[0], c[1]);
                        (z - me[2]).abs() < 1.5 && !world.clear(eye, [c[0], c[1], z + 4.5]) && !world.clear(eye, [c[0], c[1], z + 8.5])
                    })
                });
                if !cover {
                    continue;
                }
                let (l, a) = label_of(&world, &nav, &on, &fight_obs(me, foe, EMPTY, LASER_DOWN));
                if l != "reload" {
                    continue;
                }
                found += 1;
                assert!(a.mv[0].hypot(a.mv[1]) > 0.5, "no step into cover: {:?}", a.mv);
                // Laser ready: fight on with it.
                let (l2, a2) = label_of(&world, &nav, &on, &fight_obs(me, foe, EMPTY, LASER_UP));
                assert_eq!(l2, "fight", "laser ready but still took cover at {me:?}");
                assert!(a2.fire[1], "laser not used: {:?}", a2.fire);
                // Without that switch, cover anyway.
                let (l3, _) = label_of(&world, &nav, &no_w2, &fight_obs(me, foe, EMPTY, LASER_UP));
                assert_eq!(l3, "reload");
                // Reloaded: straight back to the fight.
                let (l4, _) = label_of(&world, &nav, &on, &fight_obs(me, foe, LOADED, LASER_DOWN));
                assert_ne!(l4, "reload");
                if found >= 3 {
                    break 'grid;
                }
            }
        }
    }
    assert!(found > 0, "no spot with cover found");
}
