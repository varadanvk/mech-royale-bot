//! Match-type detection and free-for-all play.
//! The practice scene needs the arena map: set `MAP_DIR`, or keep the kit next to this repo
//! (`../mechfight-kit/arena`).
use std::sync::Arc;
use tinman::brain::{Brain, Params};
use tinman::detect::Detector;
use tinman::nav::Nav;
use tinman::proto::{Action, Hello, Obs};
use tinman::world::World;

fn fnv(h: &mut u64, bytes: &[u8]) {
    for b in bytes {
        *h ^= *b as u64;
        *h = h.wrapping_mul(0x100000001b3);
    }
}

/// A practice-like scene: four mechs circling us and closing in, all aimed at us, shooting at us (bullets,
/// beams and charges at us, hits on us); our hits taking hit points off them; a drone wearing one of them down;
/// one of them killed by us. Nothing a free-for-all would show. Returns the hash of every action and the brain.
fn practice_scene(world: Arc<World>, nav: Nav, params: Params) -> (u64, Brain) {
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":77,"map":{"dir":""},"config":{}}"#).unwrap();
    let mut b = Brain::with_params(&hello, Some(world), Some(nav.clone()), params);
    let c = nav.nearest_walkable(41.3, -72.8).unwrap();
    let mut me = nav.center(c);
    let mut aim = [0.0f32, 0.0];
    let ids = [3u32, 5, 7, 9];
    let mut hp = [1400.0f32, 1400.0, 1400.0, 1400.0];
    let mut my_hp = 1400.0f32;
    let mut alive = 16;
    let mut h = 0xcbf29ce484222325u64;
    for k in 0..600u64 {
        let t = 2.0 + k as f32 * 0.1;
        let tick = 40 + 2 * k;
        let mut mechs = Vec::new();
        let mut events = Vec::new();
        let mut beams = Vec::new();
        let mut projectiles = Vec::new();
        let mut sounds = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            if hp[i] <= 0.0 {
                continue;
            }
            let r = (140.0 - t * 1.2 - i as f32 * 10.0).max(35.0);
            let a = i as f32 * 1.57 + t * 0.05;
            let p = [me[0] + a.cos() * r, me[1] + a.sin() * r, 0.0];
            let yaw = (me[1] - p[1]).atan2(me[0] - p[0]);
            let charging = i == 1 && (k % 40) > 30;
            mechs.push(format!(
                r#"{{"id":{id},"kind":"mech","pos":[{:.2},{:.2},0],"vel":[{:.2},{:.2},0],"aim":[{yaw:.3},0],"hp":{:.0},"max_hp":1400,"cubes":0,"weapons":["gun","laser"],"charging":{charging}}}"#,
                p[0],
                p[1],
                -a.sin() * r * 0.05,
                a.cos() * r * 0.05,
                hp[i]
            ));
            if k % 7 == i as u64 {
                let v = [yaw.cos() * 252.0, yaw.sin() * 252.0];
                projectiles.push(format!(r#"{{"id":{},"kind":"gun","pos":[{:.2},{:.2},7],"vel":[{:.1},{:.1},0]}}"#, 300 + k, p[0] + yaw.cos() * 6.0, p[1] + yaw.sin() * 6.0, v[0], v[1]));
                sounds.push(format!(r#"{{"kind":"gun","bearing":{:.3},"dist":"mid","age":0.05}}"#, yaw + std::f32::consts::PI));
            }
            if i == 1 && k % 40 == 31 {
                beams.push(format!(r#"{{"by":{id},"kind":"laser","path":[[{:.2},{:.2},7],[{:.2},{:.2},4]],"age":0.05}}"#, p[0], p[1], me[0], me[1]));
                events.push(format!(r#"{{"type":"damaged","amount":200,"weapon":"laser","bearing":{:.3},"by":{id}}}"#, yaw + std::f32::consts::PI));
                my_hp -= 200.0;
            }
            if k % 9 == i as u64 {
                events.push(format!(r#"{{"type":"damaged","amount":45,"weapon":"gun","bearing":{:.3},"by":{id}}}"#, yaw + std::f32::consts::PI));
                my_hp -= 45.0;
            }
        }
        // A drone wearing down enemy 7 near it.
        if hp[2] > 0.0 {
            let r = (140.0 - t * 1.2 - 20.0).max(35.0);
            let a = 2.0 * 1.57 + t * 0.05;
            mechs.push(format!(r#"{{"id":104,"kind":"npc","pos":[{:.2},{:.2},0],"vel":[0,0,0],"aim":[0,0],"hp":350,"max_hp":350}}"#, me[0] + a.cos() * (r + 20.0), me[1] + a.sin() * (r + 20.0)));
            if k % 10 == 3 {
                hp[2] -= 25.0;
            }
        }
        // Our hits.
        if k % 6 == 0 {
            let i = (k / 6 % 4) as usize;
            if hp[i] > 0.0 {
                let amt = if i == 0 && hp[0] < 200.0 { hp[0] } else { 45.0 };
                hp[i] -= amt;
                events.push(format!(r#"{{"type":"hit","target":{},"amount":{amt:.0},"weapon":"gun"}}"#, ids[i]));
                if hp[i] <= 0.0 {
                    events.push(format!(r#"{{"type":"kill","target":{}}}"#, ids[i]));
                    alive -= 1;
                }
            }
        }
        if k % 6 == 1 && hp[0] > 0.0 {
            hp[0] -= 120.0;
            events.push(r#"{"type":"hit","target":3,"amount":120,"weapon":"laser"}"#.to_string());
        }
        my_hp = (my_hp + 3.0).min(1400.0).max(300.0);
        let s = format!(
            r#"{{"type":"obs","tick":{tick},"t":{t:.2},"alive":{alive},
               "self":{{"id":0,"pos":[{:.2},{:.2},0],"vel":[0,0,0],"aim":[{:.3},{:.3}],"hp":{my_hp:.0},"max_hp":1400,"energy":80,"max_energy":100,"grounded":true,"cubes":0,"in_zone":true,
                        "weapons":[{{"kind":"gun","cooldown":0,"ammo":{},"reload":0}},{{"kind":"laser","cooldown":0,"charge":0}}]}},
               "mechs":[{}],"projectiles":[{}],"sounds":[{}],"beams":[{}],"events":[{}],
               "zone":{{"center":[41.3,-72.8],"radius":560,"next_center":[-23.7,2.2],"next_radius":380,"shrink_start":60,"shrink_end":110,"dps":20}}}}"#,
            me[0],
            me[1],
            aim[0],
            aim[1],
            12 - (k % 12),
            mechs.join(","),
            projectiles.join(","),
            sounds.join(","),
            beams.join(","),
            events.join(",")
        );
        let o: Obs = serde_json::from_str(&s).unwrap();
        let a: Action = b.act(&o);
        fnv(&mut h, serde_json::to_string(&a).unwrap().as_bytes());
        let l = a.mv[0].hypot(a.mv[1]).min(1.0);
        if l > 0.0 {
            let n = [me[0] + a.mv[0] / a.mv[0].hypot(a.mv[1]) * l * 1.17, me[1] + a.mv[1] / a.mv[0].hypot(a.mv[1]) * l * 1.17];
            if nav.walkable(n[0], n[1]) {
                me = n;
            }
        }
        aim = a.aim;
    }
    (h, b)
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

/// The practice scene's actions, as the build without the free-for-all side makes them.
const PRACTICE_HASH: u64 = 0xbdede9d60a44ac46;

/// Defaults with the practice uses of the enemy picture, enemy motion and the pincer terms off.
fn no_practice_map() -> Params {
    Params { map_preaim: 0.0, map_expose: 0.0, map_crowd: 0.0, map_reinf: 0.0, pred: 0.0, pincer: 0.0, ..Params::default() }
}

#[test]
fn practice_scene_is_played_as_before() {
    let Some((world, nav)) = load() else { return };
    let (_, b) = practice_scene(world.clone(), nav.clone(), Params::default());
    assert!(!b.ffa_latched(), "detector latched in a practice scene: {}", b.dbg_mode);
    // The break-off goal choice (`cb_*`) is newer than this scene's hash: off here.
    // The later additions off as well: the hash below is of the play before any of them.
    let p = Params {
        cb_away: 0.0,
        cb_path: 0.0,
        cb_known: 0.0,
        cb_corner: 0.0,
        cb_turn: 0.0,
        stay_fight: 0.0,
        cr_max_d: 0.0,
        cr_laser: 0.0,
        cr_strafe: 0.0,
        seek_idle: 0.0,
        lead_adapt: 0.0,
        dodge_after: 9.0,
        reload_laser: 1.0,
        lead_delay: 0.0,
        plan: 0.0,
        ..no_practice_map()
    };
    let (h, _) = practice_scene(world, nav, p);
    eprintln!("practice scene hash {h:#018x}");
    assert_eq!(h, PRACTICE_HASH, "actions differ from the build without the free-for-all side");
}

#[test]
fn practice_scene_with_the_enemy_picture_in_use() {
    let Some((world, nav)) = load() else { return };
    let p = Params { map_preaim: 1.0, map_expose: 15.0, map_crowd: 40.0, map_reinf: 15.0, ..Params::default() };
    let (h, b) = practice_scene(world, nav, p);
    eprintln!("practice scene hash with the picture in use {h:#018x}");
    assert!(!b.ffa_latched(), "detector latched in a practice scene: {}", b.dbg_mode);
}

fn obs(tick: u64, mechs: &str, beams: &str, events: &str) -> Obs {
    let s = format!(
        r#"{{"type":"obs","tick":{tick},"t":{:.2},"alive":16,
           "self":{{"id":0,"pos":[0,0,0],"vel":[0,0,0],"aim":[0,0],"hp":1400,"max_hp":1400,"energy":100,"max_energy":100,"grounded":true,"in_zone":true,
                    "weapons":[{{"kind":"gun","ammo":12}},{{"kind":"laser"}}]}},
           "mechs":[{mechs}],"beams":[{beams}],"events":[{events}],
           "zone":{{"center":[0,0],"radius":500,"next_center":[0,0],"next_radius":380,"shrink_start":100,"shrink_end":150,"dps":20}}}}"#,
        tick as f32 * 0.05
    );
    serde_json::from_str(&s).unwrap()
}

/// Mech 3 at (100, 0) and mech 5 at (100, 80), aimed at each other.
fn pair(hp3: f32, hp5: f32) -> String {
    let y35 = (80.0f32).atan2(0.0);
    format!(
        r#"{{"id":3,"kind":"mech","pos":[100,0,0],"aim":[{y35:.4},0],"hp":{hp3},"max_hp":1400,"weapons":["gun","laser"]}},
           {{"id":5,"kind":"mech","pos":[100,80,0],"aim":[{:.4},0],"hp":{hp5},"max_hp":1400,"weapons":["gun"]}}"#,
        y35 + std::f32::consts::PI
    )
}

const BEAM_3_TO_5: &str = r#"{"by":3,"kind":"laser","path":[[100,3,7],[100,78,4.5]],"age":0.05}"#;

#[test]
fn a_beam_between_two_other_mechs_settles_a_free_for_all() {
    let mut d = Detector::default();
    d.observe(&obs(100, &pair(1400.0, 1400.0), "", ""));
    d.observe(&obs(102, &pair(1400.0, 1100.0), BEAM_3_TO_5, ""));
    d.observe(&obs(104, &pair(1400.0, 1100.0), "", ""));
    assert!(d.latched, "clues {:?}", d.clues);
    assert!(d.reason.contains("beam:3>5"), "{}", d.reason);
    assert_eq!(d.fighting(5, 5.3, 5.0), Some(3));

    // The same beam with nobody hurt (practice: the squad cannot hurt itself) is not enough.
    let mut d = Detector::default();
    for k in 0..30 {
        let beam = if k == 1 { BEAM_3_TO_5 } else { "" };
        d.observe(&obs(100 + 2 * k, &pair(1400.0, 1400.0), beam, ""));
    }
    assert!(!d.latched, "clues {:?}", d.clues);

    // Through the brain, too.
    let mut b = Brain::with_params(&Hello::default(), None, None, Params::default());
    b.act(&obs(100, &pair(1400.0, 1400.0), "", ""));
    b.act(&obs(102, &pair(1400.0, 1100.0), BEAM_3_TO_5, ""));
    b.act(&obs(104, &pair(1400.0, 1100.0), "", ""));
    assert!(b.ffa_latched());
    // A beam aimed at us is never a clue.
    let mut d = Detector::default();
    for k in 0..30 {
        let beam = if k == 1 { r#"{"by":3,"kind":"laser","path":[[100,3,7],[0,0,4.5]],"age":0.05}"# } else { "" };
        d.observe(&obs(100 + 2 * k, &pair(1400.0, 1400.0 - 10.0 * k as f32), beam, ""));
    }
    assert!(!d.latched, "clues {:?}", d.clues);
}

#[test]
fn a_fight_between_two_others_is_joined_on_the_weaker() {
    let mut b = Brain::with_params(&Hello::default(), None, None, Params::default());
    // Settle it: 3 lasers 5.
    b.act(&obs(100, &pair(1300.0, 700.0), "", ""));
    b.act(&obs(102, &pair(1300.0, 450.0), BEAM_3_TO_5, ""));
    assert!(b.ffa_latched());
    // They keep at it: 5 hit again.
    let mut last = Action::default();
    let mut hp5 = 450.0;
    for k in 2..20u64 {
        let beam = if k % 6 == 0 { BEAM_3_TO_5 } else { "" };
        if k % 6 == 0 {
            hp5 -= 30.0;
        }
        last = b.act(&obs(100 + 2 * k, &pair(1300.0, hp5), beam, ""));
    }
    assert_eq!(b.ffa_third_target(), Some(5), "mode {}", b.dbg_mode);
    assert!(b.label.starts_with("ffa-third"), "label {}", b.label);
    let want = (80.0f32).atan2(100.0);
    assert!((last.aim[0] - want).abs() < 0.15, "aim {:?} want {want}", last.aim);
    let e5 = b.dbg_enemies.iter().find(|e| e.id == 5).expect("profile of 5");
    assert_eq!(e5.fighting, Some(3));
}

#[test]
fn nothing_changes_with_the_free_for_all_side_off() {
    // With the side off the brain never latches whatever it sees.
    let p = Params { ffa_detect: 0.0, ..Params::default() };
    let mut b = Brain::with_params(&Hello::default(), None, None, p);
    b.act(&obs(100, &pair(1400.0, 1400.0), "", ""));
    b.act(&obs(102, &pair(1400.0, 1100.0), BEAM_3_TO_5, ""));
    assert!(!b.ffa_latched());
}
