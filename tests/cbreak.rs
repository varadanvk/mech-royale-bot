//! Breaking off from a crowd: the goal lies away from the enemies, never past them.
//! Needs the arena map: set `MAP_DIR`, or keep the kit next to this repo (`../mechfight-kit/arena`).
use std::sync::Arc;
use tinman::brain::{Brain, Params};
use tinman::nav::Nav;
use tinman::proto::{Action, Hello, Obs};
use tinman::world::World;

fn load() -> Option<(Arc<World>, Nav)> {
    let dir = std::env::var("MAP_DIR").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../mechfight-kit/arena").to_string());
    let Ok(world) = World::load(std::path::Path::new(&dir)) else {
        eprintln!("map not found at {dir}; skipping");
        return None;
    };
    let nav = Nav::build(&world, 2.5, 9.0, 1.2, 50.0);
    Some((Arc::new(world), nav))
}

fn off() -> Params {
    Params { cb_away: 0.0, cb_path: 0.0, cb_known: 0.0, cb_corner: 0.0, cb_turn: 0.0, hold_fire: 0.0, ..Params::default() }
}

fn d2(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// One observation: us at `me` with `hp` of 2061, every enemy in `foes` (id, position, hp) aimed at us.
fn obs(k: u64, me: [f32; 3], hp: f32, energy: f32, foes: &[(u32, [f32; 3], f32)], events: &str) -> Obs {
    let mechs: Vec<String> = foes
        .iter()
        .map(|(id, p, h)| {
            let yaw = (me[1] - p[1]).atan2(me[0] - p[0]);
            format!(
                r#"{{"id":{id},"kind":"mech","pos":[{:.2},{:.2},{:.2}],"vel":[0,0,0],"aim":[{yaw:.3},0],"hp":{h:.0},"max_hp":1400,"cubes":0,"weapons":["gun","laser"],"charging":false}}"#,
                p[0], p[1], p[2]
            )
        })
        .collect();
    let s = format!(
        r#"{{"type":"obs","tick":{},"t":{:.2},"alive":12,
           "self":{{"id":0,"pos":[{:.2},{:.2},{:.2}],"vel":[0,0,0],"aim":[-2.25,0],"hp":{hp:.0},"max_hp":2061,"energy":{energy:.0},"max_energy":100,"grounded":true,"cubes":0,"in_zone":true,
                    "weapons":[{{"kind":"gun","cooldown":0,"ammo":12,"reload":0}},{{"kind":"laser","cooldown":0,"charge":0}}]}},
           "mechs":[{}],"projectiles":[],"sounds":[],"beams":[],"events":[{events}],
           "zone":{{"center":[41.3,-72.8],"radius":560,"next_center":[92.7,43.0],"next_radius":380,"shrink_start":160,"shrink_end":210,"dps":20}}}}"#,
        2440 + 2 * k,
        122.0 + k as f32 * 0.1,
        me[0],
        me[1],
        me[2],
        mechs.join(",")
    );
    serde_json::from_str(&s).unwrap()
}

/// A short warm-up seeing every enemy, then the moment the break-off starts: one enemy 43 m away, a group of
/// four 120-160 m beyond it on the same side. Returns the goal and the first actions of the break-off.
/// Moved by `off` (heights from the map).
fn crowd_scene_at(world: Arc<World>, nav: Nav, params: Params, off: [f32; 2]) -> (Option<[f32; 2]>, Vec<Action>) {
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":910,"map":{"dir":""},"config":{}}"#).unwrap();
    let at = |x: f32, y: f32, z: f32| if off == [0.0, 0.0] { [x, y, z] } else { [x + off[0], y + off[1], world.ground(x + off[0], y + off[1])] };
    let me = at(280.17, 68.85, 4.84);
    let foes = [
        (7u32, at(253.1, 35.3, 6.1), 1110.0f32),
        (9, at(212.8, -46.2, 5.3), 1456.0),
        (10, at(164.7, 28.6, 5.9), 1568.0),
        (13, at(168.6, -41.2, 5.5), 1680.0),
        (15, at(164.3, -45.4, 5.5), 1512.0),
    ];
    let mut b = Brain::with_params(&hello, Some(world.clone()), Some(nav), params);
    let mut acts = Vec::new();
    let mut goal = None;
    for k in 0..12u64 {
        let ev = if k % 3 == 0 { r#"{"type":"damaged","amount":45,"weapon":"gun","bearing":0.9,"by":7}"# } else { "" };
        let a = b.act(&obs(k, me, 696.0, 78.0, &foes, ev));
        if b.label == "cbreak" {
            goal = goal.or(b.break_off_goal());
            acts.push(a);
        }
    }
    (goal, acts)
}

/// The goal is away: farther from the enemies' centre and from the nearest, not within 60 degrees of it.
/// Returns how much the goal without the switches headed toward the nearest enemy (cosine).
fn check_away(world: Arc<World>, nav: Nav, sh: [f32; 2]) -> f32 {
    let me = [280.17 + sh[0], 68.85 + sh[1]];
    let e7 = [253.1 + sh[0], 35.3 + sh[1]];
    let to7 = [(e7[0] - me[0]) / d2(e7, me), (e7[1] - me[1]) / d2(e7, me)];
    let cos7 = |g: [f32; 2]| {
        let dir = [(g[0] - me[0]) / d2(g, me).max(1e-3), (g[1] - me[1]) / d2(g, me).max(1e-3)];
        dir[0] * to7[0] + dir[1] * to7[1]
    };
    let (before, _) = crowd_scene_at(world.clone(), nav.clone(), off(), sh);
    let (goal, acts) = crowd_scene_at(world, nav, Params::default(), sh);
    eprintln!("moved {sh:?}: goal without the switches {before:?}, now {goal:?} ({} break-off decisions)", acts.len());
    let g = goal.expect("a goal away from the enemies exists here");
    let cen = [(253.1 + 212.8 + 164.7 + 168.6 + 164.3) / 5.0 + sh[0], (35.3 - 46.2 + 28.6 - 41.2 - 45.4) / 5.0 + sh[1]];
    assert!(d2(g, cen) > d2(me, cen), "goal {g:?} not farther from the enemies");
    assert!(d2(g, e7) > d2(me, e7), "goal {g:?} not farther from the nearest enemy");
    assert!(cos7(g) < 0.5, "goal {g:?} within 60 degrees of the nearest enemy");
    before.map_or(-1.0, cos7)
}

#[test]
fn break_off_goes_away_from_the_crowd() {
    let Some((world, nav)) = load() else { return };
    check_away(world.clone(), nav.clone(), [0.0, 0.0]);
    // Here the nearest hidden cell lies past the nearest enemy, toward the group.
    let c = check_away(world, nav, [-40.0, -160.0]);
    assert!(c > 0.5, "the scene no longer tests a goal past the nearest enemy ({c})");
}

#[test]
fn no_way_out_means_no_break_off() {
    let Some((world, nav)) = load() else { return };
    // Enemies close on every side: nowhere away from all of them.
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":911,"map":{"dir":""},"config":{}}"#).unwrap();
    let mut b = Brain::with_params(&hello, Some(world), Some(nav.clone()), Params::default());
    let mut me = [280.17f32, 68.85, 4.84];
    let foes: Vec<(u32, [f32; 3], f32)> = (0..6)
        .map(|i| {
            let a = i as f32 * std::f32::consts::TAU / 6.0;
            (3 + i as u32, [me[0] + a.cos() * 35.0, me[1] + a.sin() * 35.0, 5.0], 1300.0)
        })
        .collect();
    for k in 0..12u64 {
        let ev = if k % 3 == 0 { r#"{"type":"damaged","amount":45,"weapon":"gun","bearing":0.9,"by":3}"# } else { "" };
        let a = b.act(&obs(k, me, 696.0, 78.0, &foes, ev));
        assert_ne!(b.label, "cbreak", "broke off with enemies on every side (goal {:?})", b.break_off_goal());
        if a.boost {
            let l = a.mv[0].hypot(a.mv[1]).max(1e-3);
            let u = [a.mv[0] / l, a.mv[1] / l];
            for (_, f, _) in &foes {
                let t = [(f[0] - me[0]) / 35.0, (f[1] - me[1]) / 35.0];
                assert!(u[0] * t[0] + u[1] * t[1] <= 0.5, "boosting toward an enemy ({})", b.label);
            }
        }
        // Walk as told (the enemies stay put around where we started).
        let l = a.mv[0].hypot(a.mv[1]);
        if l > 1e-3 {
            let n = [me[0] + a.mv[0] / l * 1.17, me[1] + a.mv[1] / l * 1.17];
            if nav.walkable(n[0], n[1]) {
                me = [n[0], n[1], me[2]];
            }
        }
    }
}

/// Hash of every action in the crowd scene (moved by `sh`), the tagged and the moved one, and the
/// no-way-out ring.
fn scene_hash(world: Arc<World>, nav: Nav, params: Params) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for sh in [[0.0f32, 0.0], [-40.0, -160.0]] {
        let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":910,"map":{"dir":""},"config":{}}"#).unwrap();
        let at = |x: f32, y: f32, z: f32| if sh == [0.0, 0.0] { [x, y, z] } else { [x + sh[0], y + sh[1], world.ground(x + sh[0], y + sh[1])] };
        let mut me = at(280.17, 68.85, 4.84);
        let mut foes = vec![
            (7u32, at(253.1, 35.3, 6.1), 1110.0f32),
            (9, at(212.8, -46.2, 5.3), 1456.0),
            (10, at(164.7, 28.6, 5.9), 1568.0),
            (13, at(168.6, -41.2, 5.5), 1680.0),
            (15, at(164.3, -45.4, 5.5), 1512.0),
        ];
        let mut b = Brain::with_params(&hello, Some(world.clone()), Some(nav.clone()), params.clone());
        for k in 0..60u64 {
            let ev = if k % 3 == 0 { r#"{"type":"damaged","amount":45,"weapon":"gun","bearing":0.9,"by":7}"# } else { "" };
            let a = b.act(&obs(k, me, 696.0 - k as f32 * 4.0, 78.0, &foes, ev));
            for c in serde_json::to_string(&a).unwrap().bytes() {
                h ^= c as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            let l = a.mv[0].hypot(a.mv[1]);
            if l > 1e-3 {
                let n = [me[0] + a.mv[0] / l * 1.17, me[1] + a.mv[1] / l * 1.17];
                if nav.walkable(n[0], n[1]) {
                    me = [n[0], n[1], me[2]];
                }
            }
            // Enemy 7 closes in.
            let e = &mut foes[0].1;
            let d = (me[0] - e[0]).hypot(me[1] - e[1]).max(1.0);
            e[0] += (me[0] - e[0]) / d * 0.6;
            e[1] += (me[1] - e[1]) / d * 0.6;
        }
    }
    h
}

/// The scene hash as the build without the break-off goal choice makes it.
const CROWD_HASH: u64 = 0x1e21c7d7a3d73ddf;

#[test]
fn switches_off_play_as_before() {
    let Some((world, nav)) = load() else { return };
    let h = scene_hash(world.clone(), nav.clone(), off());
    eprintln!("crowd scene hash (switches off) {h:#018x}, on {:#018x}", scene_hash(world, nav, Params::default()));
    assert_eq!(h, CROWD_HASH, "actions differ from the build without the break-off goal choice");
}

/// Moved scenes where the goal without the switches heads within 60 degrees of the nearest enemy (run by hand:
/// `cargo test --release --test cbreak scan -- --ignored --nocapture`).
#[test]
#[ignore]
fn scan() {
    let Some((world, nav)) = load() else { return };
    for i in -6..=6 {
        for j in -6..=6 {
            let sh = [i as f32 * 40.0, j as f32 * 40.0];
            let me = [280.17 + sh[0], 68.85 + sh[1]];
            let e7 = [253.1 + sh[0], 35.3 + sh[1]];
            if !nav.walkable(me[0], me[1]) {
                continue;
            }
            let (g0, _) = crowd_scene_at(world.clone(), nav.clone(), off(), sh);
            let Some(g0) = g0 else { continue };
            let to7 = [(e7[0] - me[0]) / d2(e7, me), (e7[1] - me[1]) / d2(e7, me)];
            let dir = [(g0[0] - me[0]) / d2(g0, me).max(1e-3), (g0[1] - me[1]) / d2(g0, me).max(1e-3)];
            if d2(g0, me) > 3.0 && dir[0] * to7[0] + dir[1] * to7[1] > 0.5 {
                let (g1, _) = crowd_scene_at(world.clone(), nav.clone(), Params::default(), sh);
                eprintln!("off {sh:?}: before {g0:?} now {g1:?}");
            }
        }
    }
}
