//! Hiding after a set time: switched off, the fight plays as before; on, it is unchanged before the stop and the den is held quietly after it.
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

fn params(json: &str) -> Params {
    Params::from_json(&serde_json::from_str(json).unwrap())
}

/// One observation at time `t`: us at `me`, every enemy in `foes` (id, position, hp) aimed at us.
fn obs(k: u64, t: f32, me: [f32; 3], hp: f32, foes: &[(u32, [f32; 3], f32)], events: &str) -> Obs {
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
           "self":{{"id":0,"pos":[{:.2},{:.2},{:.2}],"vel":[0,0,0],"aim":[-2.25,0],"hp":{hp:.0},"max_hp":2061,"energy":80,"max_energy":100,"grounded":true,"cubes":0,"in_zone":true,
                    "weapons":[{{"kind":"gun","cooldown":0,"ammo":12,"reload":0}},{{"kind":"laser","cooldown":0,"charge":0}}]}},
           "mechs":[{}],"projectiles":[],"sounds":[],"beams":[],"events":[{events}],
           "zone":{{"center":[41.3,-72.8],"radius":560,"next_center":[92.7,43.0],"next_radius":380,"shrink_start":160,"shrink_end":210,"dps":20}}}}"#,
        2 * k + (t * 10.0) as u64,
        t + k as f32 * 0.1,
        me[0],
        me[1],
        me[2],
        mechs.join(",")
    );
    serde_json::from_str(&s).unwrap()
}

/// A fight from time `t0`: one enemy 43 m away hitting us now and then and closing in, a second 130 m out.
/// Returns every action and label, and a hash of the actions.
fn duel(world: Arc<World>, nav: Nav, params: Params, t0: f32, n: u64) -> (Vec<(Action, &'static str)>, u64) {
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":910,"map":{"dir":""},"config":{}}"#).unwrap();
    let mut me = [280.17, 68.85, 4.84];
    let mut foes = vec![(7u32, [253.1, 35.3, 6.1], 1110.0f32), (9, [212.8, -46.2, 5.3], 1456.0)];
    let mut b = Brain::with_params(&hello, Some(world), Some(nav.clone()), params);
    let mut h = 0xcbf29ce484222325u64;
    let mut out = Vec::new();
    for k in 0..n {
        let ev = if k % 5 == 0 { r#"{"type":"damaged","amount":45,"weapon":"gun","bearing":0.9,"by":7}"# } else { "" };
        let a = b.act(&obs(k, t0, me, 1800.0 - k as f32 * 3.0, &foes, ev));
        for c in serde_json::to_string(&a).unwrap().bytes() {
            h ^= c as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let l = a.mv[0].hypot(a.mv[1]);
        if l > 1e-3 {
            let n = [me[0] + a.mv[0] * 1.17, me[1] + a.mv[1] * 1.17];
            if nav.walkable(n[0], n[1]) {
                me = [n[0], n[1], me[2]];
            }
        }
        let e = &mut foes[0].1;
        let d = (me[0] - e[0]).hypot(me[1] - e[1]).max(1.0);
        e[0] += (me[0] - e[0]) / d * 0.3;
        e[1] += (me[1] - e[1]) / d * 0.3;
        out.push((a, b.label));
    }
    (out, h)
}

/// Action hashes of the fight scene from 30 s and from 125 s as the build without `stop_t` makes them.
const DUEL_30: u64 = 0xf02832ac35461ead;
const DUEL_125: u64 = 0x3ca56fb017fc86d1;

#[test]
fn switch_off_plays_as_before() {
    let Some((world, nav)) = load() else { return };
    let off = params(r#"{"stop_t": 0}"#);
    assert_eq!(duel(world.clone(), nav.clone(), off.clone(), 30.0, 80).1, DUEL_30, "fight from 30 s differs");
    assert_eq!(duel(world, nav, off, 125.0, 80).1, DUEL_125, "fight from 125 s differs");
}


/// From `t0`, a mech 130 m off aimed at us that never comes close: walk quietly toward a den for 15 s, then
/// (moved onto it) stay there. Returns the actions on the way, the den, and the actions on it.
fn hide_from(t0: f32, json: &str) -> Option<(Vec<(Action, &'static str)>, [f32; 2], Vec<(Action, &'static str)>)> {
    let (world, nav) = load()?;
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":910,"map":{"dir":""},"config":{}}"#).unwrap();
    let mut b = Brain::with_params(&hello, Some(world.clone()), Some(nav.clone()), params(json));
    let mut me = [280.17, 68.85, 4.84];
    let foe = [(9u32, [212.8f32, -46.2, 5.3], 1456.0f32)];
    let mut way = Vec::new();
    let mut den = None;
    for k in 0..150u64 {
        let a = b.act(&obs(k, t0, me, 1800.0, &foe, ""));
        let d = b.den_goal();
        assert!(den.is_none() || d == den, "den moved from {den:?} to {d:?}");
        den = d;
        let n = [me[0] + a.mv[0] * 1.17, me[1] + a.mv[1] * 1.17];
        if nav.walkable(n[0], n[1]) {
            me = [n[0], n[1], world.ground(n[0], n[1])];
        }
        way.push((a, b.label));
    }
    let den = den.expect("no den chosen");
    let me = [den[0], den[1], world.ground(den[0], den[1])];
    let mut there = Vec::new();
    for k in 150..250u64 {
        let a = b.act(&obs(k, t0, me, 1800.0, &foe, ""));
        there.push((a, b.label));
    }
    Some((way, den, there))
}

#[test]
fn den_is_held_quietly_after_the_stop() {
    let Some((way, den, there)) = hide_from(125.0, "{}") else { return };
    let foe = [212.8f32, -46.2];
    // Next circle: centre (92.7, 43.0), radius 380.
    assert!((den[0] - 92.7).hypot(den[1] - 43.0) < 380.0 - 15.0, "den {den:?} outside the next circle");
    assert!((den[0] - foe[0]).hypot(den[1] - foe[1]) > 45.0, "den {den:?} next to the mech");
    for (a, label) in way.iter().chain(there.iter()) {
        assert_eq!(*label, "den");
        assert!(!a.boost || a.jump, "boosted other than to get unstuck");
        assert!(a.mv[0].hypot(a.mv[1]) <= 4.6 / 11.7 + 1e-3, "walked at {:?}", a.mv);
        assert!(!a.fire[0] && !a.fire[1], "shot at a mech beyond 45 m");
    }
    assert!(way.iter().filter(|(a, _)| a.mv[0].hypot(a.mv[1]) > 0.3).count() > 100, "not walking to the den");
    assert!(there[5..].iter().all(|(a, _)| a.mv == [0.0, 0.0]), "not staying in the den");
}

#[test]
fn fight_is_unchanged_before_the_stop() {
    let Some((world, nav)) = load() else { return };
    assert_eq!(duel(world, nav, params("{}"), 30.0, 80).1, DUEL_30, "fight from 30 s differs");
}
