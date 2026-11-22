//! Holding fire on distant mechs: switched off, the fight plays as before.
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

/// Action hashes of the fight scene from 30 s and from 125 s as the build without `hold_fire` makes them.
const DUEL_30: u64 = 0xf02832ac35461ead;
const DUEL_125: u64 = 0x3ca56fb017fc86d1;

#[test]
fn switch_off_plays_as_before() {
    let Some((world, nav)) = load() else { return };
    let off = params(r#"{"hold_fire": 0}"#);
    assert_eq!(duel(world.clone(), nav.clone(), off.clone(), 30.0, 80).1, DUEL_30, "fight from 30 s differs");
    assert_eq!(duel(world, nav, off, 125.0, 80).1, DUEL_125, "fight from 125 s differs");
}
