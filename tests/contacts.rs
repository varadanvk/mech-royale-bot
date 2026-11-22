//! The picture of where every enemy is believed to be, on the real map.
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

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A fight on open ground: a target in sight about 70 m away, a second enemy seen briefly and then gone, and
/// (with `heard`) gunshots every few ticks from a third enemy we never see, off to one side.
struct Scene {
    me: [f32; 3],
    target: [f32; 3],
    other: [f32; 3],
    heard_bearing: f32,
}

fn scenes(w: &World, nav: &Nav, n: usize) -> Vec<Scene> {
    let mut r = Rng(0x00c0_ffee);
    let mut v = Vec::new();
    while v.len() < n {
        let p = [w.min_x + r.f() * (w.max_x - w.min_x), w.min_y + r.f() * (w.max_y - w.min_y)];
        if !nav.walkable(p[0], p[1]) {
            continue;
        }
        let a = r.f() * std::f32::consts::TAU;
        let q = [p[0] + a.cos() * 70.0, p[1] + a.sin() * 70.0];
        if !nav.walkable(q[0], q[1]) {
            continue;
        }
        let me = [p[0], p[1], w.ground(p[0], p[1])];
        let target = [q[0], q[1], w.ground(q[0], q[1])];
        if !w.sees([me[0], me[1], me[2] + 8.0], [target[0], target[1], target[2] + 5.0]) {
            continue;
        }
        let b = a + 2.2 + r.f() * 1.8;
        let o = [p[0] + b.cos() * 90.0, p[1] + b.sin() * 90.0];
        let other = [o[0], o[1], w.ground(o[0], o[1])];
        v.push(Scene { me, target, other, heard_bearing: a + 1.4 + r.f() * 3.4 });
    }
    v
}

fn obs(s: &Scene, k: usize, heard: bool) -> Obs {
    let t = 30.0 + k as f32 * 0.1;
    let aim = (s.target[1] - s.me[1]).atan2(s.target[0] - s.me[0]);
    let mech = |id: u32, p: [f32; 3], hp: f32| {
        format!(
            r#"{{"id": {id}, "kind": "mech", "pos": [{}, {}, {}], "vel": [0, 0, 0], "aim": [{}, 0], "hp": {hp}, "max_hp": 1400, "weapons": ["gun"]}}"#,
            p[0],
            p[1],
            p[2],
            aim + std::f32::consts::PI
        )
    };
    let mut mechs = vec![mech(1, s.target, 1100.0)];
    if k < 6 {
        mechs.push(mech(2, s.other, 1300.0));
    }
    let sounds = if heard && k % 3 == 0 { format!(r#"{{"kind": "gun", "bearing": {}, "dist": "mid", "age": 0.05}}"#, s.heard_bearing) } else { String::new() };
    let j = format!(
        r#"{{"tick": {}, "t": {t}, "self": {{"pos": [{}, {}, {}], "aim": [{aim}, 0], "hp": 1300, "max_hp": 1400, "energy": 100, "grounded": true, "in_zone": true,
           "weapons": [{{"kind": "gun", "ammo": 12}}, null]}}, "mechs": [{}], "sounds": [{sounds}], "events": [],
           "zone": {{"center": [{}, {}], "radius": 900, "next_center": [{}, {}], "next_radius": 800, "shrink_start": 400}}}}"#,
        300 + k,
        s.me[0],
        s.me[1],
        s.me[2],
        mechs.join(","),
        s.me[0],
        s.me[1],
        s.me[0],
        s.me[1]
    );
    serde_json::from_str(&j).unwrap()
}

/// The actions for each scene, ticks 0..40, standing where the scene puts us.
fn run(w: &Arc<World>, nav: &Nav, scenes: &[Scene], params: &Params, heard: bool) -> Vec<Vec<Action>> {
    scenes
        .iter()
        .map(|s| {
            let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav.clone()), params.clone());
            (0..40).map(|k| b.act(&obs(s, k, heard))).collect()
        })
        .collect()
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, c| (h ^ c as u64).wrapping_mul(0x0100_0000_01b3))
}

/// Without the enemy picture the bot decides exactly as it did before the picture existed: the actions on a
/// fixed set of fights hash to the value the bot gave then (with the other switches set as they were then).
#[test]
fn without_contacts_decisions_are_unchanged() {
    let Some((w, nav)) = load() else { return };
    let sc = scenes(&w, &nav, 12);
    let params = Params::from_json(&serde_json::json!({ "contacts": 0.0, "counter": 0.0, "heal_lo": 0.0, "heal_hi": 0.7, "arrive": 0.0 }));
    let acts = run(&w, &nav, &sc, &params, true);
    let text: String = acts.iter().flatten().map(|a| serde_json::to_string(a).unwrap()).collect::<Vec<_>>().join("\n");
    let h = fnv(&text);
    eprintln!("actions hash {h:#018x}");
    assert_eq!(h, EXPECTED_HASH, "decisions without `contacts` changed");
}

const EXPECTED_HASH: u64 = 0xb96f_e2fb_ea46_bf31;

/// An enemy we only hear changes where we fight from with the picture on, and not without it.
#[test]
fn heard_enemy_moves_the_fight_position() {
    let Some((w, nav)) = load() else { return };
    let sc = scenes(&w, &nav, 12);
    let moves = |p: &Params, heard: bool| -> Vec<Vec<[f32; 2]>> { run(&w, &nav, &sc, p, heard).into_iter().map(|v| v.into_iter().map(|a| a.mv).collect()).collect() };
    let differ = |a: &[Vec<[f32; 2]>], b: &[Vec<[f32; 2]>]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    let on = Params { contacts: 1.0, ..Params::default() };
    let off = Params { contacts: 0.0, ..Params::default() };
    let d_on = differ(&moves(&on, true), &moves(&on, false));
    let d_off = differ(&moves(&off, true), &moves(&off, false));
    eprintln!("fights whose moves change when a third enemy is heard: {d_on} with the picture, {d_off} without");
    assert!(d_on >= 2, "{d_on}");
    assert_eq!(d_off, 0);
}
