//! Dueling with a better picture of each enemy (hit points, danger, help on its way, weapon state), on the real
//! map. Needs the arena map: set `MAP_DIR`, or keep the kit next to this repo (`../mechfight-kit/arena`).
use std::f32::consts::TAU;
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

/// One enemy in a scene: where it stands, its health, whether it is aimed at us and shooting, how long it stays
/// listed (decisions), and whether we hit it.
#[derive(Clone)]
struct Foe {
    id: u32,
    pos: [f32; 3],
    hp: f32,
    aimed: bool,
    shoots: bool,
    listed_for: usize,
    we_hit: bool,
}

#[derive(Clone)]
struct Scene {
    me: [f32; 3],
    hp: f32,
    foes: Vec<Foe>,
}

/// Kinds: 0 a lone weakened enemy that has not noticed us; 1 the same with a second enemy coming for us from
/// the other side, out of reach of the old duel checks; 2 a lone enemy we trade hits with that then leaves view;
/// 3 two enemies shooting from different sides.
fn scenes(w: &World, nav: &Nav, n: usize, only: Option<usize>) -> Vec<Scene> {
    let mut r = Rng(0x0d0e_1333);
    let mut v = Vec::new();
    let mut i = 0usize;
    while v.len() < n {
        let p = [w.min_x + 150.0 + r.f() * (w.max_x - w.min_x - 300.0), w.min_y + 150.0 + r.f() * (w.max_y - w.min_y - 300.0)];
        let kind = only.unwrap_or(i % 4);
        let base = r.f() * TAU;
        if !nav.walkable(p[0], p[1]) {
            continue;
        }
        let me = [p[0], p[1], w.ground(p[0], p[1])];
        let specs: Vec<(f32, f32, f32, bool, bool, usize, bool)> = match kind {
            0 => vec![(0.0, 90.0, 500.0, false, false, 99, false)],
            1 => vec![(0.0, 90.0, 900.0, false, false, 99, false), (PI_F, 190.0, 1400.0, true, false, 99, false)],
            2 => vec![(0.0, 70.0, 1300.0, true, true, 15, true)],
            _ => vec![(0.0, 70.0, 1300.0, true, true, 99, true), (2.2, 90.0, 1300.0, true, true, 99, false)],
        };
        let mut foes = Vec::new();
        for (k, (da, d, hp, aimed, shoots, listed_for, we_hit)) in specs.into_iter().enumerate() {
            let a = base + da;
            let q = [p[0] + a.cos() * d, p[1] + a.sin() * d];
            if !nav.walkable(q[0], q[1]) {
                break;
            }
            let pos = [q[0], q[1], w.ground(q[0], q[1])];
            if k == 0 && !w.sees([me[0], me[1], me[2] + 8.0], [pos[0], pos[1], pos[2] + 5.0]) {
                break;
            }
            foes.push(Foe { id: 1 + k as u32, pos, hp, aimed, shoots, listed_for, we_hit });
        }
        let want = match kind {
            0 | 2 => 1,
            _ => 2,
        };
        if foes.len() != want {
            continue;
        }
        i += 1;
        v.push(Scene { me, hp: 1400.0, foes });
    }
    v
}

const PI_F: f32 = std::f32::consts::PI;

fn obs(s: &Scene, k: usize) -> Obs {
    let t = 40.0 + k as f32 * 0.1;
    let mechs: Vec<String> = s
        .foes
        .iter()
        .filter(|f| k < f.listed_for)
        .map(|f| {
            let to_me = (s.me[1] - f.pos[1]).atan2(s.me[0] - f.pos[0]);
            let aim = if f.aimed { to_me } else { to_me + 2.0 };
            let hp = if f.we_hit { f.hp - 45.0 * (k / 3) as f32 } else { f.hp };
            format!(
                r#"{{"id": {}, "kind": "mech", "pos": [{}, {}, {}], "vel": [0, 0, 0], "aim": [{aim}, 0], "hp": {hp}, "max_hp": 1400, "weapons": ["gun"]}}"#,
                f.id, f.pos[0], f.pos[1], f.pos[2]
            )
        })
        .collect();
    let mut events: Vec<String> = s
        .foes
        .iter()
        .filter(|f| f.shoots && k % 4 == (f.id as usize) % 4 && k < f.listed_for + 10)
        .map(|f| format!(r#"{{"type": "damaged", "amount": 30, "by": {}, "bearing": {}, "weapon": "gun"}}"#, f.id, (f.pos[1] - s.me[1]).atan2(f.pos[0] - s.me[0])))
        .collect();
    events.extend(s.foes.iter().filter(|f| f.we_hit && k % 3 == 0 && k > 0).map(|f| format!(r#"{{"type": "hit", "target": {}, "amount": 45, "weapon": "gun"}}"#, f.id)));
    let sounds: Vec<String> = s
        .foes
        .iter()
        .filter(|f| f.shoots && k % 2 == 0)
        .map(|f| format!(r#"{{"kind": "gun", "bearing": {}, "dist": "mid", "age": 0.05}}"#, (f.pos[1] - s.me[1]).atan2(f.pos[0] - s.me[0])))
        .collect();
    let aim = s.foes.first().map_or(0.0, |f| (f.pos[1] - s.me[1]).atan2(f.pos[0] - s.me[0]));
    let j = format!(
        r#"{{"tick": {}, "t": {t}, "alive": 16, "self": {{"pos": [{}, {}, {}], "aim": [{aim}, 0], "hp": {}, "max_hp": 1400, "energy": 100, "grounded": true, "in_zone": true,
           "weapons": [{{"kind": "gun", "ammo": 12, "stats": {{"damage": 45, "cooldown": 0.24, "mag": 12, "reload": 2.4, "range": 230}}}}, null]}}, "mechs": [{}], "sounds": [{}], "events": [{}],
           "zone": {{"center": [{}, {}], "radius": 900, "next_center": [{}, {}], "next_radius": 800, "shrink_start": 400}}}}"#,
        400 + 2 * k,
        s.me[0],
        s.me[1],
        s.me[2],
        s.hp,
        mechs.join(","),
        sounds.join(","),
        events.join(","),
        s.me[0],
        s.me[1],
        s.me[0],
        s.me[1]
    );
    serde_json::from_str(&j).unwrap()
}

fn run(w: &Arc<World>, nav: &Nav, scenes: &[Scene], params: &Params, n: usize) -> (Vec<Vec<Action>>, u32) {
    let mut duels = 0;
    let acts = scenes
        .iter()
        .map(|s| {
            let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav.clone()), params.clone());
            let v = (0..n).map(|k| b.act(&obs(s, k))).collect();
            duels += b.duel_n;
            v
        })
        .collect();
    (acts, duels)
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, c| (h ^ c as u64).wrapping_mul(0x0100_0000_01b3))
}

fn hash(acts: &[Vec<Action>]) -> u64 {
    fnv(&acts.iter().flatten().map(|a| serde_json::to_string(a).unwrap()).collect::<Vec<_>>().join("\n"))
}

/// The new inferences all switched off: the same actions as before they existed, duels included.
fn old() -> Params {
    Params::from_json(&serde_json::json!({ "inf_hp": 0.0, "inf_danger": 0.0, "duel_eta": 0.0, "inf_weapon": 0.0, "inf_hunt": 0.0 }))
}

#[test]
fn without_new_inferences_decisions_are_unchanged() {
    let Some((w, nav)) = load() else { return };
    let sc = scenes(&w, &nav, 16, None);
    let (acts, duels) = run(&w, &nav, &sc, &old(), 40);
    let h = hash(&acts);
    eprintln!("actions hash {h:#018x}, duels {duels}");
    assert_eq!(h, BASE, "decisions with the new inferences off changed");
}

const BASE: u64 = 0x654b_f20c_ee53_4e06;

/// A weakened enemy alone, with a second one coming for us from the other side that could be shooting at us
/// well before the first is dead: no duel with the help-on-its-way check, a duel without it.
#[test]
fn duel_refused_when_help_arrives_before_the_kill() {
    let Some((w, nav)) = load() else { return };
    let sc = scenes(&w, &nav, 6, Some(1));
    let (_, on) = run(&w, &nav, &sc, &Params::default(), 20);
    let (_, off) = run(&w, &nav, &sc, &Params { duel_eta: 0.0, ..Params::default() }, 20);
    eprintln!("duels {on} of {} with the check, {off} without", sc.len());
    assert_eq!(on, 0);
    assert_eq!(off as usize, sc.len());
}

/// A lone weakened enemy that has shown nothing dangerous, nobody else anywhere: dueled, with its estimate
/// exact, its danger no more than the prior and no help able to arrive.
#[test]
fn duel_started_on_lone_weak_harmless_enemy() {
    let Some((w, nav)) = load() else { return };
    let sc = scenes(&w, &nav, 6, Some(0));
    let mut duels = 0;
    for s in &sc {
        let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav.clone()), Params::default());
        for k in 0..20 {
            b.act(&obs(s, k));
            if k == 0 {
                let e = b.dbg_enemies[0];
                assert_eq!(e[1], 500.0);
                assert!(e[3] <= 0.5 + 1e-6, "danger {}", e[3]);
                assert!(e[4] >= 0.0);
            }
        }
        duels += b.duel_n;
        assert_eq!(b.duel_abort_n, 0);
    }
    assert_eq!(duels as usize, sc.len());
}

/// Hit points of an enemy we keep hitting after it leaves view fall by what we deal it.
#[test]
fn estimate_drops_by_damage_dealt_out_of_view() {
    let Some((w, nav)) = load() else { return };
    let sc = scenes(&w, &nav, 4, Some(2));
    for s in &sc {
        let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav.clone()), Params::default());
        let mut last = 0.0;
        for k in 0..24 {
            b.act(&obs(s, k));
            let e = b.dbg_enemies.iter().find(|e| e[0] == 1.0).copied().unwrap();
            if k == 14 {
                last = e[1];
            }
            if k == 24 - 1 {
                // Listed until decision 14; hits at 15, 18 and 21 after that, 45 each.
                assert!((last - e[1] - 135.0).abs() < 0.5, "{last} -> {}", e[1]);
                assert!(e[2] > 0.0);
            }
        }
    }
}

/// Danger: an enemy that hits us often while aimed at us and moves across our line rates above one that
/// stands aiming wide and never hits; a reload heard from an enemy shows as its weapon being down.
#[test]
fn danger_and_weapon_state_are_read() {
    let mk = |aim_off: f32, vy: f32, hits: bool, reload: bool, k: usize| -> Obs {
        let t = 10.0 + k as f32 * 0.1;
        let ev = if hits && k % 2 == 0 { r#"{"type": "damaged", "amount": 45, "by": 1, "bearing": 0, "weapon": "gun"}"# } else { "" };
        let snd = if reload && k == 5 { r#"{"kind": "reload", "bearing": 0, "dist": "mid", "age": 0}"# } else { "" };
        let j = format!(
            r#"{{"tick": {}, "t": {t}, "self": {{"pos": [0,0,0], "aim": [0,0], "hp": 1400, "max_hp": 1400, "energy": 100, "grounded": true, "in_zone": true,
                "weapons": [{{"kind": "gun", "ammo": 12}}, null]}},
              "mechs": [{{"id": 1, "kind": "mech", "pos": [100,0,0], "vel": [0,{vy},0], "aim": [{}, 0], "hp": 1400, "max_hp": 1400, "weapons": ["gun"]}}],
              "sounds": [{snd}], "events": [{ev}],
              "zone": {{"center": [0,0], "radius": 900, "next_center": [0,0], "next_radius": 800, "shrink_start": 400}}}}"#,
            200 + 2 * k,
            std::f32::consts::PI + aim_off
        );
        serde_json::from_str(&j).unwrap()
    };
    let rate = |aim_off: f32, vy: f32, hits: bool, reload: bool| -> (f32, f32) {
        let mut b = Brain::with_params(&Hello::default(), None, None, Params::default());
        let mut ws = 0.0;
        for k in 0..40 {
            b.act(&mk(aim_off, vy, hits, reload, k));
            if k == 6 {
                ws = b.dbg_enemies[0][5];
            }
        }
        (b.dbg_enemies[0][3], ws)
    };
    let (sharp, _) = rate(0.0, 8.0, true, false);
    let (dull, ws) = rate(0.45, 0.0, false, true);
    eprintln!("danger {sharp:.2} vs {dull:.2}");
    assert!(sharp > 0.65 && dull < 0.4, "{sharp} {dull}");
    assert_eq!(ws, 1.0);
}
