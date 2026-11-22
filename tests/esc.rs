//! Escaping a crowd and dueling a lone enemy, on the real map.
//! Needs the arena map: set `MAP_DIR`, or keep the kit next to this repo (`../mechfight-kit/arena`).
use std::f32::consts::TAU;
use std::sync::Arc;
use tinman::brain::{Brain, Params};
use tinman::sight::{Foe as SFoe, SightTable, Zone};
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

/// One enemy in a scene: where it stands, its health, and whether it is aimed at us and shooting.
#[derive(Clone)]
struct Foe {
    id: u32,
    pos: [f32; 3],
    hp: f32,
    aimed: bool,
    shoots: bool,
}

/// Us on open ground with a few enemies round us; `hp` is ours.
#[derive(Clone)]
struct Scene {
    me: [f32; 3],
    hp: f32,
    foes: Vec<Foe>,
}

/// Mixed situations: a crowd of three aimed at us (some shooting), two shooting from different sides, a single
/// fight, and a lone weakened enemy that has not noticed us.
fn scenes(w: &World, nav: &Nav, n: usize) -> Vec<Scene> {
    let mut r = Rng(0x0e5c_a9e5);
    let mut v = Vec::new();
    while v.len() < n {
        let p = [w.min_x + 100.0 + r.f() * (w.max_x - w.min_x - 200.0), w.min_y + 100.0 + r.f() * (w.max_y - w.min_y - 200.0)];
        if !nav.walkable(p[0], p[1]) {
            continue;
        }
        let me = [p[0], p[1], w.ground(p[0], p[1])];
        let kind = v.len() % 4;
        let (count, hp) = match kind {
            0 => (3, 1100.0),
            1 => (2, 1450.0),
            2 => (1, 1300.0),
            _ => (1, 1400.0),
        };
        let base = r.f() * TAU;
        let mut foes = Vec::new();
        for k in 0..count {
            let a = if kind == 1 { base + k as f32 * 2.2 } else { base + (k as f32 - 1.0) * 0.5 };
            let d = if kind == 3 { 90.0 } else { 60.0 + 20.0 * k as f32 };
            let q = [p[0] + a.cos() * d, p[1] + a.sin() * d];
            if !nav.walkable(q[0], q[1]) {
                break;
            }
            let pos = [q[0], q[1], w.ground(q[0], q[1])];
            if !w.sees([me[0], me[1], me[2] + 8.0], [pos[0], pos[1], pos[2] + 5.0]) {
                break;
            }
            let hp = if kind == 3 { 500.0 } else { 1300.0 };
            foes.push(Foe { id: 1 + k as u32, pos, hp, aimed: kind != 3, shoots: kind <= 1 });
        }
        if foes.len() != count {
            continue;
        }
        v.push(Scene { me, hp, foes });
    }
    v
}

fn obs(s: &Scene, k: usize) -> Obs {
    let t = 40.0 + k as f32 * 0.1;
    let mechs: Vec<String> = s
        .foes
        .iter()
        // The third of a crowd is seen only at first, then known only from the picture.
        .filter(|f| !(f.id == 3 && k >= 10))
        .map(|f| {
            let to_me = (s.me[1] - f.pos[1]).atan2(s.me[0] - f.pos[0]);
            let aim = if f.aimed { to_me } else { to_me + 2.0 };
            format!(
                r#"{{"id": {}, "kind": "mech", "pos": [{}, {}, {}], "vel": [0, 0, 0], "aim": [{aim}, 0], "hp": {}, "max_hp": 1400, "weapons": ["gun"]}}"#,
                f.id, f.pos[0], f.pos[1], f.pos[2], f.hp
            )
        })
        .collect();
    let events: Vec<String> = s
        .foes
        .iter()
        .filter(|f| f.shoots && k % 4 == (f.id as usize) % 4 && !(f.id == 3 && k >= 10))
        .map(|f| format!(r#"{{"type": "damaged", "amount": 30, "by": {}, "bearing": {}, "weapon": "gun"}}"#, f.id, (f.pos[1] - s.me[1]).atan2(f.pos[0] - s.me[0])))
        .collect();
    let aim = s.foes.first().map_or(0.0, |f| (f.pos[1] - s.me[1]).atan2(f.pos[0] - s.me[0]));
    let j = format!(
        r#"{{"tick": {}, "t": {t}, "alive": 16, "self": {{"pos": [{}, {}, {}], "aim": [{aim}, 0], "hp": {}, "max_hp": 1512, "energy": 100, "grounded": true, "in_zone": true,
           "weapons": [{{"kind": "gun", "ammo": 12}}, null]}}, "mechs": [{}], "sounds": [], "events": [{}],
           "zone": {{"center": [{}, {}], "radius": 900, "next_center": [{}, {}], "next_radius": 800, "shrink_start": 400}}}}"#,
        400 + 2 * k,
        s.me[0],
        s.me[1],
        s.me[2],
        s.hp,
        mechs.join(","),
        events.join(","),
        s.me[0],
        s.me[1],
        s.me[0],
        s.me[1]
    );
    serde_json::from_str(&j).unwrap()
}

/// The actions for each scene, decisions 0..40, standing where the scene puts us.
fn run(w: &Arc<World>, nav: &Nav, scenes: &[Scene], params: &Params) -> Vec<Vec<Action>> {
    scenes
        .iter()
        .map(|s| {
            let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav.clone()), params.clone());
            (0..40).map(|k| b.act(&obs(s, k))).collect()
        })
        .collect()
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, c| (h ^ c as u64).wrapping_mul(0x0100_0000_01b3))
}

fn hash(acts: &[Vec<Action>]) -> u64 {
    fnv(&acts.iter().flatten().map(|a| serde_json::to_string(a).unwrap()).collect::<Vec<_>>().join("\n"))
}

/// With the new modes switched off the bot decides exactly as before they existed, with the enemy picture on
/// and off.
#[test]
fn without_new_modes_decisions_are_unchanged() {
    let Some((w, nav)) = load() else { return };
    let sc = scenes(&w, &nav, 16);
    let on = Params::from_json(&serde_json::json!({ "esc": 0.0, "duel": 0.0, "infer": 0.0 }));
    let off = Params::from_json(&serde_json::json!({ "contacts": 0.0 }));
    let (h_on, h_off) = (hash(&run(&w, &nav, &sc, &on)), hash(&run(&w, &nav, &sc, &off)));
    eprintln!("actions hash {h_on:#018x} (picture on) {h_off:#018x} (picture off)");
    assert_eq!(h_on, BASE_ON, "decisions with the new modes off changed");
    assert_eq!(h_off, BASE_OFF, "decisions without the enemy picture changed");
}

const BASE_ON: u64 = 0xa5d2_9493_6c54_7a16;
const BASE_OFF: u64 = 0xfe3e_76e3_a365_5019;

/// Three enemies closing in from one side on open ground: the place chosen to escape to is further from them
/// than where we stand and fewer of them can see it.
#[test]
fn swarmed_escape_goes_away_to_fewer_lines_of_sight() {
    let Some((w, nav)) = load() else { return };
    let st = SightTable::build(&w, &nav);
    let mut r = Rng(0x5a17_e5c4);
    let (mut tried, mut better, mut worst) = (0, 0, 0.0f32);
    while tried < 20 {
        let p = [w.min_x + 150.0 + r.f() * (w.max_x - w.min_x - 300.0), w.min_y + 150.0 + r.f() * (w.max_y - w.min_y - 300.0)];
        if !nav.walkable(p[0], p[1]) {
            continue;
        }
        let a = r.f() * TAU;
        let foes: Vec<SFoe> = (0..3)
            .map(|k| {
                let b = a + (k as f32 - 1.0) * 0.4;
                SFoe { pos: [p[0] + b.cos() * 80.0, p[1] + b.sin() * 80.0], hp: 1.0, r: 0.0, conf: 1.0, sure: true }
            })
            .collect();
        let here = st.expected(p, &foes);
        if here < 2.0 {
            continue;
        }
        tried += 1;
        let zone = Zone { center: p, radius: 900.0, next_center: p, next_radius: 800.0, soon: false };
        let cen = [foes.iter().map(|f| f.pos[0]).sum::<f32>() / 3.0, foes.iter().map(|f| f.pos[1]).sum::<f32>() / 3.0];
        let d = |x: [f32; 2], y: [f32; 2]| (x[0] - y[0]).hypot(x[1] - y[1]);
        let t0 = std::time::Instant::now();
        let pick = st.escape_pick(&nav, p, &foes, &zone, 11.7, None);
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        assert!(ms < 40.0, "escape search took {ms:.1} ms");
        worst = worst.max(ms);
        if let Some(pk) = pick {
            assert!(d(pk.pos, cen) > d(p, cen), "toward the enemies: {:?} from {:?}", pk.pos, p);
            assert!(st.expected(pk.pos, &foes) < here, "no fewer lines of sight");
            better += 1;
        }
    }
    eprintln!("escape places found in {better} of {tried} swarms in the open; search worst {worst:.2} ms");
    assert!(better >= tried * 3 / 4, "{better} of {tried}");
}

/// Three enemies aimed at us and shooting: the bot escapes, and only with `esc` on.
#[test]
fn swarm_triggers_escape() {
    let Some((w, nav)) = load() else { return };
    let sc: Vec<Scene> = scenes(&w, &nav, 16).into_iter().filter(|s| s.foes.len() == 3).collect();
    let count = |p: &Params| -> u32 {
        sc.iter()
            .map(|s| {
                let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav.clone()), p.clone());
                for k in 0..20 {
                    b.act(&obs(s, k));
                }
                b.esc_n
            })
            .sum()
    };
    let on = count(&Params::default());
    let off = count(&Params { esc: 0.0, ..Params::default() });
    eprintln!("escapes started: {on} of {} swarms (off: {off})", sc.len());
    assert!(on as usize >= sc.len() - 1 && off == 0);
}

/// A lone weakened enemy that has not noticed us, nobody else around: the bot takes it on (and does not
/// without `duel`).
#[test]
fn lone_weak_enemy_is_dueled() {
    let Some((w, nav)) = load() else { return };
    let sc: Vec<Scene> = scenes(&w, &nav, 16).into_iter().filter(|s| s.foes.len() == 1 && !s.foes[0].aimed).collect();
    let go = |p: &Params| -> (u32, usize) {
        let mut duels = 0;
        let mut fire = 0;
        for s in &sc {
            let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav.clone()), p.clone());
            for k in 0..20 {
                let a = b.act(&obs(s, k));
                fire += a.fire[0] as usize;
            }
            duels += b.duel_n;
        }
        (duels, fire)
    };
    let (on, fire_on) = go(&Params::default());
    let (off, fire_off) = go(&Params { duel: 0.0, ..Params::default() });
    eprintln!("duels {on} of {} (off {off}); shots {fire_on} vs {fire_off}", sc.len());
    assert_eq!(on as usize, sc.len());
    assert_eq!(off, 0);
    assert!(fire_on > fire_off);
}
