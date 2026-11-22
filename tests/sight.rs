//! Sight table and sheltered spots on the real map.
//! Needs the arena map: set `MAP_DIR`, or keep the kit next to this repo (`../mechfight-kit/arena`).
use std::time::Instant;
use tinman::nav::Nav;
use tinman::sight::{Foe, SightTable, Zone, RANGE};
use tinman::world::World;

fn load() -> Option<(World, Nav)> {
    let dir = std::env::var("MAP_DIR").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../mechfight-kit/arena").to_string());
    let Ok(world) = World::load(std::path::Path::new(&dir)) else {
        eprintln!("map not found at {dir}; skipping");
        return None;
    };
    let nav = Nav::build(&world, 2.5, 9.0, 1.2, 50.0);
    Some((world, nav))
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

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn walk_point(w: &World, nav: &Nav, r: &mut Rng) -> [f32; 2] {
    loop {
        let p = [w.min_x + r.f() * (w.max_x - w.min_x), w.min_y + r.f() * (w.max_y - w.min_y)];
        if nav.walkable(p[0], p[1]) {
            return p;
        }
    }
}

#[test]
fn table_builds_quickly_and_spots_are_sheltered() {
    let Some((w, nav)) = load() else { return };
    let t0 = Instant::now();
    let t = SightTable::build(&w, &nav);
    let ms = t0.elapsed().as_secs_f32() * 1000.0;
    let used = t.cells_used();
    eprintln!("sight table: {}x{} cells, {} used, {} bytes, {} spots, built in {:.0} ms", t.nx, t.ny, used, t.bytes(), t.spots.len(), ms);
    assert!(used > 5000);
    assert!(t.spots.len() >= 40, "{} spots", t.spots.len());
    // Exposure is a fraction; spots are among the most sheltered cells and have at most three eighths of the
    // directions open.
    let mut ex: Vec<f32> = (0..t.nx * t.ny).filter(|&c| t.point(c).is_some()).map(|c| t.exposure[c]).collect();
    assert!(ex.iter().all(|e| (0.0..=1.0).contains(e)));
    ex.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = ex[ex.len() / 2];
    let worst_spot = t.spots.iter().map(|s| s.exposure).fold(0.0f32, f32::max);
    let mean_spot = t.spots.iter().map(|s| s.exposure).sum::<f32>() / t.spots.len() as f32;
    eprintln!("exposure: median cell {:.3}, p10 {:.3}, p90 {:.3}; spots mean {:.3} worst {:.3}", med, ex[ex.len() / 10], ex[ex.len() * 9 / 10], mean_spot, worst_spot);
    assert!(med > 0.05 && med < 0.9);
    assert!(mean_spot < med * 0.5);
    for s in &t.spots {
        assert!(t.open[s.cell] as usize * 8 <= tinman::sight::SECTORS * 3);
        assert!(nav.walkable(s.pos[0], s.pos[1]));
        assert!(t.reach[s.cell]);
    }
    let with_fb = t.spots.iter().filter(|s| s.fallback.is_some()).count();
    eprintln!("spots with a fallback: {with_fb}");
    assert!(with_fb * 10 >= t.spots.len() * 7);
}

#[test]
fn table_agrees_with_raycasts() {
    let Some((w, nav)) = load() else { return };
    let t = SightTable::build(&w, &nav);
    let mut r = Rng(0x2545f4914f6cdd1d);
    let (mut agree, mut n, mut vis, mut false_hidden) = (0, 0, 0, 0);
    while n < 20000 {
        let a = walk_point(&w, &nav, &mut r);
        // Uniform over the disc of sight range.
        let ang = r.f() * std::f32::consts::TAU;
        let d = RANGE * r.f().sqrt();
        let b = [a[0] + ang.cos() * d, a[1] + ang.sin() * d];
        if !nav.walkable(b[0], b[1]) {
            continue;
        }
        n += 1;
        let direct = w.clear([b[0], b[1], w.ground(b[0], b[1]) + 8.0], [a[0], a[1], w.ground(a[0], a[1]) + 6.0]);
        let table = t.visible(a, b);
        vis += direct as usize;
        if direct == table {
            agree += 1;
        } else if direct {
            false_hidden += 1;
        }
    }
    let frac = agree as f32 / n as f32;
    eprintln!("agreement {:.3} over {n} pairs ({vis} visible; {false_hidden} seen but table says hidden)", frac);
    assert!(frac >= 0.95, "agreement {frac}");
}

/// Zone circles as the server draws them: each next circle inside the current one.
fn zone_stages(r: &mut Rng) -> Vec<([f32; 2], f32)> {
    let mut c = [41.3f32, -72.8];
    let mut rad = 560.0f32;
    let mut out = Vec::new();
    for next in [380.0f32, 240.0, 140.0, 70.0, 25.0] {
        let a = r.f() * std::f32::consts::TAU;
        let off = (rad - next) * r.f().sqrt();
        c = [c[0] + a.cos() * off, c[1] + a.sin() * off];
        rad = next;
        out.push((c, rad));
    }
    out
}

#[test]
fn spots_exist_in_every_zone_stage() {
    let Some((w, nav)) = load() else { return };
    let t = SightTable::build(&w, &nav);
    let mut r = Rng(0x9e3779b97f4a7c15);
    let trials = 300;
    let mut with_spot = [0usize; 5];
    let mut with_pick = [0usize; 5];
    for _ in 0..trials {
        let st = zone_stages(&mut r);
        for (k, &(c, rad)) in st.iter().enumerate() {
            let prev = if k == 0 { ([41.3f32, -72.8], 560.0f32) } else { st[k - 1] };
            let z = Zone { center: prev.0, radius: prev.1, next_center: c, next_radius: rad, soon: true };
            if t.spots.iter().any(|s| z.holds(s.pos, 8.0)) {
                with_spot[k] += 1;
            }
            // From a walkable point inside the circle, a place is always found.
            let me = (0..200).map(|_| {
                let a = r.f() * std::f32::consts::TAU;
                let d = rad * 0.8 * r.f().sqrt();
                [c[0] + a.cos() * d, c[1] + a.sin() * d]
            }).find(|p| nav.walkable(p[0], p[1]) && t.cell_at(p[0], p[1]).map_or(false, |c| t.reach[c]));
            if let Some(me) = me {
                if t.pick_spot(&nav, me, &[], &z, &[], 11.7).is_some() {
                    with_pick[k] += 1;
                }
            } else {
                with_pick[k] += 1;
            }
        }
    }
    eprintln!("zone stages with a spot inside (of {trials}): {:?}; with a place picked: {:?}", with_spot, with_pick);
    // Precomputed spots cover the wide circles; in the small late ones the least exposed cells stand in.
    for k in 0..3 {
        assert!(with_spot[k] * 100 >= trials * 90, "stage {k}: {} of {trials}", with_spot[k]);
    }
    assert!(with_spot[3] * 100 >= trials * 60, "stage 3: {} of {trials}", with_spot[3]);
    for k in 0..5 {
        assert!(with_pick[k] * 100 >= trials * 95, "stage {k}: {} of {trials}", with_pick[k]);
    }
}

#[test]
fn decisions_are_fast() {
    let Some((w, nav)) = load() else { return };
    let t = SightTable::build(&w, &nav);
    let mut r = Rng(0x1234567);
    let (mut worst1, mut worst2, mut sum1, mut sum2) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let (mut iso_found, n) = (0, 200);
    for _ in 0..n {
        let me = walk_point(&w, &nav, &mut r);
        let foes: Vec<Foe> = (0..15)
            .map(|_| {
                let a = r.f() * std::f32::consts::TAU;
                let d = 30.0 + r.f() * 150.0;
                Foe { pos: [me[0] + a.cos() * d, me[1] + a.sin() * d], hp: r.f() }
            })
            .collect();
        let z = Zone { center: me, radius: 300.0, next_center: me, next_radius: 200.0, soon: true };
        let t0 = Instant::now();
        let _ = t.pick_spot(&nav, me, &foes, &z, &[], 11.7);
        let a = t0.elapsed().as_secs_f32() * 1000.0;
        let t1 = Instant::now();
        let iso = t.pick_isolating(&nav, me, &foes, &z, 11.7, None);
        let b = t1.elapsed().as_secs_f32() * 1000.0;
        if let Some(p) = iso {
            iso_found += 1;
            assert_eq!(t.seen_by(p.pos, &foes), 1);
            assert!(dist(p.pos, me) < 110.0);
        }
        worst1 = worst1.max(a);
        worst2 = worst2.max(b);
        sum1 += a;
        sum2 += b;
    }
    eprintln!(
        "pick_spot mean {:.3} ms worst {:.3} ms; pick_isolating mean {:.3} ms worst {:.3} ms ({iso_found}/{n} found)",
        sum1 / n as f32, worst1, sum2 / n as f32, worst2
    );
    assert!(worst1 < 5.0 && worst2 < 5.0);
}

#[test]
fn pressed_bot_fights_from_shelter_within_budget() {
    use std::sync::Arc;
    use tinman::brain::{Brain, Params};
    use tinman::proto::{Hello, Obs};
    let Some((w, nav)) = load() else { return };
    let w = Arc::new(w);
    let mut r = Rng(0xabcdef12345);
    for level in [1.0f32, 2.0] {
        let params = Params { spot: level, ..Params::default() };
        let t0 = Instant::now();
        let mut b = Brain::with_params(&Hello::default(), Some(w.clone()), Some(nav.clone()), params);
        let build = t0.elapsed().as_secs_f32() * 1000.0;
        assert!(b.sight.is_some());
        let (mut worst, mut sum, mut n, mut sheltered) = (0.0f32, 0.0f32, 0, 0);
        for round in 0..20 {
            let me = walk_point(&w, &nav, &mut r);
            let foes: Vec<[f32; 2]> = (0..15)
                .map(|_| {
                    let a = r.f() * std::f32::consts::TAU;
                    let d = 40.0 + r.f() * 140.0;
                    [me[0] + a.cos() * d, me[1] + a.sin() * d]
                })
                .collect();
            for k in 0..20 {
                let t = 100.0 + round as f32 * 10.0 + k as f32 * 0.1;
                let mechs: Vec<String> = foes
                    .iter()
                    .enumerate()
                    .map(|(i, f)| {
                        format!(
                            r#"{{"id": {i}, "kind": "mech", "pos": [{}, {}, {}], "aim": [0, 0], "hp": 900, "max_hp": 1400, "weapons": ["gun"]}}"#,
                            f[0], f[1], w.ground(f[0], f[1])
                        )
                    })
                    .collect();
                let s = format!(
                    r#"{{"tick": {}, "t": {t}, "self": {{"pos": [{}, {}, {}], "aim": [0,0], "hp": 1400, "max_hp": 1400, "energy": 100, "grounded": true, "in_zone": true,
                       "weapons": [{{"kind": "gun", "ammo": 12}}, null]}}, "mechs": [{}],
                       "zone": {{"center": [{}, {}], "radius": 400, "next_center": [{}, {}], "next_radius": 300, "shrink_start": 400}}}}"#,
                    (t * 20.0) as u64, me[0], me[1], w.ground(me[0], me[1]), mechs.join(","), me[0], me[1], me[0], me[1]
                );
                let o: Obs = serde_json::from_str(&s).unwrap();
                let t1 = Instant::now();
                let _ = b.act(&o);
                let ms = t1.elapsed().as_secs_f32() * 1000.0;
                worst = worst.max(ms);
                sum += ms;
                n += 1;
                if matches!(b.label, "spot" | "iso") {
                    sheltered += 1;
                }
            }
        }
        eprintln!("level {level}: table built in {build:.0} ms; decision mean {:.3} ms worst {:.3} ms; sheltered {sheltered}/{n}", sum / n as f32, worst);
        assert!(worst < 10.0, "worst decision {worst} ms");
        assert!(sheltered * 10 >= n * 8, "{sheltered}/{n}");
    }
}
