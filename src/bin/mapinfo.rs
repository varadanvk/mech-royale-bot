//! Load the map, print a few sanity numbers, and time the raycast.
use tinman::world::World;
use std::time::Instant;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: mapinfo <map dir>");
    let t = Instant::now();
    let w = World::load(std::path::Path::new(&dir)).expect("load");
    println!("loaded in {:?}: {} prisms, {} trees, terrain {}x{}", t.elapsed(), w.prisms.len(), w.trees.len(), w.nx, w.ny);
    println!("ground(0,0) = {:.2}  (map origin: expect about 0)", w.ground(0.0, 0.0));
    // The recorded match's first spawn-area position from protocol.md: feet z should sit on the surface.
    for (x, y, z) in [(-217.98f32, -22.04f32, -4.73f32), (-208.24, -7.7, -4.7), (-214.6, -19.3, -4.71)] {
        println!("  surface at ({x},{y}) = {:.2}, observed feet z = {z}", w.surface_below(x, y, z + 2.0));
    }
    // Deterministic pseudo-random rays for a throughput number.
    let mut s = 0x2545F4914F6CDD1Du64;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f32 / (1u64 << 53) as f32
    };
    let n = 200_000;
    let mut hits = 0;
    let mut prism_hits = 0;
    let t = Instant::now();
    for _ in 0..n {
        let x = w.min_x + rnd() * (w.max_x - w.min_x);
        let y = w.min_y + rnd() * (w.max_y - w.min_y);
        let z = w.ground(x, y) + 8.0;
        let a = rnd() * std::f32::consts::TAU;
        let d = [a.cos(), a.sin(), (rnd() - 0.5) * 0.2];
        if let Some(h) = w.raycast([x, y, z], d, 200.0) {
            hits += 1;
            if h.prism.is_some() {
                prism_hits += 1;
            }
        }
    }
    let el = t.elapsed();
    let mut sees = 0;
    let t2 = Instant::now();
    for _ in 0..n {
        let x = w.min_x + rnd() * (w.max_x - w.min_x);
        let y = w.min_y + rnd() * (w.max_y - w.min_y);
        let a = rnd() * std::f32::consts::TAU;
        let r = 20.0 + rnd() * 180.0;
        let (bx, by) = (x + a.cos() * r, y + a.sin() * r);
        if w.sees([x, y, w.ground(x, y) + 8.0], [bx, by, w.ground(bx, by) + 4.5]) {
            sees += 1;
        }
    }
    println!("{n} sight lines (20-200 m): {:.2} us each, {:.0}% visible", t2.elapsed().as_secs_f64() * 1e6 / n as f64, 100.0 * sees as f64 / n as f64);
    println!("{n} rays of 200 m: {:.2} us each, {hits} hit ({prism_hits} on prisms)", el.as_secs_f64() * 1e6 / n as f64);
}
