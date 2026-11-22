//! Path following near building corners: stepping along the path must not turn us back the way we came.
//! Needs the arena map: set `MAP_DIR`, or keep the kit next to this repo (`../mechfight-kit/arena`).
use tinman::nav::Nav;
use tinman::world::World;

fn unit(x: f32, y: f32) -> [f32; 2] {
    let l = (x * x + y * y).sqrt();
    if l < 1e-6 { [0.0, 0.0] } else { [x / l, y / l] }
}

/// Direction the bot takes toward `g` from `p`: straight across open ground when short, else along the grid.
fn heading(nav: &mut Nav, p: [f32; 2], g: [f32; 2]) -> ([f32; 2], bool) {
    let d = (g[0] - p[0]).hypot(g[1] - p[1]);
    if d < 40.0 {
        let n = (d / 2.0).ceil().max(1.0) as usize;
        if (1..=n).all(|k| {
            let f = k as f32 / n as f32;
            nav.walkable(p[0] + (g[0] - p[0]) * f, p[1] + (g[1] - p[1]) * f)
        }) {
            return (unit(g[0] - p[0], g[1] - p[1]), false);
        }
    }
    nav.set_goal(g[0], g[1]);
    let t = nav.steer(p[0], p[1]).unwrap_or(g);
    (unit(t[0] - p[0], t[1] - p[1]), !nav.walkable(p[0], p[1]))
}

#[test]
fn steering_from_blocked_cells_does_not_reverse() {
    let dir = std::env::var("MAP_DIR").unwrap_or_else(|_| concat!(env!("CARGO_MANIFEST_DIR"), "/../mechfight-kit/arena").to_string());
    let Ok(world) = World::load(std::path::Path::new(&dir)) else {
        eprintln!("map not found at {dir}; skipping");
        return;
    };
    let mut nav = Nav::build(&world, 2.5, 9.0, 1.2, 50.0);
    let mut checked_blocked = 0;
    // Two building corners where the bot used to turn back and forth.
    for (cx, cy) in [(-68.0f32, 111.0f32), (206.0, 68.0)] {
        let mut reversals = 0;
        for gi in 0..48 {
            let a = gi as f32 * std::f32::consts::TAU / 12.0;
            let r = [20.0f32, 35.0, 60.0, 120.0][gi % 4];
            let g = [cx + a.cos() * r, cy + a.sin() * r];
            if !nav.walkable(g[0], g[1]) {
                continue;
            }
            for iy in 0..=36 {
                for ix in 0..=36 {
                    let p = [cx - 9.0 + ix as f32 * 0.5, cy - 9.0 + iy as f32 * 0.5];
                    let (d0, b0) = heading(&mut nav, p, g);
                    // About one decision of travel along that heading.
                    let (d1, b1) = heading(&mut nav, [p[0] + d0[0] * 0.8, p[1] + d0[1] * 0.8], g);
                    if b0 || b1 {
                        checked_blocked += 1;
                        if d0[0] * d1[0] + d0[1] * d1[1] < -0.3 {
                            reversals += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(reversals, 0, "heading reverses near ({cx}, {cy})");
    }
    assert!(checked_blocked > 100);
}
