//! Nav sanity: how much of the map is walkable, and how much of that is reachable from the zone centre.
use tinman::nav::Nav;
use tinman::world::World;

fn main() {
    let dir = std::env::args().nth(1).expect("usage: navinfo <map dir>");
    let w = World::load(std::path::Path::new(&dir)).expect("load");
    let t = std::time::Instant::now();
    let mut nav = Nav::build(&w, 2.5, 9.0, 1.2, 50.0);
    let built = t.elapsed();
    let t = std::time::Instant::now();
    nav.set_goal(41.3, -72.8);
    let field = t.elapsed();
    let (mut walk, mut reach) = (0, 0);
    let mut far = 0.0f32;
    for j in 0..nav.ny {
        for i in 0..nav.nx {
            let p = nav.center(j * nav.nx + i);
            if nav.walkable(p[0], p[1]) {
                walk += 1;
                if let Some(d) = nav.path_len(p[0], p[1]) {
                    reach += 1;
                    far = far.max(d);
                }
            }
        }
    }
    println!("grid {}x{}, built in {:?}, distance field in {:?}", nav.nx, nav.ny, built, field);
    println!("walkable {:.0}% of cells; {:.1}% of walkable cells reach the zone centre; longest path {:.0} m", 100.0 * nav.walkable_fraction(), 100.0 * reach as f32 / walk as f32, far);
    // Coarse picture: '#' blocked, '.' walkable, '!' walkable but cut off.
    for j in (0..nav.ny).rev().step_by(6) {
        let row: String = (0..nav.nx).step_by(3).map(|i| {
            let p = nav.center(j * nav.nx + i);
            if !nav.walkable(p[0], p[1]) { '#' } else if nav.path_len(p[0], p[1]).is_some() { '.' } else { '!' }
        }).collect();
        println!("{row}");
    }
}
