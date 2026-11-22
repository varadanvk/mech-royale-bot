//! Fight movement planning against shots in the air.
use std::sync::Arc;
use std::time::Instant;
use tinman::brain::{Brain, Params};
use tinman::nav::Nav;
use tinman::proto::{Hello, Obs};
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

/// An open spot: walkable 12 m around and a clear line 150 m along +x.
fn open_spot(w: &World, nav: &Nav) -> Option<[f32; 3]> {
    for i in 0..4000 {
        let x = -300.0 + (i % 80) as f32 * 8.0;
        let y = -300.0 + (i / 80) as f32 * 12.0;
        if !(0..16).all(|k| {
            let a = k as f32 * 0.3927;
            nav.walkable(x + a.cos() * 12.0, y + a.sin() * 12.0) && nav.walkable(x + a.cos() * 6.0, y + a.sin() * 6.0)
        }) {
            continue;
        }
        let z = w.ground(x, y);
        let far = w.ground(x + 150.0, y);
        if (far - z).abs() < 3.0 && w.clear([x, y, z + 5.0], [x + 150.0, y, far + 5.0]) {
            return Some([x, y, z]);
        }
    }
    None
}

fn obs(tick: u64, me: [f32; 3], projectiles: &str) -> Obs {
    let s = format!(
        r#"{{"tick":{tick},"t":{:.2},"alive":16,"self":{{"id":0,"pos":[{:.2},{:.2},{:.2}],"vel":[0,0,0],"aim":[0,0],"hp":1400,"max_hp":1400,"energy":100,"max_energy":100,"grounded":true,"cubes":0,"in_zone":true,"weapons":[{{"kind":"gun","cooldown":0,"ammo":12,"reload":0,"charge":0,"stats":{{}}}},null],"stats":{{}}}},"mechs":[{{"id":3,"kind":"mech","pos":[{:.2},{:.2},{:.2}],"vel":[0,0,0],"aim":[3.1416,0],"hp":1400,"max_hp":1400,"cubes":0,"weapons":["gun"],"charging":false}}],"projectiles":[{projectiles}],"zone":{{"center":[{:.2},{:.2}],"radius":560,"next_center":[{:.2},{:.2}],"next_radius":380,"shrink_start":200,"shrink_end":260,"dps":10}}}}"#,
        tick as f32 * 0.05,
        me[0], me[1], me[2],
        me[0] + 150.0, me[1], me[2],
        me[0], me[1], me[0], me[1]
    );
    serde_json::from_str(&s).unwrap()
}

fn brain(world: Arc<World>, nav: Nav, params: Params) -> Brain {
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":77,"map":{"dir":""},"config":{}}"#).unwrap();
    Brain::with_params(&hello, Some(world), Some(nav), params)
}

#[test]
fn steps_out_of_a_bullets_line() {
    let Some((world, nav)) = load() else { return };
    let Some(me) = open_spot(&world, &nav) else { return };
    // A bullet 120 m out, flying straight at our chest.
    let shot = format!(r#"{{"id":900,"kind":"gun","pos":[{:.2},{:.2},{:.2}],"vel":[-252,0,0]}}"#, me[0] + 120.0, me[1], me[2] + 5.0);
    let mut b = brain(world.clone(), nav.clone(), Params { dodge: 0.0, ..Params::default() });
    for k in 0..3 {
        b.act(&obs(40 + 2 * k, me, ""));
    }
    let a = b.act(&obs(46, me, &shot));
    assert!(a.mv[1].abs() > 0.7, "no sidestep: move {:?}", a.mv);
    // Aim and fire are the same with the planner off.
    let mut c = brain(world, nav, Params { plan: 0.0, dodge: 0.0, ..Params::default() });
    for k in 0..3 {
        c.act(&obs(40 + 2 * k, me, ""));
    }
    let a0 = c.act(&obs(46, me, &shot));
    assert_eq!((a.aim, a.fire), (a0.aim, a0.fire));
    assert!(a0.mv[1].abs() < 0.7, "sidestep without the planner: move {:?}", a0.mv);
}

#[test]
fn planning_under_heavy_fire_is_fast() {
    let Some((world, nav)) = load() else { return };
    let Some(me) = open_spot(&world, &nav) else { return };
    let mut shots = Vec::new();
    for i in 0..24 {
        let a = i as f32 * 0.26;
        let d = 60.0 + 8.0 * i as f32;
        let (px, py) = (me[0] + a.cos() * d, me[1] + a.sin() * d);
        if i % 2 == 0 {
            shots.push(format!(r#"{{"id":{},"kind":"gun","pos":[{px:.2},{py:.2},{:.2}],"vel":[{:.2},{:.2},0]}}"#, 900 + i, me[2] + 5.0, -a.cos() * 252.0, -a.sin() * 252.0));
        } else {
            // A shell lobbed to land near us.
            let t = d / 51.0;
            shots.push(format!(r#"{{"id":{},"kind":"mortar","pos":[{px:.2},{py:.2},{:.2}],"vel":[{:.2},{:.2},{:.2}]}}"#, 900 + i, me[2] + 7.0, -a.cos() * 51.0, -a.sin() * 51.0, 11.0 * t - 7.0 / t));
        }
    }
    let shots = shots.join(",");
    let mut b = brain(world, nav, Params::default());
    for k in 0..20 {
        b.act(&obs(40 + 2 * k, me, ""));
    }
    let (mut worst, mut base) = (0.0f32, 0.0f32);
    for k in 20..120u64 {
        // Keep moving a little so the decision is an ordinary one.
        let me = [me[0], me[1] + (k % 8) as f32 * 0.5, me[2]];
        let quiet = obs(40 + 4 * k, me, "");
        let t0 = Instant::now();
        b.act(&quiet);
        base = base.max(t0.elapsed().as_secs_f32() * 1000.0);
        let busy = obs(42 + 4 * k, me, &shots);
        let t1 = Instant::now();
        b.act(&busy);
        worst = worst.max(t1.elapsed().as_secs_f32() * 1000.0);
    }
    eprintln!("decision ms: quiet max {base:.2}, under fire max {worst:.2}");
    assert!(worst < 5.0, "decision under fire took {worst:.2} ms");
}
