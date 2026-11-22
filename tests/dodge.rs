//! Sidestepping incoming bullets: step away from the side a shot passes on, with boost, and only for shots
//! that would come close soon.
use tinman::brain::{Brain, Params};
use tinman::proto::{Action, Hello, Obs};

/// One observation: us at the origin, an enemy 80 m east aimed at us, and `bullets` (position, velocity).
fn obs(k: u64, bullets: &[([f32; 3], [f32; 3])]) -> Obs {
    obs_at(k, [0.0, 0.0], bullets)
}

/// As `obs`, with us at `at` (and `bullets` placed relative to us).
fn obs_at(k: u64, at: [f32; 2], bullets: &[([f32; 3], [f32; 3])]) -> Obs {
    let pr: Vec<String> = bullets
        .iter()
        .enumerate()
        .map(|(i, (p, v))| {
            format!(r#"{{"id":{},"kind":"gun","pos":[{},{},{}],"vel":[{},{},{}]}}"#, 100 + i, p[0] + at[0], p[1] + at[1], p[2], v[0], v[1], v[2])
        })
        .collect();
    let s = format!(
        r#"{{"type":"obs","tick":{},"t":{:.2},"alive":12,
           "self":{{"id":0,"pos":[{},{},0],"vel":[0,0,0],"aim":[0,0],"hp":1800,"max_hp":2061,"energy":80,"max_energy":100,"grounded":true,"cubes":0,"in_zone":true,
                    "weapons":[{{"kind":"gun","cooldown":0,"ammo":12,"reload":0}},null]}},
           "mechs":[{{"id":7,"kind":"mech","pos":[80,0,0],"vel":[0,0,0],"aim":[3.1416,0],"hp":1400,"max_hp":1400,"cubes":0,"weapons":["gun"],"charging":false}}],
           "projectiles":[{}],"sounds":[],"beams":[],"events":[],
           "zone":{{"center":[0,0],"radius":560,"next_center":[0,0],"next_radius":380,"shrink_start":160,"shrink_end":210,"dps":20}}}}"#,
        100 + 2 * k,
        20.0 + k as f32 * 0.1,
        at[0],
        at[1],
        pr.join(",")
    );
    serde_json::from_str(&s).unwrap()
}

/// A few calm decisions, then one with `bullets` in the air. Returns that decision.
fn scene(params: Params, bullets: &[([f32; 3], [f32; 3])]) -> Action {
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":5,"map":{"dir":""},"config":{}}"#).unwrap();
    let mut b = Brain::with_params(&hello, None, None, params);
    for k in 0..4 {
        b.act(&obs(k, &[]));
    }
    b.act(&obs(4, bullets))
}

/// A shot from the east, 30 m out (0.12 s away), passing `off` metres north of our centre.
fn shot(off: f32) -> ([f32; 3], [f32; 3]) {
    ([30.0, off, 5.0], [-252.0, 0.0, 0.0])
}

#[test]
fn steps_away_from_the_side_the_shot_passes() {
    for off in [1.5f32, -1.5, 2.8, -2.8] {
        let a = scene(Params::default(), &[shot(off)]);
        eprintln!("shot {off:+} m: move {:?} boost {}", a.mv, a.boost);
        assert!(a.boost, "no boost on the sidestep (shot {off:+} m)");
        assert!(a.mv[1] * off < 0.0 && a.mv[1].abs() > 0.5, "move {:?} not away from a shot {off:+} m off", a.mv);
    }
}

#[test]
fn ignores_shots_that_miss_wide_or_are_far_off() {
    let calm = scene(Params::default(), &[]);
    // 4.5 m off is past body radius + 1.1 m; 1.0 s away is past the 0.7 s horizon.
    for b in [shot(4.5), ([252.0, 0.5, 5.0], [-252.0, 0.0, 0.0])] {
        let a = scene(Params::default(), &[b]);
        assert_eq!(a.boost, calm.boost, "reacted to {b:?}");
        assert!((a.mv[0] - calm.mv[0]).abs() < 1e-3 && (a.mv[1] - calm.mv[1]).abs() < 1e-3, "moved for {b:?}: {:?} vs {:?}", a.mv, calm.mv);
    }
    // The earlier sidestep looks 1.2 s ahead.
    let a = scene(Params { dodge_tight: 0.0, ..Params::default() }, &[([252.0, 0.5, 5.0], [-252.0, 0.0, 0.0])]);
    assert!(a.boost);
}

/// While weaving in sight of the shooter the sidestep still wins over a weave reversal, and the gun keeps firing.
#[test]
fn sidestep_beats_the_weave_and_we_keep_firing() {
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":5,"map":{"dir":""},"config":{}}"#).unwrap();
    for weave in [150.0f32, 0.0] {
        for off in [1.5f32, -1.5] {
            // Shots arriving at different points in the weave, so some land just as it reverses.
            for at in 4..20u64 {
                let mut b = Brain::with_params(&hello, None, None, Params { weave_seen: weave, ..Params::default() });
                let mut pos = [0.0f32, 0.0];
                let mut calm = b.act(&obs_at(0, pos, &[]));
                for k in 1..at {
                    pos = [pos[0] + calm.mv[0] * 0.6, pos[1] + calm.mv[1] * 0.6];
                    calm = b.act(&obs_at(k, pos, &[]));
                }
                pos = [pos[0] + calm.mv[0] * 0.6, pos[1] + calm.mv[1] * 0.6];
                let a = b.act(&obs_at(at, pos, &[shot(off)]));
                assert!(a.boost, "weave {weave} at {at}: no boost on the sidestep (shot {off:+} m)");
                assert!(a.mv[1] * off < 0.0 && a.mv[1].abs() > 0.5, "weave {weave} at {at}: move {:?} not away from a shot {off:+} m off (calm {:?})", a.mv, calm.mv);
                assert!(!calm.fire[0] || a.fire[0], "weave {weave} at {at}: stopped firing on the sidestep");
            }
        }
    }
}
