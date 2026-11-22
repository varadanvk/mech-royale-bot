//! Gun lead timing: the shot leaves a moment after the look, so the target is led over that moment at its full
//! speed and the shot starts from where our muzzle will be by then.
use tinman::brain::{Brain, Params};
use tinman::proto::{Hello, Obs};

/// One observation: us at the origin moving at `vme`, an enemy 100 m east moving at `vt`.
fn obs(k: u64, vme: [f32; 2], vt: [f32; 2]) -> Obs {
    let (x, y) = (k as f32 * 0.1 * vme[0], k as f32 * 0.1 * vme[1]);
    let (tx, ty) = (100.0 + x + k as f32 * 0.1 * (vt[0] - vme[0]), y + k as f32 * 0.1 * (vt[1] - vme[1]));
    let s = format!(
        r#"{{"type":"obs","tick":{},"t":{:.2},"alive":12,
           "self":{{"id":0,"pos":[{x},{y},0],"vel":[{},{},0],"aim":[0,0],"hp":1800,"max_hp":2061,"energy":10,"max_energy":100,"grounded":true,"cubes":0,"in_zone":true,
                    "weapons":[{{"kind":"gun","cooldown":0,"ammo":12,"reload":0}},null]}},
           "mechs":[{{"id":7,"kind":"mech","pos":[{tx},{ty},0],"vel":[{},{},0],"aim":[3.1416,0],"hp":1400,"max_hp":1400,"cubes":0,"weapons":["gun"],"charging":false}}],
           "projectiles":[],"sounds":[],"beams":[],"events":[],
           "zone":{{"center":[0,0],"radius":560,"next_center":[0,0],"next_radius":380,"shrink_start":160,"shrink_end":210,"dps":20}}}}"#,
        100 + 2 * k,
        20.0 + k as f32 * 0.1,
        vme[0],
        vme[1],
        vt[0],
        vt[1],
    );
    serde_json::from_str(&s).unwrap()
}

/// Aim yaw relative to the bearing of the enemy, after a few decisions.
fn yaw_off(params: Params, vme: [f32; 2], vt: [f32; 2]) -> f32 {
    yaw_off_turning(params, vme, vt, vt)
}

/// As `yaw_off`, with the enemy's velocity `last` on the final decision.
fn yaw_off_turning(params: Params, vme: [f32; 2], vt: [f32; 2], last: [f32; 2]) -> f32 {
    let hello: Hello = serde_json::from_str(r#"{"type":"hello","slot":0,"seed":5,"map":{"dir":""},"config":{}}"#).unwrap();
    let mut b = Brain::with_params(&hello, None, None, Params { lead_adapt: 0.0, ..params });
    let mut a = None;
    let mut o = obs(0, vme, vt);
    for k in 0..4 {
        o = obs(k, vme, vt);
        if k == 3 {
            o.mechs[0].vel = [last[0], last[1], 0.0];
        }
        a = Some(b.act(&o));
    }
    let a = a.unwrap();
    assert!(a.fire[0], "not firing");
    let m = &o.mechs[0];
    let bearing = (m.pos[1] - o.me.pos[1]).atan2(m.pos[0] - o.me.pos[0]);
    a.aim[0] - bearing
}

#[test]
fn leads_over_the_delay_and_our_own_motion() {
    let p = Params::default();
    let plain = Params { lead_delay: 0.0, ..p.clone() };
    // Flight time to a target 100 m out, starting 3 m ahead of the body.
    let fl = (100.0f32 - 3.0) / 252.0;
    // A target walking north at 8 m/s: led by all of the delay and the lead share of the flight.
    let got = yaw_off(p.clone(), [0.0, 0.0], [0.0, 8.0]);
    let want = (8.0 * (p.lead_delay + p.lead_ground * fl) / 100.0).atan();
    let before = yaw_off(plain.clone(), [0.0, 0.0], [0.0, 8.0]);
    eprintln!("walking target: yaw {got:.4} want {want:.4} (before {before:.4})");
    assert!((got - want).abs() < 0.002);
    assert!(got > before);
    // A standing target while we walk north at 8 m/s: aim from where the muzzle will be, a little south.
    let got = yaw_off(p.clone(), [0.0, 8.0], [0.0, 0.0]);
    let want = (-8.0 * p.lead_delay / 100.0).atan();
    let before = yaw_off(plain.clone(), [0.0, 8.0], [0.0, 0.0]);
    eprintln!("walking shooter: yaw {got:.4} want {want:.4} (before {before:.4})");
    assert!((got - want).abs() < 0.002);
    assert!(before.abs() < 1e-3);
}

#[test]
fn leads_a_turning_target_toward_where_it_is_turning() {
    let p = Params::default();
    // Walking south, now slowing at the walking acceleration: it is turning around and will head north.
    let got = yaw_off_turning(p.clone(), [0.0, 0.0], [0.0, -11.7], [0.0, -5.85]);
    let straight = yaw_off_turning(Params { lead_accel: 0.0, ..p.clone() }, [0.0, 0.0], [0.0, -11.7], [0.0, -5.85]);
    eprintln!("turning target: yaw {got:.4} (straight on {straight:.4})");
    assert!(straight < 0.0 && got > 0.0);
    // Steady walking is led as before.
    let steady = yaw_off(p.clone(), [0.0, 0.0], [0.0, -11.7]);
    let steady0 = yaw_off(Params { lead_accel: 0.0, ..p }, [0.0, 0.0], [0.0, -11.7]);
    assert!((steady - steady0).abs() < 1e-6);
}
