//! Raised standing spots (roof tops) and how exposed they are to the ground around them.
use crate::nav::Nav;
use crate::world::World;
use std::sync::OnceLock;

#[derive(Clone, Debug)]
pub struct RoofSpot {
    pub x: f32,
    pub y: f32,
    /// Height of the standing surface.
    pub top: f32,
    /// Ground height below.
    pub ground: f32,
    /// Nearest sampled ground distance from which a standing mech here can be seen (1e9 = none).
    pub hidden_r: f32,
    /// Share of sampled ground spots that see it, in bands 20-60, 60-100, 100-150, 150-200 m.
    pub seen: [f32; 4],
    /// Nearest walkable ground cell to climb from.
    pub base: [f32; 2],
}

static SPOTS: OnceLock<Vec<RoofSpot>> = OnceLock::new();

/// Every roof spot of the map (computed once per process; the map never changes).
pub fn spots(w: &World, nav: &Nav) -> &'static Vec<RoofSpot> {
    SPOTS.get_or_init(|| compute(w, nav))
}

fn compute(w: &World, nav: &Nav) -> Vec<RoofSpot> {
    let mut out = Vec::new();
    for p in w.prisms.iter() {
        let mut x = p.min[0] + 2.0;
        while x < p.max[0] {
            let mut y = p.min[1] + 2.0;
            while y < p.max[1] {
                let s = w.surface_below(x, y, 1000.0);
                let flat = (s - p.top).abs() < 0.01
                    && w.ceiling_above(x, y, s + 0.1) == f32::MAX
                    && (0..8).all(|k| {
                        let a = k as f32 * std::f32::consts::FRAC_PI_4;
                        let s2 = w.surface_below(x + a.cos() * 3.5, y + a.sin() * 3.5, 1000.0);
                        (s2 - s).abs() < 1.0
                    });
                let g = w.ground(x, y);
                let h = s - g;
                if flat && h > 7.0 && h < 40.0 {
                    let c = [x, y, s + 4.5];
                    let t = [x, y, s + 9.0];
                    let mut hidden_r = 1e9f32;
                    let mut bands = [[0u32; 2]; 4];
                    for ri in 0..19 {
                        let r = 20.0 + ri as f32 * 10.0;
                        for k in 0..24 {
                            let a = k as f32 * std::f32::consts::TAU / 24.0;
                            let q = [x + a.cos() * r, y + a.sin() * r];
                            if !nav.walkable(q[0], q[1]) {
                                continue;
                            }
                            let e = [q[0], q[1], w.ground(q[0], q[1]) + 8.0];
                            let v = w.clear(e, c) || w.clear(e, t);
                            let b = if r < 60.0 { 0 } else if r < 100.0 { 1 } else if r < 150.0 { 2 } else { 3 };
                            bands[b][1] += 1;
                            if v {
                                bands[b][0] += 1;
                                hidden_r = hidden_r.min(r);
                            }
                        }
                    }
                    // Climb from the nearest walkable ground straight out from the spot.
                    let mut base: Option<([f32; 2], f32)> = None;
                    for k in 0..16 {
                        let a = k as f32 * std::f32::consts::TAU / 16.0;
                        let mut r = 2.0;
                        while r < 90.0 {
                            let q = [x + a.cos() * r, y + a.sin() * r];
                            if nav.walkable(q[0], q[1]) {
                                if base.map_or(true, |b| r < b.1) {
                                    base = Some((q, r));
                                }
                                break;
                            }
                            r += 2.0;
                        }
                    }
                    let base = base.map(|b| b.0);
                    if let Some(base) = base {
                        let f = |b: [u32; 2]| if b[1] > 0 { b[0] as f32 / b[1] as f32 } else { 0.0 };
                        out.push(RoofSpot { x, y, top: s, ground: g, hidden_r, seen: [f(bands[0]), f(bands[1]), f(bands[2]), f(bands[3])], base });
                    }
                }
                y += 4.0;
            }
            x += 4.0;
        }
    }
    out
}
