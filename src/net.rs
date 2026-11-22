//! Learned decision layer: a small two-layer network that reads a compact summary of the situation each decision
//! and shifts a set of behaviour thresholds (fight distances, retreat and re-engage health, target focus, crowd
//! handling, free-for-all avoidance) around their tuned values. A network whose output layer is all zero leaves
//! every threshold exactly as tuned.
//!
//! Weight file layout (little-endian f32): `n_in, n_hidden, n_out` (as f32), then W1 (n_hidden x n_in, row-major),
//! b1 (n_hidden), W2 (n_out x n_hidden), b2 (n_out).
use crate::brain::Params;
use crate::proto::Obs;

pub const N_IN: usize = 33;
pub const N_OUT: usize = 18;

/// Compiled-in weights.
static DEFAULT_WEIGHTS: &[u8] = include_bytes!("net.bin");

#[derive(Clone, Debug)]
pub struct Net {
    pub n_in: usize,
    pub n_hidden: usize,
    pub n_out: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl Net {
    pub fn from_bytes(b: &[u8]) -> Option<Net> {
        if b.len() % 4 != 0 || b.len() < 12 {
            return None;
        }
        let v: Vec<f32> = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        Net::from_flat(&v)
    }

    pub fn from_flat(v: &[f32]) -> Option<Net> {
        if v.len() < 3 {
            return None;
        }
        let (i, h, o) = (v[0] as usize, v[1] as usize, v[2] as usize);
        if i != N_IN || o != N_OUT || h == 0 || h > 1024 {
            return None;
        }
        let need = 3 + h * i + h + o * h + o;
        if v.len() != need || !v.iter().all(|x| x.is_finite()) {
            return None;
        }
        let mut k = 3;
        let mut take = |n: usize| {
            let s = v[k..k + n].to_vec();
            k += n;
            s
        };
        let w1 = take(h * i);
        let b1 = take(h);
        let w2 = take(o * h);
        let b2 = take(o);
        Some(Net { n_in: i, n_hidden: h, n_out: o, w1, b1, w2, b2 })
    }

    /// The compiled-in network.
    pub fn builtin() -> Option<Net> {
        Net::from_bytes(DEFAULT_WEIGHTS)
    }

    /// True when the output layer is all zero (the network changes nothing).
    pub fn is_neutral(&self) -> bool {
        self.w2.iter().all(|&x| x == 0.0) && self.b2.iter().all(|&x| x == 0.0)
    }

    pub fn forward(&self, x: &[f32; N_IN], out: &mut [f32; N_OUT]) {
        let mut hid = vec![0.0f32; self.n_hidden];
        for (j, hv) in hid.iter_mut().enumerate() {
            let row = &self.w1[j * self.n_in..(j + 1) * self.n_in];
            let mut s = self.b1[j];
            for (w, xv) in row.iter().zip(x.iter()) {
                s += w * xv;
            }
            *hv = s.tanh();
        }
        for (k, ov) in out.iter_mut().enumerate() {
            let row = &self.w2[k * self.n_hidden..(k + 1) * self.n_hidden];
            let mut s = self.b2[k];
            for (w, hv) in row.iter().zip(hid.iter()) {
                s += w * hv;
            }
            *ov = s.tanh();
        }
    }
}

fn wrap(a: f32) -> f32 {
    let mut a = a % std::f32::consts::TAU;
    if a > std::f32::consts::PI {
        a -= std::f32::consts::TAU;
    } else if a < -std::f32::consts::PI {
        a += std::f32::consts::TAU;
    }
    a
}

/// Situation summary fed to the network. `hp_drop`: fraction of max HP lost since the previous decision;
/// `ffa`: 1 once a free-for-all is settled.
pub fn features(o: &Obs, hp_drop: f32, ffa: bool) -> [f32; N_IN] {
    let mut x = [0.0f32; N_IN];
    let me = &o.me;
    let p = me.pos;
    x[0] = (o.t / 300.0).min(1.5);
    x[1] = if me.max_hp > 0.0 { (me.hp / me.max_hp).clamp(0.0, 1.0) } else { 1.0 };
    x[2] = if me.max_energy > 0.0 { (me.energy / me.max_energy).clamp(0.0, 1.0) } else { 0.0 };
    x[3] = if me.in_zone { 1.0 } else { 0.0 };
    x[4] = (o.zone.radius / 500.0).min(2.0);
    x[5] = (o.zone.next_radius / 500.0).min(2.0);
    let dz = ((o.zone.next_center[0] - p[0]).powi(2) + (o.zone.next_center[1] - p[1]).powi(2)).sqrt();
    x[6] = (dz / o.zone.next_radius.max(1.0)).min(3.0);
    let has = |k: &str| me.weapons.iter().flatten().any(|w| w.kind == k);
    x[7] = has("laser") as u8 as f32;
    x[8] = has("lightning") as u8 as f32;
    x[9] = has("mortar") as u8 as f32;
    x[10] = me.weapons.iter().flatten().find(|w| w.kind == "gun").map_or(0.0, |w| (w.ammo > 0.0) as u8 as f32);
    let mut foes: Vec<(f32, &crate::proto::Actor)> = o
        .mechs
        .iter()
        .filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0))
        .map(|m| (((m.pos[0] - p[0]).powi(2) + (m.pos[1] - p[1]).powi(2)).sqrt(), m))
        .collect();
    foes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    x[11] = (foes.len() as f32 / 5.0).min(2.0);
    x[12] = (foes.iter().filter(|f| f.0 < 60.0).count() as f32 / 3.0).min(2.0);
    x[13] = (foes.iter().filter(|f| f.0 < 150.0).count() as f32 / 5.0).min(2.0);
    x[14] = o.alive as f32 / 16.0;
    x[15] = ffa as u8 as f32;
    x[16] = (hp_drop * 10.0).clamp(0.0, 2.0);
    x[17] = (me.upgrades.len() as f32 / 10.0).min(2.0);
    for k in 0..3 {
        let b = 18 + 5 * k;
        match foes.get(k) {
            Some(&(d, m)) => {
                let yaw = (m.pos[1] - p[1]).atan2(m.pos[0] - p[0]);
                let rel = wrap(yaw - me.aim[0]);
                x[b] = (d / 200.0).min(1.5);
                x[b + 1] = rel.cos();
                x[b + 2] = rel.sin();
                x[b + 3] = if m.max_hp > 0.0 { (m.hp / m.max_hp).clamp(0.0, 1.0) } else { 1.0 };
                let back = (p[1] - m.pos[1]).atan2(p[0] - m.pos[0]);
                x[b + 4] = (wrap(m.aim[0] - back).abs() < 0.2) as u8 as f32;
            }
            None => {
                x[b] = 1.5;
            }
        }
    }
    x
}

/// Shift the thresholds in `p` by the network outputs (each in -1..1). Zero outputs leave `p` unchanged.
pub fn apply(p: &mut Params, y: &[f32; N_OUT]) {
    let add = |v: f32, d: f32, lo: f32, hi: f32| (v + d).clamp(lo.min(v), hi.max(v));
    let mul = |v: f32, d: f32| v * d.exp();
    p.retreat_hp = add(p.retreat_hp, 0.2 * y[0], 0.0, 0.95);
    p.reengage_hp = add(p.reengage_hp, 0.2 * y[1], 0.0, 1.0);
    p.crowd_radius = mul(p.crowd_radius, 0.5 * y[2]);
    p.focus_hp = mul(p.focus_hp, 0.7 * y[3]);
    p.finish_hp = add(p.finish_hp, 0.2 * y[4], 0.0, 1.0);
    p.finish_range = mul(p.finish_range, 0.5 * y[5]);
    p.fight_speed = add(p.fight_speed, 0.3 * y[6], 0.3, 1.0);
    p.evade_weight = mul(p.evade_weight, 0.7 * y[7]);
    p.counter_range = mul(p.counter_range, 0.6 * y[8]);
    p.pref_gun = mul(p.pref_gun, 0.4 * y[9]);
    p.pref_laser = mul(p.pref_laser, 0.4 * y[10]);
    p.lose_margin = add(p.lose_margin, 0.2 * y[11], -0.5, 0.8);
    p.close_boost = add(p.close_boost, 20.0 * y[12], 0.0, 100.0);
    p.rush_start = mul(p.rush_start, 0.5 * y[13]);
    p.ffa_avoid_r = mul(p.ffa_avoid_r, 0.5 * y[14]);
    p.ffa_third_r = mul(p.ffa_third_r, 0.5 * y[15]);
    p.cr_radius = mul(p.cr_radius, 0.5 * y[16]);
    let c = (p.crowd_count as f32 + (2.0 * y[17]).round()).max(1.0);
    p.crowd_count = c as usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_weights_load() {
        let n = Net::builtin().expect("compiled-in weights");
        assert_eq!((n.n_in, n.n_out), (N_IN, N_OUT));
    }

    #[test]
    fn zero_output_changes_nothing() {
        let base = Params::default();
        let mut p = base.clone();
        apply(&mut p, &[0.0; N_OUT]);
        assert_eq!(format!("{p:?}"), format!("{base:?}"));
    }

    #[test]
    fn forward_is_fast() {
        let n = Net::builtin().unwrap();
        let x = [0.3f32; N_IN];
        let mut y = [0.0f32; N_OUT];
        let t = std::time::Instant::now();
        for _ in 0..1000 {
            n.forward(&x, &mut y);
        }
        assert!(t.elapsed().as_secs_f32() < 0.5, "1000 forwards took {:?}", t.elapsed());
    }
}
