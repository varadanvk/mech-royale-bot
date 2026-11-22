//! Enemies on more than one side: where each enemy round us is expected to be shortly (`pred`), whether they
//! have us between them (a pincer), and what a fight position is worth for keeping them all in front of us with
//! a wall at our back (`pincer`).
use super::{d2, yaw_to, Brain};
use crate::predict;
use crate::proto::{Actor, Obs};
use std::f32::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI};

/// Believed enemies out of sight count from this confidence on.
const SURE: f32 = 0.4;
/// Enemies spread over more than this arc round us (radians; two straight opposite span half the circle)
/// have us between them.
const PINCER_SPREAD: f32 = 0.8 * PI;

/// One enemy round us.
#[derive(Clone, Copy, Debug, Default)]
pub struct Threat {
    pub id: Option<u32>,
    pub now: [f32; 2],
    /// Where it is expected to be `pred_s` from now (`now` with `pred` off).
    pub pred: [f32; 2],
    /// It is the enemy we are fighting.
    pub target: bool,
}

/// The enemies round us this decision.
#[derive(Clone, Debug, Default)]
pub struct PinPicture {
    pub threats: Vec<Threat>,
    pub pincered: bool,
}

/// Debug only: motion predictions for enemies in view, checked against where they turn out to be.
#[derive(Clone, Debug, Default)]
pub struct PredTrace {
    /// (made at, enemy, coming for us, prediction per horizon and variant).
    pending: Vec<(f32, u32, bool, [[[f32; 2]; VARIANTS]; 2])>,
    /// Per coming-for-us flag, horizon and variant: summed error and count.
    err: [[[f64; VARIANTS]; 2]; 2],
    n: [[u32; 2]; 2],
    last_t: f32,
}

/// Prediction variants traced: standing still, then (walk share, stop distance) pairs.
const VARIANTS: usize = 14;
const VARIANT_SET: [(f32, f32); 7] = [(0.0, 30.0), (0.25, 30.0), (0.5, 30.0), (0.75, 30.0), (1.0, 30.0), (0.5, 15.0), (0.5, 45.0)];

impl PredTrace {
    pub fn report(&self, slot: u32, seed: u64) {
        let mut s = format!("PRED sum slot {} seed {}", slot, seed);
        for hunt in 0..2 {
            for h in 0..2 {
                s += &format!(" | hunt{} h{} n {}", hunt, h + 1, self.n[hunt][h]);
                for v in 0..VARIANTS {
                    s += &format!(" {:.1}", self.err[hunt][h][v]);
                }
            }
        }
        eprintln!("{}", s);
    }
}

impl Brain {
    /// Whether enemy `m` (in view) is coming for us: our picture says so, it is aimed at us, or it hurt us lately.
    fn coming(&self, o: &Obs, m: &Actor, pos: &[f32; 3]) -> bool {
        self.beliefs.iter().any(|b| b.id == Some(m.id) && b.hunting)
            || super::wrap(m.aim[0] - yaw_to(&m.pos, pos)).abs() < 0.2
            || self.attackers.get(&m.id).map_or(false, |t| o.t - t < 4.0)
    }

    /// The velocity an enemy in view is predicted with: its own, with the part across our line scaled by `pred_tan`.
    pub(super) fn pred_vel(&self, m: &Actor, here: [f32; 2]) -> [f32; 2] {
        let k = self.params.pred_tan;
        let vel = [m.vel[0], m.vel[1]];
        if k == 1.0 {
            return vel;
        }
        let u = super::norm2(m.pos[0] - here[0], m.pos[1] - here[1]);
        let rad = vel[0] * u[0] + vel[1] * u[1];
        [u[0] * rad + (vel[0] - u[0] * rad) * k, u[1] * rad + (vel[1] - u[1] * rad) * k]
    }

    /// Fill in the enemies round us, where they are expected to be, and whether we are pincered.
    pub(super) fn update_pincer(&mut self, o: &Obs, pos: &[f32; 3], target: Option<&Actor>) {
        self.trace_pred(o, pos);
        self.pin = PinPicture::default();
        let p = &self.params;
        if p.pred <= 0.0 && p.pincer <= 0.0 {
            return;
        }
        let here = [pos[0], pos[1]];
        let (dt, hunt_w, stop, r) = (p.pred_s, p.pred_hunt, p.pred_stop, p.pin_r);
        let pred_on = p.pred > 0.0;
        let walk = |x: f32, y: f32| self.walkable(x, y);
        let mut th: Vec<Threat> = Vec::new();
        for m in o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0) && d2(&m.pos, pos) < r) {
            let now = [m.pos[0], m.pos[1]];
            let pred = if pred_on {
                let w = if self.coming(o, m, pos) { hunt_w } else { 0.0 };
                predict::predict(now, self.pred_vel(m, here), here, w, 0.0, stop, dt, &walk)
            } else {
                now
            };
            th.push(Threat { id: Some(m.id), now, pred, target: target.map_or(false, |t| t.id == m.id) });
        }
        for b in self.beliefs.iter().filter(|b| !b.listed && b.conf >= SURE && d2(&b.pos, pos) < r) {
            if b.id.map_or(false, |id| th.iter().any(|t| t.id == Some(id))) {
                continue;
            }
            let pred = if pred_on {
                // A track still carried along its last velocity keeps it; anything else is only known to be about.
                let vel = b.id.and_then(|id| self.contacts.tracks.get(&id)).filter(|k| k.coasting).map_or([0.0, 0.0], |k| [k.vel[0], k.vel[1]]);
                let w = if b.hunting { hunt_w } else { 0.0 };
                predict::predict(b.pos, vel, here, w, 4.0, stop, dt, &walk)
            } else {
                b.pos
            };
            th.push(Threat { id: b.id, now: b.pos, pred, target: false });
        }
        let yaws: Vec<f32> = th.iter().flat_map(|t| [yaw_to(&here, &t.now), yaw_to(&here, &t.pred)]).collect();
        let spread = if th.len() >= 2 { predict::spread(&yaws) } else { 0.0 };
        let mut pincered = th.len() >= 2 && spread > PINCER_SPREAD;
        // A flanker: someone other than the target expected to be more than 90 degrees off it.
        if let Some(t) = target {
            let ty = yaw_to(&here, &t.pos);
            pincered |= th.iter().any(|x| !x.target && predict::off(yaw_to(&here, &x.pred), ty) > FRAC_PI_2);
        }
        if std::env::var_os("TPIN").is_some() {
            let list: Vec<String> = th.iter().map(|x| format!("{}:{:.0},{:.0}->{:.0},{:.0}{}", x.id.map_or(-1, |i| i as i64), x.now[0], x.now[1], x.pred[0], x.pred[1], if x.target { "T" } else { "" })).collect();
            eprintln!("PIN t {:.1} at {:.0},{:.0} pincered {} spread {:.0} | {}", o.t, here[0], here[1], pincered as u8, spread.to_degrees(), list.join(" "));
        }
        self.pin = PinPicture { threats: th, pincered };
        self.dbg_pincer = (pincered, spread.to_degrees());
        if p.pincer > 0.0 {
            let mut s = format!("{} pin {} {:.0}", self.dbg_mode, pincered as u8, spread.to_degrees());
            s.truncate(80);
            self.dbg_mode = s;
        }
    }

    /// Whether the anti-pincer terms apply: on, and two or more enemies round us.
    pub(super) fn pin_on(&self) -> bool {
        self.params.pincer > 0.0 && self.pin.threats.len() >= 2
    }

    /// Candidate distances for a fight position: 6 m, and while pincered also every 6 m out to `pin_ring`.
    pub(super) fn pin_rings(&self) -> Vec<f32> {
        let mut v = vec![6.0];
        if self.pin_on() && self.pin.pincered {
            let mut r = 12.0;
            while r <= self.params.pin_ring + 0.01 {
                v.push(r);
                r += 6.0;
            }
        }
        v
    }

    /// What a fight position at `q` facing `t` is worth for keeping every enemy in front of us (see
    /// `Params::pincer`) from `here`; `ex_here` is the open ground round where we stand.
    pub(super) fn pin_score(&self, o: &Obs, here: [f32; 2], q: [f32; 2], t: [f32; 3], ex_here: Option<f32>) -> f32 {
        let p = &self.params;
        let aim = yaw_to(&q, &t);
        let mut s = 0.0;
        let mut yaws = vec![aim];
        for th in self.pin.threats.iter().filter(|x| !x.target) {
            // Backing off the target toward one off to the side or behind walks into it (closing on one in
            // front with the target is no worse than closing on the target).
            let side = ((predict::off(yaw_to(&here, &th.pred), yaw_to(&here, &t)) - FRAC_PI_3) / FRAC_PI_6).clamp(0.0, 1.0);
            let closer = d2(&here, &th.pred) - d2(&q, &th.pred);
            if closer > 0.0 {
                s -= p.pin_close * closer * side;
            }
            for (at, w) in [(th.now, 0.4f32), (th.pred, 0.6)] {
                let y = yaw_to(&q, &at);
                yaws.push(y);
                let d = d2(&q, &at);
                let near = if d < 60.0 { 1.0 } else { (1.0 - (d - 60.0) / 120.0).max(0.3) };
                let rear = ((predict::off(y, aim) - FRAC_PI_2) / FRAC_PI_4).clamp(0.0, 1.0);
                s -= p.pin_rear * w * rear * near;
            }
        }
        s -= p.pin_spread * ((predict::spread(&yaws) - FRAC_PI_2) / FRAC_PI_2).max(0.0);
        // Where to stand once they have us between them: a wall behind, not out in the open, inside the next circle.
        if !self.pin.pincered {
            return s;
        }
        if let Some(c) = self.sight.as_ref().and_then(|st| st.cell_at(q[0], q[1]).map(|c| (st, c))) {
            let (st, c) = c;
            let back = aim + PI;
            let rb = (-2..=2).map(|k| st.range_at(c, back + k as f32 * PI / 8.0)).sum::<f32>() / 5.0;
            s += p.pin_cover * ((40.0 - rb) / 30.0).clamp(0.0, 1.0);
            if let Some(e0) = ex_here {
                s -= p.pin_open * ((st.exposure[c] - e0) * 10.0).max(0.0);
            }
        }
        let z = &o.zone;
        let out = d2(&q, &z.next_center) - (z.next_radius - 5.0);
        if out > 0.0 {
            s -= p.pin_zone * out;
        }
        s
    }

    /// Open ground round `q` (the sight table's exposure), if known.
    pub(super) fn exposure_at(&self, q: [f32; 2]) -> Option<f32> {
        self.sight.as_ref().and_then(|st| st.cell_at(q[0], q[1]).map(|c| st.exposure[c]))
    }

    /// Debug only (`TPRED`): every half second, predict each enemy in view within 150 m one and two seconds
    /// ahead in every traced variant, and check earlier predictions that have come due against where the enemy
    /// is now (an enemy in the list is where it really is).
    fn trace_pred(&mut self, o: &Obs, pos: &[f32; 3]) {
        let Some(mut tr) = self.pred_trace.take() else { return };
        let here = [pos[0], pos[1]];
        tr.pending.retain(|(t0, id, hunt, preds)| {
            let age = o.t - t0;
            for (h, hz) in [1.0f32, 2.0].iter().enumerate() {
                if (age - hz).abs() < 0.051 {
                    if let Some(m) = o.mechs.iter().find(|m| m.id == *id) {
                        let hi = *hunt as usize;
                        tr.n[hi][h] += 1;
                        for v in 0..VARIANTS {
                            tr.err[hi][h][v] += d2(&preds[h][v], &m.pos) as f64;
                        }
                    }
                }
            }
            age < 2.1
        });
        if o.t - tr.last_t >= 0.5 - 1e-3 {
            tr.last_t = o.t;
            let walk = |x: f32, y: f32| self.walkable(x, y);
            for m in o.mechs.iter().filter(|m| m.kind == "mech" && !(m.max_hp > 0.0 && m.hp <= 0.0) && d2(&m.pos, pos) < 150.0) {
                let hunt = self.coming(o, m, pos);
                let now = [m.pos[0], m.pos[1]];
                let vel = [m.vel[0], m.vel[1]];
                let mut preds = [[now; VARIANTS]; 2];
                for (h, hz) in [1.0f32, 2.0].iter().enumerate() {
                    for (v, (w, stop)) in VARIANT_SET.iter().enumerate() {
                        preds[h][v + 1] = predict::predict(now, vel, here, *w, 0.0, *stop, *hz, &walk);
                    }
                }
                // Velocity split into the part toward or away from us and the part across our line.
                let u = super::norm2(now[0] - here[0], now[1] - here[1]);
                let rad = vel[0] * u[0] + vel[1] * u[1];
                let tan = [vel[0] - u[0] * rad, vel[1] - u[1] * rad];
                let mk = |k: f32| [u[0] * rad + tan[0] * k, u[1] * rad + tan[1] * k];
                for (h, hz) in [1.0f32, 2.0].iter().enumerate() {
                    preds[h][8] = predict::predict(now, mk(0.0), here, 0.25, 0.0, 30.0, *hz, &walk);
                    preds[h][9] = predict::predict(now, mk(0.5), here, 0.25, 0.0, 30.0, *hz, &walk);
                    preds[h][10] = predict::predict(now, mk(0.5), here, 0.5, 0.0, 45.0, *hz, &walk);
                    preds[h][11] = predict::predict(now, mk(0.7), here, 0.25, 0.0, 30.0, *hz, &walk);
                    preds[h][12] = predict::predict(now, mk(0.7), here, 0.4, 0.0, 45.0, *hz, &walk);
                    preds[h][13] = predict::predict(now, mk(0.3), here, 0.4, 0.0, 45.0, *hz, &walk);
                }
                tr.pending.push((o.t, m.id, hunt, preds));
            }
        }
        self.pred_trace = Some(tr);
    }
}
