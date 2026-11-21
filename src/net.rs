//! Small learned decision layer: situation features -> two hidden layers -> adjustments of the
//! rule parameters for the current decision. Plain f32 arithmetic in a fixed order (deterministic).
//! With a zero output layer every adjustment is zero and the rules run unchanged.

pub const NF: usize = 56;
pub const NH: usize = 32;
pub const NO: usize = 17;
/// Number of weights: two hidden layers and the output layer, each with biases.
pub const NW: usize = NF * NH + NH + NH * NH + NH + NH * NO + NO;

/// Weights compiled into the bot (empty = rules only).
static BUILTIN: &[u8] = include_bytes!("weights.bin");

/// Rational tanh approximation (exact at 0, saturates at +-1); avoids libm differences between platforms.
#[inline]
pub fn th(x: f32) -> f32 {
    let x = x.clamp(-3.0, 3.0);
    x * (27.0 + x * x) / (27.0 + 9.0 * x * x)
}

pub struct Net {
    w: Vec<f32>,
}

impl Net {
    pub fn from_vec(w: Vec<f32>) -> Option<Net> {
        (w.len() == NW && w.iter().all(|v| v.is_finite())).then(|| Net { w })
    }

    pub fn from_bytes(b: &[u8]) -> Option<Net> {
        if b.len() != NW * 4 {
            return None;
        }
        Net::from_vec(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn builtin() -> Option<Net> {
        Net::from_bytes(BUILTIN)
    }

    pub fn forward(&self, x: &[f32; NF]) -> [f32; NO] {
        let w = &self.w;
        let mut h1 = [0f32; NH];
        let b1 = NF * NH;
        for j in 0..NH {
            let mut s = w[b1 + j];
            let r = &w[j * NF..(j + 1) * NF];
            for i in 0..NF {
                s += r[i] * x[i];
            }
            h1[j] = th(s);
        }
        let o2 = b1 + NH;
        let b2 = o2 + NH * NH;
        let mut h2 = [0f32; NH];
        for j in 0..NH {
            let mut s = w[b2 + j];
            let r = &w[o2 + j * NH..o2 + (j + 1) * NH];
            for i in 0..NH {
                s += r[i] * h1[i];
            }
            h2[j] = th(s);
        }
        let o3 = b2 + NH;
        let b3 = o3 + NH * NO;
        let mut out = [0f32; NO];
        for k in 0..NO {
            let mut s = w[b3 + k];
            let r = &w[o3 + k * NH..o3 + (k + 1) * NH];
            for i in 0..NH {
                s += r[i] * h2[i];
            }
            out[k] = th(s);
        }
        out
    }
}
