//! Fight movement planner: score candidate moves against incoming fire and pick the best one.
use super::{d2, wrap, yaw_to, Brain};
use crate::proto::Obs;

/// A projectile that may reach us.
struct Incoming {
    mortar: bool,
    pos: [f32; 3],
    vel: [f32; 3],
    /// Seconds until it stops (wall, ground or blast).
    max_t: f32,
    impact: [f32; 3],
    dmg: f32,
    splash: f32,
}

const STEP: f32 = 0.05;
const MAX_STEPS: usize = 80;
const PROGRESS_STEP: usize = 12;

fn approach(v: [f32; 2], t: [f32; 2], max: f32) -> [f32; 2] {
    let d = [t[0] - v[0], t[1] - v[1]];
    let l = d[0].hypot(d[1]);
    if l <= max || l < 1e-6 {
        t
    } else {
        [v[0] + d[0] / l * max, v[1] + d[1] / l * max]
    }
}

/// Distance from point p to segment a-b in the plane, and the position along the segment (0..1).
fn seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> (f32, f32) {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let l2 = ab[0] * ab[0] + ab[1] * ab[1];
    let s = if l2 < 1e-9 { 0.0 } else { (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / l2).clamp(0.0, 1.0) };
    let q = [a[0] + ab[0] * s, a[1] + ab[1] * s];
    ((p[0] - q[0]).hypot(p[1] - q[1]), s)
}

impl Brain {
    /// Visible projectiles that could hit us in the next few seconds.
    fn incoming(&self, o: &Obs, radius: f32) -> Vec<Incoming> {
        let me = &o.me;
        let pos = me.pos;
        let mut out: Vec<Incoming> = Vec::new();
        let g = 22.0f32;
        for pr in &o.projectiles {
            if out.len() >= 16 {
                break;
            }
            let v = pr.vel;
            let sp = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if sp < 1.0 {
                continue;
            }
            let r = [pos[0] - pr.pos[0], pos[1] - pr.pos[1]];
            let d = r[0].hypot(r[1]);
            if pr.kind == "mortar" {
                if d > 260.0 {
                    continue;
                }
                // Follow the arc to where it lands.
                let h = 0.1f32;
                let (mut q, mut vz, mut t, mut hit) = (pr.pos, v[2], 0.0f32, false);
                while t < 7.0 {
                    let n = [q[0] + v[0] * h, q[1] + v[1] * h, q[2] + vz * h - 0.5 * g * h * h];
                    if !self.clear(q, n) || (self.world.is_none() && n[2] < pos[2]) {
                        hit = true;
                        break;
                    }
                    q = n;
                    vz -= g * h;
                    t += h;
                }
                if !hit {
                    continue;
                }
                let splash = 7.0 * 1.3;
                let dq = ((q[0] - pos[0]).powi(2) + (q[1] - pos[1]).powi(2) + (q[2] - pos[2]).powi(2)).sqrt();
                if dq > splash + radius + 45.0 {
                    continue;
                }
                out.push(Incoming { mortar: true, pos: pr.pos, vel: v, max_t: t + h, impact: q, dmg: 160.0, splash });
                continue;
            }
            // Straight shots: skip those flying away or passing wide.
            let along = (r[0] * v[0] + r[1] * v[1]) / sp;
            if along < -3.0 || d > 330.0 {
                continue;
            }
            let lat = (r[0] * v[1] - r[1] * v[0]).abs() / v[0].hypot(v[1]).max(1.0);
            if lat > 14.0 + radius {
                continue;
            }
            let dir = [v[0] / sp, v[1] / sp, v[2] / sp];
            let reach = self.world.as_ref().and_then(|w| w.raycast(pr.pos, dir, d + 20.0)).map_or(d + 20.0, |h| h.t);
            if reach < d - radius - 3.0 {
                continue;
            }
            let dmg = if pr.kind == "npc" { 25.0 } else { 45.0 };
            out.push(Incoming { mortar: false, pos: pr.pos, vel: v, max_t: reach / sp, impact: [0.0; 3], dmg, splash: 0.0 });
        }
        out
    }

    /// With shots in the air that can reach us, pick the move (and boost) that best keeps the wanted movement
    /// while taking the least predicted damage. `None` when nothing is incoming: the wanted move stands.
    ///
    /// A hit from behind the torso costs half as much again: shots arriving from outside the front arc of `aim_yaw`
    /// count for more, and a move that leaves a known enemy (`foes`) behind the line to `target` costs a little.
    pub(super) fn plan_move(&self, o: &Obs, want: [f32; 2], want_boost: bool, aim_yaw: f32, target: Option<[f32; 2]>, foes: &[[f32; 3]]) -> Option<([f32; 2], bool)> {
        let me = &o.me;
        if !me.grounded {
            return None;
        }
        let pos = me.pos;
        let scale = 1.0 + 0.05 * me.cubes;
        let (radius, height) = (2.5 * scale, 9.0 * scale);
        let mut threats = self.incoming(o, radius);
        let face = [aim_yaw.cos(), aim_yaw.sin()];
        // Lasers being charged at us: (muzzle, aim direction, seconds until the charge is full, damage).
        let mut beams: Vec<([f32; 3], [f32; 2], f32, f32)> = Vec::new();
        if self.params.plan_laser > 0.0 {
            let cook = if me.weapons.iter().flatten().any(|w| w.kind == "mortar") { 1.6 } else { 1.0 };
            for k in self.known.values() {
                let Some(since) = k.charge_since else { continue };
                if o.t - k.seen_t > 0.25 || d2(&k.pos, &pos) > 280.0 || wrap(k.aim_yaw - yaw_to(&k.pos, &pos)).abs() > 0.5 {
                    continue;
                }
                let rear = if (k.pos[0] - pos[0]) * face[0] + (k.pos[1] - pos[1]) * face[1] < 0.0 { 1.5 } else { 1.0 };
                let left = (0.75 - (o.t - since)).clamp(0.1, 0.75);
                beams.push(([k.pos[0], k.pos[1], k.pos[2] + 7.0], [k.aim_yaw.cos(), k.aim_yaw.sin()], left, 315.0 * cook * rear));
                if beams.len() >= 3 {
                    break;
                }
            }
        }
        if threats.is_empty() && beams.is_empty() {
            return None;
        }
        for th in threats.iter_mut().filter(|t| !t.mortar) {
            if th.vel[0] * face[0] + th.vel[1] * face[1] > 0.0 {
                th.dmg *= 1.5;
            }
        }
        let weight = self.params.plan;
        let wl = want[0].hypot(want[1]);
        let want = if wl > 1e-3 { [want[0] / wl, want[1] / wl] } else { [0.0, 0.0] };
        let slow = if me.weapons.iter().flatten().any(|w| w.kind == "laser" && w.charge > 0.0) { me.stat("charge_move_mult", 0.5) } else { 1.0 };
        let walk = me.stat("walk_speed", 11.7) * slow;
        let boost_v = me.stat("boost_speed", 19.5) * slow;
        let accel = me.stat("ground_accel", 58.5);
        let can_boost = me.energy > 12.0;
        let mut steps = 16usize;
        for th in &threats {
            steps = steps.max((th.max_t / STEP) as usize + 3);
        }
        let steps = steps.min(MAX_STEPS);

        let mut cands: Vec<([f32; 2], bool)> = Vec::with_capacity(36);
        cands.push(([0.0, 0.0], false));
        if wl > 1e-3 {
            cands.push((want, false));
        }
        for i in 0..16 {
            let a = i as f32 * std::f32::consts::PI / 8.0;
            cands.push(([a.cos(), a.sin()], false));
        }
        if can_boost {
            for i in 1..cands.len() {
                cands.push((cands[i].0, true));
            }
        }
        let nav = self.nav.as_ref();
        let start_cell = nav.map(|n| n.cell(pos[0], pos[1]));
        // Standing on a cell the grid calls unwalkable (a roof, a wall's edge): the grid says nothing useful.
        let fenced = nav.map_or(false, |n| n.walkable(pos[0], pos[1]));
        let z = &o.zone;
        let d_now = (pos[0] - z.center[0]).hypot(pos[1] - z.center[1]) - z.radius;
        let mid_z = pos[2] + height * 0.5;

        let mut best: Option<(f32, [f32; 2], bool)> = None;
        for (dir, boost) in cands {
            let sp = if boost { boost_v } else { walk };
            let tv = [dir[0] * sp, dir[1] * sp];
            let mut q = [pos[0], pos[1]];
            let mut v = [me.vel[0], me.vel[1]];
            let mut score = 0.0f32;
            let mut blocked = false;
            let mut at = q;
            let mut beam_at = [q; 3];
            for k in 1..=steps {
                v = approach(v, tv, accel * STEP);
                let mut n = [q[0] + v[0] * STEP, q[1] + v[1] * STEP];
                if fenced {
                    let nv = nav.unwrap();
                    if Some(nv.cell(n[0], n[1])) != start_cell && !nv.walkable(n[0], n[1]) {
                        blocked = true;
                        v = [0.0, 0.0];
                        n = q;
                    }
                }
                q = n;
                if k == PROGRESS_STEP {
                    at = q;
                }
                for (i, b) in beams.iter().enumerate() {
                    if k <= (b.2 / STEP).ceil() as usize {
                        beam_at[i] = q;
                    }
                }
                let (t0, t1) = ((k - 1) as f32 * STEP, k as f32 * STEP);
                for th in &threats {
                    if th.mortar {
                        if (th.max_t > t0 && th.max_t <= t1 + 1e-6) || (k == steps && th.max_t > t1) {
                            let d = ((q[0] - th.impact[0]).powi(2) + (q[1] - th.impact[1]).powi(2) + (mid_z - th.impact[2]).powi(2)).sqrt();
                            let reach = th.splash + radius;
                            if d < reach {
                                score -= th.dmg * (1.0 - 0.7 * d / reach) + 40.0;
                            }
                        }
                        continue;
                    }
                    if t0 >= th.max_t {
                        continue;
                    }
                    let te = t1.min(th.max_t);
                    let a = [th.pos[0] + th.vel[0] * t0, th.pos[1] + th.vel[1] * t0];
                    let b = [th.pos[0] + th.vel[0] * te, th.pos[1] + th.vel[1] * te];
                    let (d, s) = seg_dist(q, a, b);
                    if d < radius + 0.85 {
                        let tz = th.pos[2] + th.vel[2] * (t0 + (te - t0) * s);
                        if tz > pos[2] - 0.6 && tz < pos[2] + height + 0.6 {
                            score -= th.dmg;
                        }
                    }
                }
            }
            if steps < PROGRESS_STEP {
                at = q;
            }
            // A beam goes where its owner aims when the charge is let go: out of its sight nothing lands, and on
            // the line it is aimed along now it lands surest.
            for (i, b) in beams.iter().enumerate() {
                let c = beam_at[i];
                if !self.clear(b.0, [c[0], c[1], pos[2] + height * 0.6]) {
                    continue;
                }
                let r = [c[0] - b.0[0], c[1] - b.0[1]];
                let along = r[0] * b.1[0] + r[1] * b.1[1];
                let off = (r[0] * b.1[1] - r[1] * b.1[0]).abs();
                let sure = if along > 0.0 && off < radius + 1.5 { 1.0 } else { 0.85 };
                score -= b.3 * sure * self.params.plan_laser;
            }
            score += ((at[0] - pos[0]) * want[0] + (at[1] - pos[1]) * want[1]) * weight;
            if blocked {
                score -= 3.0;
            }
            if z.radius > 0.0 {
                let d_out = (at[0] - z.center[0]).hypot(at[1] - z.center[1]) - z.radius;
                if d_now < -0.5 && d_out > -4.0 && d_out > d_now {
                    score -= (d_out - d_now) * 8.0;
                }
            }
            let fwd = match target {
                Some(t) => {
                    let d = [t[0] - at[0], t[1] - at[1]];
                    let l = d[0].hypot(d[1]).max(1e-3);
                    [d[0] / l, d[1] / l]
                }
                None => face,
            };
            for f in foes {
                if target.map_or(false, |t| (t[0] - f[0]).hypot(t[1] - f[1]) < 4.0) {
                    continue;
                }
                if (f[0] - at[0]) * fwd[0] + (f[1] - at[1]) * fwd[1] < 0.0 {
                    score -= 2.0;
                }
            }
            if boost && !want_boost {
                score -= 1.5;
            }
            if dir == [0.0, 0.0] {
                if wl <= 1e-3 {
                    score += 0.6;
                }
            } else {
                score += 0.25 * (dir[0] * self.plan_last[0] + dir[1] * self.plan_last[1]);
            }
            if best.map_or(true, |b| score > b.0) {
                best = Some((score, dir, boost));
            }
        }
        best.map(|b| (b.1, b.2))
    }
}
