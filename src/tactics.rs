//! Breaking off to hidden ground when outnumbered or losing: where to go and which enemies to hide from.
use super::{d2, norm2, Brain};
use crate::proto::Obs;

impl Brain {
    /// Whether a mech standing at `c` (ground height `z`) is hidden from every eye at head and mid-body height.
    fn hidden_at(&self, c: [f32; 2], z: f32, eyes: &[[f32; 3]]) -> bool {
        eyes.iter().all(|e| !self.clear(*e, [c[0], c[1], z + 4.5]) && !self.clear(*e, [c[0], c[1], z + 8.5]))
    }

    /// Every enemy we know of for the break-off: seen or remembered (`memory_s`, moved on by its last speed for
    /// up to 3 s), the last threat, and believed enemies we are fairly sure of; nearest first.
    pub(super) fn cb_foes(&self, o: &Obs, threats: &[[f32; 3]]) -> Vec<[f32; 3]> {
        let me = o.me.pos;
        let mut v: Vec<[f32; 3]> = self
            .known
            .values()
            .filter(|k| o.t - k.seen_t < self.params.memory_s && d2(&k.pos, &me) < 320.0)
            .map(|k| {
                let dt = (o.t - k.seen_t).min(3.0);
                [k.pos[0] + k.vel[0] * dt, k.pos[1] + k.vel[1] * dt, k.pos[2]]
            })
            .collect();
        for t in threats {
            if !v.iter().any(|q| d2(q, t) < 20.0) {
                v.push(*t);
            }
        }
        for b in self.beliefs.iter().filter(|b| b.conf >= 0.5 && b.r <= 40.0) {
            if d2(&b.pos, &me) < 320.0 && !v.iter().any(|q| d2(q, &b.pos) < 20.0) {
                v.push([b.pos[0], b.pos[1], self.ground(b.pos[0], b.pos[1], me[2])]);
            }
        }
        v.sort_by(|a, b| d2(a, &me).partial_cmp(&d2(b, &me)).unwrap_or(std::cmp::Ordering::Equal));
        v
    }

    /// Whether going straight from `from` to `to` keeps clear of every enemy in `foes`: 25 m, or as far as we
    /// already are from one closer than that.
    pub(super) fn cb_line_clear(from: [f32; 2], to: [f32; 2], foes: &[[f32; 2]]) -> bool {
        let d = [to[0] - from[0], to[1] - from[1]];
        let l2 = d[0] * d[0] + d[1] * d[1];
        foes.iter().all(|f| {
            let now = d2(f, &from);
            let s = if l2 > 1e-6 { (((f[0] - from[0]) * d[0] + (f[1] - from[1]) * d[1]) / l2).clamp(0.0, 1.0) } else { 0.0 };
            let c = [from[0] + d[0] * s, from[1] + d[1] * s];
            d2(f, &c) >= now.min(25.0) - 0.5
        })
    }

    /// Whether a goal at `g` lies away from the enemies in `foes`: farther than we are from their centre (those
    /// within 200 m) and from the nearest, and not within 60 degrees of one within 80 m. Returns the extra cost
    /// of heading that way (directions toward enemies within 150 m cost more), or `None` when it is not away.
    pub(super) fn cb_away_cost(here: [f32; 2], g: [f32; 2], foes: &[[f32; 2]]) -> Option<f32> {
        if foes.is_empty() {
            return Some(0.0);
        }
        let near: Vec<&[f32; 2]> = foes.iter().filter(|f| d2(*f, &here) < 200.0).collect();
        let pool: Vec<&[f32; 2]> = if near.is_empty() { foes.iter().collect() } else { near };
        let n = pool.len() as f32;
        let cen = [pool.iter().map(|f| f[0]).sum::<f32>() / n, pool.iter().map(|f| f[1]).sum::<f32>() / n];
        let nearest = foes.iter().fold((f32::MAX, here), |b, f| {
            let d = d2(f, &here);
            if d < b.0 {
                (d, *f)
            } else {
                b
            }
        });
        if d2(&g, &cen) <= d2(&here, &cen) || d2(&g, &nearest.1) <= nearest.0 {
            return None;
        }
        let dir = norm2(g[0] - here[0], g[1] - here[1]);
        let mut cost = 0.0;
        for f in foes {
            let d = d2(f, &here);
            if d > 150.0 {
                continue;
            }
            let to = norm2(f[0] - here[0], f[1] - here[1]);
            let c = dir[0] * to[0] + dir[1] * to[1];
            if d < 80.0 && c > 0.5 {
                return None;
            }
            cost += 12.0 * c.max(0.0) * (1.0 - d / 150.0);
        }
        Some(cost)
    }

    /// Nearest walkable cell, searching the walk grid outward up to `cr_steps` steps, that is hidden from
    /// `threats` (at most four; six with `cb_known`), inside the zone and not right next to an enemy. `Some(here)`
    /// when we are hidden already. With `cb_away` / `cb_path`, only cells away from the enemies in `all` and
    /// reached without passing near one of them count.
    pub(super) fn retreat_cell(&self, o: &Obs, threats: &[[f32; 3]], all: &[[f32; 2]]) -> Option<[f32; 2]> {
        let pos = o.me.pos;
        let here = [pos[0], pos[1]];
        let n_eyes = if self.params.cb_known > 0.0 { 6 } else { 4 };
        let away_on = self.params.cb_away > 0.0;
        let path_on = self.params.cb_path > 0.0;
        let eyes: Vec<[f32; 3]> = threats.iter().take(n_eyes).map(|t| [t[0], t[1], t[2] + 8.0]).collect();
        if eyes.is_empty() {
            return None;
        }
        if self.hidden_at([pos[0], pos[1]], pos[2], &eyes) {
            return Some([pos[0], pos[1]]);
        }
        let nav = self.nav.as_ref()?;
        let start = nav.nearest_walkable(pos[0], pos[1])?;
        let z = &o.zone;
        let foes: Vec<[f32; 2]> = threats.iter().take(n_eyes).map(|t| [t[0], t[1]]).collect();
        // The search does not pass through ground near an enemy (closer than 25 m, or than we already are).
        let blocked = |q: [f32; 2]| path_on && all.iter().any(|f| d2(f, &q) < d2(f, &here).min(25.0) - 0.5);
        let mut seen = vec![false; nav.nx * nav.ny];
        seen[start] = true;
        let mut frontier = vec![start];
        let mut checks = 0;
        let max_steps = self.params.cr_steps.max(1.0) as usize;
        // Cost = 2 per step + a zone penalty - 0.25 per metre from the nearest enemy (up to
        // 80 m), searched until no further cell could beat the best found, over at most 350 cells.
        let weighed = true;
        let mut best_cost: Option<([f32; 2], f32)> = None;
        for step in 0..max_steps {
            if weighed {
                if let Some(b) = best_cost {
                    if 2.0 * (step + 1) as f32 - 20.0 >= b.1 {
                        return Some(b.0);
                    }
                }
            }
            let mut next = Vec::new();
            for &c in &frontier {
                for n in nav.neighbors(c) {
                    if !seen[n] {
                        seen[n] = true;
                        if blocked(nav.center(n)) {
                            continue;
                        }
                        next.push(n);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            let mut best: Option<([f32; 2], f32)> = None;
            for &c in &next {
                let q = nav.center(c);
                if d2(&q, &z.center) > z.radius - 8.0 {
                    continue;
                }
                let near = foes.iter().map(|f| d2(f, &q)).fold(f32::MAX, f32::min);
                if near < 25.0 {
                    continue;
                }
                let mut extra = 0.0;
                if away_on {
                    match Self::cb_away_cost(here, q, all) {
                        Some(c) => extra = c,
                        None => continue,
                    }
                }
                if path_on && !Self::cb_line_clear(here, q, all) {
                    continue;
                }
                // The sight table errs toward "seen": a cell it calls hidden from nobody is skipped cheaply.
                if let Some(st) = self.sight.as_ref() {
                    if foes.iter().any(|f| st.visible(q, *f) && st.cell_at(q[0], q[1]).is_some()) {
                        continue;
                    }
                }
                checks += 1;
                if checks > if weighed { 350 } else { 600 } {
                    return if weighed { best_cost.map(|b| b.0) } else { best.map(|b| b.0) };
                }
                let qz = self.ground(q[0], q[1], pos[2]);
                if weighed {
                    if self.hidden_at(q, qz, &eyes) {
                        let out_next = (d2(&q, &z.next_center) - z.next_radius).max(0.0);
                        let cost = 2.0 * (step + 1) as f32 + 0.5 * out_next.min(40.0) - 0.25 * near.min(80.0) + extra;
                        if best_cost.map_or(true, |b| cost < b.1) {
                            best_cost = Some((q, cost));
                        }
                    }
                    continue;
                }
                if self.hidden_at(q, qz, &eyes) {
                    // Among cells at the same step count, prefer the one furthest from the enemies.
                    if best.map_or(true, |b| near > b.1) {
                        best = Some((q, near));
                    }
                }
            }
            if let Some(b) = best {
                return Some(b.0);
            }
            frontier = next;
        }
        best_cost.map(|b| b.0)
    }
}
