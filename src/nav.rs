//! Ground navigation: a coarse walkability grid and a distance field toward the current goal.
//! Roofs are ignored: the bot paths around buildings, not over them.
use crate::world::{PrismKind, World};
use std::sync::{Arc, Mutex};

pub const CELL: f32 = 3.0;
const STEPS: [(isize, isize); 8] = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)];

/// Recently used distance fields by goal cell, most recent last. A distance field depends
/// only on the walkability grid and the goal, so navs built from the same grid can share one.
pub type DistCache = Arc<Mutex<Vec<(usize, Arc<Mutex<Field>>)>>>;
const DIST_CACHE_LEN: usize = 32;

/// Bucket width of the Dijkstra queue: below the cheapest step, so the cells of one bucket
/// cannot improve each other and are final when the bucket is reached.
const W: f32 = CELL * 0.95;

/// A distance field toward one goal, settled outward from the goal only as far as queries need.
/// Cells in buckets before `next` hold their final distance; every other cell holds a value no
/// smaller than its final distance, which is itself at least `next * W`. Queries that compare a
/// settled cell against its neighbours therefore get the same answers as with the full field.
pub struct Field {
    dist: Vec<f32>,
    buckets: Vec<Vec<(f32, u32)>>,
    next: usize,
}

impl Field {
    fn empty(n: usize) -> Field {
        Field { dist: vec![f32::MAX; n], buckets: Vec::new(), next: 0 }
    }

    fn new(n: usize, g: usize) -> Field {
        let mut f = Field::empty(n);
        f.dist[g] = 0.0;
        f.buckets.push(vec![(0.0, g as u32)]);
        f
    }

    fn settled(&self, c: usize) -> bool {
        let d = self.dist[c];
        d != f32::MAX && ((d / W) as usize) < self.next
    }

    /// Process buckets until cell `c` is final (or the queue is exhausted: unreachable).
    fn settle(&mut self, steps: &[u8], nx: usize, c: usize) {
        let offs: [isize; 8] = STEPS.map(|(di, dj)| dj * nx as isize + di);
        while self.next < self.buckets.len() && !self.settled(c) {
            let b = self.next;
            let mut cur = std::mem::take(&mut self.buckets[b]);
            while let Some((d, c)) = cur.pop() {
                let c = c as usize;
                if d > self.dist[c] {
                    continue;
                }
                let mask = steps[c];
                for (k, &off) in offs.iter().enumerate() {
                    if mask & (1 << k) == 0 {
                        continue;
                    }
                    let n = (c as isize + off) as usize;
                    let nd = d + if k >= 4 { CELL * 1.4142 } else { CELL };
                    if nd < self.dist[n] {
                        self.dist[n] = nd;
                        let k = ((nd / W) as usize).max(b + 1);
                        if k >= self.buckets.len() {
                            self.buckets.resize_with(k + 1, Vec::new);
                        }
                        self.buckets[k].push((nd, n as u32));
                    }
                }
            }
            self.buckets[b] = cur;
            self.next += 1;
        }
        if self.next >= self.buckets.len() {
            self.buckets = Vec::new();
        }
    }
}

/// Cloning shares the walkability grid and the current distance field.
#[derive(Clone)]
pub struct Nav {
    pub nx: usize,
    pub ny: usize,
    x0: f32,
    y0: f32,
    walk: Arc<Vec<bool>>,
    /// Per cell: bit k set if step `STEPS[k]` leads to a walkable cell without cutting a corner.
    steps: Arc<Vec<u8>>,
    dist: Arc<Mutex<Field>>,
    goal: Option<usize>,
    cache: Option<DistCache>,
}

impl Nav {
    pub fn build(w: &World, radius: f32, height: f32, step_up: f32, max_slope_deg: f32) -> Nav {
        let nx = ((w.max_x - w.min_x) / CELL).ceil() as usize;
        let ny = ((w.max_y - w.min_y) / CELL).ceil() as usize;
        let slope = max_slope_deg.to_radians().tan();
        let clear = radius + 0.8;
        let mut walk = vec![false; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                let (x, y) = (w.min_x + (i as f32 + 0.5) * CELL, w.min_y + (j as f32 + 0.5) * CELL);
                if x - clear < w.min_x || x + clear > w.max_x || y - clear < w.min_y || y + clear > w.max_y {
                    continue;
                }
                let g = w.ground(x, y);
                let (gx, gy) = ((w.ground(x + 1.0, y) - w.ground(x - 1.0, y)) * 0.5, (w.ground(x, y + 1.0) - w.ground(x, y - 1.0)) * 0.5);
                if (gx * gx + gy * gy).sqrt() > slope {
                    continue;
                }
                // Blocked if any solid that is neither steppable nor head-clearing lies within our radius.
                let mut ok = true;
                'probe: for k in 0..9 {
                    let (px, py) = if k == 8 {
                        (x, y)
                    } else {
                        let a = k as f32 * std::f32::consts::FRAC_PI_4;
                        (x + a.cos() * clear, y + a.sin() * clear)
                    };
                    for &pi in w.prisms_near(px, py) {
                        let p = &w.prisms[pi as usize];
                        let walk_under = p.kind == PrismKind::Bridge && p.base >= g + height;
                        if !walk_under && p.top > g + step_up && p.base < g + height && p.contains_xy(px, py) {
                            ok = false;
                            break 'probe;
                        }
                    }
                }
                walk[j * nx + i] = ok;
            }
        }
        let mut steps = vec![0u8; nx * ny];
        let (sx, sy) = (nx as isize, ny as isize);
        for c in 0..nx * ny {
            let (ci, cj) = ((c as isize) % sx, (c as isize) / sx);
            for (k, (di, dj)) in STEPS.iter().enumerate() {
                let (i, j) = (ci + di, cj + dj);
                if i < 0 || j < 0 || i >= sx || j >= sy || !walk[(j * sx + i) as usize] {
                    continue;
                }
                // No corner cutting on diagonals.
                if *di != 0 && *dj != 0 && (!walk[(cj * sx + i) as usize] || !walk[(j * sx + ci) as usize]) {
                    continue;
                }
                steps[c] |= 1 << k;
            }
        }
        Nav { nx, ny, x0: w.min_x, y0: w.min_y, walk: Arc::new(walk), steps: Arc::new(steps), dist: Arc::new(Mutex::new(Field::empty(nx * ny))), goal: None, cache: None }
    }

    pub fn cell(&self, x: f32, y: f32) -> usize {
        let i = (((x - self.x0) / CELL) as isize).clamp(0, self.nx as isize - 1) as usize;
        let j = (((y - self.y0) / CELL) as isize).clamp(0, self.ny as isize - 1) as usize;
        j * self.nx + i
    }

    pub fn center(&self, c: usize) -> [f32; 2] {
        [self.x0 + ((c % self.nx) as f32 + 0.5) * CELL, self.y0 + ((c / self.nx) as f32 + 0.5) * CELL]
    }

    pub fn walkable(&self, x: f32, y: f32) -> bool {
        self.walk[self.cell(x, y)]
    }

    /// Nearest walkable cell to (x, y), searching outward a few rings.
    pub fn nearest_walkable(&self, x: f32, y: f32) -> Option<usize> {
        let c = self.cell(x, y);
        if self.walk[c] {
            return Some(c);
        }
        let (ci, cj) = ((c % self.nx) as isize, (c / self.nx) as isize);
        let mut best: Option<(f32, usize)> = None;
        for r in 1..14isize {
            for dj in -r..=r {
                for di in -r..=r {
                    if di.abs() != r && dj.abs() != r {
                        continue;
                    }
                    let (i, j) = (ci + di, cj + dj);
                    if i < 0 || j < 0 || i >= self.nx as isize || j >= self.ny as isize {
                        continue;
                    }
                    let n = j as usize * self.nx + i as usize;
                    if self.walk[n] {
                        let p = self.center(n);
                        let d = (p[0] - x).powi(2) + (p[1] - y).powi(2);
                        if best.map_or(true, |b| d < b.0) {
                            best = Some((d, n));
                        }
                    }
                }
            }
            if best.is_some() {
                break;
            }
        }
        best.map(|b| b.1)
    }

    /// Share distance fields with other navs built from the same grid.
    pub fn set_cache(&mut self, cache: DistCache) {
        self.cache = Some(cache);
    }

    /// Recompute the distance field if the goal moved to another cell.
    pub fn set_goal(&mut self, x: f32, y: f32) {
        let Some(g) = self.nearest_walkable(x, y) else { return };
        if self.goal == Some(g) {
            return;
        }
        self.goal = Some(g);
        if let Some(c) = &self.cache {
            let mut c = c.lock().unwrap();
            if let Some(k) = c.iter().position(|e| e.0 == g) {
                let e = c.remove(k);
                self.dist = e.1.clone();
                c.push(e);
                return;
            }
        }
        self.dist = Arc::new(Mutex::new(Field::new(self.nx * self.ny, g)));
        if let Some(c) = &self.cache {
            let mut c = c.lock().unwrap();
            if c.len() >= DIST_CACHE_LEN {
                c.remove(0);
            }
            c.push((g, self.dist.clone()));
        }
    }

    /// Grid distances to cell `g`. Dijkstra with a bucket queue: every step costs at least CELL,
    /// so cells within one bucket (narrower than CELL) cannot improve each other and pop in any order.
    pub fn distance_field(&self, g: usize) -> Vec<f32> {
        let mut dist = vec![f32::MAX; self.nx * self.ny];
        let mut buckets: Vec<Vec<(f32, u32)>> = Vec::new();
        dist[g] = 0.0;
        buckets.push(vec![(0.0, g as u32)]);
        let nx = self.nx as isize;
        let offs: [isize; 8] = STEPS.map(|(di, dj)| dj * nx + di);
        let mut b = 0;
        while b < buckets.len() {
            let mut cur = std::mem::take(&mut buckets[b]);
            while let Some((d, c)) = cur.pop() {
                let c = c as usize;
                if d > dist[c] {
                    continue;
                }
                let mask = self.steps[c];
                for (k, &off) in offs.iter().enumerate() {
                    if mask & (1 << k) == 0 {
                        continue;
                    }
                    let n = (c as isize + off) as usize;
                    let nd = d + if k >= 4 { CELL * 1.4142 } else { CELL };
                    if nd < dist[n] {
                        dist[n] = nd;
                        let k = ((nd / W) as usize).max(b + 1);
                        if k >= buckets.len() {
                            buckets.resize_with(k + 1, Vec::new);
                        }
                        buckets[k].push((nd, n as u32));
                    }
                }
            }
            buckets[b] = cur;
            b += 1;
        }
        dist
    }

    /// Path length from (x, y) to the goal along the grid, or None if unreachable.
    pub fn path_len(&self, x: f32, y: f32) -> Option<f32> {
        let c = self.nearest_walkable(x, y)?;
        let mut f = self.dist.lock().unwrap();
        f.settle(&self.steps, self.nx, c);
        (f.dist[c] < f32::MAX).then_some(f.dist[c])
    }

    /// Point to steer toward from (x, y): a few steps down the distance field toward the goal.
    pub fn steer(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let mut c = self.nearest_walkable(x, y)?;
        let mut f = self.dist.lock().unwrap();
        f.settle(&self.steps, self.nx, c);
        if f.dist[c] == f32::MAX {
            return None;
        }
        let nx = self.nx as isize;
        // Standing in a blocked cell (pressed against a wall): step back onto the grid toward the goal.
        // The nearest walkable cell often lies behind us, and heading there and back again cycles.
        let here = self.cell(x, y);
        if !self.walk[here] {
            let (hi, hj) = ((here % self.nx) as isize, (here / self.nx) as isize);
            let mut best: Option<(f32, usize)> = None;
            for dj in -2..=2isize {
                for di in -2..=2isize {
                    let (i, j) = (hi + di, hj + dj);
                    if i < 0 || j < 0 || i >= nx || j >= self.ny as isize {
                        continue;
                    }
                    let n = (j * nx + i) as usize;
                    if !self.walk[n] {
                        continue;
                    }
                    f.settle(&self.steps, self.nx, n);
                    if f.dist[n] == f32::MAX {
                        continue;
                    }
                    let p = self.center(n);
                    let cost = (p[0] - x).hypot(p[1] - y) + f.dist[n];
                    if best.map_or(true, |b| cost < b.0) {
                        best = Some((cost, n));
                    }
                }
            }
            return Some(self.center(best.map_or(c, |b| b.1)));
        }
        let dist = &f.dist;
        for _ in 0..4 {
            let (ci, cj) = ((c as isize) % nx, (c as isize) / nx);
            let mut best = c;
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                let (i, j) = (ci + di, cj + dj);
                if i < 0 || j < 0 || i >= nx || j >= self.ny as isize {
                    continue;
                }
                let n = (j * nx + i) as usize;
                if !self.walk[n] || (di != 0 && dj != 0 && (!self.walk[(cj * nx + i) as usize] || !self.walk[(j * nx + ci) as usize])) {
                    continue;
                }
                if dist[n] < dist[best] {
                    best = n;
                }
            }
            if best == c {
                break;
            }
            c = best;
        }
        Some(self.center(c))
    }

    pub fn walkable_fraction(&self) -> f32 {
        self.walk.iter().filter(|w| **w).count() as f32 / self.walk.len() as f32
    }
}
