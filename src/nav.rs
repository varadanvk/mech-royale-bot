//! Ground navigation: a coarse walkability grid and a distance field toward the current goal.
//! Roofs are ignored: the bot paths around buildings, not over them.
use crate::world::{PrismKind, World};
use std::collections::BinaryHeap;

pub const CELL: f32 = 3.0;

pub struct Nav {
    pub nx: usize,
    pub ny: usize,
    x0: f32,
    y0: f32,
    walk: Vec<bool>,
    dist: Vec<f32>,
    goal: Option<usize>,
}

#[derive(PartialEq)]
struct Node(f32, usize);
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.partial_cmp(&self.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
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
        Nav { nx, ny, x0: w.min_x, y0: w.min_y, walk, dist: vec![f32::MAX; nx * ny], goal: None }
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

    /// Recompute the distance field if the goal moved to another cell.
    pub fn set_goal(&mut self, x: f32, y: f32) {
        let Some(g) = self.nearest_walkable(x, y) else { return };
        if self.goal == Some(g) {
            return;
        }
        self.goal = Some(g);
        self.dist.iter_mut().for_each(|d| *d = f32::MAX);
        let mut heap = BinaryHeap::new();
        self.dist[g] = 0.0;
        heap.push(Node(0.0, g));
        let nx = self.nx as isize;
        let ny = self.ny as isize;
        while let Some(Node(d, c)) = heap.pop() {
            if d > self.dist[c] {
                continue;
            }
            let (ci, cj) = ((c as isize) % nx, (c as isize) / nx);
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)] {
                let (i, j) = (ci + di, cj + dj);
                if i < 0 || j < 0 || i >= nx || j >= ny {
                    continue;
                }
                let n = (j * nx + i) as usize;
                if !self.walk[n] {
                    continue;
                }
                // No corner cutting on diagonals.
                if di != 0 && dj != 0 && (!self.walk[(cj * nx + i) as usize] || !self.walk[(j * nx + ci) as usize]) {
                    continue;
                }
                let nd = d + if di != 0 && dj != 0 { CELL * 1.4142 } else { CELL };
                if nd < self.dist[n] {
                    self.dist[n] = nd;
                    heap.push(Node(nd, n));
                }
            }
        }
    }

    /// Path length from (x, y) to the goal along the grid, or None if unreachable.
    pub fn path_len(&self, x: f32, y: f32) -> Option<f32> {
        let c = self.nearest_walkable(x, y)?;
        (self.dist[c] < f32::MAX).then_some(self.dist[c])
    }

    /// Point to steer toward from (x, y): a few steps down the distance field toward the goal.
    pub fn steer(&self, x: f32, y: f32) -> Option<[f32; 2]> {
        let mut c = self.nearest_walkable(x, y)?;
        if self.dist[c] == f32::MAX {
            return None;
        }
        let nx = self.nx as isize;
        // If we are standing in a blocked cell, first get back onto the grid.
        if !self.walk[self.cell(x, y)] {
            return Some(self.center(c));
        }
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
                if self.dist[n] < self.dist[best] {
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
