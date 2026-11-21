//! Static map: terrain height grid plus extruded-polygon prisms (buildings, stadium, bridges).
//! Frame: x east, y north, z up, meters (protocol.md section 5 and 11).
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrismKind {
    Building,
    Stadium,
    Bridge,
}

#[derive(Clone, Debug)]
pub struct Prism {
    pub kind: PrismKind,
    pub base: f32,
    pub top: f32,
    /// Outer ring first, then holes. Each loop is a closed polygon (last vertex joins the first).
    pub loops: Vec<Vec<[f32; 2]>>,
    pub min: [f32; 2],
    pub max: [f32; 2],
}

impl Prism {
    /// Even-odd test over the ring and its holes: inside the ring and outside every hole.
    pub fn contains_xy(&self, x: f32, y: f32) -> bool {
        if x < self.min[0] || x > self.max[0] || y < self.min[1] || y > self.max[1] {
            return false;
        }
        let mut inside = false;
        for lp in &self.loops {
            let n = lp.len();
            let mut j = n - 1;
            for i in 0..n {
                let (a, b) = (lp[i], lp[j]);
                if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
                    inside = !inside;
                }
                j = i;
            }
        }
        inside
    }
}

#[derive(Clone, Debug)]
pub struct Tree {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub height: f32,
    pub crown_r: f32,
    pub crown_base: f32,
    pub trunk_r: f32,
    pub crown_x: f32,
    pub crown_y: f32,
    /// Crown ellipsoid: centre height and vertical radius (horizontal radius is `crown_r`).
    pub crown_z: f32,
    pub crown_rz: f32,
    /// Trunk cylinder spans `z - 1 ..= trunk_top`.
    pub trunk_top: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub t: f32,
    /// Index into `World::prisms`, or `None` for terrain.
    pub prism: Option<u32>,
}

pub struct World {
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
    // terrain grid
    pub t_min_x: f32,
    pub t_min_y: f32,
    pub t_step: f32,
    pub nx: usize,
    pub ny: usize,
    pub heights: Vec<f32>,
    pub prisms: Vec<Prism>,
    pub trees: Vec<Tree>,
    pub canopy_density: f32,
    pub canopy_block_t: f32,
    // uniform grid over the bounds: prism ids per cell
    cell: f32,
    gx: usize,
    gy: usize,
    cells: Vec<Vec<u32>>,
    tree_cells: Vec<Vec<u32>>,
}

#[derive(Deserialize)]
struct RawBounds {
    #[serde(rename = "minX")]
    min_x: f32,
    #[serde(rename = "maxX")]
    max_x: f32,
    #[serde(rename = "minY")]
    min_y: f32,
    #[serde(rename = "maxY")]
    max_y: f32,
}
#[derive(Deserialize)]
struct RawTerrain {
    #[serde(rename = "minX")]
    min_x: f32,
    #[serde(rename = "minY")]
    min_y: f32,
    step: f32,
    nx: usize,
    ny: usize,
    scale: f32,
}
#[derive(Deserialize)]
struct RawPrism {
    kind: String,
    base: f32,
    top: f32,
    ring: Vec<f32>,
    #[serde(default)]
    holes: Vec<Vec<f32>>,
}
#[derive(Deserialize)]
struct RawTrees {
    fields: Vec<String>,
    data: Vec<f32>,
}
#[derive(Deserialize)]
struct RawRules {
    #[serde(rename = "canopyDensity", default)]
    canopy_density: f32,
    #[serde(rename = "canopyBlockT", default)]
    canopy_block_t: f32,
}
#[derive(Deserialize)]
struct RawWorld {
    bounds: RawBounds,
    terrain: RawTerrain,
    prisms: Vec<RawPrism>,
    trees: RawTrees,
    rules: RawRules,
}

fn pairs(flat: &[f32]) -> Vec<[f32; 2]> {
    flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
}

const CELL: f32 = 16.0;

impl World {
    /// Load `v2/world.json` and `v2/terrain.i16` from the map directory (`hello.map.dir`).
    pub fn load(map_dir: &Path) -> Result<World, String> {
        let wj = std::fs::read(map_dir.join("v2/world.json")).map_err(|e| format!("world.json: {e}"))?;
        let raw: RawWorld = serde_json::from_slice(&wj).map_err(|e| format!("world.json: {e}"))?;
        let tb = std::fs::read(map_dir.join("v2/terrain.i16")).map_err(|e| format!("terrain.i16: {e}"))?;
        let t = &raw.terrain;
        if tb.len() != t.nx * t.ny * 2 {
            return Err(format!("terrain.i16 has {} bytes, expected {}", tb.len(), t.nx * t.ny * 2));
        }
        let heights: Vec<f32> =
            tb.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 * t.scale).collect();

        let mut prisms = Vec::with_capacity(raw.prisms.len());
        for p in &raw.prisms {
            let kind = match p.kind.as_str() {
                "bridge" => PrismKind::Bridge,
                "stadium" => PrismKind::Stadium,
                _ => PrismKind::Building,
            };
            let mut loops = vec![pairs(&p.ring)];
            loops.extend(p.holes.iter().map(|h| pairs(h)).filter(|h| h.len() >= 3));
            if loops[0].len() < 3 {
                continue;
            }
            let (mut mn, mut mx) = ([f32::MAX; 2], [f32::MIN; 2]);
            for v in &loops[0] {
                for k in 0..2 {
                    mn[k] = mn[k].min(v[k]);
                    mx[k] = mx[k].max(v[k]);
                }
            }
            prisms.push(Prism { kind, base: p.base, top: p.top, loops, min: mn, max: mx });
        }

        let nf = raw.trees.fields.len().max(1);
        let col = |name: &str| raw.trees.fields.iter().position(|f| f == name);
        let get = |row: &[f32], c: Option<usize>, d: f32| c.map(|i| row[i]).unwrap_or(d);
        let (cx, cy, cz, ch, ccr, ccb, ctr, ccx, ccy) = (
            col("x"), col("y"), col("z"), col("height"), col("crownR"), col("crownBase"), col("trunkR"),
            col("crownX"), col("crownY"),
        );
        let trees = raw
            .trees
            .data
            .chunks_exact(nf)
            .map(|r| {
                let (x, y) = (get(r, cx, 0.0), get(r, cy, 0.0));
                let (z, height, crown_base) = (get(r, cz, 0.0), get(r, ch, 0.0), get(r, ccb, 0.0));
                Tree {
                    x,
                    y,
                    z,
                    height,
                    crown_r: get(r, ccr, 0.0),
                    crown_base,
                    crown_z: z + 0.5 * (crown_base + height),
                    crown_rz: (0.5 * (height - crown_base)).max(0.5),
                    trunk_top: z + (crown_base + 0.5 * (height - crown_base)).max(3.0),
                    trunk_r: get(r, ctr, 0.0),
                    crown_x: get(r, ccx, x),
                    crown_y: get(r, ccy, y),
                }
            })
            .collect::<Vec<Tree>>();

        let b = &raw.bounds;
        let gx = (((b.max_x - b.min_x) / CELL).ceil() as usize).max(1);
        let gy = (((b.max_y - b.min_y) / CELL).ceil() as usize).max(1);
        let mut w = World {
            min_x: b.min_x,
            min_y: b.min_y,
            max_x: b.max_x,
            max_y: b.max_y,
            t_min_x: t.min_x,
            t_min_y: t.min_y,
            t_step: t.step,
            nx: t.nx,
            ny: t.ny,
            heights,
            prisms,
            trees,
            canopy_density: raw.rules.canopy_density,
            canopy_block_t: raw.rules.canopy_block_t,
            cell: CELL,
            gx,
            gy,
            cells: vec![Vec::new(); gx * gy],
            tree_cells: vec![Vec::new(); gx * gy],
        };
        for (i, p) in w.prisms.iter().enumerate() {
            let (i0, j0) = w.cell_of(p.min[0], p.min[1]);
            let (i1, j1) = w.cell_of(p.max[0], p.max[1]);
            for j in j0..=j1 {
                for c in i0..=i1 {
                    w.cells[j * gx + c].push(i as u32);
                }
            }
        }
        for i in 0..w.trees.len() {
            let t = &w.trees[i];
            let r = t.crown_r;
            let (i0, j0) = w.cell_of((t.crown_x - r).min(t.x - t.trunk_r), (t.crown_y - r).min(t.y - t.trunk_r));
            let (i1, j1) = w.cell_of((t.crown_x + r).max(t.x + t.trunk_r), (t.crown_y + r).max(t.y + t.trunk_r));
            for j in j0..=j1 {
                for c in i0..=i1 {
                    w.tree_cells[j * gx + c].push(i as u32);
                }
            }
        }
        Ok(w)
    }

    /// Ids of the trees whose crown or trunk may touch the ground track of the segment a-b.
    fn trees_along(&self, a: [f32; 3], b: [f32; 3], out: &mut Vec<u32>) {
        out.clear();
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let steps = ((len / (self.cell * 0.5)).ceil() as usize).max(1);
        let mut last = usize::MAX;
        for s in 0..=steps {
            let f = s as f32 / steps as f32;
            let (ci, cj) = self.cell_of(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f);
            let c = cj * self.gx + ci;
            if c != last {
                last = c;
                out.extend_from_slice(&self.tree_cells[c]);
            }
        }
        out.sort_unstable();
        out.dedup();
    }

    /// Fraction of sight that survives the tree crowns between two points: exp(-density * metres of crown).
    pub fn canopy_transmittance(&self, a: [f32; 3], b: [f32; 3]) -> f32 {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let mut ids = Vec::new();
        self.trees_along(a, b, &mut ids);
        let mut through = 0.0f32;
        for &i in &ids {
            let t = &self.trees[i as usize];
            // Segment against the crown ellipsoid, in the ellipsoid's unit-sphere space.
            let (sx, sy, sz) = ((a[0] - t.crown_x) / t.crown_r, (a[1] - t.crown_y) / t.crown_r, (a[2] - t.crown_z) / t.crown_rz);
            let (ux, uy, uz) = (d[0] / t.crown_r, d[1] / t.crown_r, d[2] / t.crown_rz);
            let qa = ux * ux + uy * uy + uz * uz;
            if qa < 1e-9 {
                continue;
            }
            let qb = 2.0 * (sx * ux + sy * uy + sz * uz);
            let qc = sx * sx + sy * sy + sz * sz - 1.0;
            let disc = qb * qb - 4.0 * qa * qc;
            if disc <= 0.0 {
                continue;
            }
            let r = disc.sqrt();
            let (t0, t1) = ((-qb - r) / (2.0 * qa), (-qb + r) / (2.0 * qa));
            if t1 < 0.0 || t0 > 1.0 {
                continue;
            }
            through += (t1.min(1.0) - t0.max(0.0)) * len;
        }
        (-self.canopy_density * through).exp()
    }

    /// True if a tree trunk crosses the segment a-b. Trunks block sight but not shots or movement.
    pub fn trunk_between(&self, a: [f32; 3], b: [f32; 3]) -> bool {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let qa = d[0] * d[0] + d[1] * d[1];
        if qa < 1e-9 {
            return false;
        }
        let mut ids = Vec::new();
        self.trees_along(a, b, &mut ids);
        ids.iter().any(|&i| {
            let t = &self.trees[i as usize];
            let (fx, fy) = (a[0] - t.x, a[1] - t.y);
            let qb = 2.0 * (fx * d[0] + fy * d[1]);
            let qc = fx * fx + fy * fy - t.trunk_r * t.trunk_r;
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                return false;
            }
            let s = (-qb - disc.sqrt()) / (2.0 * qa);
            if !(0.0..1.0).contains(&s) {
                return false;
            }
            let z = a[2] + d[2] * s;
            z >= t.z - 1.0 && z <= t.trunk_top
        })
    }

    /// Sight line as the game's viewer computes it: no building, terrain or trunk in the way, and
    /// enough light left after the crowns (`canopyBlockT`).
    pub fn sees(&self, a: [f32; 3], b: [f32; 3]) -> bool {
        self.clear(a, b) && !self.trunk_between(a, b) && self.canopy_transmittance(a, b) >= self.canopy_block_t
    }

    fn cell_of(&self, x: f32, y: f32) -> (usize, usize) {
        let i = (((x - self.min_x) / self.cell) as isize).clamp(0, self.gx as isize - 1) as usize;
        let j = (((y - self.min_y) / self.cell) as isize).clamp(0, self.gy as isize - 1) as usize;
        (i, j)
    }

    /// Prism ids whose bounding box touches the cell containing (x, y).
    pub fn prisms_near(&self, x: f32, y: f32) -> &[u32] {
        let (i, j) = self.cell_of(x, y);
        &self.cells[j * self.gx + i]
    }

    /// Terrain height, bilinear over the 1 m vertex grid, clamped at the edges.
    pub fn ground(&self, x: f32, y: f32) -> f32 {
        let fx = ((x - self.t_min_x) / self.t_step).clamp(0.0, (self.nx - 1) as f32);
        let fy = ((y - self.t_min_y) / self.t_step).clamp(0.0, (self.ny - 1) as f32);
        let (i, j) = ((fx as usize).min(self.nx - 2), (fy as usize).min(self.ny - 2));
        let (u, v) = (fx - i as f32, fy - j as f32);
        let h = |i: usize, j: usize| self.heights[j * self.nx + i];
        let a = h(i, j) * (1.0 - u) + h(i + 1, j) * u;
        let b = h(i, j + 1) * (1.0 - u) + h(i + 1, j + 1) * u;
        a * (1.0 - v) + b * v
    }

    /// Highest surface (terrain or prism top) at (x, y) that is at or below `z_max`.
    pub fn surface_below(&self, x: f32, y: f32, z_max: f32) -> f32 {
        let mut s = self.ground(x, y);
        for &pi in self.prisms_near(x, y) {
            let p = &self.prisms[pi as usize];
            if p.top > s && p.top <= z_max && p.contains_xy(x, y) {
                s = p.top;
            }
        }
        s
    }

    /// Lowest prism underside above `z` at (x, y): what a jet or a head would hit. `f32::MAX` if open sky.
    pub fn ceiling_above(&self, x: f32, y: f32, z: f32) -> f32 {
        let mut c = f32::MAX;
        for &pi in self.prisms_near(x, y) {
            let p = &self.prisms[pi as usize];
            if p.base >= z && p.base < c && p.contains_xy(x, y) {
                c = p.base;
            }
        }
        c
    }

    /// True if the point is inside a prism or under the terrain.
    pub fn solid(&self, x: f32, y: f32, z: f32) -> bool {
        if z < self.ground(x, y) {
            return true;
        }
        self.prisms_near(x, y).iter().any(|&pi| {
            let p = &self.prisms[pi as usize];
            z > p.base && z < p.top && p.contains_xy(x, y)
        })
    }

    /// First hit of the segment `o + d*t`, t in [0, max_t] (`d` need not be unit) against prisms and terrain.
    pub fn raycast(&self, o: [f32; 3], d: [f32; 3], max_t: f32) -> Option<RayHit> {
        let mut best = max_t;
        let mut hit: Option<u32> = None;
        // Walk the grid cells the segment's ground track crosses, testing each prism once.
        let mut seen: [u32; 8] = [u32::MAX; 8];
        let mut seen_n = 0usize;
        let hl = (d[0] * d[0] + d[1] * d[1]).sqrt();
        let steps = ((hl * max_t / (self.cell * 0.5)).ceil() as usize).max(1);
        let mut last_cell = usize::MAX;
        for s in 0..=steps {
            let t = max_t * s as f32 / steps as f32;
            if t > best + self.cell {
                break;
            }
            let (ci, cj) = self.cell_of(o[0] + d[0] * t, o[1] + d[1] * t);
            let c = cj * self.gx + ci;
            if c == last_cell {
                continue;
            }
            last_cell = c;
            'p: for &pi in &self.cells[c] {
                for k in 0..seen_n.min(8) {
                    if seen[k] == pi {
                        continue 'p;
                    }
                }
                seen[seen_n % 8] = pi;
                seen_n += 1;
                if let Some(t) = ray_prism(&self.prisms[pi as usize], o, d, best) {
                    if t < best {
                        best = t;
                        hit = Some(pi);
                    }
                }
            }
        }
        // Terrain: march at half-grid steps up to the nearest prism hit, then bisect the crossing.
        let dt = if hl > 1e-6 { 0.5 * self.t_step / hl } else { best.max(1e-3) };
        let below = |t: f32| o[2] + d[2] * t < self.ground(o[0] + d[0] * t, o[1] + d[1] * t);
        let mut t0 = 0.0f32;
        if !below(0.0) {
            loop {
                let t1 = (t0 + dt).min(best);
                if below(t1) {
                    let (mut a, mut b) = (t0, t1);
                    for _ in 0..12 {
                        let m = 0.5 * (a + b);
                        if below(m) {
                            b = m
                        } else {
                            a = m
                        }
                    }
                    return Some(RayHit { t: b, prism: None });
                }
                if t1 >= best {
                    break;
                }
                t0 = t1;
            }
        } else {
            return Some(RayHit { t: 0.0, prism: None });
        }
        hit.map(|p| RayHit { t: best, prism: Some(p) })
    }

    /// True if nothing solid lies between the two points.
    pub fn clear(&self, a: [f32; 3], b: [f32; 3]) -> bool {
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        self.raycast(a, d, 1.0).is_none()
    }
}

/// Segment vs extruded polygon. Returns the smallest t in [0, max_t) where the segment enters the solid.
fn ray_prism(p: &Prism, o: [f32; 3], d: [f32; 3], max_t: f32) -> Option<f32> {
    // Slab test on the bounding box first.
    let (mut t_in, mut t_out) = (0.0f32, max_t);
    let lo = [p.min[0], p.min[1], p.base];
    let hi = [p.max[0], p.max[1], p.top];
    for k in 0..3 {
        if d[k].abs() < 1e-9 {
            if o[k] < lo[k] || o[k] > hi[k] {
                return None;
            }
        } else {
            let (mut a, mut b) = ((lo[k] - o[k]) / d[k], (hi[k] - o[k]) / d[k]);
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            t_in = t_in.max(a);
            t_out = t_out.min(b);
            if t_in > t_out {
                return None;
            }
        }
    }
    let mut best = f32::MAX;
    // Starting inside the solid counts as an immediate hit.
    if o[2] > p.base && o[2] < p.top && p.contains_xy(o[0], o[1]) {
        return Some(0.0);
    }
    // Side walls: every edge of every loop, between base and top.
    for lp in &p.loops {
        let n = lp.len();
        let mut j = n - 1;
        for i in 0..n {
            let (a, b) = (lp[j], lp[i]);
            j = i;
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let den = d[0] * ey - d[1] * ex;
            if den.abs() < 1e-9 {
                continue;
            }
            let (ax, ay) = (a[0] - o[0], a[1] - o[1]);
            let t = (ax * ey - ay * ex) / den;
            if t < 0.0 || t >= best || t >= max_t {
                continue;
            }
            let u = (ax * d[1] - ay * d[0]) / den;
            if !(0.0..=1.0).contains(&u) {
                continue;
            }
            let z = o[2] + d[2] * t;
            if z >= p.base && z <= p.top {
                best = t;
            }
        }
    }
    // Caps: the top from above, the base from below.
    if d[2].abs() > 1e-9 {
        for (plane, from_above) in [(p.top, true), (p.base, false)] {
            if (d[2] < 0.0) != from_above {
                continue;
            }
            let t = (plane - o[2]) / d[2];
            if t >= 0.0 && t < best && t < max_t && p.contains_xy(o[0] + d[0] * t, o[1] + d[1] * t) {
                best = t;
            }
        }
    }
    (best < max_t).then_some(best)
}
