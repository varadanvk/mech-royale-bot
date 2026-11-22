//! Sight table: for every cell of a coarse grid over walkable ground, how far away another mech can stand in
//! each of `SECTORS` compass directions and still have a clear line to a mech standing in the cell. From it
//! follow how exposed each cell is, how many approach lanes it has, and a set of sheltered spots to fight from
//! (few directions open, back and sides covered), each with a second spot nearby to fall back to.
//!
//! Lines are geometric (buildings, bridges, terrain); tree crowns are ignored, so the table errs on the side of
//! "can be seen".
use crate::nav::Nav;
use crate::world::World;
use std::f32::consts::TAU;

/// Grid cell size, metres.
pub const GRID: f32 = 6.0;
/// Compass sectors per cell.
pub const SECTORS: usize = 64;
/// Sight range the table covers, metres.
pub const RANGE: f32 = 200.0;
/// A sector counts as an open approach when the line reaches further than this.
pub const OPEN: f32 = 40.0;
/// Height above the ground of the watched point (between a mech's centre and its top) and of a watcher's eye.
const TARGET_H: f32 = 6.0;
const EYE_H: f32 = 8.0;
const SEC: f32 = TAU / SECTORS as f32;
/// Spots are at least this far apart.
const SPOT_SPACING: f32 = 20.0;

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// A sheltered place to fight from.
#[derive(Clone, Debug)]
pub struct Spot {
    pub pos: [f32; 2],
    pub cell: usize,
    /// Fraction of the ground within `RANGE` with a line to the spot.
    pub exposure: f32,
    /// Separate runs of open sectors (approach lanes).
    pub lanes: u8,
    /// Direction of the widest lane (where trouble comes from).
    pub lane_yaw: f32,
    /// Lower is better.
    pub score: f32,
    /// Nearest other spot to fall back to.
    pub fallback: Option<usize>,
}

/// The safe area as far as choosing a place is concerned.
#[derive(Clone, Copy, Debug)]
pub struct Zone {
    pub center: [f32; 2],
    pub radius: f32,
    pub next_center: [f32; 2],
    pub next_radius: f32,
    /// The shrink to the next circle is close: the place must be inside it too.
    pub soon: bool,
}

impl Zone {
    pub fn holds(&self, p: [f32; 2], margin: f32) -> bool {
        let mn = margin.min(self.next_radius * 0.3);
        dist(p, self.center) < self.radius - margin && (!self.soon || dist(p, self.next_center) < self.next_radius - mn)
    }
}

/// A known enemy for choosing a position: where it is and its health fraction.
#[derive(Clone, Copy, Debug)]
pub struct Foe {
    pub pos: [f32; 2],
    pub hp: f32,
}

/// A place chosen to go to.
#[derive(Clone, Copy, Debug)]
pub struct Pick {
    pub pos: [f32; 2],
    /// Way to face while there.
    pub yaw: f32,
    /// Index into `spots`, if it is one.
    pub spot: Option<usize>,
    /// For an isolating position: where the one enemy with a line to it is.
    pub foe: Option<[f32; 2]>,
    pub score: f32,
}

pub struct SightTable {
    x0: f32,
    y0: f32,
    pub nx: usize,
    pub ny: usize,
    /// Standing point of each cell (a walkable point near the cell centre), if any.
    pt: Vec<Option<[f32; 2]>>,
    /// Per cell and sector: metres of clear line (0 for cells without a standing point).
    range: Vec<u8>,
    pub exposure: Vec<f32>,
    pub open: Vec<u8>,
    pub lanes: Vec<u8>,
    pub lane_yaw: Vec<f32>,
    /// Reachable on foot from the main part of the map.
    pub reach: Vec<bool>,
    pub spots: Vec<Spot>,
    /// Per cell: distance to the nearest spot.
    pub spot_d: Vec<f32>,
}

impl SightTable {
    pub fn build(w: &World, nav: &Nav) -> SightTable {
        let (x0, y0) = (w.min_x, w.min_y);
        let nx = ((w.max_x - w.min_x) / GRID).ceil() as usize;
        let ny = ((w.max_y - w.min_y) / GRID).ceil() as usize;
        let n = nx * ny;
        // Walking distances from the middle of the map tell which cells can be reached.
        let mid = [(w.min_x + w.max_x) * 0.5, (w.min_y + w.max_y) * 0.5];
        let field = nav.nearest_walkable(mid[0], mid[1]).map(|g| nav.distance_field(g));
        let mut t = SightTable {
            x0,
            y0,
            nx,
            ny,
            pt: vec![None; n],
            range: vec![0; n * SECTORS],
            exposure: vec![1.0; n],
            open: vec![SECTORS as u8; n],
            lanes: vec![0; n],
            lane_yaw: vec![0.0; n],
            reach: vec![false; n],
            spots: Vec::new(),
            spot_d: vec![f32::MAX; n],
        };
        let (lo, hi) = ([w.min_x + 1.0, w.min_y + 1.0], [w.max_x - 1.0, w.max_y - 1.0]);
        let dirs: Vec<[f32; 2]> = (0..SECTORS).map(|s| [(s as f32 * SEC).cos(), (s as f32 * SEC).sin()]).collect();
        for c in 0..n {
            let g = [x0 + ((c % nx) as f32 + 0.5) * GRID, y0 + ((c / nx) as f32 + 0.5) * GRID];
            let p = [[0.0f32, 0.0], [2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]]
                .iter()
                .map(|o| [g[0] + o[0], g[1] + o[1]])
                .find(|q| nav.walkable(q[0], q[1]));
            let Some(p) = p else { continue };
            t.pt[c] = Some(p);
            t.reach[c] = field.as_ref().map_or(true, |f| f[nav.cell(p[0], p[1])] < f32::MAX);
            let o = [p[0], p[1], w.ground(p[0], p[1]) + TARGET_H];
            let r = &mut t.range[c * SECTORS..(c + 1) * SECTORS];
            for (s, d) in dirs.iter().enumerate() {
                // Distance to the map edge along this direction (nobody stands beyond it).
                let mut l = RANGE;
                for k in 0..2 {
                    if d[k] > 1e-6 {
                        l = l.min((hi[k] - p[k]) / d[k]);
                    } else if d[k] < -1e-6 {
                        l = l.min((lo[k] - p[k]) / d[k]);
                    }
                }
                let l = l.max(0.0);
                let e = [p[0] + d[0] * l, p[1] + d[1] * l];
                let ez = w.ground(e[0], e[1]) + EYE_H;
                let hit = w.raycast(o, [e[0] - o[0], e[1] - o[1], ez - o[2]], 1.0);
                let m = hit.map_or(l, |h| h.t * l);
                r[s] = m.round().clamp(0.0, 255.0) as u8;
            }
            let (expo, open, lanes, yaw) = shape(r);
            t.exposure[c] = expo;
            t.open[c] = open;
            t.lanes[c] = lanes;
            t.lane_yaw[c] = yaw;
        }
        t.find_spots(nav);
        t
    }

    fn find_spots(&mut self, nav: &Nav) {
        let mut cands: Vec<(f32, usize)> = (0..self.pt.len())
            .filter(|&c| self.pt[c].is_some() && self.reach[c] && (self.open[c] as usize) * 8 <= SECTORS * 3 && self.lanes[c] <= 3)
            .map(|c| (spot_score(self.exposure[c], self.lanes[c]), c))
            .collect();
        cands.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        let mut spots: Vec<Spot> = Vec::new();
        for (score, c) in cands {
            let p = self.pt[c].unwrap();
            if spots.iter().any(|s| dist(s.pos, p) < SPOT_SPACING) {
                continue;
            }
            spots.push(Spot { pos: p, cell: c, exposure: self.exposure[c], lanes: self.lanes[c], lane_yaw: self.lane_yaw[c], score, fallback: None });
        }
        // Fall-back partner: the nearest other spot a short walk away.
        let mut nav = nav.clone();
        for i in 0..spots.len() {
            let p = spots[i].pos;
            let mut near: Vec<(f32, usize)> =
                spots.iter().enumerate().filter(|(j, s)| *j != i && dist(s.pos, p) < 90.0).map(|(j, s)| (dist(s.pos, p), j)).collect();
            near.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
            if near.is_empty() {
                continue;
            }
            nav.set_goal(p[0], p[1]);
            spots[i].fallback = near.iter().take(4).find(|(d, j)| nav.path_len(spots[*j].pos[0], spots[*j].pos[1]).map_or(false, |l| l < d * 2.0 + 20.0)).map(|x| x.1);
        }
        for c in 0..self.pt.len() {
            if let Some(p) = self.pt[c] {
                self.spot_d[c] = spots.iter().map(|s| dist(s.pos, p)).fold(f32::MAX, f32::min);
            }
        }
        self.spots = spots;
    }

    /// Table bytes (sector ranges only).
    pub fn bytes(&self) -> usize {
        self.range.len()
    }

    /// Cells with a standing point.
    pub fn cells_used(&self) -> usize {
        self.pt.iter().filter(|p| p.is_some()).count()
    }

    /// Cell holding (x, y), if it has a standing point.
    pub fn cell_at(&self, x: f32, y: f32) -> Option<usize> {
        let i = ((x - self.x0) / GRID).floor();
        let j = ((y - self.y0) / GRID).floor();
        if i < 0.0 || j < 0.0 || i >= self.nx as f32 || j >= self.ny as f32 {
            return None;
        }
        let c = j as usize * self.nx + i as usize;
        self.pt[c].map(|_| c)
    }

    pub fn point(&self, c: usize) -> Option<[f32; 2]> {
        self.pt[c]
    }

    /// Clear-line distance from cell `c` toward `yaw` (the nearest sector ray).
    pub fn range_at(&self, c: usize, yaw: f32) -> f32 {
        let s = ((yaw.rem_euclid(TAU) / SEC).round() as usize) % SECTORS;
        self.range[c * SECTORS + s] as f32
    }

    /// Whether a mech standing at `from` has a line to a mech at `at` (approximately; unknown ground counts as
    /// seen).
    pub fn visible(&self, at: [f32; 2], from: [f32; 2]) -> bool {
        let d = dist(at, from);
        if d > RANGE {
            return false;
        }
        if d < GRID {
            return true;
        }
        match self.cell_at(at[0], at[1]) {
            Some(c) => d <= self.range_at(c, (from[1] - at[1]).atan2(from[0] - at[0])) + 3.0,
            None => true,
        }
    }

    /// How many of `foes` have a line to `at`.
    pub fn seen_by(&self, at: [f32; 2], foes: &[Foe]) -> usize {
        foes.iter().filter(|f| self.visible(at, f.pos)).count()
    }

    /// Level 1: the best sheltered spot to hold. Inside the safe area, a short walk away, not past the enemies,
    /// few of them with a line to it. Falls back to the least exposed cells nearby when no spot qualifies.
    pub fn pick_spot(&self, nav: &Nav, me: [f32; 2], foes: &[Foe], zone: &Zone, banned: &[usize], walk: f32) -> Option<Pick> {
        let mut nav = nav.clone();
        nav.set_goal(me[0], me[1]);
        let walk = walk.max(1.0);
        let judge = |nav: &mut Nav, p: [f32; 2], base: f32| -> Option<f32> {
            let travel = nav.path_len(p[0], p[1])?;
            if travel > 320.0 {
                return None;
            }
            let dm = dist(me, p);
            // An enemy nearer the place than we are, in front of it: we would be running into them.
            let past = foes.iter().filter(|f| dist(f.pos, p) < 150.0 && dist(f.pos, p) < dm * 0.8).count();
            let seen = self.seen_by(p, foes);
            Some(base * 200.0 + travel / walk * 2.0 + seen as f32 * 12.0 + past as f32 * 40.0)
        };
        let mut best: Option<Pick> = None;
        for (i, s) in self.spots.iter().enumerate() {
            if banned.contains(&i) || dist(s.pos, me) > 280.0 || !zone.holds(s.pos, 8.0) {
                continue;
            }
            if let Some(sc) = judge(&mut nav, s.pos, s.score) {
                if best.map_or(true, |b| sc < b.score) {
                    best = Some(Pick { pos: s.pos, yaw: s.lane_yaw, spot: Some(i), foe: None, score: sc });
                }
            }
        }
        if best.is_some() {
            return best;
        }
        // No spot: the least exposed cells within reach.
        let mut cells: Vec<(f32, usize)> = self
            .cells_near(me, 150.0)
            .filter(|&c| self.reach[c] && zone.holds(self.pt[c].unwrap(), 6.0))
            .map(|c| (spot_score(self.exposure[c], self.lanes[c]) * 200.0 + dist(self.pt[c].unwrap(), me) / walk * 2.0, c))
            .collect();
        cells.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        for &(_, c) in cells.iter().take(8) {
            let p = self.pt[c].unwrap();
            if let Some(sc) = judge(&mut nav, p, spot_score(self.exposure[c], self.lanes[c])) {
                if best.map_or(true, |b| sc < b.score) {
                    best = Some(Pick { pos: p, yaw: self.lane_yaw[c], spot: None, foe: None, score: sc });
                }
            }
        }
        best
    }

    /// Level 2: a position within reach where exactly one of `foes` has a line to us (the weaker and nearer the
    /// better) and the rest do not, weighing the walk there, exposure on the way, how sheltered it is and how
    /// far the nearest spot to fall back to is. `keep` is the position chosen before: it stays unless another is
    /// clearly better.
    pub fn pick_isolating(&self, nav: &Nav, me: [f32; 2], foes: &[Foe], zone: &Zone, walk: f32, keep: Option<[f32; 2]>) -> Option<Pick> {
        if foes.is_empty() {
            return None;
        }
        let mut nav = nav.clone();
        nav.set_goal(me[0], me[1]);
        let walk = walk.max(1.0);
        let eval = |p: [f32; 2], c: usize| -> Option<Pick> {
            if !self.reach[c] || !zone.holds(p, 6.0) {
                return None;
            }
            let mut seer = None;
            for (k, f) in foes.iter().enumerate() {
                if self.visible(p, f.pos) {
                    if seer.is_some() {
                        return None;
                    }
                    seer = Some(k);
                }
            }
            let k = seer?;
            let travel = nav.path_len(p[0], p[1])?;
            if travel > 140.0 {
                return None;
            }
            let mut route = 0.0;
            for f in [0.33f32, 0.67] {
                let q = [me[0] + (p[0] - me[0]) * f, me[1] + (p[1] - me[1]) * f];
                let n = self.seen_by(q, foes) as f32;
                route += n * 0.3 + (n - 1.0).max(0.0);
            }
            let foe = foes[k];
            let df = dist(foe.pos, p);
            let bonus = (1.0 - foe.hp.clamp(0.0, 1.0)) * 6.0 + if df < 80.0 { 2.0 } else { 0.0 };
            let score = travel / walk + route * 1.5 + self.exposure[c] * 10.0 + self.spot_d[c].min(150.0) / 30.0 - bonus;
            Some(Pick { pos: p, yaw: (foe.pos[1] - p[1]).atan2(foe.pos[0] - p[0]), spot: None, foe: Some(foe.pos), score })
        };
        let mut best: Option<Pick> = None;
        for c in self.cells_near(me, 72.0) {
            if let Some(pk) = eval(self.pt[c].unwrap(), c) {
                if best.map_or(true, |b| pk.score < b.score) {
                    best = Some(pk);
                }
            }
        }
        for (i, s) in self.spots.iter().enumerate() {
            let d = dist(s.pos, me);
            if d >= 72.0 - GRID && d < 100.0 {
                if let Some(mut pk) = eval(s.pos, s.cell) {
                    pk.spot = Some(i);
                    if best.map_or(true, |b| pk.score < b.score) {
                        best = Some(pk);
                    }
                }
            }
        }
        // Hysteresis: the position held so far stays while it still isolates one enemy and is nearly as good.
        if let (Some(q), Some(b)) = (keep, best) {
            if let Some(c) = self.cell_at(q[0], q[1]) {
                if let Some(pk) = eval(q, c) {
                    if pk.score <= b.score + 1.5 {
                        return Some(Pick { pos: q, ..pk });
                    }
                }
            }
        }
        best
    }

    /// Cells with a standing point whose centre lies within `r` of `p`.
    fn cells_near(&self, p: [f32; 2], r: f32) -> impl Iterator<Item = usize> + '_ {
        let i0 = (((p[0] - r - self.x0) / GRID).floor().max(0.0)) as usize;
        let j0 = (((p[1] - r - self.y0) / GRID).floor().max(0.0)) as usize;
        let i1 = ((((p[0] + r - self.x0) / GRID).floor()).max(0.0) as usize).min(self.nx - 1);
        let j1 = ((((p[1] + r - self.y0) / GRID).floor()).max(0.0) as usize).min(self.ny - 1);
        (j0..=j1).flat_map(move |j| (i0..=i1).map(move |i| j * self.nx + i)).filter(move |&c| self.pt[c].map_or(false, |q| dist(q, p) <= r))
    }
}

fn spot_score(exposure: f32, lanes: u8) -> f32 {
    exposure + 0.02 * (lanes.max(1) - 1) as f32
}

/// Exposed fraction of the disc, open sectors, lane count and the direction of the widest lane.
fn shape(r: &[u8]) -> (f32, u8, u8, f32) {
    let n = r.len();
    let expo = r.iter().map(|&x| (x as f32 / RANGE).powi(2)).sum::<f32>() / n as f32;
    let open: Vec<bool> = r.iter().map(|&x| x as f32 > OPEN).collect();
    let n_open = open.iter().filter(|&&o| o).count();
    if n_open == 0 || n_open == n {
        // No lane, or open all round: face the furthest line.
        let (s, _) = r.iter().enumerate().max_by_key(|(i, &x)| (x, n - i)).unwrap();
        return (expo, n_open as u8, (n_open == n) as u8, s as f32 * SEC);
    }
    // Start the walk at a closed sector so that runs do not wrap.
    let start = open.iter().position(|&o| !o).unwrap();
    let (mut lanes, mut best, mut best_w) = (0u8, 0.0f32, -1.0f32);
    let (mut sum, mut acc) = ([0.0f32; 2], 0.0f32);
    for k in 1..=n {
        let s = (start + k) % n;
        if open[s] {
            let w = (r[s] as f32).powi(2);
            let a = s as f32 * SEC;
            sum = [sum[0] + a.cos() * w, sum[1] + a.sin() * w];
            acc += w;
        } else if acc > 0.0 {
            lanes += 1;
            if acc > best_w {
                best_w = acc;
                best = sum[1].atan2(sum[0]);
            }
            sum = [0.0; 2];
            acc = 0.0;
        }
    }
    (expo, n_open as u8, lanes, best)
}
