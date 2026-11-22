//! Where we believe the enemies are. One track per enemy we know by id: exact while it is listed, then an
//! estimate that coasts on its last velocity for a moment and spreads out after, pulled back by what we hear
//! and by hits from it, and pushed out of places we can see empty. Sounds and hits we cannot put a name to
//! become short-lived anonymous contacts. `beliefs` gives the rest of the brain one list of both.

use std::collections::{BTreeMap, VecDeque};
use std::f32::consts::TAU;

/// How fast the area an unseen enemy may be in spreads (between walking and boosting), m/s.
const GROW: f32 = 9.0;
/// Largest uncertainty kept, metres.
const MAX_R: f32 = 120.0;
/// Seconds an unseen enemy is carried along its last velocity.
const COAST_S: f32 = 2.0;
/// Lifetime of an anonymous contact without a repeat, seconds.
const CONTACT_S: f32 = 8.0;
/// Seconds without any news after which a track is dropped.
const FORGET_S: f32 = 40.0;
/// How often a track's estimate is checked against what we can see, seconds.
const CHECK_S: f32 = 0.5;
/// Bearing error of a heard sound (one standard deviation), radians.
const BEARING_SD: f32 = 0.122;
/// Confidence of an unseen track falls to zero over this many seconds without news.
const FADE_S: f32 = 12.0;
/// Most anonymous contacts kept.
const MAX_CONTACTS: usize = 12;
/// A track's estimated position is sampled this often, and kept this long, to tell whether it is closing in.
const HIST_DT: f32 = 0.5;
const HIST_S: f32 = 5.0;
/// Hit points assumed for an enemy never listed.
const UNKNOWN_HP: f32 = 1400.0;
/// With `MapCfg::strict_pull`: largest radius of an unseen track a sound may still be put down to, metres.
const PULL_R: f32 = 60.0;
/// With `MapCfg::keep_spread`: share of a joined sound's spread the contact keeps at least.
const SPREAD_KEEP: f32 = 0.8;
/// With `MapCfg::prune`: beliefs less sure than this are left out of `beliefs`.
const MIN_CONF: f32 = 0.05;

/// Which clean-ups of the enemy picture are on (all off: the original picture).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MapCfg {
    /// A sound moves an unseen track only when that track is the one unseen track it fits and was placed
    /// tightly (within `PULL_R`); anything else becomes an anonymous contact.
    pub strict_pull: bool,
    /// Beliefs whose confidence has faded out are left out of `beliefs`.
    pub prune: bool,
    /// An anonymous contact we can see all of, empty, is dropped; a track we see empty all round keeps a quarter
    /// of its confidence instead of half.
    pub clear_seen: bool,
    /// An enemy that comes into the list takes over the anonymous contacts it explains.
    pub merge_seen: bool,
    /// A hit from an enemy we have no track for lands on the anonymous contact along its line, if there is one.
    pub hit_line: bool,
    /// A sound joins an anonymous contact only when the two places mostly overlap (centres within half their
    /// radii), so two mechs heard apart are not fused into one between them.
    pub tight_merge: bool,
    /// A sound joined into an anonymous contact (or put down to a track) narrows it to no less than `SPREAD_KEEP` of the sound's own
    /// spread: repeats heard along one line share the same distance error and do not average it out.
    pub keep_spread: bool,
    /// An estimate moved out of a place we see empty keeps its radius instead of narrowing: seeing one spot
    /// empty says nothing about which of the places round it the enemy went.
    pub hold_r: bool,
    /// An enemy that has just left the list is carried along its last velocity with a radius that grows from
    /// the moment it left, not one held at zero while it coasts.
    pub coast_grow: bool,
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Two estimates of one position (centre and radius) merged, each weighted by the inverse of its area.
fn fuse(a: [f32; 2], ra: f32, b: [f32; 2], rb: f32) -> ([f32; 2], f32) {
    let (wa, wb) = (1.0 / ra.max(1.0).powi(2), 1.0 / rb.max(1.0).powi(2));
    let s = wa + wb;
    ([(a[0] * wa + b[0] * wb) / s, (a[1] * wa + b[1] * wb) / s], (1.0 / s).sqrt())
}

/// How far a sound of this kind carries, if it tells us where an enemy mech is.
pub fn carry(kind: &str) -> Option<f32> {
    Some(match kind {
        "gun" => 320.0,
        "mortar" => 420.0,
        "laser" => 380.0,
        "laser_charge" => 140.0,
        "lightning" => 160.0,
        "jet" => 90.0,
        "boost" => 90.0,
        "step" => 40.0,
        "reload" => 60.0,
        _ => return None,
    })
}

/// Weapon codes in `Belief::weapon`: what an enemy last fired, else the best it was seen holding.
pub const W_NONE: u8 = 0;
pub const W_GUN: u8 = 1;
pub const W_MORTAR: u8 = 2;
pub const W_LIGHTNING: u8 = 3;
pub const W_LASER: u8 = 4;

/// Static name of a sound kind, for the debug source tag of a belief.
fn kind_name(kind: &str) -> &'static str {
    match kind {
        "gun" => "gun",
        "mortar" => "mortar",
        "laser" => "laser",
        "laser_charge" => "laser_charge",
        "lightning" => "lightning",
        "jet" => "jet",
        "boost" => "boost",
        "step" => "step",
        "reload" => "reload",
        _ => "other",
    }
}

fn weapon_code(kind: &str) -> u8 {
    match kind {
        "gun" => W_GUN,
        "mortar" => W_MORTAR,
        "lightning" => W_LIGHTNING,
        "laser" => W_LASER,
        _ => W_NONE,
    }
}

/// Debug source tag of an anonymous contact started by a sound of this kind.
fn contact_name(kind: &str) -> &'static str {
    match kind {
        "gun" => "a:gun",
        "mortar" => "a:mortar",
        "laser" => "a:laser",
        "laser_charge" => "a:laser_charge",
        "lightning" => "a:lightning",
        "jet" => "a:jet",
        "boost" => "a:boost",
        "step" => "a:step",
        "reload" => "a:reload",
        _ => "a:other",
    }
}

/// The weapon a sound gives away.
fn sound_weapon(kind: &str) -> Option<&'static str> {
    match kind {
        "gun" | "reload" => Some("gun"),
        "mortar" => Some("mortar"),
        "laser" | "laser_charge" => Some("laser"),
        "lightning" => Some("lightning"),
        _ => None,
    }
}

/// Distance band of a heard sound: nearest and furthest the source can be.
fn band(dist_band: &str, carry: f32) -> (f32, f32) {
    match dist_band {
        "near" => (0.0, 60.0f32.min(carry)),
        "mid" => (60.0f32.min(carry), 160.0f32.min(carry)),
        _ => (160.0f32.min(carry), carry.max(170.0)),
    }
}

#[derive(Clone, Debug)]
pub struct Track {
    /// Last exact position and velocity (while listed).
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    /// Last time it was listed (negative infinity if never: known only from its hits).
    pub seen_t: f32,
    /// Last time anything told us about it.
    pub info_t: f32,
    /// Estimate: centre, and radius `r0` as of `r_t`, growing at `GROW` after.
    pub est: [f32; 2],
    pub r0: f32,
    pub r_t: f32,
    /// Still carried along its last velocity (nothing has moved the estimate since it was listed).
    pub coasting: bool,
    pub weapons: Vec<String>,
    pub hp: f32,
    /// Last time it aimed at us, hurt us or we took it on.
    pub hunter_t: f32,
    /// The estimate was looked at and found empty all round.
    pub lost: bool,
    pub listed: bool,
    check_t: f32,
    /// Hit points when last listed, and damage we have dealt it since.
    pub hp_abs: f32,
    pub dealt: f32,
    /// Weapon it last fired (heard or hit by).
    pub fired: u8,
    /// Estimated position every `HIST_DT` seconds over the last `HIST_S`, oldest first.
    pub hist: VecDeque<(f32, [f32; 2])>,
    /// What last moved the estimate (debug only): "seen", a sound kind, "hit" or "hit_new".
    pub src: &'static str,
}

impl Track {
    pub fn radius(&self, t: f32) -> f32 {
        (self.r0 + GROW * (t - self.r_t).max(0.0)).min(MAX_R)
    }
}

/// A sound (or hit) we could not put a name to.
#[derive(Clone, Debug)]
pub struct Contact {
    pub pos: [f32; 2],
    pub r0: f32,
    /// Last time it was heard.
    pub t: f32,
    /// How many sounds went into it.
    pub n: u32,
    pub hunter: bool,
    check_t: f32,
    /// The sound kind that started it, or "hit" (debug only).
    pub src: &'static str,
}

impl Contact {
    pub fn radius(&self, t: f32) -> f32 {
        (self.r0 + GROW * 0.5 * (t - self.t).max(0.0)).min(MAX_R)
    }
}

/// One believed enemy, for the rest of the brain.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Belief {
    /// The enemy's id, for a track.
    pub id: Option<u32>,
    pub pos: [f32; 2],
    pub r: f32,
    /// 0..1: how sure we are something is there now.
    pub conf: f32,
    /// Seconds since it was listed (or since the contact was last heard).
    pub age: f32,
    pub hp: f32,
    /// It came for us lately.
    pub hunter: bool,
    /// Listed right now: its position is exact.
    pub listed: bool,
    /// Hit points: when last listed, less what we have dealt it since.
    pub est_hp: f32,
    /// Weapon it last fired, else the best it was seen holding (`W_*`).
    pub weapon: u8,
    /// Filled in by the brain (`profile`): coming for us (came for us lately, closing in over the last few
    /// seconds, or moving toward us); no other believed enemy within 80 m of it; within 60 m and outside our
    /// view; standing where few directions lead away.
    pub hunting: bool,
    pub alone: bool,
    pub behind: bool,
    pub cornered: bool,
    /// Where the belief came from (debug only): "seen", "coast", a sound kind, "hit", "hit_new", or for an
    /// anonymous contact "a:" and the sound kind (or "a:hit").
    pub src: &'static str,
}

#[derive(Clone, Debug, Default)]
pub struct Contacts {
    pub tracks: BTreeMap<u32, Track>,
    pub contacts: Vec<Contact>,
    pub cfg: MapCfg,
}

impl Contacts {
    /// Start of an update: nothing is listed until `see` says so.
    pub fn begin(&mut self) {
        for k in self.tracks.values_mut() {
            k.listed = false;
        }
    }

    /// A listed enemy mech: its exact state.
    #[allow(clippy::too_many_arguments)]
    pub fn see(&mut self, t: f32, id: u32, pos: [f32; 3], vel: [f32; 3], weapons: &[String], hp: f32, hunter: bool) {
        let k = self.tracks.entry(id).or_insert_with(|| Track {
            pos,
            vel,
            seen_t: t,
            info_t: t,
            est: [pos[0], pos[1]],
            r0: 0.0,
            r_t: t,
            coasting: true,
            weapons: Vec::new(),
            hp,
            hunter_t: f32::NEG_INFINITY,
            lost: false,
            listed: true,
            check_t: t,
            hp_abs: UNKNOWN_HP,
            dealt: 0.0,
            fired: W_NONE,
            hist: VecDeque::new(),
            src: "seen",
        });
        k.pos = pos;
        k.vel = vel;
        k.seen_t = t;
        k.info_t = t;
        k.est = [pos[0], pos[1]];
        k.r0 = 0.0;
        k.r_t = t;
        k.coasting = true;
        k.lost = false;
        k.listed = true;
        k.check_t = t;
        k.src = "seen";
        k.hp = hp;
        if !weapons.is_empty() {
            k.weapons = weapons.to_vec();
        }
        if hunter {
            k.hunter_t = t;
        }
        if self.cfg.merge_seen {
            let p = [pos[0], pos[1]];
            self.contacts.retain(|c| dist(c.pos, p) > c.radius(t));
        }
    }

    /// It is dead.
    pub fn remove(&mut self, id: u32) {
        self.tracks.remove(&id);
    }

    /// A sound heard from `me`. Explained by a listed enemy: ignored. Otherwise it pulls the best matching
    /// unseen track toward the line it came along, or adds to (or starts) an anonymous contact.
    pub fn hear(&mut self, t: f32, me: [f32; 2], kind: &str, bearing: f32, dist_band: &str, age: f32) {
        let Some(c) = carry(kind) else { return };
        let (lo, hi) = band(dist_band, c);
        let weapon = sound_weapon(kind);
        let heard_t = t - age.max(0.0);
        let dir = [bearing.cos(), bearing.sin()];
        // Where along the line, and how far off it, a point lies.
        let place = |q: [f32; 2]| -> (f32, f32) {
            let e = [q[0] - me[0], q[1] - me[1]];
            (e[0] * dir[0] + e[1] * dir[1], (e[0] * dir[1] - e[1] * dir[0]).abs())
        };
        let fits = |q: [f32; 2], r: f32| -> bool {
            let (along, off) = place(q);
            along > lo - r - 5.0 && along < hi + r + 5.0 && off < r + along.max(0.0) * 2.0 * BEARING_SD + 6.0
        };
        let armed = |k: &Track| weapon.map_or(true, |w| k.weapons.is_empty() || k.weapons.iter().any(|x| x == w));
        if self.tracks.values().any(|k| k.listed && fits(k.est, 10.0)) {
            return;
        }
        // An unseen track it fits: the latest seen first, then the nearest to the line.
        let fitting = self.tracks.iter().filter(|(_, k)| !k.listed && armed(k) && fits(k.est, k.radius(t)));
        let best = if self.cfg.strict_pull {
            // Only the one unseen track the sound fits, and only if we know tightly where it was.
            let mut it = fitting.filter(|(_, k)| k.radius(t) <= PULL_R);
            match (it.next(), it.next()) {
                (Some((id, _)), None) => Some(*id),
                _ => None,
            }
        } else {
            fitting
                .max_by(|x, y| {
                    x.1.info_t.partial_cmp(&y.1.info_t).unwrap_or(std::cmp::Ordering::Equal).then(place(y.1.est).1.partial_cmp(&place(x.1.est).1).unwrap_or(std::cmp::Ordering::Equal))
                })
                .map(|(id, _)| *id)
        };
        if let Some(id) = best {
            let k = self.tracks.get_mut(&id).unwrap();
            let r = k.radius(t);
            let along = place(k.est).0.clamp(lo, hi).max(1.0);
            let q = [me[0] + dir[0] * along, me[1] + dir[1] * along];
            let rq = (along * 2.0 * BEARING_SD).max((hi - lo) * 0.5).max(8.0);
            let (e, rn) = fuse(k.est, r, q, rq);
            k.est = e;
            k.r0 = rn.max(8.0);
            if self.cfg.keep_spread {
                k.r0 = k.r0.max(r.min(SPREAD_KEEP * rq));
            }
            k.r_t = t;
            k.info_t = k.info_t.max(heard_t);
            k.coasting = false;
            k.lost = false;
            if let Some(w) = weapon {
                k.fired = weapon_code(w);
            }
            k.src = kind_name(kind);
            return;
        }
        let d = (lo + hi) * 0.5;
        let q = [me[0] + dir[0] * d, me[1] + dir[1] * d];
        let rq = ((hi - lo) * 0.5).max(d * 2.0 * BEARING_SD).max(8.0);
        self.note(t, heard_t, q, rq, false, contact_name(kind));
    }

    /// We were hit by `by` (if known) from `bearing`. A track we cannot see is put on that line, at the distance
    /// we had it at; an enemy we had no track for gets one at a guess of the distance its weapon shoots from.
    pub fn hit_by(&mut self, t: f32, me: [f32; 2], by: Option<u32>, bearing: f32, weapon: &str) {
        let dir = [bearing.cos(), bearing.sin()];
        let guess = match weapon {
            "lightning" => 30.0,
            "gun" => 90.0,
            "laser" | "mortar" => 140.0,
            _ => 80.0,
        };
        let Some(id) = by else {
            let q = [me[0] + dir[0] * guess, me[1] + dir[1] * guess];
            self.note(t, t, q, guess * 0.6, true, "a:hit");
            return;
        };
        if let Some(k) = self.tracks.get_mut(&id) {
            if weapon_code(weapon) != W_NONE {
                k.fired = weapon_code(weapon);
            }
        }
        match self.tracks.get_mut(&id) {
            Some(k) if k.listed => k.hunter_t = t,
            Some(k) => {
                let r = k.radius(t);
                let along = dist(k.est, me).max(10.0);
                k.est = [me[0] + dir[0] * along, me[1] + dir[1] * along];
                k.r0 = r;
                k.r_t = t;
                k.info_t = t;
                k.hunter_t = t;
                k.coasting = false;
                k.lost = false;
                k.src = "hit";
            }
            None => {
                let mut q = [me[0] + dir[0] * guess, me[1] + dir[1] * guess];
                let mut r0 = guess * 0.6;
                if self.cfg.hit_line {
                    // The anonymous contact nearest us along the line of the hit, if any: that is who it was.
                    let on = self
                        .contacts
                        .iter()
                        .enumerate()
                        .filter_map(|(i, c)| {
                            let e = [c.pos[0] - me[0], c.pos[1] - me[1]];
                            let along = e[0] * dir[0] + e[1] * dir[1];
                            let off = (e[0] * dir[1] - e[1] * dir[0]).abs();
                            (along > 0.0 && off < c.radius(t) + along * 2.0 * BEARING_SD).then_some((i, along))
                        })
                        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
                    if let Some((i, along)) = on {
                        let c = self.contacts.remove(i);
                        q = [me[0] + dir[0] * along, me[1] + dir[1] * along];
                        r0 = c.radius(t).max(along * 2.0 * BEARING_SD).max(8.0);
                    }
                }
                self.tracks.insert(
                    id,
                    Track {
                        pos: [q[0], q[1], 0.0],
                        vel: [0.0; 3],
                        seen_t: f32::NEG_INFINITY,
                        info_t: t,
                        est: q,
                        r0,
                        r_t: t,
                        coasting: false,
                        weapons: if weapon == "npc" || weapon.is_empty() { Vec::new() } else { vec![weapon.to_string()] },
                        hp: 1.0,
                        hunter_t: t,
                        lost: false,
                        listed: false,
                        check_t: t,
                        hp_abs: UNKNOWN_HP,
                        dealt: 0.0,
                        fired: weapon_code(weapon),
                        hist: VecDeque::new(),
                        src: "hit_new",
                    },
                );
            }
        }
    }

    /// Add to the anonymous contact this fits, or start one.
    fn note(&mut self, t: f32, heard_t: f32, q: [f32; 2], rq: f32, hunter: bool, src: &'static str) {
        let reach = if self.cfg.tight_merge { 0.5 } else { 1.0 };
        let keep = self.cfg.keep_spread;
        if let Some(c) = self.contacts.iter_mut().find(|c| dist(c.pos, q) < (c.radius(t) + rq) * reach) {
            let rc = c.radius(t);
            let (e, rn) = fuse(c.pos, rc, q, rq);
            c.pos = e;
            c.r0 = rn.max(12.0);
            if keep {
                c.r0 = c.r0.max(rc.min(SPREAD_KEEP * rq));
            }
            c.t = c.t.max(heard_t);
            c.n += 1;
            c.hunter |= hunter;
            return;
        }
        if self.contacts.len() >= MAX_CONTACTS {
            if let Some(i) = self.contacts.iter().enumerate().min_by(|x, y| x.1.t.partial_cmp(&y.1.t).unwrap_or(std::cmp::Ordering::Equal)).map(|x| x.0) {
                self.contacts.remove(i);
            }
        }
        self.contacts.push(Contact { pos: q, r0: rq, t: heard_t, n: 1, hunter, check_t: t, src });
    }

    /// End of an update: carry unseen tracks, look at their estimates (`in_view`: a mech standing there would be
    /// listed now) and move them out of places we see empty, drop what has aged out.
    pub fn settle(&mut self, t: f32, me: [f32; 2], in_view: &dyn Fn([f32; 2]) -> bool) {
        let hold = self.cfg.hold_r;
        let grow = self.cfg.coast_grow;
        // The game keeps listing an enemy for a while after it leaves view, so an unlisted enemy is not where we
        // can see.
        for k in self.tracks.values_mut().filter(|k| !k.listed) {
            if k.coasting && k.seen_t.is_finite() {
                let dt = (t - k.seen_t).min(COAST_S);
                k.est = [k.pos[0] + k.vel[0] * dt, k.pos[1] + k.vel[1] * dt];
                k.r_t = if grow { k.seen_t } else { k.seen_t + dt };
            }
            if t - k.check_t >= CHECK_S {
                k.check_t = t;
                let r = k.radius(t);
                if let Some((q, rn, lost)) = elsewhere(k.est, r, me, in_view) {
                    k.est = q;
                    k.r0 = if hold && !lost { r } else { rn };
                    k.r_t = t;
                    k.coasting = false;
                    k.lost |= lost;
                }
            }
        }
        for c in self.contacts.iter_mut() {
            if t - c.check_t >= CHECK_S {
                c.check_t = t;
                let r = c.radius(t);
                if let Some((q, rn, lost)) = elsewhere(c.pos, r, me, in_view) {
                    c.pos = q;
                    c.r0 = if hold && !lost { r } else { rn };
                    c.t -= if lost && self.cfg.clear_seen { CONTACT_S } else { 1.0 };
                }
            }
        }
        for k in self.tracks.values_mut() {
            if k.hist.back().map_or(true, |h| t - h.0 >= HIST_DT - 1e-3) {
                k.hist.push_back((t, k.est));
            }
            while k.hist.front().map_or(false, |h| t - h.0 > HIST_S) {
                k.hist.pop_front();
            }
        }
        self.tracks.retain(|_, k| k.listed || t - k.info_t < FORGET_S);
        self.contacts.retain(|c| t - c.t < CONTACT_S);
    }

    /// Hit points of a listed enemy (when the game gives them): what we deal it from now on counts from here.
    pub fn note_hp(&mut self, id: u32, hp: f32) {
        if let Some(k) = self.tracks.get_mut(&id) {
            k.hp_abs = hp;
            k.dealt = 0.0;
        }
    }

    /// We hit enemy `id` for `amount`.
    pub fn note_dealt(&mut self, id: u32, amount: f32) {
        if let Some(k) = self.tracks.get_mut(&id) {
            k.dealt += amount.max(0.0);
        }
    }

    /// Every enemy we believe in: tracks, then anonymous contacts.
    pub fn beliefs(&self, t: f32) -> Vec<Belief> {
        let mut v: Vec<Belief> = self
            .tracks
            .iter()
            .map(|(id, k)| {
                let lost = if self.cfg.clear_seen { 0.25 } else { 0.5 };
                let conf = if k.listed { 1.0 } else { (1.0 - (t - k.info_t) / FADE_S).max(0.0) * if k.lost { lost } else { 1.0 } };
                Belief {
                    id: Some(*id),
                    pos: k.est,
                    r: if k.listed { 0.0 } else { k.radius(t) },
                    conf,
                    age: if k.seen_t.is_finite() { t - k.seen_t } else { t - k.info_t },
                    hp: k.hp,
                    hunter: t - k.hunter_t < 10.0,
                    listed: k.listed,
                    est_hp: (k.hp_abs - k.dealt).max(1.0),
                    weapon: if k.fired != W_NONE { k.fired } else { k.weapons.iter().map(|w| weapon_code(w)).max().unwrap_or(W_NONE) },
                    src: if k.src == "seen" && !k.listed { "coast" } else { k.src },
                    ..Belief::default()
                }
            })
            .collect();
        v.extend(self.contacts.iter().map(|c| Belief {
            id: None,
            pos: c.pos,
            r: c.radius(t),
            conf: (0.5 + 0.15 * (c.n - 1) as f32).min(0.9) * (1.0 - (t - c.t) / CONTACT_S).clamp(0.0, 1.0),
            age: t - c.t,
            hp: 1.0,
            hunter: c.hunter,
            listed: false,
            est_hp: UNKNOWN_HP,
            src: c.src,
            ..Belief::default()
        }));
        if self.cfg.prune {
            v.retain(|b| b.conf >= MIN_CONF);
        }
        v
    }
}

/// An estimate whose centre we can see empty: the nearest place round it that we cannot see (the one furthest
/// from us when several are as near), with a radius to match; or, when all of it is in view, the same centre
/// spread wider and marked lost. None when the centre is out of view.
fn elsewhere(c: [f32; 2], r: f32, me: [f32; 2], in_view: &dyn Fn([f32; 2]) -> bool) -> Option<([f32; 2], f32, bool)> {
    if !in_view(c) {
        return None;
    }
    for f in [0.4f32, 0.8] {
        let rr = (r * f).max(6.0);
        let mut best: Option<([f32; 2], f32)> = None;
        for k in 0..8 {
            let a = k as f32 * TAU / 8.0;
            let q = [c[0] + a.cos() * rr, c[1] + a.sin() * rr];
            if in_view(q) {
                continue;
            }
            let d = dist(q, me);
            if best.map_or(true, |b| d > b.1) {
                best = Some((q, d));
            }
        }
        if let Some((q, _)) = best {
            return Some((q, (r * (1.0 - f * 0.5)).max(6.0), false));
        }
    }
    Some((c, (r * 1.5).max(20.0).min(MAX_R), true))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guns() -> Vec<String> {
        vec!["gun".to_string()]
    }

    #[test]
    fn unseen_tracks_coast_then_spread_and_fade() {
        let mut c = Contacts::default();
        c.begin();
        c.see(10.0, 3, [100.0, 0.0, 0.0], [5.0, 0.0, 0.0], &guns(), 0.8, false);
        c.settle(10.0, [0.0, 0.0], &|_| false);
        let b = c.beliefs(10.0);
        assert_eq!(b.len(), 1);
        assert!(b[0].listed && b[0].conf == 1.0 && b[0].r == 0.0);
        // Out of the list: carried along for 2 s, then held while the radius grows.
        for k in 1..=40 {
            let t = 10.0 + k as f32 * 0.1;
            c.begin();
            c.settle(t, [0.0, 0.0], &|_| false);
        }
        let b = c.beliefs(14.0)[0];
        assert!((b.pos[0] - 110.0).abs() < 0.01, "{:?}", b.pos);
        assert!((b.r - GROW * 2.0).abs() < 0.5, "r {}", b.r);
        assert!(b.conf < 1.0 && b.conf > 0.6, "conf {}", b.conf);
        assert!(!b.listed);
        // Long after: capped, then forgotten.
        assert!(c.tracks[&3].radius(100.0) <= MAX_R);
        c.begin();
        c.settle(10.0 + FORGET_S + 1.0, [0.0, 0.0], &|_| false);
        assert!(c.tracks.is_empty());
    }

    #[test]
    fn a_sound_pulls_the_matching_track() {
        let mut c = Contacts::default();
        c.begin();
        c.see(0.0, 3, [100.0, 30.0, 0.0], [0.0; 3], &guns(), 1.0, false);
        c.see(0.0, 4, [-80.0, 0.0, 0.0], [0.0; 3], &["laser".to_string()], 1.0, false);
        c.settle(0.0, [0.0, 0.0], &|_| false);
        c.begin();
        c.settle(5.0, [0.0, 0.0], &|_| false);
        // A gunshot due east, mid range: enemy 3 (the laser holder is behind us).
        c.begin();
        c.hear(5.0, [0.0, 0.0], "gun", 0.0, "mid", 0.0);
        c.settle(5.0, [0.0, 0.0], &|_| false);
        assert!(c.contacts.is_empty());
        let k = &c.tracks[&3];
        assert!(k.est[1] < 30.0 && k.est[1] > 0.0, "{:?}", k.est);
        assert!(k.radius(5.0) < 9.0 * 5.0);
        assert_eq!(c.tracks[&4].est, [-80.0, 0.0]);
        // A gunshot to the north fits neither: an anonymous contact, made tighter by a second one heard from
        // elsewhere.
        c.begin();
        c.hear(5.0, [0.0, 0.0], "gun", 1.5708, "mid", 0.0);
        c.settle(5.0, [0.0, 0.0], &|_| false);
        assert_eq!(c.contacts.len(), 1);
        let r1 = c.contacts[0].radius(5.0);
        c.begin();
        c.hear(5.5, [40.0, 0.0], "gun", (110.0f32).atan2(-40.0), "mid", 0.0);
        c.settle(5.5, [40.0, 0.0], &|_| false);
        assert_eq!(c.contacts.len(), 1);
        assert_eq!(c.contacts[0].n, 2);
        assert!(c.contacts[0].radius(5.5) < r1);
        let b = c.beliefs(5.5);
        let anon = b.iter().find(|b| b.id.is_none()).unwrap();
        assert!(anon.conf > 0.6);
        // A sound a listed enemy explains adds nothing; contacts age out.
        c.begin();
        c.see(6.0, 4, [0.0, -50.0, 0.0], [0.0; 3], &[], 1.0, false);
        c.hear(6.0, [0.0, 0.0], "laser", -1.5708, "near", 0.0);
        c.settle(6.0, [0.0, 0.0], &|_| false);
        assert_eq!(c.contacts.len(), 1);
        c.begin();
        c.settle(5.5 + CONTACT_S + 0.1, [0.0, 0.0], &|_| false);
        assert!(c.contacts.is_empty());
    }

    #[test]
    fn a_hit_puts_the_shooter_on_its_line() {
        let mut c = Contacts::default();
        c.begin();
        c.see(0.0, 3, [100.0, 0.0, 0.0], [0.0; 3], &guns(), 1.0, false);
        c.settle(0.0, [0.0, 0.0], &|_| false);
        c.begin();
        c.hit_by(6.0, [0.0, 0.0], Some(3), 1.5708, "gun");
        c.hit_by(6.0, [0.0, 0.0], Some(7), 3.1416, "lightning");
        c.settle(6.0, [0.0, 0.0], &|_| false);
        let k = &c.tracks[&3];
        assert!(k.est[0].abs() < 0.1 && (k.est[1] - 100.0).abs() < 0.1, "{:?}", k.est);
        let b = c.beliefs(6.0);
        assert!(b.iter().all(|b| b.hunter));
        assert!((c.tracks[&7].est[0] + 30.0).abs() < 0.1);
        c.remove(7);
        assert!(!c.tracks.contains_key(&7));
    }

    #[test]
    fn an_estimate_seen_empty_moves_out_of_view() {
        let mut c = Contacts::default();
        c.begin();
        c.see(0.0, 3, [100.0, 0.0, 0.0], [0.0; 3], &guns(), 1.0, false);
        c.settle(0.0, [0.0, 0.0], &|_| false);
        // Everything with x < 110 is in view.
        let view = |q: [f32; 2]| q[0] < 110.0;
        c.begin();
        c.settle(6.0, [0.0, 0.0], &view);
        let k = &c.tracks[&3];
        assert!(k.est[0] >= 110.0, "{:?}", k.est);
        assert!(!k.lost);
        // All in view: spread and marked lost, and less trusted.
        c.begin();
        c.settle(6.6, [0.0, 0.0], &|_| true);
        assert!(c.tracks[&3].lost);
        let b = c.beliefs(6.6)[0];
        assert!(b.conf < 0.4, "conf {}", b.conf);
    }

    fn with(cfg: MapCfg) -> Contacts {
        Contacts { cfg, ..Contacts::default() }
    }

    /// Two enemies listed at time 0 at these places, unseen since `t`.
    fn two_unseen(cfg: MapCfg, a: [f32; 3], b: [f32; 3], t: f32) -> Contacts {
        let mut c = with(cfg);
        c.begin();
        c.see(0.0, 3, a, [0.0; 3], &guns(), 1.0, false);
        c.see(0.0, 4, b, [0.0; 3], &guns(), 1.0, false);
        c.settle(0.0, [0.0, 0.0], &|_| false);
        c.begin();
        c.settle(t, [0.0, 0.0], &|_| false);
        c
    }

    #[test]
    fn strict_pull_leaves_an_ambiguous_sound_anonymous() {
        let cfg = MapCfg { strict_pull: true, ..MapCfg::default() };
        // Both tracks fit a shot heard due east: neither is moved, the shot is a contact of its own.
        let mut c = two_unseen(cfg, [100.0, 10.0, 0.0], [110.0, -10.0, 0.0], 3.0);
        c.begin();
        c.hear(3.0, [0.0, 0.0], "gun", 0.0, "mid", 0.0);
        assert_eq!(c.tracks[&3].est, [100.0, 10.0]);
        assert_eq!(c.tracks[&4].est, [110.0, -10.0]);
        assert_eq!(c.contacts.len(), 1);
        // Without it the latest-seen track that fits takes the shot.
        let mut c = two_unseen(MapCfg::default(), [100.0, 10.0, 0.0], [110.0, -10.0, 0.0], 3.0);
        c.begin();
        c.hear(3.0, [0.0, 0.0], "gun", 0.0, "mid", 0.0);
        assert!(c.contacts.is_empty());
        // One track, but unseen so long it could be anywhere: not moved either.
        let mut c = two_unseen(cfg, [100.0, 10.0, 0.0], [-300.0, 0.0, 0.0], 10.0);
        assert!(c.tracks[&3].radius(10.0) > PULL_R);
        c.begin();
        c.hear(10.0, [0.0, 0.0], "gun", 0.0, "mid", 0.0);
        assert_eq!(c.tracks[&3].est, [100.0, 10.0]);
        assert_eq!(c.contacts.len(), 1);
        // One tightly placed track: moved.
        let mut c = two_unseen(cfg, [100.0, 10.0, 0.0], [-300.0, 0.0, 0.0], 3.0);
        c.begin();
        c.hear(3.0, [0.0, 0.0], "gun", 0.0, "mid", 0.0);
        assert!(c.contacts.is_empty());
        assert!(c.tracks[&3].est[1] < 10.0);
    }

    #[test]
    fn prune_leaves_out_faded_beliefs() {
        for (prune, n) in [(false, 1), (true, 0)] {
            let mut c = with(MapCfg { prune, ..MapCfg::default() });
            c.begin();
            c.see(0.0, 3, [100.0, 0.0, 0.0], [0.0; 3], &guns(), 1.0, false);
            c.settle(0.0, [0.0, 0.0], &|_| false);
            c.begin();
            c.settle(FADE_S + 1.0, [0.0, 0.0], &|_| false);
            assert_eq!(c.tracks.len(), 1);
            assert_eq!(c.beliefs(FADE_S + 1.0).len(), n, "prune {prune}");
        }
    }

    #[test]
    fn clear_seen_drops_a_contact_seen_empty() {
        for (clear, n) in [(false, 1), (true, 0)] {
            let mut c = with(MapCfg { clear_seen: clear, ..MapCfg::default() });
            c.begin();
            c.hear(0.0, [0.0, 0.0], "gun", 0.0, "near", 0.0);
            c.settle(0.0, [0.0, 0.0], &|_| false);
            assert_eq!(c.contacts.len(), 1);
            c.begin();
            c.settle(1.0, [0.0, 0.0], &|_| true);
            assert_eq!(c.contacts.len(), n, "clear {clear}");
        }
        // A track seen empty all round keeps a quarter of its confidence.
        let mut c = with(MapCfg { clear_seen: true, ..MapCfg::default() });
        c.begin();
        c.see(0.0, 3, [100.0, 0.0, 0.0], [0.0; 3], &guns(), 1.0, false);
        c.settle(0.0, [0.0, 0.0], &|_| false);
        c.begin();
        c.settle(1.0, [0.0, 0.0], &|_| true);
        let b = c.beliefs(1.0)[0];
        assert!(b.conf < 0.3 && b.conf > 0.0, "conf {}", b.conf);
    }

    #[test]
    fn merge_seen_takes_over_the_contacts_a_listed_enemy_explains() {
        for (merge, n) in [(false, 2), (true, 1)] {
            let mut c = with(MapCfg { merge_seen: merge, ..MapCfg::default() });
            c.begin();
            c.hear(0.0, [0.0, 0.0], "step", 0.0, "near", 0.0);
            c.hear(0.0, [0.0, 0.0], "gun", 3.1416, "far", 0.0);
            assert_eq!(c.contacts.len(), 2);
            c.begin();
            c.see(0.5, 3, [32.0, 3.0, 0.0], [0.0; 3], &guns(), 1.0, false);
            c.settle(0.5, [0.0, 0.0], &|_| false);
            assert_eq!(c.contacts.len(), n, "merge {merge}");
        }
    }

    #[test]
    fn hit_line_puts_an_unknown_shooter_on_the_contact_along_the_hit() {
        for hit_line in [false, true] {
            let mut c = with(MapCfg { hit_line, ..MapCfg::default() });
            c.begin();
            c.hear(0.0, [0.0, 0.0], "gun", 0.0, "far", 0.0);
            c.hit_by(0.5, [0.0, 0.0], Some(6), 0.02, "gun");
            let k = &c.tracks[&6];
            if hit_line {
                assert!(c.contacts.is_empty());
                assert!(k.est[0] > 200.0, "{:?}", k.est);
            } else {
                assert_eq!(c.contacts.len(), 1);
                assert!((k.est[0] - 90.0).abs() < 1.0, "{:?}", k.est);
            }
        }
    }

    #[test]
    fn tight_merge_keeps_two_shooters_heard_apart() {
        for (tight, n) in [(false, 1), (true, 2)] {
            let mut c = with(MapCfg { tight_merge: tight, ..MapCfg::default() });
            c.begin();
            // Mid range, 0.6 rad apart: about 66 m between the two places, each some 50 m across.
            c.hear(0.0, [0.0, 0.0], "gun", 0.0, "mid", 0.0);
            c.hear(0.0, [0.0, 0.0], "gun", 0.6, "mid", 0.0);
            assert_eq!(c.contacts.len(), n, "tight {tight}");
        }
    }

    #[test]
    fn keep_spread_stops_repeats_from_narrowing_a_contact() {
        for keep in [false, true] {
            let mut c = with(MapCfg { keep_spread: keep, ..MapCfg::default() });
            c.begin();
            for k in 0..8 {
                c.hear(k as f32 * 0.25, [0.0, 0.0], "gun", 0.0, "far", 0.0);
            }
            let r = c.contacts[0].radius(2.0);
            if keep {
                assert!(r >= SPREAD_KEEP * 80.0 - 0.1, "r {r}");
            } else {
                assert!(r < 40.0, "r {r}");
            }
        }
    }

    #[test]
    fn hold_r_keeps_the_radius_of_an_estimate_moved_out_of_view() {
        for hold in [false, true] {
            let mut c = with(MapCfg { hold_r: hold, ..MapCfg::default() });
            c.begin();
            c.see(0.0, 3, [100.0, 0.0, 0.0], [0.0; 3], &guns(), 1.0, false);
            c.settle(0.0, [0.0, 0.0], &|_| false);
            let view = |q: [f32; 2]| q[0] < 110.0;
            c.begin();
            c.settle(6.0, [0.0, 0.0], &view);
            let k = &c.tracks[&3];
            assert!(k.est[0] >= 110.0);
            let r = k.radius(6.0);
            // Carried for 2 s, then spreading for 4.
            if hold {
                assert!((r - GROW * 4.0).abs() < 0.5, "r {r}");
            } else {
                assert!(r < GROW * 4.0 * 0.9, "r {r}");
            }
        }
    }

    #[test]
    fn coast_grow_spreads_a_coasting_track() {
        for grow in [false, true] {
            let mut c = with(MapCfg { coast_grow: grow, ..MapCfg::default() });
            c.begin();
            c.see(0.0, 3, [100.0, 0.0, 0.0], [5.0, 0.0, 0.0], &guns(), 1.0, false);
            c.settle(0.0, [0.0, 0.0], &|_| false);
            c.begin();
            c.settle(1.0, [0.0, 0.0], &|_| false);
            let k = &c.tracks[&3];
            assert!((k.est[0] - 105.0).abs() < 0.01);
            let r = k.radius(1.0);
            if grow {
                assert!((r - GROW).abs() < 0.01, "r {r}");
            } else {
                assert_eq!(r, 0.0);
            }
        }
    }
}
