//! Wire types for mechfight/1 (protocol.md section 6). Everything defaults so added or missing
//! fields never break parsing.
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Stats = HashMap<String, f64>;

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct HelloMap {
    pub dir: String,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Hello {
    pub slot: u32,
    pub seed: u64,
    pub map: HelloMap,
    pub config: serde_json::Value,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Weapon {
    pub kind: String,
    pub cooldown: f32,
    pub ammo: f32,
    pub reload: f32,
    pub charge: f32,
    pub stats: Stats,
}

impl Weapon {
    pub fn stat(&self, k: &str, d: f32) -> f32 {
        self.stats.get(k).map(|v| *v as f32).unwrap_or(d)
    }
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct SelfObs {
    pub id: u32,
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub aim: [f32; 2],
    pub hp: f32,
    pub max_hp: f32,
    pub energy: f32,
    pub max_energy: f32,
    pub grounded: bool,
    pub cubes: f32,
    pub upgrades: Vec<String>,
    pub in_zone: bool,
    pub weapons: Vec<Option<Weapon>>,
    pub stats: Stats,
}

impl SelfObs {
    pub fn stat(&self, k: &str, d: f32) -> f32 {
        self.stats.get(k).map(|v| *v as f32).unwrap_or(d)
    }
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Actor {
    pub id: u32,
    pub kind: String,
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub aim: [f32; 2],
    pub hp: f32,
    pub max_hp: f32,
    pub cubes: f32,
    pub weapons: Vec<String>,
    pub charging: bool,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Item {
    pub id: u32,
    pub kind: String,
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub hp: f32,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Zone {
    pub center: [f32; 2],
    pub radius: f32,
    pub next_center: [f32; 2],
    pub next_radius: f32,
    pub shrink_start: f32,
    pub shrink_end: f32,
    pub dps: f32,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Choice {
    pub id: String,
    pub name: String,
    pub desc: String,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Offer {
    pub id: u32,
    pub expires: f32,
    pub choices: Vec<Choice>,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Drop {
    pub id: u32,
    pub pos: [f32; 3],
    pub radius: f32,
    pub lands_in: f32,
    pub capture_s: f32,
    pub progress: f32,
    pub holder: Option<u32>,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Event {
    #[serde(rename = "type")]
    pub kind: String,
    pub amount: f32,
    pub bearing: Option<f32>,
    pub by: Option<u32>,
    pub rear: bool,
    pub weapon: Option<String>,
    pub target: Option<u32>,
    pub effect: Option<String>,
    /// Pickup or drop id (number), or upgrade id (string).
    pub id: Option<serde_json::Value>,
    pub cubes: Option<f32>,
    /// Weapon kind for `took` and `dropped`.
    #[serde(rename = "kind")]
    pub item: Option<String>,
    pub slot: Option<u32>,
    pub reason: Option<String>,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Sound {
    pub kind: String,
    pub bearing: f32,
    pub dist: String,
    pub age: f32,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Beam {
    pub by: u32,
    pub kind: String,
    pub path: Vec<[f32; 3]>,
    pub age: f32,
}

#[derive(Deserialize, Default, Debug, Clone)]
#[serde(default)]
pub struct Obs {
    pub tick: u64,
    pub t: f32,
    pub alive: u32,
    #[serde(rename = "self")]
    pub me: SelfObs,
    pub mechs: Vec<Actor>,
    pub projectiles: Vec<Item>,
    pub crates: Vec<Item>,
    pub cubes: Vec<Item>,
    pub pickups: Vec<Item>,
    pub sounds: Vec<Sound>,
    pub beams: Vec<Beam>,
    pub events: Vec<Event>,
    pub drops: Vec<Drop>,
    pub zone: Zone,
    pub offer: Option<Offer>,
}

#[derive(Serialize, Default, Debug, Clone)]
pub struct Action {
    pub tick: u64,
    #[serde(rename = "move")]
    pub mv: [f32; 2],
    pub boost: bool,
    pub jump: bool,
    pub aim: [f32; 2],
    pub fire: [bool; 2],
    pub reload: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pick: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub take: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop: Option<u32>,
}
