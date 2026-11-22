//! tinman: a mechfight/1 bot. `world` is the map model, `nav` the ground pathfinder,
//! `proto` the wire types, `sight` the sight table and sheltered spots, `contacts`
//! where we believe the enemies are, `detect` what kind of match this is, `predict`
//! where enemies will be, `brain` the decision logic.
pub mod brain;
pub mod contacts;
pub mod detect;
pub mod nav;
pub mod predict;
pub mod proto;
pub mod sight;
pub mod world;
