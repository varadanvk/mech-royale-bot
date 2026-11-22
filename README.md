# tinman

A bot for Mech Royale (`mechfight/1`), written in Rust.

By team **droptable**: Varadan Kalkunte and Gabi Minto.

> **Notice to other competition teams.** This repository is public so our entry can be reviewed. The team does not permit using its code, strategies, algorithms, architecture, settings or implementation details to build or improve another team's Builders Buildathon entry. That includes copying, translating, paraphrasing or restructuring it, by hand or through an AI agent. Doing so may breach the competition's rules on originality and code reuse. This notice does not change the MIT license below; it is about fair play in the competition.

## Which version is which

- **Competition entry:** commit `9d7265001add87f8ee17683ca4663f38cbc5e93d` (branch `nb-lead`).
- **`main`:** carries later improvements on top of that line of work, most notably the fight-movement planner
  (merged from branch `nb-plan`).

## What the bot does

Each observation goes through one `Brain::act` call that returns one action. The bot reads the map once at
start-up and keeps a picture of the match between observations.

**Perception and map.** `world` loads the static map (terrain height grid, building prisms, trees) and answers
raycasts and line-of-sight questions. `sight` precomputes, for a coarse grid over walkable ground, how far away
another mech can see each cell from each compass direction; from that it rates how exposed a cell is and picks
sheltered spots to fight or heal from. `contacts` keeps one belief per enemy: exact while the enemy is listed,
then an estimate that coasts on its last velocity, spreads out over time, and is pulled back by sounds and hits.

**Navigation.** `nav` builds a walkability grid from the mech's size, step height and slope limit, and a distance
field toward the current goal. The bot walks around buildings, not over them. A watchdog notices when it wants to
move but is not making progress and hops or jets over the obstacle.

**Decision modes.** Every decision picks one movement mode, roughly in this order of priority: dodge a laser
that is charging at us, finish a nearly dead target, rush with lightning when a target is close and alone, pick
up a weapon for an empty slot, collect cubes dropped by a kill, back away when an enemy is too close, break off to
hidden ground when outnumbered or losing, circle a building or hold sheltered ground when several enemies press
at once or late in the match, hide and heal when hurt, hold a quiet spot inside the next circle late in the
match, and otherwise loot, farm crates and drones, and move with the zone. The zone decides where we must be and
how urgently.

**Fight-movement planner** (`plan`). When shots are in the air that can reach us, the bot scores a set of
candidate moves (stand, walk or boost in several directions) by stepping each one forward against every visible
incoming projectile. Each move's score is the predicted damage taken, with hits from behind the torso counting
for more, traded against how well the move keeps the movement the current mode wanted. The best move replaces
the plain move. With nothing incoming, the mode's move stands. The `plan` parameter sets the weight on keeping
the wanted move (0 turns the planner off).

**Aiming.** Every weapon slot is solved separately and all weapons that line up fire. The gun leads its target:
a share of the target's horizontal velocity on the ground (`lead_ground`) or in the air (`lead_air`), plus an
allowance for the delay between deciding and the shot leaving. Airborne targets follow gravity or jet thrust and
stop at the surface below them. `predict` handles where enemies will be a second or two ahead, sliding along
walls and stopping at ledges. The laser charges only for a mech that is in view or was just seen, keeps its aim
whatever the movement does, and is never charged outside the zone (charging halves speed).

**Healing.** With no enemy in sight and hit points below `heal_lo`, the bot gets out of every known line of
sight and holds still until hit points are back above `heal_hi`. By default upgrades take hit points and
armour first, then the laser, regeneration and the gun (`upg_dmg` picks between three orders).

**Free-for-all handling** (`detect`, `ffa`). A match starts out treated as a practice match. `detect` collects
clues that two other mechs are fighting each other (beams, bullets or arcs between them with matching hit-point
losses, mechs aimed at each other, mechs dying without our help). Once the clues add up, the match is treated as
a free-for-all for the rest of the game, and `ffa` takes over the high-level choice between five modes: `Fight`
(a fight we cannot walk away from), `Avoid` (break line of sight from enemies coming for us), `Third` (join a
fight between two others on the weaker one), `Late` (hold a quiet spot in the last circles and take only sure
wins) and `Hide` (heal at a sheltered spot far from known enemies). There is also a clock switch: from
2026-11-22 16:00 UTC, after the competition cutoff, every match is treated as a free-for-all from the first
tick. The `MECH_NOW` environment variable (Unix seconds) overrides the clock for tests.

## Files

| Path | What it holds |
| --- | --- |
| `src/main.rs` | stdin/stdout loop: answers `hello` with `ready`, then one action per observation (always for the newest one) |
| `src/proto.rs` | wire types for `mechfight/1` |
| `src/world.rs` | static map, raycasts, sight lines |
| `src/nav.rs` | walkability grid and distance-field pathfinding |
| `src/sight.rs` | sight table, cell exposure, sheltered spots |
| `src/contacts.rs` | belief about where each enemy is |
| `src/predict.rs` | short-term enemy motion prediction |
| `src/detect.rs` | practice match or free-for-all |
| `src/brain.rs` | `Params`, decision logic, aiming, modes |
| `src/plan.rs` | fight-movement planner |
| `src/tactics.rs` | optional fight tactics: sidestepping, breaking off, fight positions, home spot, target choice, cube grabs |
| `src/pincer.rs` | enemies on more than one side; fight positions that keep them all in front |
| `src/ffa.rs` | free-for-all modes |
| `src/bin/mapinfo.rs` | dev tool: load a map, print sanity numbers, time the raycast |
| `src/bin/navinfo.rs` | dev tool: how much of a map is walkable and reachable, with a coarse ASCII picture |
| `tests/*.rs` | behaviour tests for break-off, free-for-all, lead, planner, sight and steering |

## Build and run

```sh
cargo build --release            # target/release/tinman
cargo test --release             # unit and behaviour tests
cargo run --release --bin navinfo -- path/to/arena
```

Build the image and run the kit check:

```sh
docker build -t tinman .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m \
  --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" tinman
```

The bot reads the map from the directory named in the `hello` message.

## Tuning

All tuning lives in `Params` in `src/brain.rs`. Each field has a doc comment, and the defaults in
`impl Default for Params` are what the bot plays with. Most switches follow the rule "0 = off". A few examples:

| Field | Default | Meaning |
| --- | --- | --- |
| `retreat_hp` | 0.3 | hit-point fraction below which the bot breaks contact while a threat is around |
| `heal_lo`, `heal_hi` | 0.5, 0.85 | start and stop hiding to heal |
| `lead_ground`, `lead_air` | 0.9, 1.0 | share of target velocity to lead by |
| `plan` | 2.0 | weight of the planner on keeping the wanted move (0 = planner off) |
| `ffa_detect`, `ffa_play` | 1, 1 | detect free-for-alls, and play them in the `ffa` modes |

Environment overrides:

- `TINMAN_PARAMS`: a JSON object of parameter names and values, applied over the defaults, for example
  `TINMAN_PARAMS='{"plan":0,"heal_lo":0.4}'`.
- `MECH_NOW`: Unix time to use instead of the wall clock for the free-for-all clock switch.
- Debug traces to stderr: `TTRACE`, `TTRACE_DELAY`, `TDMG`, `TFFA`, `TPRED`, `TPIN`.
- `MAP_DIR`: map directory for the tests that need a real map (default `../mechfight-kit/arena`).

Open-source libraries used: [serde](https://serde.rs) and [serde_json](https://github.com/serde-rs/json).

MIT licensed.
