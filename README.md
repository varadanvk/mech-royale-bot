# tinman

A bot for Mech Royale (`mechfight/1`), written in Rust.

By team **droptable**: Varadan Kalkunte and Gabi Minto.

> **Notice to other competition teams.** This repository is public so our entry can be reviewed. The team does not permit using its code, strategies, algorithms, architecture, settings or implementation details to build or improve another team's Builders Buildathon entry. That includes copying, translating, paraphrasing or restructuring it, by hand or through an AI agent. Doing so may breach the competition's rules on originality and code reuse. This notice does not change the MIT license below; it is about fair play in the competition.

- `src/world.rs`: the map (terrain grid, building prisms, trees) with raycasts and sight lines.
- `src/nav.rs`: ground pathfinding on a 3 m grid.
- `src/brain.rs`: decisions (aim, fight or flee, loot, zone).
- `src/main.rs`: the stdin/stdout loop.

Build and check:

```sh
docker build -t tinman .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m \
  --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" tinman
```

Open-source libraries used: [serde](https://serde.rs) and [serde_json](https://github.com/serde-rs/json).

MIT licensed.
