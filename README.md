# tinman

A bot for Mech Royale (`mechfight/1`), written in Rust.

By team **droptable**: Varadan Kalkunte and Gabi Minto.

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
