# Build from this folder: docker build -t tinman .
# The match runs it with no network, a read-only root (writable /tmp), 1 CPU, 512 MB, and the map at /arena.
FROM rust:1.91.0-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --bin tinman

FROM debian:bookworm-20251103-slim
COPY --from=build /src/target/release/tinman /bot/tinman
ENTRYPOINT ["/bot/tinman"]
