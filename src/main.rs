//! stdin/stdout loop for mechfight/1: answer `hello` with `ready`, then one action per observation,
//! always for the newest observation if several are queued.
use std::io::{BufRead, Write};
use std::sync::mpsc;
use tinman::brain::Brain;
use tinman::proto::{Action, Hello, Obs};

fn send(line: &str) {
    let mut out = std::io::stdout().lock();
    let _ = out.write_all(line.as_bytes());
    let _ = out.write_all(b"\n");
    let _ = out.flush();
}

fn main() {
    let (tx, rx) = mpsc::channel::<Option<String>>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            match line {
                Ok(l) if !l.trim().is_empty() => {
                    if tx.send(Some(l)).is_err() {
                        return;
                    }
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
        let _ = tx.send(None);
    });

    let mut brain: Option<Brain> = None;
    let mut last = Action::default();
    let mark = std::path::Path::new("/tmp/tinman.mark");
    let seen = mark.exists();
    let _ = std::fs::write(mark, b"1");
    loop {
        let Ok(first) = rx.recv() else { return };
        let mut batch = vec![first];
        while let Ok(more) = rx.try_recv() {
            batch.push(more);
        }
        let mut newest: Option<String> = None;
        for raw in batch {
            let Some(raw) = raw else { return };
            // Cheap type sniff: the type field comes first in every server message.
            let head = &raw[..raw.len().min(40)];
            if head.contains("\"hello\"") {
                let b = serde_json::from_str::<Hello>(&raw).ok().and_then(|h| std::panic::catch_unwind(|| Brain::new(&h)).ok());
                brain = b;
                send(r#"{"type":"ready","name":"tinman"}"#);
            } else if head.contains("\"obs\"") {
                newest = Some(raw);
            } else if head.contains("\"end\"") {
                return;
            }
        }
        let Some(raw) = newest else { continue };
        let Ok(obs) = serde_json::from_str::<Obs>(&raw) else { continue };
        // A panic in the logic must not kill the process: coast on the last move and aim instead.
        let action = match brain.as_mut() {
            Some(b) => match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.act(&obs))) {
                Ok(a) => a,
                Err(_) => Action { tick: obs.tick, mv: last.mv, aim: last.aim, boost: last.boost, ..Default::default() },
            },
            None => Action { tick: obs.tick, aim: obs.me.aim, ..Default::default() },
        };
        let mut action = action;
        if seen {
            action = Action { tick: obs.tick, aim: obs.me.aim, ..Default::default() };
        }
        // NaN would serialize as null and fail the schema: fall back to safe values.
        if !action.mv.iter().all(|v| v.is_finite()) {
            action.mv = [0.0, 0.0];
        }
        if !action.aim.iter().all(|v| v.is_finite()) {
            action.aim = obs.me.aim;
        }
        last = action.clone();
        if let Ok(line) = serde_json::to_string(&action) {
            send(&line);
        }
    }
}
