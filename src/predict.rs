//! Where an enemy will be a second or two from now, and how widely the enemies around a point are spread.
//!
//! Motion: an enemy keeps its velocity, sliding along walls and stopping where the ground ends. One that is
//! coming for us is also walked toward us around obstacles (stopping at the distance it fights from), and the
//! two guesses are blended.
use std::f32::consts::{PI, TAU};

/// Time step of the walks, seconds.
const STEP: f32 = 0.25;

fn wrap(a: f32) -> f32 {
    let mut a = (a + PI) % TAU;
    if a < 0.0 {
        a += TAU;
    }
    a - PI
}

/// Carry `pos` along `vel` for `dt` seconds over walkable ground: a step into a wall keeps the part along it
/// (the larger of the two axis parts that stays walkable), and the walk stops where neither does.
pub fn extrapolate(pos: [f32; 2], vel: [f32; 2], dt: f32, walk: &dyn Fn(f32, f32) -> bool) -> [f32; 2] {
    let mut p = pos;
    let mut t = 0.0;
    while t < dt - 1e-4 {
        let h = STEP.min(dt - t);
        let d = [vel[0] * h, vel[1] * h];
        let full = [p[0] + d[0], p[1] + d[1]];
        if walk(full[0], full[1]) {
            p = full;
        } else {
            // Slide: keep the larger axis part that stays on walkable ground.
            let xs = [p[0] + d[0], p[1]];
            let ys = [p[0], p[1] + d[1]];
            let (a, b) = if d[0].abs() >= d[1].abs() { (xs, ys) } else { (ys, xs) };
            if walk(a[0], a[1]) {
                p = a;
            } else if walk(b[0], b[1]) {
                p = b;
            } else {
                break;
            }
        }
        t += h;
    }
    p
}

/// Walk from `pos` toward `goal` at `speed` for `dt` seconds, turning up to 90 degrees either way around
/// whatever is not walkable, and stopping once within `stop` metres of the goal.
pub fn approach(pos: [f32; 2], goal: [f32; 2], speed: f32, dt: f32, stop: f32, walk: &dyn Fn(f32, f32) -> bool) -> [f32; 2] {
    let mut p = pos;
    let mut t = 0.0;
    while t < dt - 1e-4 {
        let h = STEP.min(dt - t);
        let (dx, dy) = (goal[0] - p[0], goal[1] - p[1]);
        let d = dx.hypot(dy);
        if d <= stop {
            break;
        }
        let len = (speed * h).min(d - stop);
        let base = dy.atan2(dx);
        let mut moved = false;
        for turn in [0.0f32, 0.5, -0.5, 1.0, -1.0, 1.5708, -1.5708] {
            let a = base + turn;
            let q = [p[0] + a.cos() * len, p[1] + a.sin() * len];
            if walk(q[0], q[1]) {
                p = q;
                moved = true;
                break;
            }
        }
        if !moved {
            break;
        }
        t += h;
    }
    p
}

/// Predicted position `dt` seconds ahead. `hunt`: the share given to a walk toward `me` (0 for an enemy not
/// coming for us), at the speed it moves now but at least `min_speed`, stopping `stop` metres short of us.
#[allow(clippy::too_many_arguments)]
pub fn predict(pos: [f32; 2], vel: [f32; 2], me: [f32; 2], hunt: f32, min_speed: f32, stop: f32, dt: f32, walk: &dyn Fn(f32, f32) -> bool) -> [f32; 2] {
    let v = extrapolate(pos, vel, dt, walk);
    if hunt <= 0.0 {
        return v;
    }
    let speed = vel[0].hypot(vel[1]).max(min_speed);
    let h = approach(pos, me, speed, dt, stop, walk);
    let w = hunt.min(1.0);
    [v[0] + (h[0] - v[0]) * w, v[1] + (h[1] - v[1]) * w]
}

/// Widest empty angle (radians) between the bearings `yaws`; TAU with fewer than two.
pub fn widest_gap(yaws: &[f32]) -> f32 {
    if yaws.len() < 2 {
        return TAU;
    }
    let mut a: Vec<f32> = yaws.iter().map(|y| y.rem_euclid(TAU)).collect();
    a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let mut gap = a[0] + TAU - a[a.len() - 1];
    for w in a.windows(2) {
        gap = gap.max(w[1] - w[0]);
    }
    gap
}

/// The arc (radians) the bearings `yaws` take up: the whole circle less the widest gap.
pub fn spread(yaws: &[f32]) -> f32 {
    if yaws.len() < 2 {
        0.0
    } else {
        TAU - widest_gap(yaws)
    }
}

/// How far (radians, 0..PI) bearing `yaw` is off `aim`.
pub fn off(yaw: f32, aim: f32) -> f32 {
    wrap(yaw - aim).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open(_: f32, _: f32) -> bool {
        true
    }

    #[test]
    fn velocity_is_carried_and_stops_at_walls() {
        let p = extrapolate([0.0, 0.0], [10.0, 0.0], 2.0, &open);
        assert!((p[0] - 20.0).abs() < 1e-3 && p[1].abs() < 1e-3);
        // A wall at x = 12 running north-south: the walk ends against it.
        let wall = |x: f32, _: f32| x < 12.0;
        let p = extrapolate([0.0, 0.0], [10.0, 0.0], 2.0, &wall);
        assert!(p[0] < 12.0 && p[0] > 9.0, "{:?}", p);
        // Moving diagonally into it: slides north along it.
        let p = extrapolate([0.0, 0.0], [8.0, 8.0], 2.0, &wall);
        assert!(p[0] < 12.0 && p[1] > 15.0, "{:?}", p);
    }

    #[test]
    fn a_hunter_is_walked_around_a_wall_toward_us() {
        // A wall between it and us, open to the north (y > 10).
        let wall = |x: f32, y: f32| !(x > 8.0 && x < 12.0 && y < 10.0);
        let me = [40.0, 0.0];
        let p = approach([0.0, 0.0], me, 10.0, 3.0, 20.0, &wall);
        assert!(wall(p[0], p[1]));
        assert!(((p[0] - me[0]).hypot(p[1] - me[1])) < 38.0, "{:?}", p);
        // It stops at its fighting distance.
        let p = approach([0.0, 0.0], [30.0, 0.0], 10.0, 3.0, 20.0, &open);
        assert!((p[0] - 10.0).abs() < 0.5, "{:?}", p);
        // Blend half way between holding course (north) and coming for us (east).
        let p = predict([0.0, 0.0], [0.0, 10.0], [100.0, 0.0], 0.5, 0.0, 20.0, 1.0, &open);
        assert!((p[0] - 5.0).abs() < 0.5 && (p[1] - 5.0).abs() < 0.5, "{:?}", p);
    }

    #[test]
    fn spread_of_bearings() {
        assert_eq!(spread(&[1.0]), 0.0);
        assert!((spread(&[0.1, -0.1]) - 0.2).abs() < 1e-4);
        assert!((spread(&[0.0, PI * 0.5, PI]) - PI).abs() < 1e-4);
        // Three round us: most of the circle.
        assert!(spread(&[0.0, 2.1, 4.2]) > 4.0);
        assert!((off(3.0, -3.0) - (TAU - 6.0)).abs() < 1e-4);
    }
}
