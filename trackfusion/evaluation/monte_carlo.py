import numpy as np
from scipy.optimize import linear_sum_assignment

from trackfusion.evaluation.metrics import rmse
from trackfusion.sensors.doppler_radar import DopplerRadar
from trackfusion.sensors.radar import Radar
from trackfusion.simulation.scenario import generate_scenario
from trackfusion.tracking.tracker import MultiTargetTracker


def create_radar_a(seed):
    return Radar(
        sensor_id="radar_a",
        position=np.array([-8000.0, 0.0]),
        max_range=30000.0,
        detection_probability=0.90,
        clutter_rate=1.0,
        range_std=120.0,
        bearing_std=np.deg2rad(1.0),
        scan_interval=1.0,
        start_time=0.0,
        rng=np.random.default_rng(seed)
    )


def create_radar_b(seed):
    return DopplerRadar(
        sensor_id="radar_b",
        position=np.array([8000.0, 0.0]),
        max_range=25000.0,
        detection_probability=0.95,
        clutter_rate=1.0,
        range_std=60.0,
        bearing_std=np.deg2rad(0.5),
        scan_interval=0.5,
        start_time=0.25,
        rng=np.random.default_rng(seed),
        range_rate_std=5.0
    )


def track_state_at(track, timestamp):
    dt = timestamp - track.last_time

    return (
        track.position + track.velocity * dt,
        track.velocity.copy()
    )


def evaluate_tracks(
    tracks,
    targets,
    timestamp,
    max_match_distance=3000.0
):
    active_targets = [
        target
        for target in targets
        if target.active
    ]

    if not tracks or not active_targets:
        return [], [], len(active_targets), len(tracks)

    states = [
        track_state_at(track, timestamp)
        for track in tracks
    ]

    costs = np.zeros(
        (len(tracks), len(active_targets))
    )

    for i, (position, _) in enumerate(states):
        for j, target in enumerate(active_targets):
            costs[i, j] = np.linalg.norm(
                position - target.position
            )

    rows, columns = linear_sum_assignment(costs)

    position_errors = []
    velocity_errors = []

    matched_tracks = set()
    matched_targets = set()

    for i, j in zip(rows, columns):
        if costs[i, j] > max_match_distance:
            continue

        position, velocity = states[i]
        target = active_targets[j]

        position_errors.append(
            np.linalg.norm(
                position - target.position
            )
        )

        velocity_errors.append(
            np.linalg.norm(
                velocity - target.velocity
            )
        )

        matched_tracks.add(i)
        matched_targets.add(j)

    missed_targets = (
        len(active_targets) - len(matched_targets)
    )

    false_tracks = (
        len(tracks) - len(matched_tracks)
    )

    return (
        position_errors,
        velocity_errors,
        missed_targets,
        false_tracks
    )


def run_configuration(
    scenario_seed,
    use_radar_a,
    use_radar_b,
    n_targets=5,
    duration=60.0
):
    scenario = generate_scenario(
        n_targets=n_targets,
        seed=scenario_seed
    )

    sensors = []

    if use_radar_a:
        sensors.append(
            create_radar_a(1000 + scenario_seed)
        )

    if use_radar_b:
        sensors.append(
            create_radar_b(2000 + scenario_seed)
        )

    tracker = MultiTargetTracker(
        confirmation_hits=3,
        tentative_max_misses=1,
        confirmed_max_misses=5,
        acceleration_std=8.0,
        initial_velocity_std=100.0,
        gate_probability=0.99
    )

    position_errors = []
    velocity_errors = []

    missed_targets = 0
    false_tracks = 0
    target_instances = 0
    track_instances = 0

    dt = 0.25
    time = 0.0
    next_evaluation = 1.0

    while time <= duration:
        for sensor in sensors:
            if time >= sensor.next_scan_time - 1e-9:
                measurements = sensor.measure(
                    scenario.targets,
                    time
                )

                if use_radar_a:
                    allow_initiation = (
                        sensor.sensor_id == "radar_a"
                    )
                else:
                    allow_initiation = True

                tracker.process_scan(
                    measurements=measurements,
                    sensor=sensor,
                    timestamp=time,
                    allow_initiation=allow_initiation
                )

                sensor.schedule_next_scan()

        if time >= next_evaluation - 1e-9:
            tracks = tracker.confirmed_tracks

            (
                position_error,
                velocity_error,
                missed,
                false
            ) = evaluate_tracks(
                tracks,
                scenario.targets,
                time
            )

            position_errors.extend(position_error)
            velocity_errors.extend(velocity_error)

            missed_targets += missed
            false_tracks += false

            target_instances += sum(
                target.active
                for target in scenario.targets
            )

            track_instances += len(tracks)

            next_evaluation += 1.0

        if time >= duration:
            break

        for target in scenario.targets:
            target.step(dt)

        time += dt

    coverage = (
        (target_instances - missed_targets)
        / target_instances
        if target_instances > 0
        else np.nan
    )

    false_track_fraction = (
        false_tracks / track_instances
        if track_instances > 0
        else 0.0
    )

    return {
        "position_rmse": rmse(position_errors),
        "velocity_rmse": rmse(velocity_errors),
        "coverage": coverage,
        "false_track_fraction": false_track_fraction
    }