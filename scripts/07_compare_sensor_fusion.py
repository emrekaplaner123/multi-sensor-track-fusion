import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import linear_sum_assignment

from trackfusion.evaluation.metrics import rmse
from trackfusion.sensors.doppler_radar import DopplerRadar
from trackfusion.sensors.radar import Radar
from trackfusion.simulation.scenario import generate_scenario
from trackfusion.tracking.tracker import MultiTargetTracker


def create_radar_a():
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
        rng=np.random.default_rng(100)
    )


def create_radar_b():
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
        rng=np.random.default_rng(200),
        range_rate_std=5.0
    )


def track_state_at(track, timestamp):
    dt = timestamp - track.last_time

    position = (
        track.position
        + track.velocity * dt
    )

    return position, track.velocity.copy()


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

    track_states = [
        track_state_at(track, timestamp)
        for track in tracks
    ]

    cost_matrix = np.zeros(
        (len(tracks), len(active_targets))
    )

    for i, (position, _) in enumerate(track_states):
        for j, target in enumerate(active_targets):
            cost_matrix[i, j] = np.linalg.norm(
                position - target.position
            )

    rows, columns = linear_sum_assignment(
        cost_matrix
    )

    position_errors = []
    velocity_errors = []
    matched_tracks = set()
    matched_targets = set()

    for i, j in zip(rows, columns):
        if cost_matrix[i, j] > max_match_distance:
            continue

        estimated_position, estimated_velocity = track_states[i]
        target = active_targets[j]

        position_errors.append(
            np.linalg.norm(
                estimated_position - target.position
            )
        )

        velocity_errors.append(
            np.linalg.norm(
                estimated_velocity - target.velocity
            )
        )

        matched_tracks.add(i)
        matched_targets.add(j)

    missed_targets = (
        len(active_targets)
        - len(matched_targets)
    )

    false_tracks = (
        len(tracks)
        - len(matched_tracks)
    )

    return (
        position_errors,
        velocity_errors,
        missed_targets,
        false_tracks
    )


def run_configuration(
    name,
    use_radar_a,
    use_radar_b,
    scenario_seed=42,
    n_targets=5,
    duration=60.0
):
    scenario = generate_scenario(
        n_targets=n_targets,
        seed=scenario_seed
    )

    sensors = []

    if use_radar_a:
        sensors.append(create_radar_a())

    if use_radar_b:
        sensors.append(create_radar_b())

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
            active_targets = [
                target
                for target in scenario.targets
                if target.active
            ]

            tracks = tracker.confirmed_tracks

            (
                position_error,
                velocity_error,
                missed,
                false
            ) = evaluate_tracks(
                tracks,
                active_targets,
                time
            )

            position_errors.extend(
                position_error
            )

            velocity_errors.extend(
                velocity_error
            )

            missed_targets += missed
            false_tracks += false

            target_instances += len(
                active_targets
            )

            track_instances += len(
                tracks
            )

            next_evaluation += 1.0

        if time >= duration:
            break

        for target in scenario.targets:
            target.step(dt)

        time += dt

    matched_targets = (
        target_instances
        - missed_targets
    )

    coverage = (
        matched_targets / target_instances
        if target_instances > 0
        else np.nan
    )

    false_track_fraction = (
        false_tracks / track_instances
        if track_instances > 0
        else 0.0
    )

    return {
        "name": name,
        "position_rmse": rmse(position_errors),
        "velocity_rmse": rmse(velocity_errors),
        "coverage": coverage,
        "false_track_fraction": false_track_fraction
    }


results = [
    run_configuration(
        name="Radar A",
        use_radar_a=True,
        use_radar_b=False
    ),
    run_configuration(
        name="Radar B",
        use_radar_a=False,
        use_radar_b=True
    ),
    run_configuration(
        name="A + B",
        use_radar_a=True,
        use_radar_b=True
    )
]


print()

for result in results:
    print(result["name"])
    print(
        f"  Position RMSE: "
        f"{result['position_rmse']:.1f} m"
    )
    print(
        f"  Velocity RMSE: "
        f"{result['velocity_rmse']:.1f} m/s"
    )
    print(
        f"  Track coverage: "
        f"{100 * result['coverage']:.1f}%"
    )
    print(
        f"  False-track fraction: "
        f"{100 * result['false_track_fraction']:.1f}%"
    )
    print()


names = [
    result["name"]
    for result in results
]

position_rmse = [
    result["position_rmse"]
    for result in results
]

velocity_rmse = [
    result["velocity_rmse"]
    for result in results
]

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    names,
    position_rmse
)

ax.set_ylabel("Position RMSE [m]")
ax.set_title("Position tracking accuracy")
ax.grid(axis="y")

plt.show()


fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    names,
    velocity_rmse
)

ax.set_ylabel("Velocity RMSE [m/s]")
ax.set_title("Velocity tracking accuracy")
ax.grid(axis="y")

plt.show()