import matplotlib.pyplot as plt
import numpy as np

from trackfusion.sensors.radar import Radar
from trackfusion.simulation.scenario import generate_scenario
from trackfusion.tracking.tracker import MultiTargetTracker


scenario = generate_scenario(
    n_targets=5,
    seed=42
)

radar = Radar(
    sensor_id="radar_a",
    position=np.array([0.0, 0.0]),
    max_range=30000.0,
    detection_probability=0.90,
    clutter_rate=1.0,
    range_std=120.0,
    bearing_std=np.deg2rad(1.0),
    scan_interval=1.0,
    start_time=0.0,
    rng=np.random.default_rng(123)
)

tracker = MultiTargetTracker(
    confirmation_hits=3,
    tentative_max_misses=1,
    confirmed_max_misses=5,
    acceleration_std=8.0,
    initial_velocity_std=200.0,
    gate_probability=0.99
)

duration = 60.0
dt = 0.25

truth_history = {
    target.target_id: []
    for target in scenario.targets
}

track_history = {}
detections = []
clutter = []

time = 0.0

while time <= duration:
    for target in scenario.targets:
        if target.active:
            truth_history[target.target_id].append(
                target.position.copy()
            )

    if time >= radar.next_scan_time - 1e-9:
        measurements = radar.measure(
            scenario.targets,
            time
        )

        tracker.process_scan(
            measurements=measurements,
            sensor=radar,
            timestamp=time
        )

        for measurement in measurements:
            r = measurement.z[0]
            bearing = measurement.z[1]

            x = radar.position[0] + r * np.cos(bearing)
            y = radar.position[1] + r * np.sin(bearing)

            if measurement.truth_target_id is None:
                clutter.append([x, y])
            else:
                detections.append([
                    x,
                    y,
                    measurement.truth_target_id
                ])

        for track in tracker.confirmed_tracks:
            if track.track_id not in track_history:
                track_history[track.track_id] = []

            track_history[track.track_id].append(
                track.position.copy()
            )

        radar.schedule_next_scan()

    if time >= duration:
        break

    for target in scenario.targets:
        target.step(dt)

    time += dt


fig, ax = plt.subplots(figsize=(11, 10))

colors = plt.get_cmap("tab10")

for target in scenario.targets:
    path = np.array(
        truth_history[target.target_id]
    )

    if len(path) == 0:
        continue

    color = colors(target.target_id - 1)

    ax.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        linewidth=1.5,
        alpha=0.6,
        label=f"Truth {target.target_id}"
    )

for target_id in range(1, 11):
    points = [
        point[:2]
        for point in detections
        if point[2] == target_id
    ]

    if not points:
        continue

    points = np.array(points)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=10,
        color=colors(target_id - 1),
        alpha=0.5
    )

if clutter:
    clutter_points = np.array(clutter)

    ax.scatter(
        clutter_points[:, 0],
        clutter_points[:, 1],
        color="0.75",
        marker=".",
        s=10,
        label="Clutter"
    )

for track_id, history in track_history.items():
    path = np.array(history)

    if len(path) < 2:
        continue

    ax.plot(
        path[:, 0],
        path[:, 1],
        color="black",
        linewidth=2,
        linestyle="--"
    )

    ax.text(
        path[-1, 0],
        path[-1, 1],
        f"T{track_id}",
        fontsize=8
    )

ax.scatter(
    radar.position[0],
    radar.position[1],
    marker="^",
    color="black",
    s=120,
    label="Radar"
)

half_size = scenario.half_size

ax.plot(
    [-half_size, half_size, half_size, -half_size, -half_size],
    [-half_size, -half_size, half_size, half_size, -half_size],
    linestyle="--",
    color="black",
    linewidth=1
)

ax.set_xlim(
    -1.2 * half_size,
    1.2 * half_size
)

ax.set_ylim(
    -1.2 * half_size,
    1.2 * half_size
)

ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(
    f"Multi-target tracking | scenario seed = {scenario.seed}"
)

ax.grid()
ax.legend(
    ncol=2,
    fontsize=8
)

plt.show()