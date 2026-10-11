import matplotlib.pyplot as plt
import numpy as np

from trackfusion.sensors.doppler_radar import DopplerRadar
from trackfusion.sensors.radar import Radar
from trackfusion.simulation.scenario import generate_scenario
from trackfusion.tracking.tracker import MultiTargetTracker


scenario = generate_scenario(
    n_targets=5,
    seed=42
)

radar_a = Radar(
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

radar_b = DopplerRadar(
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

sensors = [radar_a, radar_b]

tracker = MultiTargetTracker(
    confirmation_hits=3,
    tentative_max_misses=1,
    confirmed_max_misses=5,
    acceleration_std=8.0,
    initial_velocity_std=100.0,
    gate_probability=0.99
)

duration = 60.0
dt = 0.25

truth_history = {
    target.target_id: []
    for target in scenario.targets
}

track_history = {}

detections_a = {
    target.target_id: []
    for target in scenario.targets
}

detections_b = {
    target.target_id: []
    for target in scenario.targets
}

clutter_a = []
clutter_b = []

time = 0.0

while time <= duration:
    for target in scenario.targets:
        if target.active:
            truth_history[target.target_id].append(
                target.position.copy()
            )

    for sensor in sensors:
        if time >= sensor.next_scan_time - 1e-9:
            measurements = sensor.measure(
                scenario.targets,
                time
            )

            tracker.process_scan(
                measurements=measurements,
                sensor=sensor,
                timestamp=time,
                allow_initiation=sensor.sensor_id == "radar_a"
            )

            for measurement in measurements:
                r = measurement.z[0]
                bearing = measurement.z[1]

                x = (
                    sensor.position[0]
                    + r * np.cos(bearing)
                )

                y = (
                    sensor.position[1]
                    + r * np.sin(bearing)
                )

                if measurement.truth_target_id is None:
                    if sensor.sensor_id == "radar_a":
                        clutter_a.append([x, y])
                    else:
                        clutter_b.append([x, y])

                else:
                    if sensor.sensor_id == "radar_a":
                        detections_a[
                            measurement.truth_target_id
                        ].append([x, y])

                    else:
                        detections_b[
                            measurement.truth_target_id
                        ].append([x, y])

            for track in tracker.confirmed_tracks:
                if track.track_id not in track_history:
                    track_history[track.track_id] = []

                track_history[track.track_id].append(
                    track.position.copy()
                )

            sensor.schedule_next_scan()

    if time >= duration:
        break

    for target in scenario.targets:
        target.step(dt)

    time += dt


fig, ax = plt.subplots(figsize=(11, 10))

colors = plt.get_cmap("tab10")

for target in scenario.targets:
    target_id = target.target_id
    path = np.array(
        truth_history[target_id]
    )

    if len(path) == 0:
        continue

    color = colors(target_id - 1)

    ax.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        linewidth=1.5,
        alpha=0.6,
        label=f"Truth {target_id}"
    )

    points_a = detections_a[target_id]

    if points_a:
        points_a = np.array(points_a)

        ax.scatter(
            points_a[:, 0],
            points_a[:, 1],
            facecolors="none",
            edgecolors=color,
            marker="o",
            s=22
        )

    points_b = detections_b[target_id]

    if points_b:
        points_b = np.array(points_b)

        ax.scatter(
            points_b[:, 0],
            points_b[:, 1],
            facecolors="none",
            edgecolors=color,
            marker="s",
            s=22
        )

if clutter_a:
    points = np.array(clutter_a)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        color="0.75",
        marker=".",
        s=10,
        label="Radar A clutter"
    )

if clutter_b:
    points = np.array(clutter_b)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        color="0.6",
        marker="x",
        s=14,
        label="Radar B clutter"
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
    radar_a.position[0],
    radar_a.position[1],
    marker="^",
    color="black",
    s=120,
    label="Radar A"
)

ax.scatter(
    radar_b.position[0],
    radar_b.position[1],
    marker="D",
    color="black",
    s=80,
    label="Radar B"
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
    f"Multi-sensor tracking | scenario seed = {scenario.seed}"
)

ax.grid()

ax.legend(
    ncol=2,
    fontsize=8
)

plt.show()