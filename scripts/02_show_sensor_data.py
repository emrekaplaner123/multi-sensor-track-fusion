import matplotlib.pyplot as plt
import numpy as np

from trackfusion.sensors.doppler_radar import DopplerRadar
from trackfusion.sensors.radar import Radar
from trackfusion.simulation.scenario import generate_scenario


scenario = generate_scenario(
    n_targets=10,
    seed=42
)

radar_a = Radar(
    sensor_id="radar_a",
    position=np.array([-8000.0, 0.0]),
    max_range=30000.0,
    detection_probability=0.90,
    clutter_rate=2.0,
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

duration = 180.0
dt = 0.25

histories = {
    target.target_id: [target.position.copy()]
    for target in scenario.targets
}

detections = {
    "radar_a": {
        target.target_id: []
        for target in scenario.targets
    },
    "radar_b": {
        target.target_id: []
        for target in scenario.targets
    }
}

clutter = {
    "radar_a": [],
    "radar_b": []
}

time = 0.0

while time <= duration:
    for sensor in sensors:
        if time >= sensor.next_scan_time - 1e-9:
            measurements = sensor.measure(
                scenario.targets,
                time
            )

            for measurement in measurements:
                r = measurement.z[0]
                bearing = measurement.z[1]

                x = sensor.position[0] + r * np.cos(bearing)
                y = sensor.position[1] + r * np.sin(bearing)

                if measurement.truth_target_id is None:
                    clutter[sensor.sensor_id].append([x, y])
                else:
                    detections[sensor.sensor_id][
                        measurement.truth_target_id
                    ].append([x, y])

            sensor.schedule_next_scan()

    if time >= duration:
        break

    for target in scenario.targets:
        target.step(dt)

        histories[target.target_id].append(
            target.position.copy()
        )

    time += dt


fig, ax = plt.subplots(figsize=(11, 10))

colors = plt.get_cmap("tab10")

for target in scenario.targets:
    target_id = target.target_id
    color = colors(target_id - 1)

    path = np.array(
        histories[target_id]
    )

    ax.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        linewidth=1.8,
        label=f"Target {target_id}"
    )

    radar_a_points = detections["radar_a"][target_id]

    if radar_a_points:
        radar_a_points = np.array(radar_a_points)

        ax.scatter(
            radar_a_points[:, 0],
            radar_a_points[:, 1],
            facecolors="none",
            edgecolors=color,
            marker="o",
            s=24
        )

    radar_b_points = detections["radar_b"][target_id]

    if radar_b_points:
        radar_b_points = np.array(radar_b_points)

        ax.scatter(
            radar_b_points[:, 0],
            radar_b_points[:, 1],
            facecolors="none",
            edgecolors=color,
            marker="s",
            s=24
        )

if clutter["radar_a"]:
    points = np.array(clutter["radar_a"])

    ax.scatter(
        points[:, 0],
        points[:, 1],
        marker=".",
        color="0.7",
        s=12,
        label="Radar A clutter"
    )

if clutter["radar_b"]:
    points = np.array(clutter["radar_b"])

    ax.scatter(
        points[:, 0],
        points[:, 1],
        marker="x",
        color="0.6",
        s=15,
        label="Radar B clutter"
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
    f"Multi-sensor measurements | scenario seed = {scenario.seed}"
)

ax.grid()

ax.legend(
    ncol=2,
    fontsize=8
)

plt.show()