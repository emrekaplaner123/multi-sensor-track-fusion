import matplotlib.pyplot as plt
import numpy as np

from trackfusion.filtering.ekf import ExtendedKalmanFilter
from trackfusion.sensors.doppler_radar import DopplerRadar
from trackfusion.sensors.radar import Radar
from trackfusion.simulation.scenario import generate_scenario


scenario = generate_scenario(
    n_targets=1,
    seed=42
)

target = scenario.targets[0]

radar_a = Radar(
    sensor_id="radar_a",
    position=np.array([-8000.0, 0.0]),
    max_range=40000.0,
    detection_probability=1.0,
    clutter_rate=0.0,
    range_std=120.0,
    bearing_std=np.deg2rad(1.0),
    scan_interval=1.0,
    start_time=0.0,
    rng=np.random.default_rng(100)
)

radar_b = DopplerRadar(
    sensor_id="radar_b",
    position=np.array([8000.0, 0.0]),
    max_range=40000.0,
    detection_probability=1.0,
    clutter_rate=0.0,
    range_std=60.0,
    bearing_std=np.deg2rad(0.5),
    scan_interval=0.5,
    start_time=0.25,
    rng=np.random.default_rng(200),
    range_rate_std=5.0
)

sensors = [radar_a, radar_b]

initial_state = target.state + np.array([
    800.0,
    -600.0,
    40.0,
    -30.0
])

initial_covariance = np.diag([
    1000.0**2,
    1000.0**2,
    100.0**2,
    100.0**2
])

ekf = ExtendedKalmanFilter(
    state=initial_state,
    covariance=initial_covariance,
    acceleration_std=8.0
)

duration = 180.0
dt = 0.25

truth_history = [target.position.copy()]
estimate_history = [ekf.x[:2].copy()]

radar_a_measurements = []
radar_b_measurements = []

time = 0.0
filter_time = 0.0

while time <= duration and target.active:
    for sensor in sensors:
        if time >= sensor.next_scan_time - 1e-9:
            prediction_dt = time - filter_time

            if prediction_dt > 0:
                ekf.predict(prediction_dt)
                filter_time = time

            measurements = sensor.measure(
                [target],
                time
            )

            if measurements:
                measurement = measurements[0]

                ekf.update(
                    z=measurement.z,
                    h=sensor.h,
                    jacobian=sensor.jacobian,
                    R=measurement.R,
                    angle_indices=(1,)
                )

                r = measurement.z[0]
                bearing = measurement.z[1]

                x = sensor.position[0] + r * np.cos(bearing)
                y = sensor.position[1] + r * np.sin(bearing)

                if sensor.sensor_id == "radar_a":
                    radar_a_measurements.append([x, y])
                else:
                    radar_b_measurements.append([x, y])

                estimate_history.append(
                    ekf.x[:2].copy()
                )

            sensor.schedule_next_scan()

    if time >= duration:
        break

    target.step(dt)

    truth_history.append(
        target.position.copy()
    )

    time += dt


truth_history = np.array(truth_history)
estimate_history = np.array(estimate_history)

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(
    truth_history[:, 0],
    truth_history[:, 1],
    linewidth=2,
    label="True trajectory"
)

ax.plot(
    estimate_history[:, 0],
    estimate_history[:, 1],
    linestyle="--",
    linewidth=2,
    label="EKF estimate"
)

if radar_a_measurements:
    points = np.array(radar_a_measurements)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        facecolors="none",
        marker="o",
        s=25,
        label="Radar A measurements"
    )

if radar_b_measurements:
    points = np.array(radar_b_measurements)

    ax.scatter(
        points[:, 0],
        points[:, 1],
        facecolors="none",
        marker="s",
        s=25,
        label="Radar B measurements"
    )

ax.scatter(
    radar_a.position[0],
    radar_a.position[1],
    marker="^",
    s=120,
    label="Radar A"
)

ax.scatter(
    radar_b.position[0],
    radar_b.position[1],
    marker="D",
    s=80,
    label="Radar B"
)

half_size = scenario.half_size

ax.plot(
    [-half_size, half_size, half_size, -half_size, -half_size],
    [-half_size, -half_size, half_size, half_size, -half_size],
    linestyle="--",
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
ax.set_title("Single-target EKF")
ax.grid()
ax.legend()

plt.show()