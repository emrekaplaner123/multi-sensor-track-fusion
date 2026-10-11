import matplotlib.pyplot as plt
import numpy as np

from trackfusion.association.gating import gate_measurement
from trackfusion.filtering.ekf import ExtendedKalmanFilter
from trackfusion.sensors.radar import Radar
from trackfusion.simulation.scenario import generate_scenario


scenario = generate_scenario(
    n_targets=1,
    seed=42
)

target = scenario.targets[0]

radar = Radar(
    sensor_id="radar_a",
    position=np.array([0.0, 0.0]),
    max_range=30000.0,
    detection_probability=1.0,
    clutter_rate=25.0,
    range_std=120.0,
    bearing_std=np.deg2rad(1.0),
    scan_interval=1.0,
    start_time=0.0,
    rng=np.random.default_rng(123)
)

ekf = ExtendedKalmanFilter(
    state=target.state + np.array([
        500.0,
        -500.0,
        30.0,
        -20.0
    ]),
    covariance=np.diag([
        800.0**2,
        800.0**2,
        80.0**2,
        80.0**2
    ]),
    acceleration_std=8.0
)

duration = 25.0
dt = 0.25

time = 0.0
filter_time = 0.0

selected_measurements = None
selected_truth_position = None
selected_estimate = None
selected_timestamp = None

while time <= duration and target.active:
    if time >= radar.next_scan_time - 1e-9:
        prediction_dt = time - filter_time

        if prediction_dt > 0:
            ekf.predict(prediction_dt)
            filter_time = time

        measurements = radar.measure(
            [target],
            time
        )

        if 10.0 <= time <= 15.0:
            selected_measurements = measurements
            selected_truth_position = target.position.copy()
            selected_estimate = ekf.x.copy()
            selected_timestamp = time
            break

        true_measurements = [
            measurement
            for measurement in measurements
            if measurement.truth_target_id == target.target_id
        ]

        if true_measurements:
            ekf.update(
                z=true_measurements[0].z,
                h=radar.h,
                jacobian=radar.jacobian,
                R=true_measurements[0].R,
                angle_indices=(1,)
            )

        radar.schedule_next_scan()

    target.step(dt)
    time += dt


accepted = []
rejected = []

for measurement in selected_measurements:
    inside_gate, distance = gate_measurement(
        ekf,
        measurement,
        radar,
        gate_probability=0.99
    )

    r, bearing = measurement.z

    x = radar.position[0] + r * np.cos(bearing)
    y = radar.position[1] + r * np.sin(bearing)

    if inside_gate:
        accepted.append([x, y])
    else:
        rejected.append([x, y])


fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(
    radar.position[0],
    radar.position[1],
    marker="^",
    s=120,
    label="Radar"
)

ax.scatter(
    selected_truth_position[0],
    selected_truth_position[1],
    marker="x",
    s=100,
    label="True target"
)

ax.scatter(
    selected_estimate[0],
    selected_estimate[1],
    marker="X",
    s=100,
    label="Predicted track"
)

if accepted:
    accepted = np.array(accepted)

    ax.scatter(
        accepted[:, 0],
        accepted[:, 1],
        marker="o",
        s=60,
        facecolors="none",
        label="Inside gate"
    )

if rejected:
    rejected = np.array(rejected)

    ax.scatter(
        rejected[:, 0],
        rejected[:, 1],
        marker=".",
        s=30,
        label="Rejected"
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
ax.set_title(
    f"Measurement gating at t = {selected_timestamp:.1f} s"
)

ax.grid()
ax.legend()

plt.show()