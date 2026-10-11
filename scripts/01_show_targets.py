import matplotlib.pyplot as plt
import numpy as np

from trackfusion.simulation.scenario import generate_scenario


scenario = generate_scenario(
    n_targets=10,
    seed=12
)

duration = 180.0
dt = 0.5

histories = {
    target.target_id: [target.position.copy()]
    for target in scenario.targets
}

time = 0.0

while time < duration:
    for target in scenario.targets:
        target.step(dt)
        histories[target.target_id].append(
            target.position.copy()
        )

    time += dt

fig, ax = plt.subplots(figsize=(10, 10))

colors = plt.get_cmap("tab10")

for target in scenario.targets:
    path = np.array(
        histories[target.target_id]
    )

    color = colors(
        target.target_id - 1
    )

    ax.plot(
        path[:, 0],
        path[:, 1],
        color=color,
        linewidth=2,
        label=f"Target {target.target_id}"
    )

    ax.scatter(
        path[0, 0],
        path[0, 1],
        color=color,
        marker="o",
        s=50
    )

    ax.scatter(
        path[-1, 0],
        path[-1, 1],
        color=color,
        marker="x",
        s=70
    )

half_size = scenario.half_size

ax.set_xlim(
    -1.1 * half_size,
    1.1 * half_size
)

ax.set_ylim(
    -1.1 * half_size,
    1.1 * half_size
)

ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_title(
    f"Target trajectories | seed = {scenario.seed}"
)

ax.grid()
ax.legend(
    ncol=2,
    loc="upper right"
)

plt.show()