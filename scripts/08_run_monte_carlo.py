import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trackfusion.evaluation.monte_carlo import run_configuration


configurations = [
    ("Radar A", True, False),
    ("Radar B", False, True),
    ("A + B", True, True)
]

n_scenarios = 20
n_targets = 5
duration = 60.0

results = {
    name: {
        "position_rmse": [],
        "velocity_rmse": [],
        "coverage": [],
        "false_track_fraction": []
    }
    for name, _, _ in configurations
}

for seed in range(n_scenarios):
    print(f"Scenario {seed + 1}/{n_scenarios}")

    for name, use_radar_a, use_radar_b in configurations:
        result = run_configuration(
            scenario_seed=seed,
            use_radar_a=use_radar_a,
            use_radar_b=use_radar_b,
            n_targets=n_targets,
            duration=duration
        )

        for metric in results[name]:
            results[name][metric].append(
                result[metric]
            )


print()

for name in results:
    print(name)

    print(
        f"  Position RMSE: "
        f"{np.nanmean(results[name]['position_rmse']):.1f} "
        f"± {np.nanstd(results[name]['position_rmse']):.1f} m"
    )

    print(
        f"  Velocity RMSE: "
        f"{np.nanmean(results[name]['velocity_rmse']):.1f} "
        f"± {np.nanstd(results[name]['velocity_rmse']):.1f} m/s"
    )

    print(
        f"  Track coverage: "
        f"{100 * np.nanmean(results[name]['coverage']):.1f}%"
    )

    print(
        f"  False-track fraction: "
        f"{100 * np.nanmean(results[name]['false_track_fraction']):.1f}%"
    )

    print()


output_directory = Path("outputs/results")
output_directory.mkdir(
    parents=True,
    exist_ok=True
)

csv_path = output_directory / "monte_carlo_results.csv"

with open(
    csv_path,
    "w",
    newline=""
) as file:
    writer = csv.writer(file)

    writer.writerow([
        "scenario_seed",
        "configuration",
        "position_rmse",
        "velocity_rmse",
        "coverage",
        "false_track_fraction"
    ])

    for seed in range(n_scenarios):
        for name in results:
            writer.writerow([
                seed,
                name,
                results[name]["position_rmse"][seed],
                results[name]["velocity_rmse"][seed],
                results[name]["coverage"][seed],
                results[name]["false_track_fraction"][seed]
            ])


names = [
    name
    for name, _, _ in configurations
]

position_means = [
    np.nanmean(
        results[name]["position_rmse"]
    )
    for name in names
]

position_stds = [
    np.nanstd(
        results[name]["position_rmse"]
    )
    for name in names
]

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    names,
    position_means,
    yerr=position_stds,
    capsize=5
)

ax.set_ylabel("Position RMSE [m]")
ax.set_title(
    f"Position accuracy across {n_scenarios} scenarios"
)
ax.grid(axis="y")

plt.show()


velocity_means = [
    np.nanmean(
        results[name]["velocity_rmse"]
    )
    for name in names
]

velocity_stds = [
    np.nanstd(
        results[name]["velocity_rmse"]
    )
    for name in names
]

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    names,
    velocity_means,
    yerr=velocity_stds,
    capsize=5
)

ax.set_ylabel("Velocity RMSE [m/s]")
ax.set_title(
    f"Velocity accuracy across {n_scenarios} scenarios"
)
ax.grid(axis="y")

plt.show()