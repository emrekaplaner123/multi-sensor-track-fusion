from dataclasses import dataclass

import numpy as np

from trackfusion.simulation.target import Target


@dataclass
class Scenario:
    targets: list[Target]
    half_size: float
    seed: int


def generate_scenario(
    n_targets: int = 10,
    seed: int = 42,
    half_size: float = 20000.0,
    min_speed: float = 160.0,
    max_speed: float = 250.0
) -> Scenario:

    rng = np.random.default_rng(seed)

    targets = []

    for target_id in range(1, n_targets + 1):
        entry_edge = rng.integers(0, 4)

        exit_edges = [
            edge
            for edge in range(4)
            if edge != entry_edge
        ]

        exit_edge = rng.choice(exit_edges)

        start = _point_on_edge(
            rng,
            entry_edge,
            half_size
        )

        exit_point = _point_on_edge(
            rng,
            exit_edge,
            half_size
        )

        waypoints = _generate_waypoints(
            rng,
            start,
            exit_point,
            half_size
        )

        speed = rng.uniform(
            min_speed,
            max_speed
        )

        direction = waypoints[0] - start

        heading = np.arctan2(
            direction[1],
            direction[0]
        )

        targets.append(
            Target(
                target_id=target_id,
                position=start,
                speed=speed,
                heading=heading,
                waypoints=waypoints,
                max_turn_rate=np.deg2rad(
                    rng.uniform(1.5, 3.0)
                ),
                waypoint_tolerance=300.0
            )
        )

    return Scenario(
        targets=targets,
        half_size=half_size,
        seed=seed
    )


def _point_on_edge(
    rng: np.random.Generator,
    edge: int,
    half_size: float
) -> np.ndarray:

    offset = rng.uniform(
        -0.85 * half_size,
        0.85 * half_size
    )

    if edge == 0:
        return np.array([
            offset,
            half_size
        ])

    if edge == 1:
        return np.array([
            half_size,
            offset
        ])

    if edge == 2:
        return np.array([
            offset,
            -half_size
        ])

    return np.array([
        -half_size,
        offset
    ])


def _generate_waypoints(
    rng: np.random.Generator,
    start: np.ndarray,
    end: np.ndarray,
    half_size: float
) -> list[np.ndarray]:

    direction = end - start
    length = np.linalg.norm(direction)

    perpendicular = np.array([
        -direction[1],
        direction[0]
    ]) / length

    fractions = [0.33, 0.66]

    waypoints = []

    for fraction in fractions:
        point = start + fraction * direction

        offset = rng.uniform(
            -0.12 * half_size,
            0.12 * half_size
        )

        point = point + offset * perpendicular

        point = np.clip(
            point,
            -0.9 * half_size,
            0.9 * half_size
        )

        waypoints.append(point)

    waypoints.append(end)

    return waypoints