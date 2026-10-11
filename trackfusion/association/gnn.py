import numpy as np

from scipy.optimize import linear_sum_assignment

from trackfusion.association.gating import gate_measurement


def associate(
    filters,
    measurements,
    sensor,
    gate_probability: float = 0.99
):
    n_tracks = len(filters)
    n_measurements = len(measurements)

    if n_tracks == 0:
        return [], [], list(range(n_measurements))

    if n_measurements == 0:
        return [], list(range(n_tracks)), []

    large_cost = 1e9

    cost_matrix = np.full(
        (n_tracks, n_measurements),
        large_cost
    )

    for track_index, ekf in enumerate(filters):
        for measurement_index, measurement in enumerate(measurements):
            inside_gate, distance = gate_measurement(
                ekf,
                measurement,
                sensor,
                gate_probability
            )

            if inside_gate:
                cost_matrix[
                    track_index,
                    measurement_index
                ] = distance

    row_indices, column_indices = linear_sum_assignment(
        cost_matrix
    )

    assignments = []
    assigned_tracks = set()
    assigned_measurements = set()

    for track_index, measurement_index in zip(
        row_indices,
        column_indices
    ):
        cost = cost_matrix[
            track_index,
            measurement_index
        ]

        if cost >= large_cost:
            continue

        assignments.append(
            (
                track_index,
                measurement_index,
                cost
            )
        )

        assigned_tracks.add(track_index)
        assigned_measurements.add(measurement_index)

    unassigned_tracks = [
        index
        for index in range(n_tracks)
        if index not in assigned_tracks
    ]

    unassigned_measurements = [
        index
        for index in range(n_measurements)
        if index not in assigned_measurements
    ]

    return (
        assignments,
        unassigned_tracks,
        unassigned_measurements
    )