import numpy as np


def position_error(
    estimated_position: np.ndarray,
    true_position: np.ndarray
) -> float:
    return float(
        np.linalg.norm(
            estimated_position - true_position
        )
    )


def velocity_error(
    estimated_velocity: np.ndarray,
    true_velocity: np.ndarray
) -> float:
    return float(
        np.linalg.norm(
            estimated_velocity - true_velocity
        )
    )


def rmse(errors) -> float:
    errors = np.asarray(errors, dtype=float)

    if len(errors) == 0:
        return np.nan

    return float(
        np.sqrt(
            np.mean(errors**2)
        )
    )


def mean_error(errors) -> float:
    errors = np.asarray(errors, dtype=float)

    if len(errors) == 0:
        return np.nan

    return float(
        np.mean(errors)
    )