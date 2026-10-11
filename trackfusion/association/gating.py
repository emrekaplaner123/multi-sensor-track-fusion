import numpy as np
from scipy.stats import chi2


def mahalanobis_distance(
    innovation: np.ndarray,
    innovation_covariance: np.ndarray
) -> float:
    return float(
        innovation.T
        @ np.linalg.solve(
            innovation_covariance,
            innovation
        )
    )


def gate_measurement(
    ekf,
    measurement,
    sensor,
    gate_probability: float = 0.99
) -> tuple[bool, float]:
    innovation, S, _ = ekf.innovation(
        z=measurement.z,
        h=sensor.h,
        jacobian=sensor.jacobian,
        R=measurement.R,
        angle_indices=(1,)
    )

    distance = mahalanobis_distance(
        innovation,
        S
    )

    threshold = chi2.ppf(
        gate_probability,
        df=len(measurement.z)
    )

    return distance <= threshold, distance