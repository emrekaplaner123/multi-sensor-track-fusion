import numpy as np


class ExtendedKalmanFilter:
    def __init__(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        acceleration_std: float
    ):
        self.x = np.asarray(state, dtype=float)
        self.P = np.asarray(covariance, dtype=float)
        self.acceleration_std = acceleration_std

    def predict(self, dt: float):
        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        q = self.acceleration_std**2

        Q = q * np.array([
            [dt**4 / 4, 0.0, dt**3 / 2, 0.0],
            [0.0, dt**4 / 4, 0.0, dt**3 / 2],
            [dt**3 / 2, 0.0, dt**2, 0.0],
            [0.0, dt**3 / 2, 0.0, dt**2]
        ])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation(
        self,
        z: np.ndarray,
        h,
        jacobian,
        R: np.ndarray,
        angle_indices: tuple[int, ...] = ()
    ):
        z = np.asarray(z, dtype=float)
        R = np.asarray(R, dtype=float)

        z_pred = h(self.x)
        H = jacobian(self.x)

        y = z - z_pred

        for index in angle_indices:
            y[index] = (
                y[index] + np.pi
            ) % (2 * np.pi) - np.pi

        S = H @ self.P @ H.T + R

        return y, S, H

    def update(
        self,
        z: np.ndarray,
        h,
        jacobian,
        R: np.ndarray,
        angle_indices: tuple[int, ...] = ()
    ):
        y, S, H = self.innovation(
            z,
            h,
            jacobian,
            R,
            angle_indices
        )

        K = np.linalg.solve(
            S,
            (self.P @ H.T).T
        ).T

        self.x = self.x + K @ y

        I = np.eye(len(self.x))
        A = I - K @ H

        self.P = (
            A @ self.P @ A.T
            + K @ R @ K.T
        )