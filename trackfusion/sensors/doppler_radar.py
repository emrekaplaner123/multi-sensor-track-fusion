from dataclasses import dataclass

import numpy as np

from trackfusion.sensors.measurement import Measurement, MeasurementType
from trackfusion.sensors.radar import Radar
from trackfusion.simulation.target import Target


@dataclass
class DopplerRadar(Radar):
    range_rate_std: float = 5.0
    clutter_max_abs_range_rate: float = 300.0

    @property
    def R(self) -> np.ndarray:
        return np.diag([
            self.range_std**2,
            self.bearing_std**2,
            self.range_rate_std**2
        ])

    def h(self, state: np.ndarray) -> np.ndarray:
        dx = state[0] - self.position[0]
        dy = state[1] - self.position[1]
        vx = state[2]
        vy = state[3]

        r = np.sqrt(dx**2 + dy**2)

        if r < 1e-12:
            raise ValueError("State is too close to radar position.")

        range_rate = (
            dx * vx + dy * vy
        ) / r

        return np.array([
            r,
            np.arctan2(dy, dx),
            range_rate
        ])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        dx = state[0] - self.position[0]
        dy = state[1] - self.position[1]
        vx = state[2]
        vy = state[3]

        q = dx**2 + dy**2

        if q < 1e-12:
            raise ValueError("State is too close to radar position.")

        r = np.sqrt(q)

        range_rate = (
            dx * vx + dy * vy
        ) / r

        return np.array([
            [dx / r, dy / r, 0.0, 0.0],
            [-dy / q, dx / q, 0.0, 0.0],
            [
                vx / r - range_rate * dx / q,
                vy / r - range_rate * dy / q,
                dx / r,
                dy / r
            ]
        ])

    def measure(
        self,
        targets: list[Target],
        timestamp: float
    ) -> list[Measurement]:

        measurements = []

        for target in targets:
            if not target.active:
                continue

            true_measurement = self.h(target.state)
            distance = true_measurement[0]

            if distance > self.max_range:
                continue

            if self.rng.random() > self.detection_probability:
                continue

            z = true_measurement + np.array([
                self.rng.normal(0.0, self.range_std),
                self.rng.normal(0.0, self.bearing_std),
                self.rng.normal(0.0, self.range_rate_std)
            ])

            z[1] = (
                z[1] + np.pi
            ) % (2 * np.pi) - np.pi

            measurements.append(
                Measurement(
                    timestamp=timestamp,
                    sensor_id=self.sensor_id,
                    z=z,
                    R=self.R,
                    measurement_type=MeasurementType.RANGE_BEARING_DOPPLER,
                    truth_target_id=target.target_id
                )
            )

        measurements.extend(
            self._generate_clutter(timestamp)
        )

        return measurements

    def _generate_clutter(
        self,
        timestamp: float
    ) -> list[Measurement]:

        count = self.rng.poisson(
            self.clutter_rate
        )

        measurements = []

        for _ in range(count):
            clutter_range = (
                self.max_range
                * np.sqrt(self.rng.random())
            )

            clutter_bearing = self.rng.uniform(
                -np.pi,
                np.pi
            )

            clutter_range_rate = self.rng.uniform(
                -self.clutter_max_abs_range_rate,
                self.clutter_max_abs_range_rate
            )

            measurements.append(
                Measurement(
                    timestamp=timestamp,
                    sensor_id=self.sensor_id,
                    z=np.array([
                        clutter_range,
                        clutter_bearing,
                        clutter_range_rate
                    ]),
                    R=self.R,
                    measurement_type=MeasurementType.RANGE_BEARING_DOPPLER,
                    truth_target_id=None
                )
            )

        return measurements