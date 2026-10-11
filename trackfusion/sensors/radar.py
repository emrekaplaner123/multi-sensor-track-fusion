from dataclasses import dataclass, field

import numpy as np

from trackfusion.sensors.measurement import Measurement, MeasurementType
from trackfusion.simulation.target import Target


@dataclass
class Radar:
    sensor_id: str
    position: np.ndarray
    max_range: float
    detection_probability: float
    clutter_rate: float
    range_std: float
    bearing_std: float
    scan_interval: float
    start_time: float = 0.0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.next_scan_time = self.start_time

        if self.position.shape != (2,):
            raise ValueError("position must have shape (2,).")

        if self.max_range <= 0:
            raise ValueError("max_range must be positive.")

        if not 0 <= self.detection_probability <= 1:
            raise ValueError(
                "detection_probability must be between 0 and 1."
            )

        if self.clutter_rate < 0:
            raise ValueError("clutter_rate cannot be negative.")

        if self.scan_interval <= 0:
            raise ValueError("scan_interval must be positive.")

    @property
    def R(self) -> np.ndarray:
        return np.diag([
            self.range_std**2,
            self.bearing_std**2
        ])

    def h(self, state: np.ndarray) -> np.ndarray:
        dx = state[0] - self.position[0]
        dy = state[1] - self.position[1]

        return np.array([
            np.sqrt(dx**2 + dy**2),
            np.arctan2(dy, dx)
        ])

    def jacobian(self, state: np.ndarray) -> np.ndarray:
        dx = state[0] - self.position[0]
        dy = state[1] - self.position[1]

        q = dx**2 + dy**2

        if q < 1e-12:
            raise ValueError("State is too close to radar position.")

        r = np.sqrt(q)

        return np.array([
            [dx / r, dy / r, 0.0, 0.0],
            [-dy / q, dx / q, 0.0, 0.0]
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

            relative_position = target.position - self.position
            distance = np.linalg.norm(relative_position)

            if distance > self.max_range:
                continue

            if self.rng.random() > self.detection_probability:
                continue

            bearing = np.arctan2(
                relative_position[1],
                relative_position[0]
            )

            measured_range = distance + self.rng.normal(
                0.0,
                self.range_std
            )

            measured_bearing = bearing + self.rng.normal(
                0.0,
                self.bearing_std
            )

            measured_bearing = (
                measured_bearing + np.pi
            ) % (2 * np.pi) - np.pi

            measurements.append(
                Measurement(
                    timestamp=timestamp,
                    sensor_id=self.sensor_id,
                    z=np.array([
                        measured_range,
                        measured_bearing
                    ]),
                    R=self.R,
                    measurement_type=MeasurementType.RANGE_BEARING,
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

            measurements.append(
                Measurement(
                    timestamp=timestamp,
                    sensor_id=self.sensor_id,
                    z=np.array([
                        clutter_range,
                        clutter_bearing
                    ]),
                    R=self.R,
                    measurement_type=MeasurementType.RANGE_BEARING,
                    truth_target_id=None
                )
            )

        return measurements

    def schedule_next_scan(self):
        self.next_scan_time += self.scan_interval