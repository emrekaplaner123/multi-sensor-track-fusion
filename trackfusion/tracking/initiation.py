from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Candidate:
    position: np.ndarray
    timestamp: float
    sensor_id: str


@dataclass
class TrackSeed:
    position: np.ndarray
    velocity: np.ndarray
    timestamp: float
    measurement: object


class TrackInitiator:
    def __init__(
        self,
        max_age: float = 2.0,
        max_speed: float = 350.0,
        position_margin: float = 500.0
    ):
        self.max_age = max_age
        self.max_speed = max_speed
        self.position_margin = position_margin
        self.candidates = []

    def process(self, measurements, sensor, timestamp):
        self._remove_old(timestamp)

        positions = [
            self._to_cartesian(measurement, sensor)
            for measurement in measurements
        ]

        candidates = [
            candidate
            for candidate in self.candidates
            if candidate.sensor_id == sensor.sensor_id
        ]

        matched_measurements = set()
        matched_candidates = set()
        seeds = []

        if candidates and measurements:
            large_cost = 1e9

            costs = np.full(
                (len(candidates), len(measurements)),
                large_cost
            )

            for i, candidate in enumerate(candidates):
                dt = timestamp - candidate.timestamp

                if dt <= 0 or dt > self.max_age:
                    continue

                max_distance = (
                    self.max_speed * dt
                    + self.position_margin
                )

                for j, position in enumerate(positions):
                    distance = np.linalg.norm(
                        position - candidate.position
                    )

                    if distance <= max_distance:
                        costs[i, j] = distance

            rows, columns = linear_sum_assignment(costs)

            for i, j in zip(rows, columns):
                if costs[i, j] >= large_cost:
                    continue

                candidate = candidates[i]
                dt = timestamp - candidate.timestamp

                velocity = (
                    positions[j] - candidate.position
                ) / dt

                seeds.append(
                    TrackSeed(
                        position=positions[j],
                        velocity=velocity,
                        timestamp=timestamp,
                        measurement=measurements[j]
                    )
                )

                matched_candidates.add(id(candidate))
                matched_measurements.add(j)

        self.candidates = [
            candidate
            for candidate in self.candidates
            if id(candidate) not in matched_candidates
        ]

        for index, position in enumerate(positions):
            if index in matched_measurements:
                continue

            self.candidates.append(
                Candidate(
                    position=position,
                    timestamp=timestamp,
                    sensor_id=sensor.sensor_id
                )
            )

        return seeds

    def _remove_old(self, timestamp):
        self.candidates = [
            candidate
            for candidate in self.candidates
            if timestamp - candidate.timestamp <= self.max_age
        ]

    @staticmethod
    def _to_cartesian(measurement, sensor):
        r = measurement.z[0]
        bearing = measurement.z[1]

        return sensor.position + np.array([
            r * np.cos(bearing),
            r * np.sin(bearing)
        ])