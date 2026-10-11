from dataclasses import dataclass, field

import numpy as np


@dataclass
class Target:
    target_id: int
    position: np.ndarray
    speed: float
    heading: float
    waypoints: list[np.ndarray]
    max_turn_rate: float
    waypoint_tolerance: float = 250.0
    current_waypoint_index: int = field(init=False, default=0)
    initial_position: np.ndarray = field(init=False)
    active: bool = field(init=False, default=True)

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.initial_position = self.position.copy()

        self.waypoints = [
            np.asarray(waypoint, dtype=float)
            for waypoint in self.waypoints
        ]

        if self.position.shape != (2,):
            raise ValueError("position must have shape (2,).")

        if any(waypoint.shape != (2,) for waypoint in self.waypoints):
            raise ValueError("Each waypoint must have shape (2,).")

        if self.speed <= 0:
            raise ValueError("speed must be positive.")

        if self.max_turn_rate <= 0:
            raise ValueError("max_turn_rate must be positive.")

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * np.array([
            np.cos(self.heading),
            np.sin(self.heading)
        ])

    @property
    def state(self) -> np.ndarray:
        return np.array([
            self.position[0],
            self.position[1],
            self.velocity[0],
            self.velocity[1]
        ])

    @property
    def current_waypoint(self) -> np.ndarray | None:
        if self.current_waypoint_index >= len(self.waypoints):
            return None

        return self.waypoints[self.current_waypoint_index]

    def step(self, dt: float):
        if not self.active:
            return

        self._advance_waypoint()

        if not self.active:
            return

        waypoint = self.current_waypoint

        direction = waypoint - self.position

        desired_heading = np.arctan2(
            direction[1],
            direction[0]
        )

        heading_error = (
            desired_heading - self.heading + np.pi
        ) % (2 * np.pi) - np.pi

        max_change = self.max_turn_rate * dt

        self.heading += np.clip(
            heading_error,
            -max_change,
            max_change
        )

        self.heading = (
            self.heading + np.pi
        ) % (2 * np.pi) - np.pi

        self.position += self.velocity * dt

        self._advance_waypoint()

    def _advance_waypoint(self):
        while self.current_waypoint is not None:
            waypoint = self.current_waypoint

            distance = np.linalg.norm(
                waypoint - self.position
            )

            if self.current_waypoint_index == 0:
                segment_start = self.initial_position
            else:
                segment_start = self.waypoints[
                    self.current_waypoint_index - 1
                ]

            segment = waypoint - segment_start

            progress = np.dot(
                self.position - segment_start,
                segment
            ) / np.dot(segment, segment)

            if distance <= self.waypoint_tolerance or progress >= 1.0:
                self.current_waypoint_index += 1

                if self.current_waypoint_index >= len(self.waypoints):
                    self.active = False
                    return

                continue

            break