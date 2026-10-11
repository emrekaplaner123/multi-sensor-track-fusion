import numpy as np

from trackfusion.filtering.ekf import ExtendedKalmanFilter
from trackfusion.tracking.track import Track, TrackStatus


class TrackManager:
    def __init__(
        self,
        confirmation_hits: int = 3,
        tentative_max_misses: int = 1,
        confirmed_max_misses: int = 5,
        acceleration_std: float = 8.0,
        initial_velocity_std: float = 100.0
    ):
        self.confirmation_hits = confirmation_hits
        self.tentative_max_misses = tentative_max_misses
        self.confirmed_max_misses = confirmed_max_misses
        self.acceleration_std = acceleration_std
        self.initial_velocity_std = initial_velocity_std

        self.tracks = []
        self.next_track_id = 1

    def create_track(self, seed, sensor):
        state = np.array([
            seed.position[0],
            seed.position[1],
            seed.velocity[0],
            seed.velocity[1]
        ])

        r = seed.measurement.z[0]

        position_std = max(
            sensor.range_std,
            r * sensor.bearing_std
        )

        covariance = np.diag([
            position_std**2,
            position_std**2,
            self.initial_velocity_std**2,
            self.initial_velocity_std**2
        ])

        ekf = ExtendedKalmanFilter(
            state=state,
            covariance=covariance,
            acceleration_std=self.acceleration_std
        )

        track = Track(
            track_id=self.next_track_id,
            ekf=ekf,
            created_at=seed.timestamp,
            last_time=seed.timestamp,
            hits=2
        )

        self.next_track_id += 1
        self.tracks.append(track)

        return track

    def update_statuses(self):
        for track in self.tracks:
            if (
                track.status == TrackStatus.TENTATIVE
                and track.hits >= self.confirmation_hits
            ):
                track.confirm()

    def delete_stale_tracks(self):
        surviving_tracks = []

        for track in self.tracks:
            if (
                track.status == TrackStatus.TENTATIVE
                and track.misses > self.tentative_max_misses
            ):
                continue

            if (
                track.status == TrackStatus.CONFIRMED
                and track.misses > self.confirmed_max_misses
            ):
                continue

            surviving_tracks.append(track)

        self.tracks = surviving_tracks

    def maintain_tracks(self):
        self.update_statuses()
        self.delete_stale_tracks()