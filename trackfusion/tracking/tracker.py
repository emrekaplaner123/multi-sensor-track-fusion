import numpy as np

from trackfusion.association.gnn import associate
from trackfusion.tracking.initiation import TrackInitiator
from trackfusion.tracking.track import TrackStatus
from trackfusion.tracking.track_manager import TrackManager


class MultiTargetTracker:
    def __init__(
        self,
        confirmation_hits: int = 3,
        tentative_max_misses: int = 1,
        confirmed_max_misses: int = 5,
        acceleration_std: float = 8.0,
        initial_velocity_std: float = 100.0,
        gate_probability: float = 0.99
    ):
        self.track_manager = TrackManager(
            confirmation_hits=confirmation_hits,
            tentative_max_misses=tentative_max_misses,
            confirmed_max_misses=confirmed_max_misses,
            acceleration_std=acceleration_std,
            initial_velocity_std=initial_velocity_std
        )

        self.initiator = TrackInitiator()
        self.gate_probability = gate_probability

    @property
    def tracks(self):
        return self.track_manager.tracks

    @property
    def confirmed_tracks(self):
        return [
            track
            for track in self.tracks
            if track.status == TrackStatus.CONFIRMED
        ]

    def process_scan(
        self,
        measurements,
        sensor,
        timestamp: float,
        allow_initiation: bool = True
    ):
        tracks = self.track_manager.tracks

        for track in tracks:
            track.predict_to(timestamp)

        filters = [
            track.ekf
            for track in tracks
        ]

        assignments, unassigned_tracks, unassigned_measurements = associate(
            filters=filters,
            measurements=measurements,
            sensor=sensor,
            gate_probability=self.gate_probability
        )

        for track_index, measurement_index, _ in assignments:
            tracks[track_index].update(
                measurements[measurement_index],
                sensor
            )

        for track_index in unassigned_tracks:
            track = tracks[track_index]

            if self._inside_sensor_coverage(
                track,
                sensor
            ):
                track.mark_missed()

        created_tracks = []

        if allow_initiation:
            unassigned = [
                measurements[index]
                for index in unassigned_measurements
            ]

            seeds = self.initiator.process(
                measurements=unassigned,
                sensor=sensor,
                timestamp=timestamp
            )

            created_tracks = [
                self.track_manager.create_track(
                    seed=seed,
                    sensor=sensor
                )
                for seed in seeds
            ]

        self.track_manager.maintain_tracks()

        return {
            "assignments": assignments,
            "unassigned_tracks": unassigned_tracks,
            "unassigned_measurements": unassigned_measurements,
            "created_tracks": created_tracks
        }

    @staticmethod
    def _inside_sensor_coverage(
        track,
        sensor
    ):
        distance = np.linalg.norm(
            track.position - sensor.position
        )

        return distance <= sensor.max_range