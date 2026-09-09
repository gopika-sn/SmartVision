import os
import time

import cv2

from datetime import datetime


class EventDetector:

    def __init__(self):

        self.alerted_ids = set()

        self.latest_event = "SYSTEM NORMAL"

        self.event_time = 0

        self.event_log = []

        self.event_folder = "events"

        os.makedirs(
            self.event_folder,
            exist_ok=True
        )

    def check_zone_entry(
        self,
        tracks,
        zone_detector,
        zone,
        frame
    ):

        for track in tracks:

            track_id = track["id"]

            object_name = track["name"]

            center = track["center"]

            inside_zone = zone_detector.check_inside(
                center,
                zone
            )

            # Restricted-zone alert
            if (
                inside_zone
                and object_name == "person"
                and track_id not in self.alerted_ids
            ):

                current_time = datetime.now()

                time_string = current_time.strftime(
                    "%H:%M:%S"
                )

                file_time = current_time.strftime(
                    "%Y%m%d_%H%M%S"
                )

                event_message = (
                    f"Person #{track_id} "
                    f"entered Zone A"
                )

                self.latest_event = (
                    f"ALERT: {event_message}"
                )

                self.event_time = time.time()

                # Screenshot filename
                filename = (
                    f"{file_time}_"
                    f"person_{track_id}.jpg"
                )

                filepath = os.path.join(
                    self.event_folder,
                    filename
                )

                # Save evidence
                cv2.imwrite(
                    filepath,
                    frame
                )

                # Save event information
                self.event_log.append({

                    "time": time_string,

                    "event": event_message,

                    "track_id": track_id,

                    "object": object_name,

                    "screenshot": filepath

                })

                self.alerted_ids.add(track_id)

    def get_latest_event(self):

        if time.time() - self.event_time < 5:

            return self.latest_event

        return "SYSTEM NORMAL"

    def get_event_log(self):

        return self.event_log