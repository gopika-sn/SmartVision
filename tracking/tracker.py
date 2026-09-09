from collections import defaultdict


class ObjectTracker:

    def __init__(self, model_names):

        self.model_names = model_names

        self.track_history = defaultdict(list)

    def update(self, result):

        tracks = []

        if result.boxes.id is None:
            return tracks

        boxes = result.boxes.xywh.cpu()

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        class_ids = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        for box, track_id, class_id in zip(
            boxes,
            track_ids,
            class_ids
        ):

            x, y, w, h = box

            center = (
                int(x),
                int(y)
            )

            object_name = self.model_names[class_id]

            self.track_history[track_id].append(
                center
            )

            # Keep only the latest 30 positions
            if len(self.track_history[track_id]) > 30:

                self.track_history[track_id].pop(0)

            tracks.append({

                "id": track_id,

                "class_id": class_id,

                "name": object_name,

                "center": center,

                "history": self.track_history[track_id]

            })

        return tracks