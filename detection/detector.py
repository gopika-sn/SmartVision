from ultralytics import YOLO


class ObjectDetector:

    def __init__(self, model_path="yolo26n.pt"):

        self.model = YOLO(model_path)

    def detect_and_track(self, frame):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        return results[0]