from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load YOLO once when server starts
model = YOLO("yolo26n.pt")

# Tracking information
last_sides = {}
entered = 0
exited = 0


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process_frame():

    global entered, exited

    try:
        # Get image from browser
        data = request.json["image"]

        # Remove base64 header
        image_data = data.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(image_data)

        array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({"error": "Invalid frame"}), 400

        height, width = frame.shape[:2]

        # ------------------------------------------------
        # YOLO + BYTE TRACK
        # ------------------------------------------------

        results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        imgsz=320,
        conf=0.25
        )

        result = results[0]

        # Draw YOLO results
        output = result.plot()

        # ------------------------------------------------
        # COUNTING LINE
        # ------------------------------------------------

        line_y = height // 2

        cv2.line(
            output,
            (0, line_y),
            (width, line_y),
            (80, 255, 150),
            2
        )

        cv2.putText(
            output,
            "COUNTING LINE",
            (15, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (80, 255, 150),
            1
        )

        # ------------------------------------------------
        # RESTRICTED ZONE
        # ------------------------------------------------

        zx1 = int(width * 0.25)
        zy1 = int(height * 0.25)

        zx2 = int(width * 0.75)
        zy2 = int(height * 0.75)

        cv2.rectangle(
            output,
            (zx1, zy1),
            (zx2, zy2),
            (80, 255, 150),
            2
        )

        cv2.putText(
            output,
            "ZONE A - RESTRICTED",
            (zx1, zy1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 150),
            1
        )

        # ------------------------------------------------
        # ANALYTICS
        # ------------------------------------------------

        object_counts = {}

        zone_count = 0

        alerts = []

        tracked = 0

        if result.boxes.id is not None:

            boxes = result.boxes.xywh.cpu()

            ids = (
                result.boxes.id
                .int()
                .cpu()
                .tolist()
            )

            classes = (
                result.boxes.cls
                .int()
                .cpu()
                .tolist()
            )

            tracked = len(ids)

            for box, track_id, class_id in zip(
                boxes,
                ids,
                classes
            ):

                x, y, w, h = box

                cx = int(x)
                cy = int(y)

                name = model.names[class_id]

                # Object count
                object_counts[name] = (
                    object_counts.get(name, 0) + 1
                )

                # ------------------------------------------------
                # ZONE
                # ------------------------------------------------

                inside_zone = (
                    zx1 <= cx <= zx2
                    and
                    zy1 <= cy <= zy2
                )

                if inside_zone:

                    zone_count += 1

                    if name == "person":

                        alerts.append(
                            f"Person #{track_id} in Zone A"
                        )

                # ------------------------------------------------
                # LINE COUNTING
                # ------------------------------------------------

                current_side = (
                    "top"
                    if cy < line_y
                    else "bottom"
                )

                if track_id in last_sides:

                    previous_side = last_sides[track_id]

                    if (
                        previous_side == "top"
                        and current_side == "bottom"
                    ):
                        entered += 1

                    elif (
                        previous_side == "bottom"
                        and current_side == "top"
                    ):
                        exited += 1

                last_sides[track_id] = current_side

        # ------------------------------------------------
        # JPEG ENCODE
        # ------------------------------------------------

        success, buffer = cv2.imencode(
            ".jpg",
            output,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                80
            ]
        )

        if not success:
            return jsonify({"error": "Encoding failed"}), 500

        encoded = base64.b64encode(
            buffer
        ).decode("utf-8")

        return jsonify({

            "image": encoded,

            "tracked": tracked,

            "objects": object_counts,

            "zone": zone_count,

            "entered": entered,

            "exited": exited,

            "alert": alerts[0] if alerts else ""

        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )