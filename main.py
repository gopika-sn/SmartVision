import cv2
import os
from collections import Counter
from datetime import datetime

from detection.detector import ObjectDetector
from tracking.tracker import ObjectTracker

from analytics.counting import LineCounter
from analytics.zones import ZoneDetector
from analytics.events import EventDetector

from utils.fps import FPSCounter


# ============================================================
# SMARTVISION
# Real-Time AI Video Analytics
# ============================================================


# ============================================================
# CONFIGURATION
# ============================================================

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 820

BACKGROUND_PATH = os.path.join(
    "assets",
    "ui_background.png"
)

RECORDING_FOLDER = "recordings"

os.makedirs(
    RECORDING_FOLDER,
    exist_ok=True
)


# ============================================================
# COLOR PALETTE
# Green / Black Theme
# ============================================================

BG_DARK = (5, 12, 9)

GREEN = (70, 255, 150)

GREEN_BRIGHT = (100, 255, 170)

GREEN_DARK = (25, 100, 60)

WHITE = (235, 245, 240)

GRAY = (150, 165, 158)

DARK_PANEL = (8, 20, 14)

PANEL_BORDER = (30, 90, 55)

RED = (60, 70, 230)

YELLOW = (60, 220, 255)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def draw_panel(
    image,
    x1,
    y1,
    x2,
    y2,
    fill=DARK_PANEL,
    border=PANEL_BORDER,
    alpha=0.88
):
    """
    Draw a semi-transparent dashboard panel.
    """

    overlay = image.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        fill,
        -1
    )

    cv2.addWeighted(
        overlay,
        alpha,
        image,
        1 - alpha,
        0,
        image
    )

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        border,
        1
    )


def draw_text(
    image,
    text,
    position,
    size=0.6,
    color=WHITE,
    thickness=1
):
    cv2.putText(
        image,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def fit_image(image, target_width, target_height):

    if image is None:
        return None

    return cv2.resize(
        image,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )


def create_background():

    if os.path.exists(BACKGROUND_PATH):

        background = cv2.imread(
            BACKGROUND_PATH
        )

        if background is not None:

            background = cv2.resize(
                background,
                (
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT
                ),
                interpolation=cv2.INTER_AREA
            )

            # Darken the background so UI remains readable
            dark_overlay = background.copy()

            dark_overlay[:] = (0, 10, 5)

            background = cv2.addWeighted(
                background,
                0.28,
                dark_overlay,
                0.72,
                0
            )

            return background

    # Fallback if background image isn't found
    return np.zeros(
        (
            WINDOW_HEIGHT,
            WINDOW_WIDTH,
            3
        ),
        dtype="uint8"
    )


# ============================================================
# NUMPY
# ============================================================

import numpy as np


# ============================================================
# INITIALIZE AI COMPONENTS
# ============================================================

print()
print("==============================================")
print("             SMARTVISION")
print("       REAL-TIME AI ANALYTICS")
print("==============================================")
print()

print("[SYSTEM] Loading YOLO model...")

detector = ObjectDetector()

print("[SYSTEM] YOLO model loaded.")

tracker = ObjectTracker(
    detector.model.names
)

line_counter = LineCounter()

zone_detector = ZoneDetector()

event_detector = EventDetector()

fps_counter = FPSCounter()

print("[SYSTEM] All modules initialized.")


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("[ERROR] Could not open webcam.")

    exit()


# ============================================================
# VIDEO WRITER
# ============================================================

video_writer = None

recording = False


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # READ FRAME
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:

        print(
            "[ERROR] Could not read webcam frame."
        )

        break


    # --------------------------------------------------------
    # FRAME INFORMATION
    # --------------------------------------------------------

    frame_height, frame_width = frame.shape[:2]

    line_y = frame_height // 2


    # --------------------------------------------------------
    # ZONE
    # --------------------------------------------------------

    zone = zone_detector.get_zone(
        frame_width,
        frame_height
    )

    zone_x1, zone_y1, zone_x2, zone_y2 = zone


    # --------------------------------------------------------
    # YOLO
    # --------------------------------------------------------

    result = detector.detect_and_track(
        frame
    )


    # --------------------------------------------------------
    # TRACKING
    # --------------------------------------------------------

    tracks = tracker.update(
        result
    )


    # --------------------------------------------------------
    # OBJECT COUNTS
    # --------------------------------------------------------

    object_counts = Counter()

    zone_counts = Counter()

    for track in tracks:

        object_name = track["name"]

        center = track["center"]

        object_counts[
            object_name
        ] += 1


        # Check zone
        if zone_detector.check_inside(
            center,
            zone
        ):

            zone_counts[
                object_name
            ] += 1


    # --------------------------------------------------------
    # LINE COUNTING
    # --------------------------------------------------------

    line_counter.update(
        tracks,
        line_y
    )

    entered, exited = (
        line_counter.get_counts()
    )


    # --------------------------------------------------------
    # EVENT DETECTION
    # --------------------------------------------------------

    event_detector.check_zone_entry(

        tracks,

        zone_detector,

        zone,

        frame

    )

    latest_event = (
        event_detector.get_latest_event()
    )

    event_log = (
        event_detector.get_event_log()
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    fps = fps_counter.update()


    # ========================================================
    # CAMERA OUTPUT
    # ========================================================

    camera_output = result.plot()


    # --------------------------------------------------------
    # MOVEMENT TRAILS
    # --------------------------------------------------------

    for track in tracks:

        points = track["history"]

        for i in range(
            1,
            len(points)
        ):

            cv2.line(

                camera_output,

                points[i - 1],

                points[i],

                GREEN,

                2,

                cv2.LINE_AA

            )


    # --------------------------------------------------------
    # TRACK CENTER POINTS
    # --------------------------------------------------------

    for track in tracks:

        center = track["center"]

        cv2.circle(

            camera_output,

            center,

            4,

            GREEN_BRIGHT,

            -1,

            cv2.LINE_AA

        )


    # --------------------------------------------------------
    # COUNTING LINE
    # --------------------------------------------------------

    cv2.line(

        camera_output,

        (0, line_y),

        (
            frame_width,
            line_y
        ),

        GREEN,

        2,

        cv2.LINE_AA

    )

    draw_text(

        camera_output,

        "COUNTING LINE",

        (
            15,
            line_y - 10
        ),

        0.5,

        GREEN_BRIGHT,

        1

    )


    # --------------------------------------------------------
    # RESTRICTED ZONE
    # --------------------------------------------------------

    cv2.rectangle(

        camera_output,

        (
            zone_x1,
            zone_y1
        ),

        (
            zone_x2,
            zone_y2
        ),

        GREEN_BRIGHT,

        2,

        cv2.LINE_AA

    )

    draw_text(

        camera_output,

        "ZONE A  |  RESTRICTED",

        (
            zone_x1,
            zone_y1 - 10
        ),

        0.55,

        GREEN_BRIGHT,

        1

    )


    # ========================================================
    # CREATE DASHBOARD BACKGROUND
    # ========================================================

    if os.path.exists(
        BACKGROUND_PATH
    ):

        background = cv2.imread(
            BACKGROUND_PATH
        )

        background = cv2.resize(
            background,
            (
                WINDOW_WIDTH,
                WINDOW_HEIGHT
            ),
            interpolation=cv2.INTER_AREA
        )

        # Darken theme image
        dark_layer = np.zeros_like(
            background
        )

        dark_layer[:] = BG_DARK

        background = cv2.addWeighted(
            background,
            0.24,
            dark_layer,
            0.76,
            0
        )

    else:

        background = np.zeros(
            (
                WINDOW_HEIGHT,
                WINDOW_WIDTH,
                3
            ),
            dtype=np.uint8
        )

        background[:] = BG_DARK


    dashboard = background.copy()


    # ========================================================
    # HEADER
    # ========================================================

    draw_panel(
        dashboard,
        25,
        20,
        WINDOW_WIDTH - 25,
        100,
        alpha=0.82
    )


    # Small green indicator
    cv2.circle(

        dashboard,

        (55, 55),

        7,

        GREEN,

        -1,

        cv2.LINE_AA

    )


    draw_text(

        dashboard,

        "SMARTVISION",

        (75, 62),

        1.15,

        WHITE,

        2

    )


    draw_text(

        dashboard,

        "REAL-TIME AI VIDEO ANALYTICS",

        (77, 88),

        0.48,

        GREEN,

        1

    )


    # System status
    cv2.circle(

        dashboard,

        (
            WINDOW_WIDTH - 165,
            52
        ),

        6,

        GREEN,

        -1,

        cv2.LINE_AA

    )


    draw_text(

        dashboard,

        "SYSTEM ONLINE",

        (
            WINDOW_WIDTH - 150,
            57
        ),

        0.55,

        GREEN_BRIGHT,

        1

    )


    draw_text(

        dashboard,

        "YOLO + BYTE TRACK",

        (
            WINDOW_WIDTH - 150,
            82
        ),

        0.42,

        GRAY,

        1

    )


    # ========================================================
    # MAIN LAYOUT
    # ========================================================

    camera_x1 = 25
    camera_y1 = 135

    camera_x2 = 985
    camera_y2 = 675

    sidebar_x1 = 1000
    sidebar_y1 = 135

    sidebar_x2 = WINDOW_WIDTH - 25
    sidebar_y2 = 675


    # ========================================================
    # LIVE CAMERA PANEL
    # ========================================================

    draw_panel(

        dashboard,

        camera_x1,
        camera_y1,

        camera_x2,
        camera_y2,

        alpha=0.92

    )


    draw_text(

        dashboard,

        "LIVE CAMERA FEED",

        (
            camera_x1 + 20,
            camera_y1 + 30
        ),

        0.62,

        WHITE,

        1

    )


    draw_text(

        dashboard,

        "REAL-TIME",

        (
            camera_x2 - 105,
            camera_y1 + 30
        ),

        0.45,

        GREEN,

        1

    )


    # Camera area
    feed_x1 = camera_x1 + 15
    feed_y1 = camera_y1 + 45

    feed_width = 930
    feed_height = 500


    camera_resized = fit_image(
        camera_output,
        feed_width,
        feed_height
    )


    dashboard[
        feed_y1:feed_y1 + feed_height,
        feed_x1:feed_x1 + feed_width
    ] = camera_resized


    # ========================================================
    # SIDEBAR - LIVE ANALYTICS
    # ========================================================

    draw_panel(

        dashboard,

        sidebar_x1,
        sidebar_y1,

        sidebar_x2,
        sidebar_y2,

        alpha=0.94

    )


    draw_text(

        dashboard,

        "LIVE ANALYTICS",

        (
            sidebar_x1 + 20,
            sidebar_y1 + 35
        ),

        0.72,

        WHITE,

        1

    )


    draw_text(

        dashboard,

        "CURRENT FRAME",

        (
            sidebar_x1 + 20,
            sidebar_y1 + 58
        ),

        0.4,

        GRAY,

        1

    )


    # --------------------------------------------------------
    # OBJECT STATISTICS
    # --------------------------------------------------------

    stats_y = sidebar_y1 + 95


    # Show important object categories first
    priority_objects = [
        "person",
        "car",
        "motorcycle",
        "bicycle",
        "bus",
        "truck"
    ]


    shown = []

    for name in priority_objects:

        if name in object_counts:

            shown.append(name)


    # Add other detected objects
    for name in object_counts:

        if name not in shown:

            shown.append(name)


    shown = shown[:5]


    for index, name in enumerate(
        shown
    ):

        y = stats_y + index * 38

        count = object_counts[name]

        # Object name
        draw_text(

            dashboard,

            name.upper(),

            (
                sidebar_x1 + 20,
                y
            ),

            0.5,

            GRAY,

            1

        )

        # Count
        draw_text(

            dashboard,

            f"{count:02d}",

            (
                sidebar_x2 - 55,
                y
            ),

            0.72,

            GREEN_BRIGHT,

            2

        )


    # Divider
    divider_y = sidebar_y1 + 300

    cv2.line(

        dashboard,

        (
            sidebar_x1 + 20,
            divider_y
        ),

        (
            sidebar_x2 - 20,
            divider_y
        ),

        PANEL_BORDER,

        1

    )


    # --------------------------------------------------------
    # ENTRY / EXIT
    # --------------------------------------------------------

    draw_text(

        dashboard,

        "MOVEMENT",

        (
            sidebar_x1 + 20,
            divider_y + 35
        ),

        0.52,

        WHITE,

        1

    )


    draw_text(

        dashboard,

        "ENTERED",

        (
            sidebar_x1 + 20,
            divider_y + 70
        ),

        0.42,

        GRAY,

        1

    )

    draw_text(

        dashboard,

        f"{entered:02d}",

        (
            sidebar_x1 + 110,
            divider_y + 70
        ),

        0.65,

        GREEN_BRIGHT,

        2

    )


    draw_text(

        dashboard,

        "EXITED",

        (
            sidebar_x1 + 180,
            divider_y + 70
        ),

        0.42,

        GRAY,

        1

    )

    draw_text(

        dashboard,

        f"{exited:02d}",

        (
            sidebar_x1 + 260,
            divider_y + 70
        ),

        0.65,

        GREEN_BRIGHT,

        2

    )


    # Zone occupancy
    zone_total = sum(
        zone_counts.values()
    )


    draw_text(

        dashboard,

        "ZONE A",

        (
            sidebar_x1 + 20,
            divider_y + 105
        ),

        0.42,

        GRAY,

        1

    )

    draw_text(

        dashboard,

        f"{zone_total:02d} OBJECTS",

        (
            sidebar_x1 + 110,
            divider_y + 105
        ),

        0.5,

        GREEN,

        1

    )


    # FPS
    draw_text(

        dashboard,

        "PERFORMANCE",

        (
            sidebar_x1 + 20,
            divider_y + 140
        ),

        0.42,

        GRAY,

        1

    )

    draw_text(

        dashboard,

        f"{fps:.1f} FPS",

        (
            sidebar_x1 + 125,
            divider_y + 140
        ),

        0.52,

        WHITE,

        1

    )


    # ========================================================
    # BOTTOM SYSTEM PANEL
    # ========================================================

    bottom_y1 = 695
    bottom_y2 = 795


    # Left system panel
    draw_panel(

        dashboard,

        25,
        bottom_y1,
        470,
        bottom_y2,

        alpha=0.9

    )


    draw_text(

        dashboard,

        "SYSTEM",

        (
            45,
            bottom_y1 + 30
        ),

        0.58,

        WHITE,

        1

    )


    draw_text(

        dashboard,

        "STATUS",

        (
            45,
            bottom_y1 + 62
        ),

        0.4,

        GRAY,

        1

    )

    draw_text(

        dashboard,

        "ONLINE",

        (
            105,
            bottom_y1 + 62
        ),

        0.45,

        GREEN,

        1

    )


    draw_text(

        dashboard,

        "TRACKS",

        (
            200,
            bottom_y1 + 62
        ),

        0.4,

        GRAY,

        1

    )

    draw_text(

        dashboard,

        f"{len(tracks):02d}",

        (
            260,
            bottom_y1 + 62
        ),

        0.48,

        WHITE,

        1

    )


    # Recording status
    draw_text(

        dashboard,

        "RECORDING",

        (
            320,
            bottom_y1 + 62
        ),

        0.4,

        GRAY,

        1

    )


    if recording:

        draw_text(

            dashboard,

            "ON",

            (
                405,
                bottom_y1 + 62
            ),

            0.45,

            RED,

            1

        )

    else:

        draw_text(

            dashboard,

            "OFF",

            (
                405,
                bottom_y1 + 62
            ),

            0.45,

            GRAY,

            1

        )


    # Controls
    draw_text(

        dashboard,

        "Q  QUIT",

        (
            45,
            bottom_y1 + 85
        ),

        0.38,

        GRAY,

        1

    )

    draw_text(

        dashboard,

        "R  RECORD",

        (
            120,
            bottom_y1 + 85
        ),

        0.38,

        GRAY,

        1

    )


    # ========================================================
    # EVENTS PANEL
    # ========================================================

    events_x1 = 490
    events_x2 = WINDOW_WIDTH - 25


    draw_panel(

        dashboard,

        events_x1,
        bottom_y1,
        events_x2,
        bottom_y2,

        alpha=0.9

    )


    draw_text(

        dashboard,

        "RECENT EVENTS",

        (
            events_x1 + 20,
            bottom_y1 + 30
        ),

        0.58,

        WHITE,

        1

    )


    recent_events = event_log[-2:]


    if not recent_events:

        draw_text(

            dashboard,

            "No security events detected",

            (
                events_x1 + 20,
                bottom_y1 + 63
            ),

            0.43,

            GRAY,

            1

        )

    else:

        event_y = bottom_y1 + 60


        for event in recent_events:

            event_text = (

                f"{event['time']}   "

                f"{event['event']}"

            )


            draw_text(

                dashboard,

                event_text[:65],

                (
                    events_x1 + 20,
                    event_y
                ),

                0.42,

                GREEN,

                1

            )

            event_y += 28


    # ========================================================
    # ALERT OVERLAY
    # ========================================================

    if latest_event != "SYSTEM NORMAL":

        alert_height = 55

        alert_y = (
            WINDOW_HEIGHT -
            alert_height -
            15
        )


        overlay = dashboard.copy()


        cv2.rectangle(

            overlay,

            (
                25,
                alert_y
            ),

            (
                WINDOW_WIDTH - 25,
                alert_y + alert_height
            ),

            (20, 35, 20),

            -1

        )


        cv2.addWeighted(

            overlay,

            0.94,

            dashboard,

            0.06,

            0,

            dashboard

        )


        cv2.rectangle(

            dashboard,

            (
                25,
                alert_y
            ),

            (
                WINDOW_WIDTH - 25,
                alert_y + alert_height
            ),

            GREEN,

            1

        )


        draw_text(

            dashboard,

            "SECURITY EVENT",

            (
                45,
                alert_y + 34
            ),

            0.48,

            GREEN_BRIGHT,

            1

        )


        draw_text(

            dashboard,

            latest_event,

            (
                205,
                alert_y + 34
            ),

            0.48,

            WHITE,

            1

        )


    # ========================================================
    # VIDEO RECORDING
    # ========================================================

    if recording:

        if video_writer is None:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            video_path = os.path.join(

                RECORDING_FOLDER,

                f"session_{timestamp}.mp4"

            )


            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )


            video_writer = cv2.VideoWriter(

                video_path,

                fourcc,

                20.0,

                (
                    WINDOW_WIDTH,
                    WINDOW_HEIGHT
                )

            )


            print(
                f"[RECORDING] Started: {video_path}"
            )


        video_writer.write(
            dashboard
        )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(

        "SMARTVISION | AI VIDEO ANALYTICS",

        dashboard

    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Q = Quit
    if key == ord("q"):

        break


    # R = Recording
    elif key == ord("r"):

        recording = not recording


        if recording:

            print(
                "[RECORDING] Starting..."
            )

        else:

            if video_writer is not None:

                video_writer.release()

                video_writer = None

            print(
                "[RECORDING] Stopped."
            )


# ============================================================
# CLEANUP
# ============================================================

cap.release()


if video_writer is not None:

    video_writer.release()


cv2.destroyAllWindows()


print()
print("==============================================")
print("         SMARTVISION SESSION ENDED")
print("==============================================")