class ZoneDetector:

    def __init__(
        self,
        x1_ratio=0.25,
        y1_ratio=0.25,
        x2_ratio=0.75,
        y2_ratio=0.75
    ):

        self.x1_ratio = x1_ratio
        self.y1_ratio = y1_ratio
        self.x2_ratio = x2_ratio
        self.y2_ratio = y2_ratio

    def get_zone(self, width, height):

        x1 = int(width * self.x1_ratio)

        y1 = int(height * self.y1_ratio)

        x2 = int(width * self.x2_ratio)

        y2 = int(height * self.y2_ratio)

        return x1, y1, x2, y2

    def check_inside(self, center, zone):

        x, y = center

        x1, y1, x2, y2 = zone

        return (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        )