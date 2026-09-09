class LineCounter:

    def __init__(self):

        self.entered = 0
        self.exited = 0

        self.last_side = {}

        self.class_entered = {}
        self.class_exited = {}

    def update(self, tracks, line_y):

        for track in tracks:

            track_id = track["id"]

            object_name = track["name"]

            points = track["history"]

            if len(points) < 2:
                continue

            current_y = points[-1][1]

            previous_y = points[-2][1]

            # Determine current side
            if current_y < line_y:

                current_side = "top"

            else:

                current_side = "bottom"

            # First time seeing this object
            if track_id not in self.last_side:

                self.last_side[track_id] = current_side

                continue

            previous_side = self.last_side[track_id]

            # ----------------------------------
            # TOP -> BOTTOM = ENTER
            # ----------------------------------

            if (
                previous_side == "top"
                and current_side == "bottom"
            ):

                self.entered += 1

                self.class_entered[object_name] = (
                    self.class_entered.get(
                        object_name,
                        0
                    ) + 1
                )

            # ----------------------------------
            # BOTTOM -> TOP = EXIT
            # ----------------------------------

            elif (
                previous_side == "bottom"
                and current_side == "top"
            ):

                self.exited += 1

                self.class_exited[object_name] = (
                    self.class_exited.get(
                        object_name,
                        0
                    ) + 1
                )

            # Update side
            self.last_side[track_id] = current_side

    def get_counts(self):

        return self.entered, self.exited

    def get_class_counts(self):

        return (
            self.class_entered,
            self.class_exited
        )