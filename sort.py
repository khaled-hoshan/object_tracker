from filterpy.kalman import KalmanFilter
import numpy as np

# ---------------IOU FUNCTION-------------------------------------------------------------------------------------------------------


def iou_calculation(boxA, boxB):  # claculate bboxes overlapping.

    # calculate top_left, and bottom_right:
    top_left = (max(boxA[0], boxB[0]), max(boxA[1], boxB[1]))

    bottom_right = (min(boxA[2], boxB[2]), min(boxA[3], boxB[3]))

    # calculate intersection area:
    intersection_width = max(
        0, bottom_right[0] - top_left[0]
    )  # taking the max as 0 if the substraction is negative
    intersection_hight = max(
        0, bottom_right[1] - top_left[1]
    )  # taking the max as 0 if the substraction is negative

    intersection_area = intersection_width * intersection_hight

    # calculate union:
    # union = area(box1) + area(box2) - intersection area
    # area = w x h, w = x_bottom - x_top, h = y_bottom - y_top
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - intersection_area

    # calculate IOU:
    iou = intersection_area / union_area if union_area > 0 else 0

    return iou


# --------------TRACKER CLASS--------------------------------------------------------------------------------------------------------


class tracker:  # holds the prediction phase(kalman filter, predict, update)
    count = 0  # global ID counter

    # kalman filter contains 2 main steps: 1. predection, 2. updating.
    def __init__(self, bbox):
        # initialize tracker from first detection.
        # bbox: [x1, y1, x2, y2]

        # step1: create a kalman filter
        # 7 state variables: [u, v, s, r, u', v', s']
        # 4 measurements: [u, v, s, r], no velocities are measured.
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # step2: convert bbox to [u, v, s, r]
        # convert [x1, y1, x2, y2]:
        x1, y1, x2, y2 = bbox
        w = x2 - x1  # width
        h = y2 - y1  # height

        u = x1 + w / 2.0  # center x
        v = y1 + h / 2.0  # center y
        s = w * h  # area
        r = w / float(h)  # aspect ratio

        # initialize state while velocities starts at 0:
        self.kf.x = np.array([[u], [v], [s], [r], [0], [0], [0]])
        # step3: setup the matrices
        # F = state transition: next state of the system.
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],  # u = u + u_velocity
                [0, 1, 0, 0, 0, 1, 0],  # v = v + v_velocity
                [0, 0, 1, 0, 0, 0, 1],  # s = s + s_velocity
                [0, 0, 0, 1, 0, 0, 0],  # r = r
                [0, 0, 0, 0, 1, 0, 0],  # u_velocity
                [0, 0, 0, 0, 0, 1, 0],  # v_velocity
                [0, 0, 0, 0, 0, 0, 1],  # s_velocity
            ],
            dtype=float,
        )

        # H = measurement: what can we measure.
        self.kf.H = np.array(  # type: ignore
            [
                [1, 0, 0, 0, 0, 0, 0],  # measure u
                [0, 1, 0, 0, 0, 0, 0],  # measure v
                [0, 0, 1, 0, 0, 0, 0],  # measure s
                [0, 0, 0, 1, 0, 0, 0],  # measure r
            ],
            dtype=float,
        )

        # Measurement noise (how much we trust detections): copied from SORT paper.
        self.kf.R[2:, 2:] *= 10.0  # s and r are noisier

        # Initial uncertainty (we don't know velocities!)
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty in velocities
        self.kf.P *= 10.0

        # Process noise (how much randomness in motion)
        self.kf.Q[-1, -1] *= 0.01  # Scale doesn't change much
        self.kf.Q[4:, 4:] *= 0.01  # Velocities are stable

        # step4: tracking metadata
        self.id = tracker.count  # give each object a unique ID.
        tracker.count += 1  # give next object the next ID.
        self.time_since_update = (
            0  # number of frames since last successful update(last seen of the object).
        )
        self.hits = 0  # how many times the object got detected.
        self.age = 0  # number of frames since the object was first detected.
        self.hit_streak = 0  # number of continues frames the object got detected in.

    def predict(self):
        self.kf.predict()  # calling this does all the math.
        self.age += 1
        # what happenes if the object isn't detected in the frame?
        # the hits doesn't increment.
        # time since update increment by one.

        self.time_since_update += 1
        self.hit_streak = 0

    def update(self, bbox):
        # an object is detected, what happenes?
        # the box is in forman [x1, y1, x2, y2] so we need to convert it to [u, v, s, r].
        # time_since_update is set to 0.
        # hits increment by one.
        # we need to call update to update the predections.

        # convert bbox to [u, v, s, r]:
        x1, y1, x2, y2 = bbox
        w = x2 - x1  # width
        h = y2 - y1  # height

        u = x1 + w / 2.0  # center x
        v = y1 + h / 2.0  # center y
        s = w * h  # area
        r = w / float(h)  # aspect ratio

        # Update Kalman filter with measurement
        z = np.array([[u], [v], [s], [r]])
        self.kf.update(z)

        # Update tracking stats
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0


# --------------TRACKER MANAGER CLASS--------------------------------------------------------------------------------------------------------


















































if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    print("=" * 60)
    print("TEST 1: IOU sanity check")
    print("=" * 60)
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 15, 15]
    boxC = [20, 20, 30, 30]

    print("IOU overlap:", iou_calculation(boxA, boxB))  # ~0.14
    print("IOU no overlap:", iou_calculation(boxA, boxC))  # 0.0

    print("\n" + "=" * 60)
    print("TEST 2: Tracker lifecycle")
    print("=" * 60)

    # Initial detection
    initial_bbox = [10, 20, 20, 30]
    trk = tracker(initial_bbox)

    print(f"Tracker ID: {trk.id}")
    print("Initial state:", trk.kf.x.flatten())
    print(
        "age:",
        trk.age,
        "hits:",
        trk.hits,
        "hit_streak:",
        trk.hit_streak,
        "time_since_update:",
        trk.time_since_update,
    )

    print("\n--- Frame 1: predict only (missed detection) ---")
    trk.predict()
    print("State:", trk.kf.x.flatten())
    print(
        "age:",
        trk.age,
        "hits:",
        trk.hits,
        "hit_streak:",
        trk.hit_streak,
        "time_since_update:",
        trk.time_since_update,
    )

    print("\n--- Frame 2: predict + update (detected) ---")
    detection = [12, 22, 22, 32]  # object moved slightly
    trk.predict()
    trk.update(detection)
    print("State:", trk.kf.x.flatten())
    print(
        "age:",
        trk.age,
        "hits:",
        trk.hits,
        "hit_streak:",
        trk.hit_streak,
        "time_since_update:",
        trk.time_since_update,
    )

    print("\n--- Frame 3: predict only (missed again) ---")
    trk.predict()
    print("State:", trk.kf.x.flatten())
    print(
        "age:",
        trk.age,
        "hits:",
        trk.hits,
        "hit_streak:",
        trk.hit_streak,
        "time_since_update:",
        trk.time_since_update,
    )

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
