from scipy.optimize import linear_sum_assignment
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


class Tracker:  # holds the prediction phase(kalman filter, predict, update)
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
        self.id = Tracker.count  # give each object a unique ID.
        Tracker.count += 1  # give next object the next ID.
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

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        return self.get_state()

    def get_state(self):
        """
        this function does the following:
        # Translate the Kalman filter state-space representation(convert [u,v,s,r] back to [x1,y1,x2,y2]).
        # into a drawable and comparable bounding box in image space.

        """
        u, v, s, r = self.kf.x[:4].flatten()

        # if the state is bad, force the area to a tiny positive number.
        if s <= 0:
            s = 1e-4

            # update the kalman state with the constrained value to maintain stabelity.
            self.kf.x[2, 0] = s

        # Reverse the conversion
        w = np.sqrt(s * r)  # width from area and ratio
        h = s / w  # height from area and width

        # Calculate corners from center
        x1 = u - w / 2
        y1 = v - h / 2
        x2 = u + w / 2
        y2 = v + h / 2

        return np.array([x1, y1, x2, y2])

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


class TrackerManager:
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        # tracks manager initialization
        self.tracks = []  # this holds an empty list of tracks.
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections):
        """
        tracker manager loop:
        1- predict all existing tracks.
        2- compute cost matrix.
        3- data association(Hungarian Algorithm).
        4- update matched tracks.
        5- handle unmatched tracks.
        6- craete new tracks for unmatched detections.
        7- delete bad tracks.
        8- output confirmed tracks.
        """

        # 1- predict all existing tracks:
        predictions = []
        for t in self.tracks:
            t.predict()  # Advance the Kalman filter one time step and get the predicted bounding box
            predictions.append(
                t.get_state()
            )  # Save the predicted bounding box for later use(IOU computation, cost matrix, data association)

        # 2- compare cost matrix:
        # first calulate the iou matrix:
        iou_matrix = np.zeros((len(predictions), len(detections)))
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = iou_calculation(pred, det)
        # calculate the cost matrix(just flipping the iou matrix, because iou maxamize numbers, and hungarian minimize numbers).
        cost_matrix = 1 - iou_matrix

        # 3- data association(Hungarian Algorithm).
        row_idx, col_idx = linear_sum_assignment(
            cost_matrix
        )  # calling the hungarian algorithm on cost_matrix(this matches everything, even bad matches, so we need to filter them).
        """
         Now must answer three questions:
            which matches are good enough? (IOU threshold)
            which tracks got no detection?
            which detections are new objects?
        """
        # calculating IOU threshold to filter bad matches:
        iou_threshold = (
            self.iou_threshold
        )  # basically because IOU below ~0.3 is usually bullshit, reject weak overlaps; allow noisy but plausible matches

        matches = []
        unmatched_tracks = set(
            range(len(self.tracks))
        )  # create a set containing the indices of all tracks.
        unmatched_detections = set(
            range(len(detections))
        )  # create a set containing the indices of all detections.

        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= self.iou_threshold:
                matches.append((r, c))
                unmatched_tracks.remove(r)  # track index
                unmatched_detections.remove(c)  # detection index

        # 4- update matched tracks:
        """
        Loop over confirmed pairs from Hungarian + IOU filtering
            For each pair:
            Take one track
            Feed it one detection
            Call update() → Kalman filter correction + bookkeeping
            (this is preventing deletion in step 7).
        """
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # 5- handle unmatched tracks:
        """
        Meaning:
        These are tracks that did not get any detection matched to them this frame.

        What we do to unmatched tracks?
        we do almost nothing. On purpose.
        we do NOT update them with a detection
        we let the Kalman prediction stand
        we increment time_since_update (already happened in predict())

        Why this step exists at all?
        because objects:
        get occluded
        miss detections
        blink out for a frame or two
        """

        # 6- create new tracks for unmatched detections:
        # any detection that wasn't matched to an existing track must become a new track.
        for det_idx in unmatched_detections:
            new_track = Tracker(detections[det_idx])
            self.tracks.append(new_track)

        # 7- deletion of tracks logic:
        # too many misses(time_since_update > max_age).
        # not enough confirmations(age <= min_hits).
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # 8 - output confirmed tracks:
        """
        Only output tracks that are:
        alive (not deleted).
        stable (seen enough times).
        recently updated (not ghosts from 10 frames ago).

        In SORT-style logic, a track is confirmed if:
        hits >= min_hits OR still in the first few frames
        time_since_update == 0 (it was matched this frame)
        
        """
        # 8 - output confirmed tracks:

        outputs = []

        for t in self.tracks:
            # ONLY output tracks that have been seen enough times to be confirmed.
            # The box drawn is t.get_state(), which is the Kalman Filter's prediction.
            if t.hits >= self.min_hits:
                bbox = t.get_state()
                outputs.append(np.concatenate([bbox, [t.id]]))

        return np.array(outputs)  # [x1, y1, x2, y2, track_id]


# ------TEST----------


"""if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    print("=" * 70)
    print("TEST 1: IOU Function")
    print("=" * 70)
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 15, 15]
    boxC = [20, 20, 30, 30]
    print(f"IOU(overlap): {iou_calculation(boxA, boxB):.3f}")  # ~0.143
    print(f"IOU(no overlap): {iou_calculation(boxA, boxC):.3f}")  # 0.0

    print("\n" + "=" * 70)
    print("TEST 2: Single Tracker Lifecycle")
    print("=" * 70)
    trk = Tracker([10, 20, 20, 30])
    print(f"Initial - ID:{trk.id}, bbox:{trk.get_state()}")

    pred1 = trk.predict()
    print(f"Frame 1 (predict) - bbox:{pred1}, time_since:{trk.time_since_update}")

    trk.update([12, 22, 22, 32])
    print(f"Frame 2 (update) - bbox:{trk.get_state()}, hits:{trk.hits}")
    pred2 = trk.predict()
    print(f"Frame 3 (predict) - bbox:{pred2} (should predict forward motion)")

    print("\n" + "=" * 70)
    print("TEST 3: Multi-Object Tracking")
    print("=" * 70)

    tracker_mgr = TrackerManager(max_age=3, min_hits=1, iou_threshold=0.3)

    # Frame 1: Two objects appear
    dets1 = [[10, 10, 50, 50], [100, 100, 150, 150]]
    result1 = tracker_mgr.update(dets1)
    print(f"Frame 1: {len(result1)} confirmed tracks")
    print(result1)

    # Frame 2: Objects move
    dets2 = [[12, 12, 52, 52], [102, 102, 152, 152]]
    result2 = tracker_mgr.update(dets2)
    print(f"\nFrame 2: {len(result2)} confirmed tracks (IDs should be consistent)")
    print(result2)

    # Frame 3: One object disappears
    dets3 = [[14, 14, 54, 54]]
    result3 = tracker_mgr.update(dets3)
    print(f"\nFrame 3: {len(result3)} confirmed tracks (one missing)")
    print(result3)

    # Frame 4-6: First object still there, second still missing
    for i in range(4, 7):
        dets = [[14 + 2 * i, 14 + 2 * i, 54 + 2 * i, 54 + 2 * i]]
        result = tracker_mgr.update(dets)
        print(
            f"\nFrame {i}: {len(result)} tracks, total alive: {len(tracker_mgr.tracks)}"
        )

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE ✓")
    print("=" * 70)"""
