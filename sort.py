from filterpy.kalman import KalmanFilter
import numpy as np


class kalmanboxtracker:
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
        # F = state transition
        # H = measurement

        # step4: tracking metadata
        self.id = kalmanboxtracker.count  # give each tracker a unique ID.
        kalmanboxtracker.count += 1  # give next tracker the next ID.
        self.time_since_update = 0  # number of frames since object detected.
        self.hits = 0  # how many times tracker matched.
        self.age = 0  # number of frames the tracker detected in.

    def predect(self):
        pass

    def update(self):
        pass


if __name__ == "__main__":
    bbox = [10, 20, 20, 30]
    tracker = kalmanboxtracker(bbox)
    print("initial_state:", tracker.kf.x.flatten())


def iou_calculation(boxA, boxB):

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


"""if __name__ == "__main__":
    # overlap perfect:
    boxA = [0, 0, 10, 10]
    boxB = [0, 0, 10, 10]
    print(f"same box: {iou_calculation(boxA,boxB)}")

    # Test 2: No overlap
    boxA = [0, 0, 10, 10]
    boxB = [20, 20, 30, 30]
    print(f"No overlap IOU: {iou_calculation(boxA, boxB)}")  # Should be 0.0

    # Test 3: Partial overlap
    boxA = [0, 0, 10, 10]
    boxB = [5, 5, 15, 15]
    print(f"Partial overlap IOU: {iou_calculation(boxA, boxB)}")  # Should be ~0.14"""


"""
F = np.array(
    [
        [1, 0, 0, 0, 1, 0, 0],  # u = u + u_velocity
        [0, 1, 0, 0, 0, 1, 0],  # v = v + v_velocity
        [0, 0, 1, 0, 0, 0, 1],  # s = s + s_velocity
        [0, 0, 0, 1, 0, 0, 0],  # r = r
        [0, 0, 0, 0, 1, 0, 0],  # u_velocity
        [0, 0, 0, 0, 0, 1, 0],  # v_velocity
        [0, 0, 0, 0, 0, 0, 1],  # s_velocity
    ]
) 
"""
