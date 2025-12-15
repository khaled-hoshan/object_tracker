import numpy as np
from sort import TrackerManager

"""tracker = TrackerManager(max_age=3, min_hits=1, iou_threshold=0.3)

frames = [
    np.array([[10, 10, 20, 20]]),
    np.array([[12, 12, 22, 22]]),
    np.array([[14, 14, 24, 24]]),
]

for i, dets in enumerate(frames):
    tracks = tracker.update(dets)
    print(f"Frame {i}: {tracks}")"""

"""detections = [
    np.array([[10, 10, 20, 20], [50, 50, 60, 60]]),  # Frame 0: two objects
    np.array([[12, 12, 22, 22], [48, 48, 58, 58]]),  # Frame 1: slightly moved
    np.array([[14, 14, 24, 24], [46, 46, 56, 56]]),  # Frame 2: continue moving
]

tm = TrackerManager(min_hits=1)

for i, dets in enumerate(detections):
    output = tm.update(dets)
    print(f"Frame {i}: {output}")"""

"""detections = [
    np.array([[10, 10, 20, 20], [50, 50, 60, 60]]),  # Frame 0
    np.array([[12, 12, 22, 22]]),  # Frame 1: only first object
    np.array([[14, 14, 24, 24], [46, 46, 56, 56]]),  # Frame 2: both appear again
]

tm = TrackerManager(min_hits=1, max_age=2)

for i, dets in enumerate(detections):
    output = tm.update(dets)
    print(f"Frame {i}: {output}")"""

frames = [
    # Frame 0
    [[10, 10, 20, 20], [50, 50, 60, 60]],
    # Frame 1
    [[12, 12, 22, 22], [48, 48, 58, 58]],
    # Frame 2 - second object occluded
    [[14, 14, 24, 24]],
    # Frame 3 - both appear again, second object moved faster
    [[16, 16, 26, 26], [45, 45, 55, 55]],
    # Frame 4 - objects crossing paths
    [[18, 18, 28, 28], [18, 18, 28, 28]],
]


tm = TrackerManager(iou_threshold=0.15 - 0.3)
for f_idx, dets in enumerate(frames):
    out = tm.update(dets)
    print(f"Frame {f_idx}: {out}")
