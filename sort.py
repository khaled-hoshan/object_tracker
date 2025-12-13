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


if __name__ == "__main__":
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
    print(f"Partial overlap IOU: {iou_calculation(boxA, boxB)}")  # Should be ~0.14
