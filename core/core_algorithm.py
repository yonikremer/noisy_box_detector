import cv2
import numpy as np
import yaml
from pydantic import ValidationError
from logging import debug

from .rectangle import Rectangle

MAX_UINT8 = 255


with open("algorithm_configs.yaml") as f:
    algorithm_configs = yaml.load(f, Loader=yaml.FullLoader)
    BLURRING_KERNEL_SIZE = algorithm_configs["blurring_kernel_size"]
    STRUCTURING_KERNEL_SIZE = algorithm_configs["structuring_kernel_size"]
    MAX_RECTANGLE_AREA_INCREMENT_RATIO = algorithm_configs[
        "max_rectangle_area_increment_ratio"
    ]
    MAX_DISTANCE_BETWEEN_RECTANGLES = algorithm_configs[
        "max_distance_between_rectangles"
    ]


def merge_overlapping_rectangles(rect_list: list[Rectangle]) -> list[Rectangle]:
    """
    Given a list of rectangles, merges close rectangles
    Merge conditions:
    1. The distance between the two rectangles is less than or equal to MAX_DISTANCE_BETWEEN_RECTANGLES
    2. The area of the merged rectangle, divided by the sum of the areas of the two rectangles,
       is less than or equal to MAX_RECTANGLE_AREA_INCREMENT_RATIO
    3. The merged rectangle is a valid rectangle

    Merging algorithm:
        Initialize a flag merged_occurred to True.
        This flag will be used to track whether any merging occurred in the current iteration.
        Initialize an empty list new_rectangles to store the merged rectangles.
        Initialize a list used to keep track of which rectangles have been used in the merging process.
        This list is initialized with False values for each rectangle in the input list.
        Iterate over each rectangle in the input list:
            If the rectangle has already been used (i.e., used[i] is True), skip to the next rectangle.
            Otherwise, set used[i] to True.
            Iterate over each rectangle in the input list that comes after the current rectangle (i.e., j ranges from i+1 to the end of the list):
                If the current rectangle overlaps with the other rectangle (i.e., current.distance(other) <= MAX_DISTANCE_BETWEEN_RECTANGLES),
                merge them:
                    Create a new rectangle that is the union of the two rectangles (i.e., merged = current.merge(other)).
                    Check if the merged rectangle is valid (i.e., its area is less than or equal to MAX_RECTANGLE_AREA_INCREMENT_RATIO times the sum of the areas of the two original rectangles). If it's not valid, skip to the next rectangle.
                    Set merged_occurred to True.
                    Update the current rectangle to be the merged rectangle.
                    Set used[j] to True to mark the other rectangle as used.
                    Add the updated current rectangle to the new_rectangles list.
                Repeat steps 4-5 until no more merging occurs (i.e., merged_occurred is False).
                Return the new_rectangles list.

    :param rect_list: A list of rectangles
    :return: a list of merged rectangles
    """
    merged_occurred = True
    while merged_occurred:
        merged_occurred = False
        new_rectangles = []
        used = [False] * len(rect_list)
        for i in range(len(rect_list)):
            if used[i]:
                continue
            current = rect_list[i]
            for j in range(i + 1, len(rect_list)):
                if used[j]:
                    continue
                other = rect_list[j]
                # If they overlap, merge them
                if current.distance(other) <= MAX_DISTANCE_BETWEEN_RECTANGLES:
                    try:
                        merged = current.merge(other)
                    except ValidationError as e:
                        debug(f"Could not merge rectangles: {current} and {other}")
                        debug(f"Error: {e}")
                        continue
                    area_before_merge = current.area() + other.area()
                    area_increment_ratio = merged.area() / area_before_merge
                    if area_increment_ratio <= MAX_RECTANGLE_AREA_INCREMENT_RATIO:
                        current = merged
                        used[j] = True
                        merged_occurred = True
            new_rectangles.append(current)
        rect_list = new_rectangles
    return rect_list


def contours_to_rectangles(contours, hierarchy):
    filtered_contours = [
        cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] == -1
    ]
    rect_list = []
    for cnt in filtered_contours:
        x, y, width, height = cv2.boundingRect(cnt)
        try:
            found_rectangle = Rectangle(x=x, y=y, height=height, length=width)
        except ValueError:
            continue
        rect_list.append(found_rectangle)
    rect_list = merge_overlapping_rectangles(rect_list)
    return rect_list


def preprocess(image):
    """
    Given a black and white image with noise low-valued background and high-valued rectangles,
    Preprocess the image to remove as much noise as possible

    Algorithm:
    1. Blur the image
    2. Find the noise threshold
    3. Zero out the pixel less than the threshold
    4. remove small noise points using Morphological opening
    5. fill small holes in rectangles using Morphological closing

    :param image: a black and white image with noise low-valued background and high-valued rectangles
    :return: a less noisy version of the image
    """
    blurred = cv2.GaussianBlur(image, (BLURRING_KERNEL_SIZE, BLURRING_KERNEL_SIZE), 0)
    # Convert to uint8 before thresholding
    max_value, min_value = blurred.max(), blurred.min()
    
    # If the image is uniform (all black or all white), return it as is
    if max_value == min_value:
        return image
        
    blurred_normalised = np.uint8(
        MAX_UINT8 * ((blurred - min_value) / (max_value - min_value))
    )
    thresh, binary_image = cv2.threshold(
        blurred_normalised, 0, MAX_UINT8, cv2.THRESH_OTSU
    )
    STRUCTURING_ELEMENT_KERNEL = cv2.getStructuringElement(
        cv2.MORPH_RECT, (STRUCTURING_KERNEL_SIZE, STRUCTURING_KERNEL_SIZE)
    )
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, STRUCTURING_ELEMENT_KERNEL)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, STRUCTURING_ELEMENT_KERNEL)
    return closed
