import random
from logging import debug

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pydantic import ValidationError

from rectangle import Rectangle

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


def generate_rectangles(num_rectangles):
    placed_rectangles: list[Rectangle] = []
    for _ in range(num_rectangles):
        while True:
            curr_rectangle = Rectangle.random()
            # Check for overlap
            overlap = False
            for placed_rectangle in placed_rectangles:
                if curr_rectangle.overlap(placed_rectangle):
                    overlap = True
                    break

            if not overlap:
                placed_rectangles.append(curr_rectangle)
                break  # Move to the next rectangle

    return placed_rectangles


def generate_data():
    # The data are greyscale images with a noisy background and black rectangles
    # of different shapes, all parallel to the image's axes.
    with open("signal_configs.yaml") as signal_configs_file:
        configs = yaml.load(signal_configs_file, Loader=yaml.FullLoader)
        NOISE_STD = configs["noise_std"]
        SIGNAL_TO_NOISE_RATIO = configs["signal_to_noise_ratio"]
        BACKGROUND_MEAN = configs["background_mean"]

    RECTANGLE_MEAN = BACKGROUND_MEAN * SIGNAL_TO_NOISE_RATIO
    RECTANGLE_STD = NOISE_STD * SIGNAL_TO_NOISE_RATIO

    with open("data_configs.yaml") as data_configs_file:
        data_configs = yaml.load(data_configs_file, Loader=yaml.FullLoader)
        IMAGE_WIDTH = data_configs["image_width"]
        IMAGE_HEIGHT = data_configs["image_height"]

    image = np.random.normal(
        loc=BACKGROUND_MEAN, scale=NOISE_STD, size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    )
    image = image.astype(np.float64)

    # Add a random number of black (0-valued) rectangles without overlap.
    num_rectangles = random.randint(5, 15)
    placed_rectangles = generate_rectangles(num_rectangles)

    for curr_rectangle in placed_rectangles:
        rectangle_values = np.random.normal(
            loc=RECTANGLE_MEAN,
            scale=RECTANGLE_STD,
            size=(curr_rectangle.height, curr_rectangle.width),
        )
        image[curr_rectangle.slice] = rectangle_values

    return image


def main():
    image = generate_data()
    fig, ax = plt.subplots()
    ax.imshow(image, cmap="binary", vmin=0, vmax=MAX_UINT8)
    clean = preprocess(image)
    contours, hierarchy = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    former_rectangles = contours_to_rectangles(contours, hierarchy)

    for found_rectangle in former_rectangles:
        inside_rectangle = image[found_rectangle.slice]
        mean = np.mean(inside_rectangle)
        print(f"Rectangle mean: {mean}")
        plotting_rectangle = plt.Rectangle(
            (found_rectangle.x, found_rectangle.y),
            found_rectangle.width,
            found_rectangle.height,
            linewidth=1,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(plotting_rectangle)

    plt.title("Detected Rectangles")
    plt.show()


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
            found_rectangle = Rectangle(x=x, y=y, height=height, width=width)
        except ValueError:
            continue
        rect_list.append(found_rectangle)
    rect_list = merge_overlapping_rectangles(rect_list)
    return rect_list


def plot_image(image, title):
    plt.imshow(image, cmap="binary", vmin=0, vmax=MAX_UINT8)
    plt.title(title)
    plt.show()


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


if __name__ == "__main__":
    main()
