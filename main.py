import itertools
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

from core_algorithm import contours_to_rectangles, preprocess
from rectangle import Rectangle

MAX_UINT8 = 255


def generate_rectangles(num_rectangles: int) -> list[Rectangle]:
    """
    Generate a list of random non-overlapping rectangles
    :param num_rectangles: the number of rectangles
    :return: a list of non-overlapping rectangles
    """
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

    return placed_rectangles, image


def main():
    ground_truth_rectangles, image = generate_data()
    plt.imshow(image, cmap="binary", vmin=0, vmax=MAX_UINT8)
    total_ground_truth_area = 0
    clean = preprocess(image)
    contours, hierarchy = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    predicted_rectangles = contours_to_rectangles(contours, hierarchy)
    total_predicted_area = 0
    for predicted_rectangle in predicted_rectangles:
        total_predicted_area += predicted_rectangle.area()
        predicted_rectangle.plot(color="red")

    for ground_truth_rectangle in ground_truth_rectangles:
        ground_truth_rectangle.plot(color="green")
        total_ground_truth_area += ground_truth_rectangle.area()

    plt.title("Detected Rectangles (red) Compared to Ground Truth (green)")
    plt.show()

    total_intersection_area = 0
    total_union_area = 0
    for ground_truth_rectangle, computed_rectangle in itertools.product(ground_truth_rectangles, predicted_rectangles):
        if ground_truth_rectangle.overlap(computed_rectangle):
            current_intersection_area = ground_truth_rectangle.intersetion_area(computed_rectangle)
            total_intersection_area += current_intersection_area
            total_union_area += ground_truth_rectangle.area() + computed_rectangle.area() - current_intersection_area

    print(f"Total ground truth area: {total_ground_truth_area}")
    print(f"Total predicted area: {total_predicted_area}")
    print(f"Total intersection area: {total_intersection_area}")
    print(f"Total union area: {total_union_area}")
    print(f"Intersection Over Union: {total_intersection_area / total_union_area}")
    print(f"Intersection over ground truth: {total_intersection_area / total_ground_truth_area}")
    print(f"Intersection over predicted: {total_intersection_area / total_predicted_area}")
    print(f"Number of ground truth rectangles: {len(ground_truth_rectangles)}")
    print(f"Number of predicted rectangles: {len(predicted_rectangles)}")


def plot_image(image, title):
    plt.imshow(image, cmap="binary", vmin=0, vmax=MAX_UINT8)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    main()
