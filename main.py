from typing import Self

import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pydantic import BaseModel, Field, model_validator

MAX_UINT8 = 255

with open("signal_configs.yaml") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    NOISE_STD = configs["noise_std"]
    SIGNAL_TO_NOISE_RATIO = configs["signal_to_noise_ratio"]
    BACKGROUND_MEAN = configs["background_mean"]

RECTANGLE_MEAN = BACKGROUND_MEAN * SIGNAL_TO_NOISE_RATIO
RECTANGLE_STD = NOISE_STD * SIGNAL_TO_NOISE_RATIO
MIN_RECTANGLE_AVG = RECTANGLE_MEAN - RECTANGLE_STD / 2

with open("data_configs.yaml") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    IMAGE_WIDTH = configs["image_width"]
    IMAGE_HEIGHT = configs["image_height"]
    MIN_RECTANGLE_HEIGHT = configs["min_rectangle_height"]
    MAX_RECTANGLE_HEIGHT = configs["max_rectangle_height"]
    MIN_RECTANGLE_WIDTH = configs["min_rectangle_width"]
    MIN_RECTANGLE_AREA = configs["min_rectangle_area"]

with open("algorithm_configs.yaml") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    BLURRING_KERNEL_SIZE = configs["blurring_kernel_size"]
    STRUCTURING_KERNEL_SIZE = configs["structuring_kernel_size"]
    MAX_RECTANGLE_AREA_INCREMENT_RATIO = configs["max_rectangle_area_increment_ratio"]


class Rectangle(BaseModel):
    x: int = Field(ge=0, lt=IMAGE_WIDTH)
    y: int = Field(ge=0, lt=IMAGE_HEIGHT)
    height: int = Field(ge=MIN_RECTANGLE_HEIGHT, le=MAX_RECTANGLE_HEIGHT)
    width: int = Field(ge=MIN_RECTANGLE_WIDTH, le=IMAGE_WIDTH)

    def max_x(self):
        return self.x + self.width

    @model_validator(mode='after')
    def validate_max_x(self):
        if self.max_x() >= IMAGE_WIDTH:
            raise ValueError("x + width must be less than IMAGE_WIDTH")
        return self

    def max_y(self):
        return self.y + self.height

    @model_validator(mode='after')
    def validate_max_y(self):
        if self.max_y() >= IMAGE_HEIGHT:
            raise ValueError("y + height must be less than IMAGE_HEIGHT")
        return self

    def area(self):
        return self.height * self.width

    @model_validator(mode='after')
    def validate_area(self):
        if self.area() < MIN_RECTANGLE_AREA:
            raise ValueError("Area of rectangle must be greater than MIN_RECTANGLE_AREA")
        return self

    @property
    def slice(self):
        """
        :return: a slice of the image corresponding to the rectangle
        """
        return slice(self.y, self.max_y()), slice(self.x, self.max_x())

    def overlap(self, other: Self):
        return not (self.max_x() <= other.x or other.max_x() <= self.x
                    or self.max_y() <= other.y or other.max_y() <= self.y)

    def merge(self, other: Self):
        merged_x = min(self.x, other.x)
        merged_y = min(self.y, other.y)
        merged_w = max(self.max_x(), other.max_x()) - merged_x
        merged_h = max(self.max_y(), other.max_y()) - merged_y
        merged_rectangle = Rectangle(x=merged_x, y=merged_y, height=merged_h, width=merged_w)
        return merged_rectangle

    def __hash__(self):
        return hash((self.x, self.y, self.height, self.width))

    def plot(self, ax, color='red'):
        rect = plt.Rectangle((self.x, self.y),
                             self.width, self.height, linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)


def generate_rectangles(num_rectangles):
    placed_rectangles: list[Rectangle] = []
    for _ in range(num_rectangles):
        while True:
            rect_width = np.random.randint(MIN_RECTANGLE_WIDTH, IMAGE_WIDTH)
            rect_height = np.random.randint(MIN_RECTANGLE_HEIGHT, IMAGE_HEIGHT)
            max_x = IMAGE_WIDTH - rect_width
            max_y = IMAGE_HEIGHT - rect_height
            if max_x <= 0 or max_y <= 0:
                continue
            x = np.random.randint(0, max_x)
            y = np.random.randint(0, max_y)
            try:
                curr_rectangle = Rectangle(x=x, y=y, height=rect_height, width=rect_width)
            except ValueError:
                print("Rectangle out of bounds")
                continue

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
    image = np.random.normal(loc=BACKGROUND_MEAN, scale=NOISE_STD, size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    image = image.astype(np.float64)

    # Add a random number of black (0-valued) rectangles without overlap.
    num_rectangles = np.random.randint(5, 15)
    placed_rectangles = generate_rectangles(num_rectangles)

    for curr_rectangle in placed_rectangles:
        rectangle_values = np.random.normal(loc=RECTANGLE_MEAN, scale=RECTANGLE_STD,
                                            size=(curr_rectangle.height, curr_rectangle.width))
        image[curr_rectangle.slice] = rectangle_values

    return image


def main():
    image = generate_data()
    fig, ax = plt.subplots()
    ax.imshow(image, cmap='binary', vmin=0, vmax=MAX_UINT8)
    clean = preprocess(image)
    contours, hierarchy = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    former_rectangles = contours_to_rectangles(contours, hierarchy)

    for found_rectangle in former_rectangles:
        inside_rectangle = image[found_rectangle.slice]
        mean = np.mean(inside_rectangle)
        print(f"Rectangle mean: {mean}")
        plotting_rectangle = plt.Rectangle((found_rectangle.x, found_rectangle.y), found_rectangle.width,
                                           found_rectangle.height, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(plotting_rectangle)

    plt.title("Detected Rectangles")
    plt.show()


def contours_to_rectangles(contours, hierarchy):
    filtered_contours = [cnt for i, cnt in enumerate(contours) if hierarchy[0][i][3] == -1]
    rect_list = []
    for cnt in filtered_contours:
        x, y, width, height = cv2.boundingRect(cnt)
        try:
            found_rectangle = Rectangle(x=x, y=y, height=height, width=width)
        except ValueError:
            continue
        rect_list.append(found_rectangle)

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
                if current.overlap(other):
                    merged = current.merge(other)
                    area_before_merge = current.area() + other.area()
                    area_increment_ratio = merged.area() / area_before_merge
                    if area_increment_ratio <= MAX_RECTANGLE_AREA_INCREMENT_RATIO:
                        current = merged
                        used[j] = True
                        merged_occurred = True
            new_rectangles.append(current)

        rect_list = new_rectangles

    return rect_list


def plot_image(image, title):
    plt.imshow(image, cmap='binary', vmin=0, vmax=MAX_UINT8)
    plt.title(title)
    plt.show()


def preprocess(image):
    blurred = cv2.GaussianBlur(image, (BLURRING_KERNEL_SIZE, BLURRING_KERNEL_SIZE), 0)
    # Convert to uint8 before thresholding
    max_value, min_value = blurred.max(), blurred.min()
    blurred_normalised = np.uint8(MAX_UINT8 * ((blurred - min_value) / (max_value - min_value)))
    thresh, binary_image = cv2.threshold(blurred_normalised, 0, MAX_UINT8, cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (STRUCTURING_KERNEL_SIZE, STRUCTURING_KERNEL_SIZE))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


if __name__ == "__main__":
    main()
