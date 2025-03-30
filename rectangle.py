import random
from typing import Self

import yaml
from matplotlib import pyplot as plt
from pydantic import BaseModel, Field, model_validator

with open("data_configs.yaml") as f:
    configs = yaml.load(f, Loader=yaml.FullLoader)
    IMAGE_WIDTH = configs["image_width"]
    IMAGE_HEIGHT = configs["image_height"]
    MIN_RECTANGLE_HEIGHT = configs["min_rectangle_height"]
    MAX_RECTANGLE_HEIGHT = configs["max_rectangle_height"]
    MIN_RECTANGLE_WIDTH = configs["min_rectangle_width"]
    MIN_RECTANGLE_AREA = configs["min_rectangle_area"]


class Rectangle(BaseModel):
    x: int = Field(ge=0, lt=IMAGE_WIDTH)
    y: int = Field(ge=0, lt=IMAGE_HEIGHT)
    height: int = Field(ge=MIN_RECTANGLE_HEIGHT, le=MAX_RECTANGLE_HEIGHT)
    width: int = Field(ge=MIN_RECTANGLE_WIDTH, le=IMAGE_WIDTH)

    def max_x(self):
        return self.x + self.width

    @model_validator(mode="after")
    def validate_max_x(self):
        if self.max_x() >= IMAGE_WIDTH:
            raise ValueError("x + width must be less than IMAGE_WIDTH")
        return self

    def max_y(self):
        return self.y + self.height

    @model_validator(mode="after")
    def validate_max_y(self):
        if self.max_y() >= IMAGE_HEIGHT:
            raise ValueError("y + height must be less than IMAGE_HEIGHT")
        return self

    def area(self):
        return self.height * self.width

    @model_validator(mode="after")
    def validate_area(self):
        if self.area() < MIN_RECTANGLE_AREA:
            raise ValueError(
                "Area of rectangle must be greater than MIN_RECTANGLE_AREA"
            )
        return self

    @property
    def slice(self):
        """
        :return: a slice of the image corresponding to the rectangle
        """
        return slice(self.y, self.max_y()), slice(self.x, self.max_x())

    def overlap(self, other: Self):
        return not (
            self.max_x() <= other.x
            or other.max_x() <= self.x
            or self.max_y() <= other.y
            or other.max_y() <= self.y
        )

    def merge(self, other: Self):
        merged_x = min(self.x, other.x)
        merged_y = min(self.y, other.y)
        merged_w = max(self.max_x(), other.max_x()) - merged_x
        merged_h = max(self.max_y(), other.max_y()) - merged_y
        merged_rectangle = Rectangle(
            x=merged_x, y=merged_y, height=merged_h, width=merged_w
        )
        return merged_rectangle

    def __hash__(self):
        return hash((self.x, self.y, self.height, self.width))

    def plot(self, ax, color="red"):
        rect = plt.Rectangle(
            (self.x, self.y),
            self.width,
            self.height,
            linewidth=1,
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(rect)

    @classmethod
    def random(cls):
        rect_width = random.randint(MIN_RECTANGLE_WIDTH, IMAGE_WIDTH)
        rect_height = random.randint(MIN_RECTANGLE_HEIGHT, MAX_RECTANGLE_HEIGHT)
        max_x = IMAGE_WIDTH - rect_width
        max_y = IMAGE_HEIGHT - rect_height
        if max_x <= 0 or max_y <= 0:
            raise ValueError
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        return Rectangle(x=x, y=y, height=rect_height, width=rect_width)
