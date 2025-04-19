import pytest
from pydantic import ValidationError
import numpy as np
from matplotlib import pyplot as plt

from core import (
    Rectangle,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    MIN_RECTANGLE_HEIGHT,
    MAX_RECTANGLE_HEIGHT,
    MIN_RECTANGLE_WIDTH,
    MIN_RECTANGLE_AREA,
)


@pytest.fixture
def valid_rectangle():
    return Rectangle(x=100, y=100, height=50, length=50)


def test_rectangle_creation(valid_rectangle):
    # Test valid rectangle creation
    assert valid_rectangle.x == 100
    assert valid_rectangle.y == 100
    assert valid_rectangle.height == 50
    assert valid_rectangle.length == 50

    # Test invalid x coordinate
    with pytest.raises(ValidationError):
        Rectangle(x=-1, y=100, height=50, length=50)
    with pytest.raises(ValidationError):
        Rectangle(x=IMAGE_WIDTH, y=100, height=50, length=50)

    # Test invalid y coordinate
    with pytest.raises(ValidationError):
        Rectangle(x=100, y=-1, height=50, length=50)
    with pytest.raises(ValidationError):
        Rectangle(x=100, y=IMAGE_HEIGHT, height=50, length=50)

    # Test invalid height
    with pytest.raises(ValidationError):
        Rectangle(x=100, y=100, height=MIN_RECTANGLE_HEIGHT-1, length=50)
    with pytest.raises(ValidationError):
        Rectangle(x=100, y=100, height=MAX_RECTANGLE_HEIGHT+1, length=50)

    # Test invalid length
    with pytest.raises(ValidationError):
        Rectangle(x=100, y=100, height=50, length=MIN_RECTANGLE_WIDTH-1)
    with pytest.raises(ValidationError):
        Rectangle(x=100, y=100, height=50, length=IMAGE_WIDTH+1)

    # Test that middle_y calculation is correct
    assert valid_rectangle.middle_y() == 125  # 100 + 50//2


def test_rectangle_boundaries():
    # Test rectangle at image boundaries with minimum area
    min_side = int(np.ceil(np.sqrt(MIN_RECTANGLE_AREA)))
    rect = Rectangle(
        x=0, y=0,
        height=min_side,
        length=min_side
    )
    assert rect.x == 0
    assert rect.y == 0

    # Test rectangle extending beyond boundaries
    with pytest.raises(ValidationError):
        Rectangle(
            x=IMAGE_WIDTH-10,
            y=IMAGE_HEIGHT-10,
            height=20,
            length=20
        )


def test_rectangle_area(valid_rectangle):
    # Test area calculation
    assert valid_rectangle.area() == 2500  # 50 * 50

    # Test minimum area validation
    with pytest.raises(ValidationError):
        Rectangle(
            x=0, y=0,
            height=int(np.sqrt(MIN_RECTANGLE_AREA-1)),
            length=int(np.sqrt(MIN_RECTANGLE_AREA-1))
        )


def test_rectangle_overlap():
    rect1 = Rectangle(x=0, y=0, height=50, length=50)
    
    # Test no overlap
    rect2 = Rectangle(x=100, y=100, height=50, length=50)
    assert not rect1.overlap(rect2)
    
    # Test partial overlap
    rect3 = Rectangle(x=25, y=25, height=50, length=50)
    assert rect1.overlap(rect3)
    
    # Test complete overlap
    rect4 = Rectangle(x=0, y=0, height=25, length=25)
    assert rect1.overlap(rect4)
    
    # Test edge touch
    rect5 = Rectangle(x=50, y=0, height=50, length=50)
    assert not rect1.overlap(rect5)


def test_rectangle_merge():
    rect1 = Rectangle(x=0, y=0, height=50, length=50)
    rect2 = Rectangle(x=25, y=25, height=50, length=50)
    
    merged = rect1.merge(rect2)
    assert merged.x == 0
    assert merged.y == 0
    assert merged.height == 75
    assert merged.length == 75

    # Test merge with non-overlapping rectangles
    rect3 = Rectangle(x=100, y=100, height=50, length=50)
    merged = rect1.merge(rect3)
    assert merged.x == 0
    assert merged.y == 0
    assert merged.height == 150
    assert merged.length == 150

    # Test merging identical rectangles
    rect1 = Rectangle(x=0, y=0, height=50, length=50)
    merged = rect1.merge(rect1)
    assert merged.x == 0
    assert merged.y == 0
    assert merged.height == 50
    assert merged.length == 50


def test_rectangle_distance():
    rect1 = Rectangle(x=0, y=0, height=50, length=50)
    
    # Test overlapping rectangles (distance should be 0)
    rect2 = Rectangle(x=25, y=25, height=50, length=50)
    assert rect1.distance(rect2) == 0
    
    # Test adjacent rectangles
    rect3 = Rectangle(x=50, y=0, height=50, length=50)
    # dx = max(abs(50-50), abs(0-100)) = max(0, 100) = 100
    # dy = max(abs(0-50), abs(0-50)) = max(50, 50) = 50
    # min(dx, dy) = min(100, 50) = 50
    assert rect1.distance(rect3) == 50
    
    # Test separated rectangles
    rect4 = Rectangle(x=100, y=100, height=50, length=50)
    # dx = max(abs(100-50), abs(0-150)) = max(50, 150) = 150
    # dy = max(abs(100-50), abs(0-150)) = max(50, 150) = 150
    # min(dx, dy) = min(150, 150) = 150
    assert rect1.distance(rect4) == 150
    
    # Test diagonal separation
    rect5 = Rectangle(x=75, y=75, height=50, length=50)
    # dx = max(abs(75-50), abs(0-125)) = max(25, 125) = 125
    # dy = max(abs(75-50), abs(0-125)) = max(25, 125) = 125
    # min(dx, dy) = min(125, 125) = 125
    assert rect1.distance(rect5) == 125
    
    # Test with one rectangle completely above another
    rect6 = Rectangle(x=0, y=75, height=50, length=50)
    # dx = max(abs(0-50), abs(0-50)) = max(50, 50) = 50
    # dy = max(abs(75-50), abs(0-125)) = max(25, 125) = 125
    # min(dx, dy) = min(50, 125) = 50
    assert rect1.distance(rect6) == 50


def test_rectangle_intersection_area():
    rect1 = Rectangle(x=0, y=0, height=50, length=50)
    
    # Test no intersection
    rect2 = Rectangle(x=100, y=100, height=50, length=50)
    assert rect1.intersetion_area(rect2) == 0
    
    # Test partial intersection
    rect3 = Rectangle(x=25, y=25, height=50, length=50)
    assert rect1.intersetion_area(rect3) == 625  # 25 * 25
    
    # Test complete intersection (smaller inside larger)
    rect4 = Rectangle(x=0, y=0, height=25, length=25)
    assert rect1.intersetion_area(rect4) == 625  # 25 * 25

    # Test complete intersection (larger contains smaller)
    assert rect4.intersetion_area(rect1) == 625  # Should be symmetric


def test_rectangle_random():
    # Test multiple random rectangles
    for _ in range(10000):
        rect = Rectangle.random()
        assert 0 <= rect.x < IMAGE_WIDTH
        assert 0 <= rect.y < IMAGE_HEIGHT
        assert MIN_RECTANGLE_HEIGHT <= rect.height <= MAX_RECTANGLE_HEIGHT
        assert MIN_RECTANGLE_WIDTH <= rect.length <= IMAGE_WIDTH
        assert rect.area() >= MIN_RECTANGLE_AREA
        assert rect.max_x() < IMAGE_WIDTH
        assert rect.max_y() < IMAGE_HEIGHT

        # Test that random rectangles are within bounds with margin
        assert rect.x < IMAGE_WIDTH - MIN_RECTANGLE_WIDTH
        assert rect.y < IMAGE_HEIGHT - MIN_RECTANGLE_HEIGHT


def test_rectangle_plot(valid_rectangle):
    # Test that plotting doesn't raise errors
    plt.figure()
    valid_rectangle.plot()
    plt.close()

    # Test with custom color
    plt.figure()
    valid_rectangle.plot(color='blue')
    plt.close()


def test_rectangle_csv_methods(valid_rectangle):
    # Test CSV header
    header = Rectangle.csv_header()
    assert isinstance(header, str)
    assert 'x' in header
    assert 'length' in header
    assert 'middle_y' in header
    assert 'height' in header
    
    # Test CSV row
    row = valid_rectangle.to_csv_row()
    assert isinstance(row, str)
    values = row.strip().split(',')
    assert len(values) == 4
    assert int(values[0]) == valid_rectangle.x
    assert int(values[1]) == valid_rectangle.length
    assert int(values[2]) == valid_rectangle.middle_y()
    assert int(values[3]) == valid_rectangle.height 


def test_validate_max_y():
    # Test valid rectangle that extends to just before image height
    rect = Rectangle(x=0, y=IMAGE_HEIGHT-51, height=50, length=50)
    assert rect.max_y() == IMAGE_HEIGHT-1

    # Test rectangle that would extend beyond image height
    with pytest.raises(ValidationError):
        Rectangle(x=0, y=IMAGE_HEIGHT-40, height=50, length=50)


def test_rectangle_slice(valid_rectangle):
    # Test that slice returns correct tuple
    slice_tuple = valid_rectangle.slice
    assert isinstance(slice_tuple, tuple)
    assert len(slice_tuple) == 2
    assert slice_tuple[0].start == valid_rectangle.y
    assert slice_tuple[0].stop == valid_rectangle.y + valid_rectangle.height
    assert slice_tuple[1].start == valid_rectangle.x
    assert slice_tuple[1].stop == valid_rectangle.x + valid_rectangle.length


def test_rectangle_hash():
    # Test that identical rectangles have the same hash
    rect1 = Rectangle(x=100, y=100, height=50, length=50)
    rect2 = Rectangle(x=100, y=100, height=50, length=50)
    assert hash(rect1) == hash(rect2)

    # Test that different rectangles have different hashes
    rect3 = Rectangle(x=101, y=100, height=50, length=50)
    assert hash(rect1) != hash(rect3) 