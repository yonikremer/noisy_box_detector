import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from core_algorithm import (
    merge_overlapping_rectangles,
    contours_to_rectangles,
    preprocess,
    MAX_DISTANCE_BETWEEN_RECTANGLES,
    MAX_RECTANGLE_AREA_INCREMENT_RATIO,
)
from rectangle import Rectangle


@pytest.fixture
def sample_rectangles():
    return [
        Rectangle(x=0, y=0, length=20, height=20),  # Area = 400
        Rectangle(x=10, y=10, length=20, height=20),  # Area = 400
        Rectangle(x=50, y=50, length=20, height=20),  # Area = 400
    ]


@pytest.fixture
def sample_image():
    # Create a 100x100 black image
    image = np.zeros((100, 100), dtype=np.uint8)
    # Add some rectangles
    cv2.rectangle(image, (10, 10), (30, 30), 255, -1)  # Area = 400
    cv2.rectangle(image, (40, 40), (60, 60), 255, -1)  # Area = 400
    # Add some noise
    image[70:80, 70:80] = 128
    return image


def test_merge_overlapping_rectangles(sample_rectangles):
    # Test merging of overlapping rectangles
    merged = merge_overlapping_rectangles(sample_rectangles)
    assert len(merged) == 2  # Two rectangles should be merged, one should remain separate

    # Test with non-overlapping rectangles
    non_overlapping = [
        Rectangle(x=0, y=0, length=20, height=20),
        Rectangle(x=50, y=50, length=20, height=20),
    ]
    merged = merge_overlapping_rectangles(non_overlapping)
    assert len(merged) == 2  # No merging should occur

    # Test with empty list
    assert merge_overlapping_rectangles([]) == []

    # Test with single rectangle
    single = [Rectangle(x=0, y=0, length=20, height=20)]
    assert merge_overlapping_rectangles(single) == single


def test_merge_overlapping_rectangles_edge_cases():
    # Test rectangles that are exactly at the maximum distance
    rect1 = Rectangle(x=0, y=0, length=20, height=20)
    rect2 = Rectangle(
        x=20 + MAX_DISTANCE_BETWEEN_RECTANGLES, y=0, length=20, height=20
    )
    merged = merge_overlapping_rectangles([rect1, rect2])
    assert len(merged) == 2  # Should not merge at exactly max distance

    # Test rectangles that would exceed area ratio when merged
    rect1 = Rectangle(x=0, y=0, length=20, height=20)
    rect2 = Rectangle(x=10, y=10, length=30, height=30)
    merged = merge_overlapping_rectangles([rect1, rect2])
    assert len(merged) == 2  # Should not merge if area ratio would be exceeded


def test_merge_overlapping_rectangles_complex_cases():
    # Test chain merging (A->B->C)
    chain_rectangles = [
        Rectangle(x=0, y=0, length=20, height=20),
        Rectangle(x=15, y=0, length=20, height=20),
        Rectangle(x=30, y=0, length=20, height=20),
    ]
    merged = merge_overlapping_rectangles(chain_rectangles)
    assert len(merged) == 1  # All three should merge into one

    # Test circular arrangement
    circle_rectangles = [
        Rectangle(x=0, y=0, length=20, height=20),
        Rectangle(x=15, y=15, length=20, height=20),
        Rectangle(x=0, y=15, length=20, height=20),
    ]
    merged = merge_overlapping_rectangles(circle_rectangles)
    assert len(merged) == 1  # All should merge into one

    # Test with multiple groups
    groups_rectangles = [
        # Group 1
        Rectangle(x=0, y=0, length=20, height=20),
        Rectangle(x=15, y=0, length=20, height=20),
        # Group 2
        Rectangle(x=100, y=100, length=20, height=20),
        Rectangle(x=115, y=100, length=20, height=20),
    ]
    merged = merge_overlapping_rectangles(groups_rectangles)
    assert len(merged) == 2  # Should merge into two groups


def test_contours_to_rectangles(sample_image):
    # Find contours in the sample image
    contours, hierarchy = cv2.findContours(
        sample_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Convert contours to rectangles
    rectangles = contours_to_rectangles(contours, hierarchy)
    
    # Should find 2 rectangles (the noise should be filtered out)
    assert len(rectangles) == 2
    
    # Test with empty contours
    empty_contours = []
    empty_hierarchy = np.array([[]])
    assert contours_to_rectangles(empty_contours, empty_hierarchy) == []


def test_contours_to_rectangles_edge_cases():
    # Create an image with a single pixel
    single_pixel = np.zeros((100, 100), dtype=np.uint8)
    single_pixel[50, 50] = 255
    contours, hierarchy = cv2.findContours(
        single_pixel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    rectangles = contours_to_rectangles(contours, hierarchy)
    assert len(rectangles) == 0  # Should be filtered out due to minimum area

    # Create an image with nested contours
    nested = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(nested, (10, 10), (90, 90), 255, -1)  # Outer rectangle
    cv2.rectangle(nested, (30, 30), (70, 70), 0, -1)   # Inner hole
    contours, hierarchy = cv2.findContours(
        nested, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    rectangles = contours_to_rectangles(contours, hierarchy)
    assert len(rectangles) == 1  # Should only detect the outer rectangle


def test_preprocess(sample_image):
    # Test preprocessing
    processed = preprocess(sample_image)
    
    # Check output type and shape
    assert isinstance(processed, np.ndarray)
    assert processed.shape == sample_image.shape
    assert processed.dtype == np.uint8
    
    # Check that noise is removed (the 128 value pixels should be gone)
    assert not np.any(processed[70:80, 70:80] == 128)
    
    # Check that rectangles are preserved
    assert np.any(processed[10:30, 10:30] > 0)  # Check for any non-zero values
    assert np.any(processed[40:60, 40:60] > 0)  # Check for any non-zero values
    
    # Test with empty image
    empty_image = np.zeros((100, 100), dtype=np.uint8)
    processed_empty = preprocess(empty_image)
    assert np.all(processed_empty == 0)
    
    # Test with all white image - after preprocessing, values might not be exactly 255
    white_image = np.full((100, 100), 255, dtype=np.uint8)
    processed_white = preprocess(white_image)
    assert np.mean(processed_white) > 128  # Check that the image remains mostly white


def test_preprocess_edge_cases():
    # Test with single pixel image
    single_pixel = np.zeros((1, 1), dtype=np.uint8)
    single_pixel[0, 0] = 255
    processed = preprocess(single_pixel)
    assert processed.shape == (1, 1)  # Shape should be preserved
    
    # Test with very small image
    small_image = np.zeros((3, 3), dtype=np.uint8)
    small_image[1, 1] = 255
    processed = preprocess(small_image)
    assert processed.shape == (3, 3)  # Shape should be preserved
    
    # Test with image containing only intermediate values
    gray_image = np.full((100, 100), 128, dtype=np.uint8)
    processed = preprocess(gray_image)
    assert processed.dtype == np.uint8
    
    # Test with alternating pattern
    checker = np.zeros((100, 100), dtype=np.uint8)
    checker[::2, ::2] = 255  # Create checkerboard pattern
    processed = preprocess(checker)
    assert processed.dtype == np.uint8 