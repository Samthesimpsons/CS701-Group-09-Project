"""
This module provides a set of general functions for preprocessing CT images, 
including contrast/brightness adjustment, and bounding box extraction.
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict


def adjust_contrast_brightness(
    image: np.ndarray, alpha: float, beta: int
) -> np.ndarray:
    """Adjust the contrast and brightness of an image.

    Args:
        image (np.ndarray): Input image, can be grayscale or color (3-channel).
        alpha (float): Contrast factor. Values > 1 increase contrast, < 1 decrease contrast.
        beta (int): Brightness factor. Positive values increase brightness, negative values decrease it.

    Returns:
        np.ndarray: Image with adjusted contrast and brightness.
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def apply_preprocessing_to_input_image(
    image: np.ndarray,
    alpha: float = 1.1,
    beta: int = 5,
) -> np.ndarray:
    """Apply preprocessing steps to an input image.

    Args:
        image (np.ndarray): Input 2D grayscale iamge (NumPy array with 2 dimensions).
        alpha (float): Contrast adjustment factor.
        beta (int): Brightness adjustment factor.

    Returns:
        np.ndarray: Preprocessed image.
    """

    image = adjust_contrast_brightness(image, alpha, beta)

    return image


def apply_preprocessing_to_label_mask(
    labels: np.ndarray,
) -> np.ndarray:
    """Apply preprocessing steps to a label mask.

    Args:
        labels (np.ndarray): Input 2D grayscale label iamge (NumPy array with 2 dimensions).

    Returns:
        np.ndarray: Preprocessed label mask.
    """

    labels = np.where(labels > 0, 1, 0)

    return labels


def get_bounding_boxes(
    mask_array: np.ndarray, CT_number: str, scan_number: int
) -> Dict[Tuple[int, int, int], List[int]]:
    """
    Computes bounding boxes for each unique label in a given ground truth mask.

    Args:
        mask_array (np.ndarray): 2D numpy array representing the ground truth mask.
                                 Non-zero values indicate the presence of objects.
        CT_number (str): The CT scan identifier (e.g., '01', '02').
        scan_number (int): The slice number of the scan.

    Returns:
        Dict[Tuple[int, int, int], List[int]]: A dictionary where keys are tuples of
                                               (CT_number, scan_number, label), and
                                               values are bounding box coordinates [x_min, y_min, x_max, y_max].
    """
    if mask_array.ndim != 2:
        raise ValueError("mask_array must be a 2D array.")

    H, W = mask_array.shape
    bounding_boxes = {}

    valid_labels = set(range(1, 13))

    unique_labels = np.unique(mask_array)
    unique_labels = [label for label in unique_labels if label != 0]

    invalid_labels = [label for label in unique_labels if label not in valid_labels]
    if invalid_labels:
        print(
            f"Warning: Found invalid labels {invalid_labels}. Expected labels are from 1 to 12."
        )

    for label in unique_labels:
        y_indices, x_indices = np.where(mask_array == label)

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))

        bounding_boxes[f"<{CT_number}, {scan_number}, {label}>"] = [
            x_min,
            y_min,
            x_max,
            y_max,
        ]

    return bounding_boxes
