"""
This module provides a set of general functions for preprocessing CT images, 
including contrast/brightness adjustment, and bounding box extraction.
"""

import cv2
import random
import numpy as np
import SimpleITK as sitk
from typing import Tuple, List, Dict


def adjust_contrast_brightness(
    image: np.ndarray, 
    alpha_range: tuple = (0.9, 1.3), 
    beta_range: tuple = (-10, 10)
) -> np.ndarray:
    """Randomly adjust the contrast and brightness of an image.

    Args:
        image (np.ndarray): Input image, can be grayscale or color (3-channel).
        alpha_range (tuple): Range for contrast factor. Values > 1 increase contrast, < 1 decrease contrast.
        beta_range (tuple): Range for brightness adjustment. Positive values increase brightness, negative values decrease it.

    Returns:
        np.ndarray: Image with randomly adjusted contrast and brightness.
    """
    alpha = random.uniform(*alpha_range)
    beta = random.randint(*beta_range)
    
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_gaussian_noise(
    image: np.ndarray, 
    mean_range: tuple = (0, 0), 
    std_dev_range: tuple = (10, 50)
) -> np.ndarray:
    """Randomly add Gaussian noise to an image.

    Args:
        image (np.ndarray): Input image, can be grayscale or color (3-channel).
        mean_range (tuple): Range for mean of the Gaussian noise.
        std_dev_range (tuple): Range for standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Image with Gaussian noise added.
    """
    mean = random.uniform(*mean_range)
    std_dev = random.uniform(*std_dev_range)
    
    noise = np.random.normal(mean, std_dev, image.shape).astype(np.float32)
    noisy_image = cv2.add(image.astype(np.float32), noise)

    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image


def normalize_to_unit_range(image: np.ndarray) -> np.ndarray:
    """Normalize the image to the range [0, 1].

    Args:
        image (np.ndarray): Input image with pixel values in the range [0, 255].

    Returns:
        np.ndarray: Normalized image with pixel values in the range [0, 1].
    """
    return image.astype(np.float32) / 255.0


def resample_and_resize(
    image_np: np.ndarray,
    original_spacing: Tuple[float, float],
    new_spacing: Tuple[float, float] = (1.0, 1.0),
    interpolator: int = sitk.sitkNearestNeighbor,
) -> np.ndarray:
    """Resamples a 2D grayscale image to a new pixel spacing and resizes it back to the original size.
    
    Args:
        image_np (np.ndarray): Input grayscale image as a NumPy array with shape (H, W).
        original_spacing (Tuple[float, float]): Original pixel spacing as (spacing_y, spacing_x).
        new_spacing (Tuple[float, float], optional): Desired pixel spacing as (new_spacing_y, new_spacing_x).
            Default is (1.0, 1.0).
        interpolator (int, optional): Interpolation method (default is sitk.sitkNearestNeighbor).
    
    Returns:
        np.ndarray: Resampled and resized 2D grayscale image as a NumPy array with the original size.
    """
    target_size = image_np.shape

    sitk_image = sitk.GetImageFromArray(image_np)
    sitk_image.SetSpacing(original_spacing)

    original_size = np.array(sitk_image.GetSize())
    new_size = (
        np.round(original_size * np.array(original_spacing) / np.array(new_spacing))
        .astype(int)
        .tolist()
    )

    # Resample the image
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampled_image = sitk.GetArrayFromImage(resampler.Execute(sitk_image))

    # Center crop or pad the image to match the original size
    pad_or_crop_image = np.zeros(target_size, dtype=resampled_image.dtype)
    height, width = resampled_image.shape
    y_start = max((height - target_size[0]) // 2, 0)
    x_start = max((width - target_size[1]) // 2, 0)
    y_end = y_start + min(target_size[0], height)
    x_end = x_start + min(target_size[1], width)
    target_y_start = max((target_size[0] - height) // 2, 0)
    target_x_start = max((target_size[1] - width) // 2, 0)
    target_y_end = target_y_start + (y_end - y_start)
    target_x_end = target_x_start + (x_end - x_start)

    pad_or_crop_image[target_y_start:target_y_end, target_x_start:target_x_end] = resampled_image[
        y_start:y_end, x_start:x_end
    ]

    return pad_or_crop_image

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
