"""
This module provides a set of functions for pre-processing CT images.
"""

import cv2
import SimpleITK as sitk
import numpy as np
from typing import Tuple


def resample_pixel_space(
    image_np: np.ndarray,
    original_spacing: Tuple[float, float],
    new_spacing: Tuple[float, float],
    interpolator: int = sitk.sitkNearestNeighbor,
) -> np.ndarray:
    """Resamples a 2D color image to a new pixel spacing using SimpleITK.

    Args:
        image_np (np.ndarray): Input color image as a NumPy array with shape (H, W, 3).
        original_spacing (Tuple[float, float]): Original pixel spacing as (spacing_y, spacing_x).
        new_spacing (Tuple[float, float]): Desired pixel spacing as (new_spacing_y, new_spacing_x).
        interpolator (int, optional): Interpolation method (default: sitk.sitkNearestNeighbor).

    Returns:
        np.ndarray: Resampled 2D color image as a NumPy array with shape (new_H, new_W, 3).

    Example:
        >>> original_spacing = (0.9, 0.9)
        >>> new_spacing = (1.0, 1.0)
        >>> image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        >>> resampled_image = resample_pixel_space(image, original_spacing, new_spacing)
    """
    sitk_image = sitk.GetImageFromArray(image_np, isVector=True)
    sitk_image.SetSpacing(original_spacing)

    original_size = np.array(sitk_image.GetSize())
    new_size = (
        np.round(original_size * np.array(original_spacing) / np.array(new_spacing))
        .astype(int)
        .tolist()
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(interpolator)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())

    resampled_image = resampler.Execute(sitk_image)
    resampled_image = sitk.GetArrayFromImage(resampled_image)

    return resampled_image


def center_crop_or_pad(
    image: np.ndarray, target_size: int = 512, background_value: int = 0
) -> np.ndarray:
    """Center crop or pad the image to the target size.

    Args:
        image (np.ndarray): Input image as a 2D (grayscale) or 3D (color) array.
        target_size (int): Desired size for both height and width (default is 512).
        background_value (int): Background value used for padding (default is 0).

    Returns:
        np.ndarray: Image resized to the target size with center cropping or padding.
    """
    height, width = image.shape[:2]

    pad_or_crop_image = np.full(
        (
            (target_size, target_size, image.shape[2])
            if image.ndim == 3
            else (target_size, target_size)
        ),
        background_value,
        dtype=image.dtype,
    )

    y_start = max((height - target_size) // 2, 0)
    x_start = max((width - target_size) // 2, 0)

    y_end = y_start + min(target_size, height)
    x_end = x_start + min(target_size, width)

    target_y_start = max((target_size - height) // 2, 0)
    target_x_start = max((target_size - width) // 2, 0)

    target_y_end = target_y_start + (y_end - y_start)
    target_x_end = target_x_start + (x_end - x_start)

    pad_or_crop_image[target_y_start:target_y_end, target_x_start:target_x_end] = image[
        y_start:y_end, x_start:x_end
    ]

    return pad_or_crop_image


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
    original_spacing: Tuple[float, float],
    new_spacing: Tuple[float, float] = (0.9, 0.9),
    alpha: float = 1.5,
    beta: int = 20,
    target_size: int = 512,
    background_value: int = 0,
) -> np.ndarray:
    """Apply resampling, contrast/brightness adjustment, and center crop or padding.

    Args:
        image (np.ndarray): Input image to preprocess.
        original_spacing (Tuple[float, float]): Original pixel spacing.
        new_spacing (Tuple[float, float]): New pixel spacing for resampling.
        alpha (float): Contrast factor (> 1 increases contrast).
        beta (int): Brightness factor (positive increases brightness).
        target_size (int): Target size for height and width after cropping or padding.
        background_value (int): Value used for padding if the image is smaller.

    Returns:
        np.ndarray: Preprocessed image.
    """
    resampled_image = resample_pixel_space(image, original_spacing, new_spacing)

    adjusted_image = adjust_contrast_brightness(resampled_image, alpha, beta)

    final_image = center_crop_or_pad(adjusted_image, target_size, background_value)

    return final_image


def apply_preprocessing_to_label_mask(
    labels: np.ndarray,
    original_spacing: Tuple[float, float] = (1.0, 1.0),
    new_spacing: Tuple[float, float] = (0.9, 0.9),
    target_size: int = 512,
    background_value: int = 0,
) -> np.ndarray:
    """Apply resampling and center crop or padding to label image mask.

    Args:
        labels (np.ndarray): Input label image to preprocess.
        original_spacing (Tuple[float, float]): Original pixel spacing.
        new_spacing (Tuple[float, float]): New pixel spacing for resampling.
        target_size (int): Target size for height and width after cropping or padding.
        background_value (int): Value used for padding if the label image is smaller.

    Returns:
        np.ndarray: Preprocessed label image.
    """
    resampled_label = resample_pixel_space(labels, original_spacing, new_spacing)

    final_label = center_crop_or_pad(resampled_label, target_size, background_value)

    return final_label
