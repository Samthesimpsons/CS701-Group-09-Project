"""
This module provides functionality to run inference using a Segment Anything Model (SAM)
on a dataset and save the resulting predicted segmentation masks as grayscale images.
"""

import os
import cv2
import torch
import numpy as np

from tqdm import tqdm
from typing import Generator, List, Dict


def run_SAM_inference_and_save_masks(
    model: torch.nn.Module,
    test_dataset: torch.utils.data.Dataset,
    device: str = None,
) -> None:
    """
    Runs inference using a Segment Anything Model (SAM) on a test dataset and saves the predicted masks as images.

    This function processes the input dataset in batches, runs SAM inference to generate segmentation masks,
    and saves the resulting binary masks as grayscale images. The segmentation masks are thresholded to convert
    probabilities into binary masks, which are saved in the output directory.

    Args:
        model (torch.nn.Module): The SAM model to use for generating segmentation masks.
        test_dataset (torch.utils.data.Dataset): The dataset for inference, with each sample containing
                                                 'pixel_values', 'input_boxes', and 'image_path'.
        device (str, optional): The device to run the inference on ('cuda' or 'cpu'). If not provided,
                                it is auto-detected based on availability.

    Returns:
        None
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    model.eval()

    for sample in tqdm(test_dataset):
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.to(device)

        with torch.no_grad():
            outputs = model(**sample, multimask_output=False)

        predicted_probabilities = (
            torch.sigmoid(outputs.pred_masks.squeeze(0).squeeze(1)).cpu().numpy()
        )

        masks = (predicted_probabilities > 0.5).astype(np.uint8)

        label_image = np.max(masks, axis=0).astype(np.uint8)

        label_image_resized = cv2.resize(
            label_image, (512, 512), interpolation=cv2.INTER_NEAREST
        )

        original_image_path = sample["image_path"]
        mask_output_path = original_image_path.replace("test_images", "test_labels")

        output_directory = os.path.dirname(mask_output_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        cv2.imwrite(mask_output_path, grayscale_mask_resized)
