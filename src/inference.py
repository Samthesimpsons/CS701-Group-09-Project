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
    batch_size: int = 10,
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
        batch_size (int, optional): Number of samples to process in each batch for SAM inference. Defaults to 2.
        device (str, optional): The device to run the inference on ('cuda' or 'cpu'). If not provided,
                                it is auto-detected based on availability.

    Returns:
        None
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    model.eval()

    masks_by_path: Dict[str, np.ndarray] = {}

    def batch_generator(
        dataset: torch.utils.data.Dataset, batch_size: int
    ) -> Generator[List[Dict], None, None]:
        """
        Generates batches of samples from the dataset.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to divide into batches.
            batch_size (int): Number of samples in each batch.

        Yields:
            List[Dict]: A batch of records containing image data and associated metadata (e.g., pixel values and input boxes).
        """
        current_batch = []
        for idx in range(len(dataset)):
            sample = dataset[idx]
            current_batch.append(sample)
            if len(current_batch) == batch_size:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch

    for batch in tqdm(batch_generator(test_dataset, batch_size)):
        pixel_values = torch.stack([sample["pixel_values"] for sample in batch]).to(
            device
        )
        input_boxes = torch.stack([sample["input_boxes"] for sample in batch]).to(
            device
        )
        organ_labels = [sample["organ_class"] for sample in batch]
        inputs = {
            "pixel_values": pixel_values,
            "input_boxes": input_boxes,
        }

        with torch.no_grad():
            outputs = model(**inputs, multimask_output=False)

        for i, sample in enumerate(batch):
            predicted_probabilities = (
                torch.sigmoid(outputs.pred_masks[i]).cpu().numpy().squeeze()
            )

            binary_mask = (predicted_probabilities > 0.5).astype(np.uint8)

            binary_mask = cv2.resize(
                # To match back the train labels where the segmentation mask values are equal to the class organ label index
                src=(predicted_probabilities > 0.5).astype(np.uint8)*organ_labels[i],
                dsize=(512, 512),
                interpolation=cv2.INTER_NEAREST
            )

            mask_output_path = sample["image_path"].replace("test_images", "test_labels")

            if mask_output_path in masks_by_path:
                masks_by_path[mask_output_path] = np.maximum(
                    masks_by_path[mask_output_path], binary_mask
                )
            else:
                masks_by_path[mask_output_path] = binary_mask

    for mask_output_path, mask in masks_by_path.items():
        output_directory = os.path.dirname(mask_output_path)

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        cv2.imwrite(mask_output_path, mask)
