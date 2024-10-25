"""
This module provides a dataset class and utility functions for handling CT scan image segmentation tasks,
specifically for the Segment Anything Model (SAM).

It includes functionality for loading images, preprocessing them, applying bounding boxes, and preparing
inputs for SAM during both training and testing phases.
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Any, Optional
from transformers import SamProcessor
from .preprocessing import (
    apply_preprocessing_to_input_image,
    apply_preprocessing_to_label_mask,
    get_bounding_boxes,
)
from .visualization import parse_spacing_file


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    shuffle: bool = True,
    num_workers: int = 0,
    batch_size: int = 1,
) -> DataLoader:
    """
    Creates a DataLoader object for a dataset, which is useful for batching inputs.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load into the DataLoader.
        shuffle (bool): Whether to shuffle the data (default: True).
        num_workers (int): Number of worker processes to load data (default: 0).
        batch_size (int): Number of samples per batch for inference or training (default: 1).

    Returns:
        DataLoader: A DataLoader object for batching and iterating over a dataset.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


class SAMSegmentationDataset(Dataset):
    """
    A dataset class for SAM-based segmentation tasks, supporting both training and testing modes.

    Args:
        image_dir (str): Directory containing the input images.
        processor (str): Pretrained SAM processor for preparing model inputs.
        spacing_metadata_dir (str): Directory containing the spacing metadata for the CT scans.
        mask_dir (Optional[str]): Directory containing the segmentation masks for training (default: None).
        bbox_file_dir (Optional[str]): Directory containing bounding boxes for testing (default: None).

    This dataset handles both training (with masks) and testing (with bounding boxes).
    If `mask_dir` is provided, the dataset operates in training mode. If only `bbox_file_dir` is provided,
    it operates in testing mode.
    """

    def __init__(
        self,
        image_dir: str,
        processor: str,
        spacing_metadata_dir: str,
        mask_dir: Optional[str] = None,
        bbox_file_dir: Optional[str] = None,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.bbox_file_dir = bbox_file_dir
        self.processor = SamProcessor.from_pretrained(processor)
        self.spacing_metadata_df = parse_spacing_file(spacing_metadata_dir)

        if self.mask_dir:
            self.image_files, self.mask_files = self._load_image_mask_paths()
            self.inputs_with_boxes = self._prepare_training_inputs()
        elif self.bbox_file_dir:
            self.bbox_data = self._load_bounding_boxes(self.bbox_file_dir)
            self.inputs_with_boxes = self._prepare_testing_inputs()

    def _load_image_mask_paths(self) -> Tuple[List[str], List[str]]:
        """
        Loads the image and corresponding mask file paths for training mode.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists: one for image paths and one for mask paths.
        """
        image_paths, mask_paths = [], []

        for patient_folder in sorted(os.listdir(self.image_dir)):
            image_folder = os.path.join(self.image_dir, patient_folder)
            mask_folder = os.path.join(self.mask_dir, patient_folder)

            if os.path.isdir(image_folder) and os.path.isdir(mask_folder):
                images = sorted(os.listdir(image_folder))
                masks = sorted(os.listdir(mask_folder))

                for img_file, mask_file in zip(images, masks):
                    image_paths.append(os.path.join(image_folder, img_file))
                    mask_paths.append(os.path.join(mask_folder, mask_file))

        return image_paths, mask_paths

    def _prepare_training_inputs(
        self,
    ) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
        """
        Prepares inputs consisting of images, masks, and bounding boxes for SAM training.

        Returns:
            List[Tuple[str, str, Tuple[int, int, int, int]]]: A list of tuples containing image paths, mask paths,
            and bounding boxes.
        """
        inputs_with_boxes = []

        for idx in range(len(self.image_files)):
            image_path = self.image_files[idx]
            mask_path = self.mask_files[idx]

            ct_id, slice_num = self._extract_metadata_from_filename(image_path)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            bounding_boxes = get_bounding_boxes(mask, ct_id, slice_num)

            for box in bounding_boxes.values():
                inputs_with_boxes.append((image_path, mask_path, box))

        return inputs_with_boxes

    def _load_bounding_boxes(
        self, bbox_file_dir: str
    ) -> Dict[Tuple[str, int, int], List[int]]:
        """
        Loads bounding boxes from a file for testing mode.

        Args:
            bbox_file_dir (str): Path to the file containing bounding box coordinates.

        Returns:
            Dict[Tuple[str, int, int], List[int]]: A dictionary where keys are tuples of CT ID, slice number, and organ label,
            and values are bounding box coordinates.
        """
        bbox_data = {}

        with open(bbox_file_dir, "r") as file:
            for line in file:
                key, value = line.strip().split(": ")
                ct_id, slice_num, organ_label = key.strip("<>").split(", ")
                bbox = eval(value)
                bbox_data[(str(ct_id), slice_num, organ_label)] = bbox

        return bbox_data

    def _prepare_testing_inputs(self) -> List[Tuple[str, List[int]]]:
        """
        Prepares inputs consisting of images and bounding boxes for SAM testing.

        Returns:
            List[Tuple[str, List[int]]]: A list of tuples containing image paths and bounding boxes.
        """
        inputs_with_boxes = []

        for (ct_id, slice_num, _), bbox in self.bbox_data.items():
            image_path = os.path.join(self.image_dir, ct_id, f"{slice_num}.png")

            if os.path.exists(image_path):
                inputs_with_boxes.append((image_path, bbox))
            else:
                print(f"Warning: Image {image_path} not found, skipping.")

        return inputs_with_boxes

    def __len__(self) -> int:
        """
        Returns the total number of inputs based on bounding boxes.

        Returns:
            int: The total number of input samples (bounding box-based).
        """
        return len(self.inputs_with_boxes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves the input at the specified index.

        Args:
            idx (int): Index of the input to retrieve.

        Returns:
            Dict[str, Any]: The processed input data ready for SAM, which includes pixel values, bounding boxes,
            and either the ground truth mask (for training) or image path (for testing).
        """
        if self.mask_dir:  # Training mode
            image_path, mask_path, bounding_box = self.inputs_with_boxes[idx]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            ct_id, slice_num = self._extract_metadata_from_filename(image_path)

            # spacing = tuple(
            #     self.spacing_metadata_df[self.spacing_metadata_df["CT_ID"] == ct_id][
            #         ["spacing_X", "spacing_Y"]
            #     ].iloc[0]
            # )

            processed_image = apply_preprocessing_to_input_image(
                image,
                # spacing,
            )
            processed_mask = apply_preprocessing_to_label_mask(
                mask,
                # spacing,
            )

            inputs = self.processor(
                cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB),
                input_boxes=[[bounding_box]],  # A single bounding box
                return_tensors="pt",
            )

            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["ground_truth_mask"] = torch.tensor(
                processed_mask, dtype=torch.float32
            )

            return inputs

        else:  # Testing mode
            image_path, bounding_box = self.inputs_with_boxes[idx]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            inputs = self.processor(
                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB),
                input_boxes=[[bounding_box]],  # A single bounding box
                return_tensors="pt",
            )

            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
            inputs["image_path"] = image_path

            return inputs

    def _extract_metadata_from_filename(self, filename: str) -> Tuple[str, int]:
        """
        Extracts the CT ID and slice number from the given file path.

        Args:
            filename (str): The path to the image file.

        Returns:
            Tuple[str, int]: The CT ID and the slice number extracted from the filename.
        """
        parts = filename.split(os.sep)
        ct_id = parts[-2]
        slice_num = int(parts[-1].split(".png")[0])

        return ct_id, slice_num
