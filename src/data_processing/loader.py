"""
This module provides a set of classes for loading model specific Datasets.
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
from .EDA import parse_spacing_file


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    num_workers: int = 0,
    batch_size: int = 1,
) -> DataLoader:
    """
    Creates DataLoader object for training.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        train_ratio (float): Proportion of the dataset to use for training (default: 0.8).
        shuffle (bool): Whether to shuffle the dataset before splitting (default: True).
        num_workers (int): Number of subprocesses to use for data loading (default: 0).
        batch_size (int): Number of samples per batch to load (default: 1).

    Returns:
        DataLoader: DataLoader object for training.
    """
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataloader


class SAMDataset(Dataset):
    """
    A unified dataset class for both training (with masks) and testing (with bounding boxes from a file).

    This dataset can switch between training and testing based on whether a mask directory or a bounding box file is provided.

    Args:
        image_dir (str): Directory containing input images.
        processor (str): A processor to prepare inputs for the segmentation model.
        spacing_metadata_dir (str): Directory containing the spacing metadata file.
        mask_dir (Optional[str]): Directory containing label masks (for training). Default is None.
        bbox_file_dir (Optional[str]): Directory to the text file containing bounding box annotations (for testing). Default is None.
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
            self.inputs_with_boxes = self._prepare_inputs_with_bounding_boxes_training()
        elif self.bbox_file_dir:
            self.bbox_data = self._load_bounding_boxes(self.bbox_file_dir)
            self.inputs_with_boxes = self._prepare_inputs_with_bounding_boxes_testing()

    def _load_image_mask_paths(self) -> Tuple[List[str], List[str]]:
        """
        Loads all matching image and mask file paths for training.
        Only called when mask_dir is provided (training).

        Returns:
            Tuple[List[str], List[str]]: Lists of matching image and mask file paths.
        """
        image_paths = []
        mask_paths = []

        for ct_folder in sorted(os.listdir(self.image_dir)):
            ct_image_folder = os.path.join(self.image_dir, ct_folder)
            ct_mask_folder = os.path.join(self.mask_dir, ct_folder)

            if os.path.isdir(ct_image_folder) and os.path.isdir(ct_mask_folder):
                images = sorted(os.listdir(ct_image_folder))
                masks = sorted(os.listdir(ct_mask_folder))

                for image_file, mask_file in zip(images, masks):
                    image_paths.append(os.path.join(ct_image_folder, image_file))
                    mask_paths.append(os.path.join(ct_mask_folder, mask_file))

        return image_paths, mask_paths

    def _prepare_inputs_with_bounding_boxes_training(
        self,
    ) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
        """
        Prepares a list of (image path, mask path, bounding box) for each bounding box in the training dataset.

        Returns:
            List[Tuple[str, str, Tuple[int, int, int, int]]]: A list of tuples, each containing the image path, mask path, and a bounding box.
        """
        inputs_with_boxes = []

        for idx in range(len(self.image_files)):
            image_path = self.image_files[idx]
            mask_path = self.mask_files[idx]

            ct_number, slice_number = self._extract_metadata_from_filename(image_path)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            bounding_boxes = get_bounding_boxes(mask, ct_number, slice_number)

            for box in bounding_boxes.values():
                inputs_with_boxes.append((image_path, mask_path, box))

        return inputs_with_boxes

    def _load_bounding_boxes(
        self, bbox_file_dir: str
    ) -> Dict[Tuple[str, int, int], List[int]]:
        """
        Loads the bounding box annotations from the text file for testing.

        Args:
            bbox_file_dir (str): Directory to the text file containing bounding box annotations.

        Returns:
            Dict[Tuple[str, int, int], List[int]]: A dictionary with keys as
            (CT number, slice number, organ label) and values as bounding box coordinates.
        """
        bbox_data = {}

        with open(bbox_file_dir, "r") as file:
            for line in file:
                key, value = line.strip().split(": ")
                ct_number, slice_number, organ_label = key.strip("<>").split(", ")
                bbox = eval(value)
                bbox_data[(str(ct_number), slice_number, organ_label)] = bbox

        return bbox_data

    def _prepare_inputs_with_bounding_boxes_testing(
        self,
    ) -> List[Tuple[str, List[int]]]:
        """
        Prepares a list of (image path, bounding box) for each bounding box in the testing dataset.

        Returns:
            List[Tuple[str, List[int]]]: A list of tuples, each containing the image path and a bounding box.
        """
        inputs_with_boxes = []

        for (ct_number, slice_number, _), bbox in self.bbox_data.items():
            image_path = os.path.join(self.image_dir, ct_number, f"{slice_number}.png")

            if os.path.exists(image_path):
                inputs_with_boxes.append((image_path, bbox))
            else:
                print(f"Warning: Image {image_path} not found, skipping.")

        return inputs_with_boxes

    def __len__(self) -> int:
        """
        Returns the total number of inputs (bounding boxes) in the dataset.

        Returns:
            int: The total number of bounding box-based inputs in the dataset.
        """
        return len(self.inputs_with_boxes)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieves an image and a bounding box, applies preprocessing,
        and prepares the input for the segmentation model.

        Args:
            idx (int): Index of the input to retrieve.

        Returns:
            Dict[str, Any]: A dictionary containing processed input tensors.
        """
        if self.mask_dir:  # For training set
            image_path, mask_path, bounding_box = self.inputs_with_boxes[idx]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            ct_number, slice_number = self._extract_metadata_from_filename(image_path)

            spacing = tuple(
                self.spacing_metadata_df[
                    self.spacing_metadata_df["CT_ID"] == ct_number
                ][["spacing_X", "spacing_Y"]].iloc[0]
            )

            # Apply our custom preprocessing
            new_image = apply_preprocessing_to_input_image(image, spacing)
            new_mask = apply_preprocessing_to_label_mask(mask, spacing)

            inputs = self.processor(
                cv2.cvtColor(new_image, cv2.COLOR_GRAY2RGB),
                input_boxes=[[bounding_box]],  # A single bounding box
                return_tensors="pt",
            )

            # Remove the batch dimension added by the processor (default behavior)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            inputs["ground_truth_mask"] = torch.tensor(new_mask, dtype=torch.float32)

            return inputs

        else:  # For testing
            image_path, bounding_box = self.inputs_with_boxes[idx]

            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # TODO: If we apply preprocessing, then we need to remap it back to inital volume
            inputs = self.processor(
                cv2.cvtColor(image, cv2.COLOR_GRAY2RGB),
                input_boxes=[[bounding_box]],  # A single bounding box
                return_tensors="pt",
            )

            # Remove the batch dimension added by the processor (default behavior)
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            return inputs

    def _extract_metadata_from_filename(self, filename: str) -> Tuple[str, int]:
        """
        Extracts the CT number and slice number from the filename.

        Args:
            filename (str): Full path to the image file.

        Returns:
            Tuple[str, int]: A tuple containing the CT number and the slice number.
        """
        parts = filename.split(os.sep)
        ct_number = parts[-2]
        slice_number = int(parts[-1].split(".png")[0])

        return ct_number, slice_number
