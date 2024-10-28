"""
This module provides functions for processing CT scan data. It includes segmentation mask visualization,
organ counting, voxel spacing analysis, and generating Sweetviz reports for exploratory data analysis.
"""

import os
import re
import cv2
import numpy as np
import pandas as pd
import sweetviz as sv
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Any, Optional


def count_slices_in_ct(
    ct_scan_directory: str, data_split_category: str
) -> pd.DataFrame:
    """
    Counts the number of slices in each CT scan folder and returns the result as a DataFrame.
    A new column is added to indicate the data split category.

    Args:
        ct_scan_directory (str): Directory containing CT scan folders.
        data_split_category (str): Data split type (e.g., train or test).

    Returns:
        pd.DataFrame: A DataFrame with columns ['CT_ID', 'slice_count', 'data_split_type'].

    Example:
        If a folder contains 3 slices, it will return {'CT_ID': 'folder_name', 'slice_count': 3, 'data_split_type': 'train'}
    """
    slice_counts = {}

    for folder in os.listdir(ct_scan_directory):
        folder_path = os.path.join(ct_scan_directory, folder)
        if not os.path.isdir(folder_path):
            continue

        valid_files = [f for f in os.listdir(folder_path) if f != ".gitkeep"]
        slice_counts[folder] = len(valid_files)

    slice_counts_df = pd.DataFrame(
        list(slice_counts.items()), columns=["CT_ID", "slice_count"]
    )
    slice_counts_df["data_split_type"] = data_split_category

    return slice_counts_df


def parse_spacing_file(file_path: str) -> pd.DataFrame:
    """
    Parses voxel spacing information from a text file.

    Args:
        file_path (str): Path to the spacing file.

    Returns:
        pd.DataFrame: A DataFrame with columns ['CT_ID', 'spacing_X', 'spacing_Y', 'spacing_Z'].

    Example:
        Parses lines like '1: [0.8, 0.8, 1.5]' and returns CT_ID 1 with corresponding spacing in X, Y, Z axes.
    """
    spacing_data = []

    with open(file_path, "r") as file:
        for line in file:
            match = re.match(r"(\d+): \[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]", line)
            if match:
                ct_id, spacing_x, spacing_y, spacing_z = match.groups()
                spacing_data.append(
                    {
                        "CT_ID": ct_id,
                        "spacing_X": float(spacing_x),
                        "spacing_Y": float(spacing_y),
                        "spacing_Z": float(spacing_z),
                    }
                )

    return pd.DataFrame(spacing_data)


def parse_bounding_box_file(file_path: str) -> pd.DataFrame:
    """
    Parses a bounding box file and counts annotated organs per slice.

    Args:
        file_path (str): Path to the bounding box file.

    Returns:
        pd.DataFrame: A DataFrame with columns ['CT_ID', 'Slice_Num', 'Organ_Count'].

    Example:
        Parses lines like '<1, 3, 5>: [some_box_values]' to count organs in each slice.
    """
    organ_counts = {}

    with open(file_path, "r") as file:
        for line in file:
            match = re.match(r"<(\d+), (\d+), (\d+)>: \[.*\]", line)
            if match:
                ct_id, slice_num, organ_id = match.groups()

                if ct_id not in organ_counts:
                    organ_counts[ct_id] = {}

                if slice_num not in organ_counts[ct_id]:
                    organ_counts[ct_id][slice_num] = set()

                organ_counts[ct_id][slice_num].add(organ_id)

    organ_records = [
        {"CT_ID": ct_id, "Slice_Num": slice_num, "Organ_Count": len(organs)}
        for ct_id, slices in organ_counts.items()
        for slice_num, organs in slices.items()
    ]

    return pd.DataFrame(organ_records)


def visualize_segmentation_from_numpy_arrays(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    bounding_boxes: Optional[List[List[int]]] = None,
) -> None:
    """
    Visualizes a CT scan image with its segmentation mask and optional bounding boxes.
    | Label | Organ                  | Color            |
    |-------|------------------------|------------------|
    | 0     | Background             | Black            |
    | 1     | Gallbladder            | Yellow           |
    | 2     | Stomach                | Red              |
    | 3     | Esophagus              | Green            |
    | 4     | Right Kidney           | Blue             |
    | 5     | Right Adrenal Gland    | Orange           |
    | 6     | Left Adrenal Gland     | Purple           |
    | 7     | Liver                  | Magenta          |
    | 8     | Left Kidney            | Cyan             |
    | 9     | Aorta                  | Pink             |
    | 10    | Spleen                 | Dark Green       |
    | 11    | Inferior Vena Cava     | Gray             |
    | 12    | Pancreas               | Dark Blue        |
    Args:
        image_array (np.ndarray): CT scan image in 2D format.
        mask_array (np.ndarray): Segmentation mask in 2D format.
        bounding_boxes (Optional[List[List[int]]]): Bounding boxes in format [x_min, y_min, x_max, y_max].

    Raises:
        ValueError: If input arrays don't have 2D shape.

    Example:
        Visualizes an image with organ labels such as liver, stomach, and kidneys, overlays segmentation mask, and bounding boxes if provided.
    """
    if image_array.ndim != 2 or mask_array.ndim != 2:
        raise ValueError("Both image_array and mask_array must be 2D")

    unique_labels = np.unique(mask_array)

    label_mapping = {
        0: "Background",
        1: "Gallbladder",
        2: "Stomach",
        3: "Esophagus",
        4: "Right Kidney",
        5: "Right Adrenal Gland",
        6: "Left Adrenal Gland",
        7: "Liver",
        8: "Left Kidney",
        9: "Aorta",
        10: "Spleen",
        11: "Inferior Vena Cava",
        12: "Pancreas",
    }

    label_colors = {
        0: [0, 0, 0],
        1: [255, 255, 0],
        2: [255, 0, 0],
        3: [0, 255, 0],
        4: [0, 0, 255],
        5: [255, 165, 0],
        6: [128, 0, 128],
        7: [255, 0, 255],
        8: [0, 255, 255],
        9: [255, 192, 203],
        10: [0, 128, 0],
        11: [128, 128, 128],
        12: [0, 0, 128],
    }

    mask_rgb = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    for label, color in label_colors.items():
        mask_rgb[mask_array == label] = color

    alpha = 0.5
    overlay_image = np.stack([image_array] * 3, axis=-1)
    overlay_image = cv2.addWeighted(overlay_image, 1 - alpha, mask_rgb, alpha, 0)

    if bounding_boxes:
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box
            cv2.rectangle(
                overlay_image,
                (x_min, y_min),
                (x_max, y_max),
                color=(255, 255, 255),
                thickness=2,
            )

    print("Labels in segmentation mask and corresponding organs:")
    for label in unique_labels:
        label_name = label_mapping.get(label, "Unknown Label")
        print(f"Label {label}: {label_name}")

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image_array, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay_image)
    plt.title("Image with Mask Label")
    plt.axis("off")

    plt.show()


def count_organs_and_area_fractions(mask_file_path: str) -> Dict[str, Any]:
    """
    Counts the distinct organs in a segmentation mask and computes the area fraction occupied by each organ.

    Args:
        mask_file_path (str): Path to the segmentation mask file.

    Returns:
        dict: A dictionary with:
            - 'num_organs' (int): Number of distinct organs.
            - 'organ_fractions' (Dict[str, float]): Fraction of image area occupied by each organ.

    Example:
        Parses a mask file to find the percentage of the image occupied by each organ like liver, kidneys, etc.
    """
    mask_image = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    unique_labels, pixel_counts = np.unique(mask_image, return_counts=True)
    organ_labels = unique_labels[unique_labels != 0]
    total_pixels = mask_image.size
    organ_fractions = {}

    label_mapping = {
        0: "Background",
        1: "Gallbladder",
        2: "Stomach",
        3: "Esophagus",
        4: "Right Kidney",
        5: "Right Adrenal Gland",
        6: "Left Adrenal Gland",
        7: "Liver",
        8: "Left Kidney",
        9: "Aorta",
        10: "Spleen",
        11: "Inferior Vena Cava",
        12: "Pancreas",
    }

    for organ_label in organ_labels:
        pixel_count = pixel_counts[unique_labels == organ_label][0]
        fraction = pixel_count / total_pixels
        organ_fractions[label_mapping.get(organ_label, "NA")] = round(fraction, 4)

    return {"num_organs": len(organ_labels), "organ_fractions": organ_fractions}


def process_all_slices_in_labels(base_mask_directory: str) -> pd.DataFrame:
    """
    Processes all slices in the given directory, computes organ counts and area fractions, and expands organ fractions.

    Args:
        base_mask_directory (str): Directory containing segmentation masks.

    Returns:
        pd.DataFrame: A DataFrame containing 'CT_ID', 'slice_id', 'num_organs', and individual columns for organ area fractions.

    Example:
        Processes multiple slices in folders and returns organ area distribution in a structured format.
    """
    data_records = []

    for root_dir, sub_dirs, file_names in os.walk(base_mask_directory):
        for file_name in file_names:
            if file_name == ".gitkeep":
                continue

            if file_name.endswith(".png"):
                mask_file_path = os.path.join(root_dir, file_name)
                slice_id = os.path.splitext(file_name)[0]

                if slice_id.isdigit():
                    slice_id = int(slice_id)

                ct_id = os.path.basename(os.path.dirname(mask_file_path))
                organ_analysis = count_organs_and_area_fractions(mask_file_path)

                data_records.append(
                    {
                        "CT_ID": ct_id,
                        "slice_id": slice_id,
                        "num_organs": organ_analysis["num_organs"],
                        "organ_fractions": organ_analysis["organ_fractions"],
                    }
                )

    analysis_df = pd.DataFrame(data_records)

    label_mapping = {
        1: "Gallbladder",
        2: "Stomach",
        3: "Esophagus",
        4: "Right Kidney",
        5: "Right Adrenal Gland",
        6: "Left Adrenal Gland",
        7: "Liver",
        8: "Left Kidney",
        9: "Aorta",
        10: "Spleen",
        11: "Inferior Vena Cava",
        12: "Pancreas",
    }

    for organ_id, organ_name in label_mapping.items():
        analysis_df[organ_name] = 0.0

    for idx, row in analysis_df.iterrows():
        organ_fractions = row["organ_fractions"]
        for organ, fraction in organ_fractions.items():
            if organ in label_mapping.values():
                analysis_df.at[idx, organ] = fraction

    analysis_df = analysis_df.drop(columns=["organ_fractions"])

    return analysis_df


def process_training_ct_scan_metadata(
    train_images_directory: str, train_labels_directory: str, spacing_file_path: str
) -> pd.DataFrame:
    """
    Processes training CT scan data, including slice counting, label processing, and spacing metadata merging.

    Args:
        train_images_directory (str): Directory containing training CT scan images.
        train_labels_directory (str): Directory containing training label masks.
        spacing_file_path (str): Path to the spacing metadata file.

    Returns:
        pd.DataFrame: A merged DataFrame containing slice count, spacing, and organ fraction data for the training set.

    Example:
        Returns combined slice count, voxel spacing, and organ fractions for training images.
    """
    slice_counts_df = count_slices_in_ct(
        ct_scan_directory=train_images_directory,
        data_split_category="train",
    )

    processed_labels = process_all_slices_in_labels(train_labels_directory)
    spacing_df = parse_spacing_file(spacing_file_path)

    merged_df = pd.merge(slice_counts_df, spacing_df, on=["CT_ID"], how="inner").merge(
        processed_labels, on=["CT_ID"], how="inner"
    )

    return merged_df


def process_test_ct_scan_metadata(
    test_images_directory: str, spacing_file_path: str
) -> pd.DataFrame:
    """
    Processes testing CT scan data by counting slices and merging voxel spacing metadata.

    Args:
        test_images_directory (str): Directory containing testing CT scan images.
        spacing_file_path (str): Path to the spacing metadata file.

    Returns:
        pd.DataFrame: A merged DataFrame containing slice count and spacing data for the testing set.

    Example:
        Returns combined slice count and voxel spacing for test images.
    """
    slice_counts_df = count_slices_in_ct(
        ct_scan_directory=test_images_directory,
        data_split_category="test",
    )

    spacing_df = parse_spacing_file(spacing_file_path)

    merged_df = pd.merge(slice_counts_df, spacing_df, on=["CT_ID"], how="inner")

    return merged_df


def generate_sweetviz_report(dataframe: pd.DataFrame, report_filename: str) -> None:
    """
    Generates and saves a Sweetviz report for a given DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
        report_filename (str): Path and name of the report file to save.

    Example:
        Generates a Sweetviz report for a DataFrame, useful for exploratory data analysis.
    """
    report_dir = Path(report_filename).parent
    report_dir.mkdir(parents=True, exist_ok=True)

    report = sv.analyze(dataframe)
    report.show_html(filepath=report_filename, open_browser=False)

    return None
