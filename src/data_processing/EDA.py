"""
This module provides a set of functions for processing CT scan data, including segmentation mask visualization,
organ counting, voxel spacing analysis, and the generation of Sweetviz reports for exploratory data analysis.
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
    """Counts the number of slices in each CT scan folder and returns the result as a DataFrame.
    Also adds a new column for the data split category.

    Args:
        ct_scan_directory (str): Path to the directory containing CT scan folders.
        data_split_category (str): Category for data split, e.g., train or test.

    Returns:
        pd.DataFrame: A DataFrame with columns ['CT_ID', 'slice_count', 'data_split_type'].
    """
    slice_counts = {}

    for ct_folder in os.listdir(ct_scan_directory):
        folder_path = os.path.join(ct_scan_directory, ct_folder)
        if not os.path.isdir(folder_path):
            continue

        files_in_folder = [f for f in os.listdir(folder_path) if f != ".gitkeep"]
        slice_counts[ct_folder] = len(files_in_folder)

    slice_counts_dataframe = pd.DataFrame(
        list(slice_counts.items()), columns=["CT_ID", "slice_count"]
    )
    slice_counts_dataframe["data_split_type"] = data_split_category

    return slice_counts_dataframe


def parse_spacing_file(file_path: str) -> pd.DataFrame:
    """Parses the voxel spacing information from the given file.

    Args:
        file_path (str): Path to the file containing spacing information.

    Returns:
        pd.DataFrame: A DataFrame containing CT_ID and voxel spacing in X, Y, and Z dimensions.
    """
    spacing_data_records = []

    with open(file_path, "r") as file:
        for line in file:
            match = re.match(r"(\d+): \[(\d+\.\d+), (\d+\.\d+), (\d+\.\d+)\]", line)
            if match:
                ct_scan_id = match.group(1)
                spacing_X = float(match.group(2))
                spacing_y = float(match.group(3))
                spacing_Z = float(match.group(4))
                spacing_data_records.append(
                    {
                        "CT_ID": ct_scan_id,
                        "spacing_X": spacing_X,
                        "spacing_Y": spacing_y,
                        "spacing_Z": spacing_Z,
                    }
                )

    return pd.DataFrame(spacing_data_records)


def parse_bounding_box_file(file_path: str) -> pd.DataFrame:
    """Parses the bounding box file to count the number of annotated organs per slice.

    Args:
        file_path (str): Path to the bounding box file.

    Returns:
        pd.DataFrame: A DataFrame with columns ['CT_ID', 'Slice_Num', 'Organ_Count'].
    """
    organ_count_data: Dict[str, Dict[str, set]] = {}

    with open(file_path, "r") as file:
        for line in file:
            match = re.match(r"<(\d+), (\d+), (\d+)>: \[.*\]", line)
            if match:
                ct_scan_id = match.group(1)
                slice_number = match.group(2)
                organ_id = match.group(3)

                if ct_scan_id not in organ_count_data:
                    organ_count_data[ct_scan_id] = {}

                if slice_number not in organ_count_data[ct_scan_id]:
                    organ_count_data[ct_scan_id][slice_number] = set()

                organ_count_data[ct_scan_id][slice_number].add(organ_id)

    organ_count_records: List[Dict[str, str]] = []

    for ct_scan_id, slices in organ_count_data.items():
        for slice_number, organs in slices.items():
            organ_count_records.append(
                {
                    "CT_ID": ct_scan_id,
                    "Slice_Num": slice_number,
                    "Organ_Count": len(organs),
                }
            )

    return pd.DataFrame(organ_count_records)


def visualize_segmentation_from_numpy_arrays(
    image_array: np.ndarray,
    mask_array: np.ndarray,
    bounding_boxes: Optional[List[List[int]]] = None,
) -> None:
    """
    Visualize a CT scan image along with its segmentation mask, overlay the mask on the image,
    and optionally display bounding boxes.

    Args:
        image_array (np.ndarray): The CT scan image as a NumPy array with shape (x, x).
        mask_array (np.ndarray): The segmentation mask as a NumPy array with shape (x, x).
        bounding_boxes (Optional[List[List[int]]]): List of bounding boxes with format
            [x_min, y_min, x_max, y_max]. Defaults to None.

    Raises:
        ValueError: If the input arrays do not have the expected shapes.
    """
    if image_array.ndim != 2 or mask_array.ndim != 2:
        raise ValueError("Both image_array and mask_array must have shape (x, x)")

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
        0: [0, 0, 0],  # Background (black)
        1: [255, 255, 0],  # Gallbladder (yellow)
        2: [255, 0, 0],  # Stomach (red)
        3: [0, 255, 0],  # Esophagus (green)
        4: [0, 0, 255],  # Right Kidney (blue)
        5: [255, 165, 0],  # Right Adrenal Gland (orange)
        6: [128, 0, 128],  # Left Adrenal Gland (purple)
        7: [255, 0, 255],  # Liver (magenta)
        8: [0, 255, 255],  # Left Kidney (cyan)
        9: [255, 192, 203],  # Aorta (pink)
        10: [0, 128, 0],  # Spleen (dark green)
        11: [128, 128, 128],  # Inferior Vena Cava (gray)
        12: [0, 0, 128],  # Pancreas (dark blue)
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

    print("Unique labels in the segmentation mask and their corresponding organs:")
    for label in unique_labels:
        label_name = label_mapping.get(label, "Unknown Label")
        print(f"Label {label}: {label_name}")

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(image_array, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_rgb)
    plt.title("Color-Coded Segmentation Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay_image)
    plt.title("Image with Mask Overlay and Bounding Boxes")
    plt.axis("off")

    plt.show()


def count_organs_and_area_fractions(mask_file_path: str) -> Dict[str, Any]:
    """
    Counts the number of distinct organs in a given segmentation mask slice and calculates the fraction
    of the image area occupied by each organ.

    Args:
        mask_file_path (str): Path to the segmentation mask (e.g., 'train_labels/01/1.png').

    Returns:
        dict: A dictionary containing:
            - 'num_organs' (int): The number of distinct organs in the slice.
            - 'organ_fractions' (Dict[int, float]): The fraction of the image area occupied by each organ,
              keyed by the organ label and rounded to 4 decimal places.
    """
    mask_image = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)
    unique_labels, pixel_counts = np.unique(mask_image, return_counts=True)
    organ_labels = unique_labels[unique_labels != 0]
    total_pixel_count = mask_image.size
    organ_fractions: Dict[int, float] = {}

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
        organ_pixel_count = pixel_counts[unique_labels == organ_label][0]
        organ_fraction = organ_pixel_count / total_pixel_count
        organ_fractions[label_mapping.get(organ_label, "NA")] = round(organ_fraction, 4)

    number_of_organs = len(organ_labels)

    return {"num_organs": number_of_organs, "organ_fractions": organ_fractions}


def process_all_slices_in_labels(base_mask_directory: str) -> pd.DataFrame:
    """
    Processes all slices in the given base directory (including subfolders), calculates the number of organs,
    computes the area fractions for each organ in every slice, and expands the organ fractions into individual columns.

    This function traverses the directory tree starting from the given base directory, processes all PNG mask files
    corresponding to CT scan slices, and computes the number of distinct organs and the fraction of the image area
    each organ occupies. The organ fractions are then expanded into separate columns for each organ in the resulting
    DataFrame.

    Args:
        base_mask_directory (str): Base directory path containing subdirectories with segmentation masks.

    Returns:
        pd.DataFrame: A DataFrame containing the following columns:
            - 'CT_ID' (str): The CT scan ID, corresponding to the folder name.
            - 'slice_id' (int): The numeric slice ID (filename without extension).
            - 'num_organs' (int): The number of distinct organs in the slice.
            - 'organ_fractions' (Dict[int, float]): The fraction of the image area occupied by each organ.
            - Individual columns for each organ, named according to the label_mapping, containing the fraction
              of the slice area occupied by that specific organ. Organs not present in a slice will have a value of 0.0.
    """
    data_records = []

    for directory_root, directory_subfolders, file_names in os.walk(
        base_mask_directory
    ):
        for file_name in file_names:
            if file_name == ".gitkeep":
                continue

            if file_name.endswith(".png"):
                mask_file_path = os.path.join(directory_root, file_name)
                slice_identifier = os.path.splitext(file_name)[0]

                if slice_identifier.isdigit():
                    slice_identifier = int(slice_identifier)

                computed_ct_scan_identifier = os.path.basename(
                    os.path.dirname(mask_file_path)
                )
                organ_analysis_result = count_organs_and_area_fractions(mask_file_path)

                data_records.append(
                    {
                        "CT_ID": computed_ct_scan_identifier,
                        "slice_id": slice_identifier,
                        "num_organs": organ_analysis_result["num_organs"],
                        "organ_fractions": organ_analysis_result["organ_fractions"],
                    }
                )

    analysis_dataframe = pd.DataFrame(data_records)

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
        analysis_dataframe[organ_name] = 0.0

    for index, row in analysis_dataframe.iterrows():
        organ_fractions: Dict[str, float] = row["organ_fractions"]
        for organ, fraction in organ_fractions.items():
            if organ in label_mapping.values():
                analysis_dataframe.at[index, organ] = fraction

    analysis_dataframe = analysis_dataframe.drop(columns=["organ_fractions"])

    return analysis_dataframe


def process_training_ct_scan_data(
    train_images_directory: str, train_labels_directory: str, spacing_file_path: str
) -> pd.DataFrame:
    """
    Processes training CT scan data by counting slices, processing labels, and merging metadata such as spacing information.

    This function processes the training CT scan directory to count the slices, processes the label masks to compute organ
    fractions, and parses a common file containing slice spacing metadata. It merges the resulting DataFrames on the 'CT_ID'
    to create a comprehensive dataset for the training set.

    Args:
        train_images_directory (str): Directory containing training CT scan images.
        train_labels_directory (str): Directory containing training label masks.
        spacing_file_path (str): File path for the slice spacing metadata.

    Returns:
        pd.DataFrame: A merged DataFrame containing slice count, spacing, and organ fractions data for the training set.
    """
    train_slices_df = count_slices_in_ct(
        ct_scan_directory=train_images_directory,
        data_split_category="train",
    )

    processed_labels = process_all_slices_in_labels(train_labels_directory)

    spacing_df = parse_spacing_file(spacing_file_path)

    train_merged_df = pd.merge(
        train_slices_df, spacing_df, on=["CT_ID"], how="inner"
    ).merge(processed_labels, on=["CT_ID"], how="inner")

    return train_merged_df


def process_test_ct_scan_data(
    test_images_directory: str, spacing_file_path: str
) -> pd.DataFrame:
    """
    Processes testing CT scan data by counting slices and merging metadata such as spacing information.

    This function processes the testing CT scan directory to count the slices and parses a common file containing slice
    spacing metadata. It merges the resulting DataFrames on the 'CT_ID' to create a comprehensive dataset for the test set.

    Args:
        test_images_directory (str): Directory containing testing CT scan images.
        spacing_file_path (str): File path for the slice spacing metadata.

    Returns:
        pd.DataFrame: A merged DataFrame containing slice count and spacing data for the testing set.
    """
    test_slices_df = count_slices_in_ct(
        ct_scan_directory=test_images_directory,
        data_split_category="test",
    )

    spacing_df = parse_spacing_file(spacing_file_path)

    test_merged_df = pd.merge(test_slices_df, spacing_df, on=["CT_ID"], how="inner")

    return test_merged_df


def generate_sweetviz_report(dataframe: pd.DataFrame, report_filename: str) -> None:
    """
    Generates a Sweetviz report for the given DataFrame, displays it inline in a notebook,
    and saves it as an HTML file.

    Args:
        dataframe (pd.DataFrame): The DataFrame to analyze.
        report_filename (str): The file name (with path) where the Sweetviz report will be saved as an HTML file.

    Returns:
        None
    """
    report_path = Path(report_filename).parent
    report_path.mkdir(parents=True, exist_ok=True)

    report = sv.analyze(dataframe)

    report.show_notebook(
        w=None, h=None, scale=None, layout="vertical", filepath=report_filename
    )

    return None
