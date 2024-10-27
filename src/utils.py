"""Module for Utils functions"""

import os
import re
from datetime import datetime


def get_latest_model_path(base_directory: str):
    """Get the path of the most recent model directory based on timestamp.

    Args:
        base_directory (str): The base directory where model directories are saved.

    Returns:
        str: The path to the most recent model directory if found, else None.
    """
    pattern = re.compile(r"medsam_finetuned_(\d{8}_\d{6})")

    subdirs = [
        (subdir, datetime.strptime(pattern.search(subdir).group(1), "%Y%m%d_%H%M%S"))
        for subdir in os.listdir(base_directory)
        if pattern.search(subdir)
    ]

    latest_subdir = max(subdirs, key=lambda x: x[1])[0] if subdirs else None

    if latest_subdir:
        latest_model_path = os.path.join(base_directory, latest_subdir)

        return latest_model_path
