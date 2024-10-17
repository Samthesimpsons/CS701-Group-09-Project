<h1>CS701 Deep Learning and Vision Project - Group 09</h1>

<h2>Authors</h2>

- TAN Siao Wei​
- Keith CHIANG​
- YI Hang​
- XIAO Xiangjie​
- Samuel SIM Wei Xuan

<h2>Table of Contents</h2>

- [1. Problem Overview](#1-problem-overview)
- [2. Dataset](#2-dataset)
  - [2.1. Labels](#21-labels)
  - [2.2. Folder Structure](#22-folder-structure)
  - [2.3. Predictions](#23-predictions)
- [3. Technical Details](#3-technical-details)
  - [3.1. Folder Structure](#31-folder-structure)
  - [3.2. Dependency Management](#32-dependency-management)
  - [3.3. Running Checks](#33-running-checks)

<hr>

## 1. Problem Overview
Medical Image Segmentation on 12 Abdominal Organs

- **Evaluation**
   1. **Dice Similarity Coefficient (DSC)**
   2. **Normalized Surface Dice (NSD)**, tolerance = 1mm

## 2. Dataset

### 2.1. Labels
- **13 Total Classes**:
  - **0**: Background
  - **1**: Gallbladder
  - **2**: Stomach
  - **3**: Esophagus
  - **4**: Right Kidney
  - **5**: Right Adrenal Gland
  - **6**: Left Adrenal Gland
  - **7**: Liver
  - **8**: Left Kidney
  - **9**: Aorta
  - **10**: Spleen
  - **11**: Inferior Vena Cava
  - **12**: Pancreas

### 2.2. Folder Structure

```bash
data/
├── train_images/                           # Contains 50 sets of CT scans for training
│   ├── 01/                                 # Folder for CT volume 1
│   │   ├── 1.png                           # Individual slices of CT volume 1
│   │   └── ...                             # More slices for CT volume 1
│   └── ... (total 50 sets)                 # Total of 50 folders for training CT volumes
├── train_labels/                           # Contains segmentation masks for the 50 sets of training CT scans
│   ├── 01/                                 # Folder for segmentation masks corresponding to CT volume 1
│   │   ├── 1.png                           # Segmentation mask for slice 1 of CT volume 1
│   │   └── ...
│   └── ... (total 50 sets)
├── test_images/                            # Contains 15 sets of CT scans for testing
│   ├── 51/
│   │   ├── 1.png
│   │   └── ...
│   └── ... (total 15 sets)
├── test2_images/                           # Contains 15 sets of CT scans for final competition evaluation (Not yet provided)
│   └── ... (total 15 sets)
└── metadata/                               # Metadata and additional information for CT volumes
    ├── spacing_mm.txt                      # Contains voxel spacing information for all 80 CT scans (training, test, and hidden test)
    │   Example Entries:
    │   01: [0.976562, 0.976562, 2.5]       # Spacing for CT volume 1 in mm (X, Y, Z)
    │   02: [0.908203, 0.908203, 2.5]       # Spacing for CT volume 2 in mm (X, Y, Z)
    │   ...
    ├── test1_bbox.txt                       # Contains bounding box prompts for the test_images/ (CT 51 to 65)
    │   Example Entries:
    │   <65, 76, 11>: [235, 247, 307, 286]  # Bounding box for <CT 65, 76th slice, 11th organ (Inferior Vena Cava)> [x_min, y_min, x_max, y_max]
    └── <65, 76, 3>: [232, 211, 267, 271]   # Bounding box for <CT 65, 76th slice, 3rd organ (Esophagus)> [x_min, y_min, x_max, y_max]
```

### 2.3. Predictions
- **Predicted Labels**:
   - After running the model, predicted labels will be saved in a format similar to training labels.  
   - Example: `test_labels/05/3.png` for the model's prediction of `test_images/05/3.png`.

## 3. Technical Details

### 3.1. Folder Structure
The current folder structure follows a pattern similar to **Kedro**:
- **`./data`**: Contains all saved data (raw, processed).
- **`./notebooks`**: Contains any Jupyter notebooks.
- **`./src`**: Contains all the scripts.
- **`./models`**: Hosts any trained or pretrained models.

### 3.2. Dependency Management
**Poetry** is used for managing project dependencies and builds, replacing the outdated requirements.txt and setup.cfg approach with the modern `pyproject.toml` build system.

- **Benefits**:
  - Reliable and efficient way to manage dependencies.
  - Ensures consistency across different environments.

- **Steps to Install and Manage Dependencies**:

  ```bash
  # Install Poetry
  pip install poetry

  # Navigate to the directory containing `pyproject.toml`
  cd path/to/your/project

  # Install project dependencies listed in `pyproject.toml`
  poetry install

  # Optional: Activate the virtual environment that Poetry automatically creates by running:
  poetry shell

  # If you prefer not to use `poetry shell`, you can work within the virtual environment
  # by activating it similarly to any other virtual environment.

- **Adding Packages**:

   ``` bash
   # To add a package to your project
   poetry add <package-name>

   # For example, to add 'requests'
   poetry add requests

   # To specify a version or version range
   poetry add requests@^2.26.0

   # To add a package as a development dependency
   poetry add --dev <package-name>

   # For example, to add 'pytest' as a development dependency
   poetry add --dev pytest
   ```

- **Addtional Commands**:

   ``` bash
   # To see a list of installed packages and their versions
   poetry show

   # To update all packages to the latest compatible versions according to pyproject.toml
   poetry update
   ```

### 3.3. Running Checks
To ensure code consistency, correct type annotations, and comprehensive testing, you can run linting, typing checks, and tests with coverage together. The dependencies for the packages are already added to `pyproject.toml` via `poetry`.

**Prerequisite:** Ensure Make is installed.

Steps to run all checks:

   ```bash
   make check-all

   # Uncomment and run individually if needed:
   # make test
   # make typing
   # make lint
   ```