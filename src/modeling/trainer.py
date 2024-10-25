"""
This module defines a `SAMTrainer` class to train a SAM (Segment Anything Model) 
segmentation model using K-Fold Cross-Validation. The model is initialized from 
the "facebook/sam-vit-base" checkpoint, and the training process utilizes 
the DiceCELoss function for segmentation.

Additionally, an `EarlyStopping` class is implemented to halt training if 
the validation loss does not improve after a set number of epochs (patience).

The module leverages popular libraries like PyTorch, MONAI, Transformers, 
and Matplotlib for the training process and visualization.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai.losses import DiceCELoss
from sklearn.model_selection import KFold
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import SamModel
from peft import LoraConfig, get_peft_model
from statistics import mean
from typing import List

# Enable interactive plotting for visualization
plt.ion()


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """Initialize the EarlyStopping mechanism.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            delta (float): Minimum change to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """Check whether to stop training based on validation loss.

        Args:
            val_loss (float): The current validation loss.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


class SAMTrainer:
    """A class to encapsulate the SAM model, optimizer, and training logic."""

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str = "cpu",
        learning_rate: float = 1e-5,
        weight_decay: float = 0,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        quantize_8bits: bool = False,
    ):
        """Initialize the SAMTrainer class with model, optimizer, and loss function.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
            device (str): The device to use for training ('cpu' or 'cuda').
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            lora_r (int): Rank of the LoRA decomposition.
            lora_alpha (int): Scaling factor for the LoRA parameters.
            lora_dropout (float): Dropout rate for the LoRA layers.
            quantize_8bits (bool): Whether to load the model in 8-bit quantization.
        """
        self.device = device
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.quantize_8bits = quantize_8bits
        self.model = self._initialize_model(model_name)
        self.optimizer = self._get_optimizer(learning_rate, weight_decay)
        self.loss_function = DiceCELoss(
            sigmoid=True, softmax=False, squared_pred=True, reduction="mean"
        )

    def _initialize_model(self, model_name: str) -> SamModel:
        """Load and initialize the SAM model, applying LoRA configuration.

        Args:
            model_name (str): The name of the model to load from Hugging Face.

        Returns:
            SamModel: The initialized SAM model with LoRA configuration.
        """
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
            lora_dropout=self.lora_dropout,
            bias="none",
        )
        if self.quantize_8bits:
            model = get_peft_model(
                SamModel.from_pretrained(model_name, load_in_8bit=True), config
            )
        else:
            model = get_peft_model(SamModel.from_pretrained(model_name), config)
        model.print_trainable_parameters()
        model.to(self.device)
        return model

    def _get_optimizer(self, learning_rate: float, weight_decay: float) -> Adam:
        """Configure the optimizer.

        Args:
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for regularization.

        Returns:
            Adam: The configured optimizer.
        """
        return Adam(
            self.model.mask_decoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _compute_loss(
        self, predicted_masks: torch.Tensor, ground_truth_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute the segmentation loss.

        Args:
            predicted_masks (torch.Tensor): The predicted masks from the model.
            ground_truth_masks (torch.Tensor): The true masks from the dataset.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.loss_function(predicted_masks, ground_truth_masks.unsqueeze(1))

    def train_one_epoch(self, train_dataloader: DataLoader) -> float:
        """Train the model for one epoch.

        Args:
            train_dataloader (DataLoader): The dataloader for the training set.

        Returns:
            float: The mean training loss for the epoch.
        """
        self.model.train()
        epoch_losses = []

        for batch in tqdm(train_dataloader):
            outputs = self.model(
                pixel_values=batch["pixel_values"].to(self.device),
                input_boxes=batch["input_boxes"].to(self.device),
                multimask_output=False,
            )
            predicted_masks = outputs.pred_masks.squeeze(1)
            predicted_masks = F.interpolate(
                predicted_masks, size=(512, 512), mode="nearest-exact",
            )

            ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
            loss = self._compute_loss(predicted_masks, ground_truth_masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())

        return mean(epoch_losses)

    def validate(self, val_dataloader: DataLoader) -> float:
        """Validate the model on the validation set.

        Args:
            val_dataloader (DataLoader): The dataloader for the validation set.

        Returns:
            float: The mean validation loss for the epoch.
        """
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                outputs = self.model(
                    pixel_values=batch["pixel_values"].to(self.device),
                    input_boxes=batch["input_boxes"].to(self.device),
                    multimask_output=False,
                )
                predicted_masks = outputs.pred_masks.squeeze(1)
                predicted_masks = F.interpolate(
                    predicted_masks,
                    size=(512, 512),
                    mode="nearest-exact",
                )

                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
                loss = self._compute_loss(predicted_masks, ground_truth_masks)

                val_losses.append(loss.item())

        return mean(val_losses)

    def k_fold_cross_validation(
        self, dataloader: DataLoader, k_folds: int = 5, num_epochs: int = 10
    ):
        """Perform K-Fold Cross-Validation on the dataset.

        Args:
            dataloader (DataLoader): The dataloader containing the dataset.
            k_folds (int): Number of folds for cross-validation.
            num_epochs (int): Number of epochs to train for each fold.
        """
        dataset = dataloader.dataset  # Extract the dataset from the dataloader
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_training_losses, fold_validation_losses = [], []
        early_stopping = EarlyStopping(patience=3)

        # Loop through each fold
        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"Fold {fold + 1}/{k_folds}")

            # Create DataLoaders for the current fold
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            train_loader = DataLoader(
                train_subset, batch_size=dataloader.batch_size, shuffle=True
            )
            val_loader = DataLoader(val_subset, batch_size=dataloader.batch_size)

            # Store losses for visualization
            training_losses, validation_losses = [], []

            for epoch in range(num_epochs):
                train_loss = self.train_one_epoch(train_loader)
                val_loss = self.validate(val_loader)

                training_losses.append(train_loss)
                validation_losses.append(val_loss)

                print(
                    f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            # Store fold losses
            fold_training_losses.append(training_losses)
            fold_validation_losses.append(validation_losses)

            # Update plot for each fold
            self._update_plot(fold + 1, training_losses, validation_losses)

        print("Cross-validation completed.")

    def _update_plot(
        self, fold: int, training_losses: List[float], validation_losses: List[float]
    ):
        """Update the loss plot for each fold.

        Args:
            fold (int): The current fold number.
            training_losses (List[float]): List of training losses for each epoch.
            validation_losses (List[float]): List of validation losses for each epoch.
        """
        plt.figure(fold)
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss for Fold {fold}")
        plt.legend()
        plt.pause(0.1)  # Allow plot to update
