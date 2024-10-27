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

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers import SamModel
from monai.losses import DiceCELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import KFold
from peft import LoraConfig, get_peft_model
from statistics import mean
from typing import List

# Enable interactive plotting for visualization
plt.ion()


class EarlyStopping:
    def __init__(self, patience: int = 5, delta: float = 0, restore_best: bool = True):
        """
        Initialize the EarlyStopping mechanism.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            delta (float): Minimum change to qualify as an improvement.
            restore_best (bool): Whether to restore the model to its best state.
        """
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.restore_best = restore_best
        self.best_model_state = None

    def __call__(self, val_loss: float, model: torch.nn.Module):
        """
        Check whether to stop training based on validation loss.

        Args:
            val_loss (float): The current validation loss.
            model (torch.nn.Module): The model being trained.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best:
                self.best_model_state = {
                    k: v.clone() for k, v in model.state_dict().items()
                }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best:
                    model.load_state_dict(self.best_model_state)


class SAMTrainer:
    """A class to encapsulate the SAM model, optimizer, and training logic."""

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str = "cpu",
        learning_rate: float = 0.005,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0,
        quantize_8bits: bool = False,
    ):
        """
        Initialize the SAMTrainer class with model, optimizer, and loss function.

        Args:
            model_name (str): The name of the model to load from Hugging Face.
            device (str): The device to use for training ('cpu' or 'cuda').
            learning_rate (float): The learning rate for the optimizer.
            lora_r (int): Rank of the LoRA decomposition.
            lora_alpha (int): Scaling factor for the LoRA parameters.
            lora_dropout (float): Dropout rate for the LoRA layers.
            quantize_8bits (bool): Whether to load the model in 8bits quantization.
        """
        self.device = device
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.quantize_8bits = quantize_8bits
        self.model = self._initialize_model(model_name)
        self.optimizer = self._get_optimizer(learning_rate)
        self.loss_function = DiceCELoss(
            sigmoid=True,
            squared_pred=True,
            reduction="mean",
        )

    def _initialize_model(self, model_name: str) -> SamModel:
        """
        Load and initialize the SAM model, applying LoRA configuration.

        Args:
            model_name (str): The name of the model to load from Hugging Face.

        Returns:
            SamModel: The initialized SAM model with LoRA configuration.
        """
        config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["qkv"],
            lora_dropout=self.lora_dropout,
            bias="none",
        )

        if self.quantize_8bits:
            model = SamModel.from_pretrained(model_name, load_in_8bit=True)
        else:
            model = SamModel.from_pretrained(model_name)

        for name, param in model.named_parameters():
            if name.startswith("vision_encoder"):
                param.requires_grad_(False)

        model.vision_encoder = get_peft_model(model.vision_encoder, config)

        percentage_of_trainable_parameters = (
            sum(p.numel() for p in model.parameters() if p.requires_grad)
            / sum(p.numel() for p in model.parameters())
            * 100
        )
        print(
            f"Percentage of trainable parameters: {percentage_of_trainable_parameters:.2f}%"
        )

        model.to(self.device)
        return model

    def _get_optimizer(self, learning_rate: float) -> AdamW:
        """
        Configure the optimizer with AdamW.

        Args:
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            AdamW: The configured optimizer with specific betas and weight decay.
        """
        optimizer = AdamW(
            self.model.mask_decoder.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.1,
        )
        self.scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        return optimizer

    def _resize_ground_truth(
        self, ground_truth: torch.Tensor, target_size=(256, 256)
    ) -> torch.Tensor:
        """
        Resize ground truth masks to a target size.

        Args:
            ground_truth (torch.Tensor): The ground truth mask tensor.
            target_size (tuple): Target dimensions for resizing.

        Returns:
            torch.Tensor: Resized ground truth masks.
        """
        return F.interpolate(
            ground_truth.float().unsqueeze(1), size=target_size, mode="nearest-exact"
        ).squeeze(1)

    def _compute_loss(
        self, predicted_masks: torch.Tensor, ground_truth_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the segmentation loss.

        Args:
            predicted_masks (torch.Tensor): The predicted masks from the model.
            ground_truth_masks (torch.Tensor): The true masks from the dataset.

        Returns:
            torch.Tensor: The computed loss.
        """
        return self.loss_function(predicted_masks, ground_truth_masks.unsqueeze(1))

    def save_pretrained(self, base_directory: str):
        """
        Save the model with a folder named by model name and timestamp.

        Args:
            base_directory (str): The base directory to save the model.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_directory = os.path.join(base_directory, f"medsam_finetuned_{timestamp}")

        os.makedirs(save_directory, exist_ok=True)

        self.model.save_pretrained(
            save_directory, safe_serialization=False, map_location="cpu"
        )

        print(f"Model saved to {save_directory}")

    def train_one_epoch(self, train_dataloader: DataLoader) -> float:
        """
        Train the model for one epoch.

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

            ground_truth_masks = self._resize_ground_truth(
                batch["ground_truth_mask"].to(self.device)
            )

            loss = self._compute_loss(predicted_masks, ground_truth_masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())

        return mean(epoch_losses)

    def validate(self, val_dataloader: DataLoader) -> float:
        """
        Validate the model on the validation set.

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

                ground_truth_masks = self._resize_ground_truth(
                    batch["ground_truth_mask"].to(self.device)
                )

                loss = self._compute_loss(predicted_masks, ground_truth_masks)
                val_losses.append(loss.item())

        return mean(val_losses)

    def k_fold_cross_validation(
        self, dataloader: DataLoader, k_folds: int = 5, num_epochs: int = 10
    ):
        """
        Perform K-Fold Cross-Validation on the dataset.

        Args:
            dataloader (DataLoader): The dataloader containing the dataset.
            k_folds (int): Number of folds for cross-validation.
            num_epochs (int): Number of epochs to train for each fold.
        """
        dataset = dataloader.dataset

        kfold = KFold(n_splits=k_folds, shuffle=True)

        fold_training_losses, fold_validation_losses = [], []

        early_stopping = EarlyStopping(patience=3)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"[Fold {fold + 1}/{k_folds}]")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=dataloader.batch_size, shuffle=True
            )

            val_loader = DataLoader(val_subset, batch_size=dataloader.batch_size)

            training_losses, validation_losses = [], []

            for epoch in range(num_epochs):
                train_loss = self.train_one_epoch(train_loader)
                val_loss = self.validate(val_loader)

                training_losses.append(train_loss)
                validation_losses.append(val_loss)

                print(
                    f"For Epoch {epoch + 1}/{num_epochs} | Train DiceCE Loss: {train_loss:.4f} | Val DiceCE Loss: {val_loss:.4f}"
                )

                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            fold_training_losses.append(training_losses)
            fold_validation_losses.append(validation_losses)

            self._update_plot(fold + 1, training_losses, validation_losses)

    def _update_plot(
        self,
        fold: int,
        training_losses: List[float],
        validation_losses: List[float],
    ):
        """
        Update the loss plot for each fold.

        Args:
            fold (int): The current fold number.
            training_losses (List[float]): List of training losses for each epoch.
            validation_losses (List[float]): List of validation losses for each epoch.
        """
        plt.figure(fold)

        plt.plot(training_losses, label="Training Loss", color="blue")
        plt.plot(validation_losses, label="Validation Loss", color="orange")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss for Fold {fold}")
        plt.legend(loc="upper right")

        plt.show(block=False)
        plt.pause(0.1)
        plt.close()
