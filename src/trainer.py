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

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import SamModel
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
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
        learning_rate: float = 0.005,
        lora_r: int = 4,
        lora_alpha: int = 8,
        lora_dropout: float = 0,
        quantize_8bits: bool = False,
    ):
        """Initialize the SAMTrainer class with model, optimizer, and loss function.

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
        # TODO: Add NSC, but need parse in the spacing from spacing_mm.txt
        self.metric_function = DiceMetric(
            include_background=True,
            reduction="mean"
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
        """Configure the optimizer with AdamW.

        Args:
            learning_rate (float): The learning rate for the optimizer.

        Returns:
            AdamW: The configured optimizer with specific betas and weight decay.
        """
        return AdamW(
            self.model.mask_decoder.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.1,
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

    def _compute_metric(
        self, predicted_masks: torch.Tensor, ground_truth_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute the segmentation metric.

        Args:
            predicted_masks (torch.Tensor): The predicted masks from the model.
            ground_truth_masks (torch.Tensor): The true masks from the dataset.

        Returns:
            torch.Tensor: The computed metric.
        """
        metric = self.metric_function(y_pred=predicted_masks, y=ground_truth_masks)


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

            predicted_masks = F.interpolate(
                outputs.pred_masks.squeeze(),
                size=(512, 512),
                mode="nearest-exact",
            )

            ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
        
            loss = self._compute_loss(predicted_masks, ground_truth_masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())

        return mean(epoch_losses)

    def validate(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        """Validate the model on the validation set.

        Args:
            val_dataloader (DataLoader): The dataloader for the validation set.

        Returns:
            Tuple[float, float]: A tuple containing the mean validation loss and the mean validation score for the epoch.
        """
        self.model.eval()
        val_losses, val_scores = [], []

        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                outputs = self.model(
                    pixel_values=batch["pixel_values"].to(self.device),
                    input_boxes=batch["input_boxes"].to(self.device),
                    multimask_output=False,
                )

                predicted_masks = F.interpolate(
                    outputs.pred_masks.squeeze(),
                    size=(512, 512),
                    mode="nearest-exact",
                )

                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)

                loss = self._compute_loss(predicted_masks, ground_truth_masks)
                score = self._compute_metric(predicted_masks, ground_truth_masks)

                val_losses.append(loss.item())
                val_scores.append(score.item())

        mean_loss = mean(val_losses)
        mean_score = mean(val_scores)

        return mean_loss, mean_score

    def k_fold_cross_validation(
        self, dataloader: DataLoader, k_folds: int = 5, num_epochs: int = 10
    ):
        """Perform K-Fold Cross-Validation on the dataset.

        Args:
            dataloader (DataLoader): The dataloader containing the dataset.
            k_folds (int): Number of folds for cross-validation.
            num_epochs (int): Number of epochs to train for each fold.
        """
        dataset = dataloader.dataset

        kfold = KFold(n_splits=k_folds, shuffle=True)

        fold_training_losses, fold_validation_losses, fold_validation_scores = [], [], []

        early_stopping = EarlyStopping(patience=3)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
            print(f"[Fold {fold + 1}/{k_folds}]")

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=dataloader.batch_size, shuffle=True
            )

            val_loader = DataLoader(val_subset, batch_size=dataloader.batch_size)

            training_losses, validation_losses, validation_scores = [], [], []

            for epoch in range(num_epochs):
                train_loss = self.train_one_epoch(train_loader)
                val_loss, val_score = self.validate(val_loader)

                training_losses.append(train_loss)
                validation_losses.append(val_loss)
                validation_scores.append(val_score)

                print(
                    f"For Epoch {epoch + 1}/{num_epochs} | Train DiceCE Loss: {train_loss:.4f} | Val DiceCE Loss: {val_loss:.4f} | Val DSC Score: {val_score:.4f}"
                )

                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            fold_training_losses.append(training_losses)
            fold_validation_losses.append(validation_losses)
            fold_validation_scores.append(validation_scores)

            self._update_plot(fold + 1, training_losses, validation_losses, fold_validation_scores)

    def _update_plot(
        self, fold: int, training_losses: List[float], validation_losses: List[float], validation_scores: List[float]
    ):
        """Update the loss and score plot for each fold.

        Args:
            fold (int): The current fold number.
            training_losses (List[float]): List of training losses for each epoch.
            validation_losses (List[float]): List of validation losses for each epoch.
            validation_scores (List[float]): List of validation scores for each epoch.
        """
        plt.figure(fold)

        plt.plot(training_losses, label="Training Loss", color="blue")
        plt.plot(validation_losses, label="Validation Loss", color="orange")

        # Plot validation score on a secondary y-axis
        ax1 = plt.gca()  # Get the current axis for losses
        ax2 = ax1.twinx()  # Create a secondary axis for the score
        ax2.plot(validation_scores, label="Validation Score", color="green", linestyle="--")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_ylabel("Score")
        plt.title(f"Training and Validation Loss and Score for Fold {fold}")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.show(block=False)
        plt.pause(0.1)
        plt.close()