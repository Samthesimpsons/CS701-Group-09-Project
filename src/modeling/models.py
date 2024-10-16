from typing import List
from transformers import SamModel
from torch.optim import Adam
from monai.losses import DiceCELoss
from tqdm import tqdm
from statistics import mean
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Enable interactive plotting for visualization
plt.ion()


class SAMModel:
    """A class to encapsulate the SAM model, optimizer, and training logic."""

    def __init__(
        self,
        model_name: str = "facebook/sam-vit-base",
        device: str = "cpu",
        learning_rate: float = 1e-5,
        weight_decay: float = 0,
    ):
        self.device = device
        self.model = self._initialize_model(model_name)
        self.optimizer = self._get_optimizer(learning_rate, weight_decay)
        self.loss_function = DiceCELoss(
            sigmoid=True, squared_pred=True, reduction="mean"
        )

    def _initialize_model(self, model_name: str) -> SamModel:
        """Load and initialize the model."""
        model = SamModel.from_pretrained(model_name)
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        model.to(self.device)
        return model

    def _get_optimizer(self, learning_rate: float, weight_decay: float) -> Adam:
        """Configure the optimizer."""
        return Adam(
            self.model.mask_decoder.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def _compute_loss(
        self, predicted_masks: torch.Tensor, ground_truth_masks: torch.Tensor
    ) -> torch.Tensor:
        """Compute the segmentation loss."""
        return self.loss_function(predicted_masks, ground_truth_masks.unsqueeze(1))

    def train_one_epoch(self, train_dataloader: DataLoader) -> float:
        """Train the model for one epoch."""
        self.model.train()
        epoch_losses = []

        for batch in tqdm(train_dataloader):
            outputs = self.model(
                pixel_values=batch["pixel_values"].to(self.device),
                input_boxes=batch["input_boxes"].to(self.device),
                multimask_output=False,
            )
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
            loss = self._compute_loss(predicted_masks, ground_truth_masks)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())

        return mean(epoch_losses)

    def validate(self, val_dataloader: DataLoader) -> float:
        """Validate the model for one epoch."""
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
                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
                loss = self._compute_loss(predicted_masks, ground_truth_masks)

                val_losses.append(loss.item())

        return mean(val_losses)

    def k_fold_cross_validation(
        self, dataloader: DataLoader, k_folds: int = 5, num_epochs: int = 10
    ):
        """Perform K-Fold Cross-Validation on the dataset inside the DataLoader."""
        dataset = dataloader.dataset  # Extract the dataset from the dataloader
        kfold = KFold(n_splits=k_folds, shuffle=True)
        fold_training_losses, fold_validation_losses = [], []

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

            # Store fold losses
            fold_training_losses.append(training_losses)
            fold_validation_losses.append(validation_losses)

            # Update plot for each fold
            self._update_plot(fold + 1, training_losses, validation_losses)

        print("Cross-validation completed.")

    def _update_plot(
        self, fold: int, training_losses: List[float], validation_losses: List[float]
    ):
        """Update the plot for each fold."""
        plt.figure(fold)
        plt.plot(training_losses, label="Training Loss")
        plt.plot(validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss for Fold {fold}")
        plt.legend()
        plt.pause(0.1)  # Allow plot to update
