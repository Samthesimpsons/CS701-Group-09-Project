"""
This module defines a `SAMTrainer` class to train a SAM (Segment Anything Model) 
segmentation model using K-Fold Cross-Validation. The model is initialized from 
the "facebook/sam-vit-base" checkpoint, and the training process utilizes 
the DiceCELoss function for segmentation.
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd

from tqdm.auto import tqdm
from datetime import datetime
from transformers import SamModel
from monai.losses import DiceCELoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
from peft import LoraConfig, get_peft_model
from statistics import mean


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
        val_split: float = 0.2,
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
        self.val_split = val_split
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
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
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

        for batch in tqdm(
            train_dataloader, desc="Training Batches", total=len(train_dataloader)
        ):
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
            print(f"Batch Train Loss: {loss.item():.4f}")

            del outputs, predicted_masks, ground_truth_masks, loss
            torch.cuda.empty_cache()

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
            for batch in tqdm(
                val_dataloader, desc="Validation Batches", total=len(val_dataloader)
            ):
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

                print(f"Batch Val Loss: {loss.item():.4f}")

                del outputs, predicted_masks, ground_truth_masks, loss
                torch.cuda.empty_cache()

        return mean(val_losses)

    def split_dataset(self, dataloader: DataLoader):
        """
        Split the dataset into training and validation subsets.

        Args:
            dataloader (DataLoader): The dataloader containing the full dataset.

        Returns:
            tuple: Train and validation DataLoaders.
        """
        dataset = dataloader.dataset
        val_size = int(len(dataset) * self.val_split)
        train_size = len(dataset) - val_size
        train_subset, val_subset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_subset, batch_size=dataloader.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_subset, batch_size=dataloader.batch_size)

        return train_loader, val_loader

    def train(self, dataloader: DataLoader, num_epochs: int = 10) -> pd.DataFrame:
        """
        Train the model over multiple epochs with validation.

        Args:
            dataloader (DataLoader): The dataloader containing the full dataset.
            num_epochs (int): Number of epochs to train.

        Returns:
            pd.DataFrame: A DataFrame containing epoch, train loss, and validation loss.
        """
        train_dataloader, val_dataloader = self.split_dataset(dataloader)
        losses = []

        for epoch in tqdm(range(num_epochs), desc="Epochs", total=num_epochs):
            train_loss = self.train_one_epoch(train_dataloader)
            val_loss = self.validate(val_dataloader)

            losses.append(
                {
                    "Epoch": epoch + 1,
                    "Train_DiceCE_Loss": train_loss,
                    "Val_DiceCE_Loss": val_loss,
                }
            )

            print(
                f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        losses_df = pd.DataFrame(losses)
        return losses_df
