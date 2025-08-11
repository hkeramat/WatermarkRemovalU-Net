import os
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dataset import SegmentationDataset
from model import UNet
import torch.nn.init as init

def compute_iou(preds: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Computes the Intersection over Union (IoU) for binary segmentation outputs.

    Args:
        preds   (torch.Tensor): Binary predictions of shape [N, 1, H, W].
        targets (torch.Tensor): Ground truth masks of shape [N, 1, H, W].
        eps     (float)       : Small epsilon for numerical stability.

    Returns:
        float: Average IoU across the batch.
    """
    # Convert everything to float for safe summations
    preds   = preds.float()
    targets = targets.float()

    # Compute intersection and union
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union        = (preds + targets).sum(dim=(1, 2, 3)) - intersection

    # IoU for each image in the batch
    iou = (intersection + eps) / (union + eps)

    # Return average IoU in the batch
    return iou.mean().item()


def init_weights(m):
    """
    Applies random weight initialization.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # Kaiming normal initialization
        init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        # Xavier initialization
        init.xavier_normal_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)


if __name__ == "__main__":
    # hyper-parameters
    LEARNING_RATE   = 3e-4
    BATCH_SIZE      = 16
    EPOCHS          = 13

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the full dataset
    full_dataset = SegmentationDataset("./")

    # compute 80/20 split sizes
    total_samples = len(full_dataset)
    train_size    = int(0.8 * total_samples)
    val_size      = total_samples - train_size

    # reproducible split
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    # data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    # model, loss, optimizer
    model     = UNet(in_channels=3, num_classes=1).to(device)
    model.apply(init_weights) 
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        # ——— training ———
        model.train()
        train_loss = 0.0
        train_iou  = 0.0  # accumulate IoU
        for imgs, masks in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss  = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Compute IoU
            # Convert logits -> probabilities -> binary predictions (threshold=0.5).
            pred_probs  = torch.sigmoid(preds)
            pred_binary = (pred_probs > 0.5).float()
            batch_iou   = compute_iou(pred_binary, masks)
            train_iou   += batch_iou

        train_loss /= len(train_loader)
        train_iou  /= len(train_loader)

        # ——— validation ———
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Val   Epoch {epoch}"):
                imgs  = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)
                loss  = criterion(preds, masks)
                val_loss += loss.item()

                # Compute IoU for validation
                pred_probs  = torch.sigmoid(preds)
                pred_binary = (pred_probs > 0.5).float()
                batch_iou   = compute_iou(pred_binary, masks)
                val_iou     += batch_iou

        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)

        print(f"\nEpoch {epoch} — "
              f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} || "
              f"Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}\n")

    # save final weights
    os.makedirs(os.path.dirname('./'), exist_ok=True)
    torch.save(model.state_dict(), "./model.pth")
