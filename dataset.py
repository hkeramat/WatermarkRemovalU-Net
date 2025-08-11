import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

""""
Prepreocessing the data and creating dataset
"""
class SegmentationDataset(Dataset):
    def __init__(self, root_path, transform=None):
        img_dir  = os.path.join(root_path, "img")
        mask_dir = os.path.join(root_path, "mask")

        # only .jpg/.jpeg for images
        self.images = sorted([
            os.path.join(img_dir, f)
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg"))
        ])
        # only .bmp for masks
        self.masks = sorted([
            os.path.join(mask_dir, f)
            for f in os.listdir(mask_dir)
            if f.lower().endswith(".bmp")
        ])

        assert len(self.images) == len(self.masks), (
            f"Got {len(self.images)} images but {len(self.masks)} masks."
        )

        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img  = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        return self.transform(img), self.transform(mask)

    def __len__(self):
        return len(self.images)


