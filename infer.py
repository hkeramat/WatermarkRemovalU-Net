import os
import sys
import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import UNet

MODEL_PATH = "./model.pth"  # The path to your saved model

def single_image_inference(model, device, image_path):
    """Run inference on a single image and return the predicted mask as a torch.Tensor."""
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_tensor))
        # Remove all dimensions of size 1 (batch and channel)
        pred = (prob > 0.5).float().cpu().squeeze()  # shape => [H, W]

    return pred

def run_inference_on_directory(image_dir, output_dir, device):
    """Load model from MODEL_PATH, run inference on every image in image_dir, and save masks to output_dir."""
    # 1) Load the model
    model = UNet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 2) Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # 3) Iterate over images
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(image_dir, filename)
            pred_mask = single_image_inference(model, device, image_path)
            
            # Convert [H, W] mask to a PIL Image (values 0 or 255)
            mask_img = Image.fromarray((pred_mask.numpy() * 255).astype('uint8'))
            save_path = os.path.join(output_dir, filename)
            mask_img.save(save_path)
            print(f"Saved mask: {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Inference script: python infer.py <image_dir> <output_dir>"
    )
    parser.add_argument("image_dir", type=str, help="Directory containing input images.")
    parser.add_argument("output_dir", type=str, help="Directory to save predicted masks.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_inference_on_directory(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=device
    )

if __name__ == "__main__":
    main()
