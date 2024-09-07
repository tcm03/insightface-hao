import argparse
import os
import cv2
import torch
from backbones import get_model
from pathlib import Path

@torch.no_grad()
def batch_inference(weight, model_name, test_dir):
    # Load the model architecture and the saved weights
    net = get_model(model_name, fp16=False).cuda()
    net.load_state_dict(torch.load(weight))
    net.eval()

    # Dictionary to store the image path and its corresponding embedding
    embeddings = {}

    # Loop over the test directory, process all images in the subfolders
    for subdir in Path(test_dir).iterdir():
        if subdir.is_dir():  # Check if it's a subfolder
            for img_path in subdir.iterdir():
                if img_path.suffix in [".jpg", ".jpeg", ".png"]:  # Check for valid image files
                    # Read and preprocess the image
                    img = cv2.imread(str(img_path))
                    img = cv2.resize(img, (112, 112))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().cuda()
                    img.div_(255).sub_(0.5).div_(0.5)  # Normalize the image

                    # Get the embedding from the model
                    feat = net(img).cpu().numpy()

                    # Store the embedding
                    embeddings[str(img_path)] = feat

    return embeddings


if __name__ == "__main__":
    # Define arguments for the script
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Inference')
    parser.add_argument('--network', type=str, default='r50', help='backbone network (e.g., r50)')
    parser.add_argument('--weight', type=str, default='/kaggle/working/model.pt', help='path to the saved model')
    parser.add_argument('--test_dir', type=str, default='/kaggle/input/splited-casia-webmaskedface/splited-CASIA-WebMaskedFace/test', help='path to the test directory')
    args = parser.parse_args()

    # Perform batch inference and get embeddings
    embeddings = batch_inference(args.weight, args.network, args.test_dir)

    # Print or save embeddings (for further comparison or evaluation)
    print(f"Inference completed on test set. Number of images processed: {len(embeddings)}")
