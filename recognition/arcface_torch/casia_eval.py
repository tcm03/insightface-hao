import argparse
import os
import cv2
import torch
import random
import numpy as np
from pathlib import Path
from numpy.linalg import norm
from sklearn.metrics import accuracy_score
from backbones import get_model

# Cosine similarity function
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (norm(embedding1) * norm(embedding2))

# Generate pairs of images (positive and negative pairs)
def generate_pairs(embeddings, labels, num_pairs=10000):
    positive_pairs = []
    negative_pairs = []
    identities = list(set(labels))

    for _ in range(num_pairs // 2):
        # Generate a positive pair (same identity)
        same_identity = random.choice(identities)
        same_indices = [i for i, label in enumerate(labels) if label == same_identity]
        img1, img2 = random.sample(same_indices, 2)
        positive_pairs.append((img1, img2, 1))  # (index1, index2, label 1 for same identity)

        # Generate a negative pair (different identity)
        diff_identity1, diff_identity2 = random.sample(identities, 2)
        img1 = random.choice([i for i, label in enumerate(labels) if label == diff_identity1])
        img2 = random.choice([i for i, label in enumerate(labels) if label == diff_identity2])
        negative_pairs.append((img1, img2, 0))  # (index1, index2, label 0 for different identity)

    return positive_pairs + negative_pairs

# Compute verification accuracy
def compute_verification_accuracy(embeddings, labels, threshold=0.5, num_pairs=10000):
    pairs = generate_pairs(embeddings, labels, num_pairs)
    y_true = []
    y_pred = []

    for (img1, img2, label) in pairs:
        embedding1 = embeddings[img1]
        embedding2 = embeddings[img2]
        similarity = cosine_similarity(embedding1, embedding2)
        y_true.append(label)
        y_pred.append(1 if similarity > threshold else 0)  # 1 if similarity > threshold, else 0

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

@torch.no_grad()
def batch_inference(weight, model_name, test_dir):
    # Load the model architecture and the saved weights
    net = get_model(model_name, fp16=False)
    net.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
    net.eval()

    # Dictionary to store the image path, its corresponding embedding, and labels
    embeddings = []
    labels = []
    label_map = {}  # To map folder names to identity labels
    current_label = 0

    # Loop over the test directory, process all images in the subfolders
    for subdir in Path(test_dir).iterdir():
        if subdir.is_dir():  # Check if it's a subfolder
            if subdir.name not in label_map:
                label_map[subdir.name] = current_label
                current_label += 1
            for img_path in subdir.iterdir():
                if img_path.suffix in [".jpg", ".jpeg", ".png"]:  # Check for valid image files
                    # Read and preprocess the image
                    img = cv2.imread(str(img_path))
                    img = cv2.resize(img, (112, 112))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                    img.div_(255).sub_(0.5).div_(0.5)  # Normalize the image

                    # Get the embedding from the model
                    feat = net(img).cpu().numpy().flatten()
                    embeddings.append(feat)
                    labels.append(label_map[subdir.name])

    return np.array(embeddings), labels


if __name__ == "__main__":
    # Define arguments for the script
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Inference')
    parser.add_argument('--network', type=str, default='r50', help='backbone network (e.g., r50)')
    parser.add_argument('--weight', type=str, default='/kaggle/working/model.pt', help='path to the saved model')
    parser.add_argument('--test_dir', type=str, default='/kaggle/input/splited-casia-webmaskedface/splited-CASIA-WebMaskedFace/test', help='path to the test directory')
    args = parser.parse_args()

    # Perform batch inference and get embeddings and labels
    embeddings, labels = batch_inference(args.weight, args.network, args.test_dir)

    print(f'Number of processed images: {len(embeddings)}')

    # Compute verification accuracy
    accuracy = compute_verification_accuracy(embeddings, labels, threshold=0.5, num_pairs=10000)
    print(f"Verification Accuracy: {accuracy * 100:.2f}%")
