import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def load_images(folder_path):
    images = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
    for filename in image_files:
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images, image_files

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_frames(images):
    mse_scores = []
    ssim_scores = []
    for i in range(len(images) - 1):
        mse_score = mse(images[i], images[i + 1])
        ssim_score, _ = ssim(images[i], images[i + 1], full=True)
        mse_scores.append(mse_score)
        ssim_scores.append(ssim_score)
    
    return mse_scores, ssim_scores

def detect_missing_frames(mse_scores, ssim_scores, mse_threshold=None, ssim_threshold=None):
    potential_missing_frames = []
    for i in range(len(mse_scores)):
        if (mse_threshold is not None and mse_scores[i] > mse_threshold) or \
           (ssim_threshold is not None and ssim_scores[i] < ssim_threshold):
            potential_missing_frames.append(i + 1)
    return potential_missing_frames

# Example usage:
given_frames_folder = "/Users/chensiyu/Desktop/IDETC 2024/Hackathon data/pre_release/sample_data/original"

# Load images without preprocessing
images, image_files = load_images(given_frames_folder)

# Compare consecutive frames using MSE and SSIM
mse_scores, ssim_scores = compare_frames(images)

# Adjust thresholds to make detection more sensitive
mse_threshold = np.mean(mse_scores) + 2.5 * np.std(mse_scores)  # If two frames have larger mse between them, meaning potential missing frame
ssim_threshold = np.mean(ssim_scores) - 2.5 * np.std(ssim_scores)  # If two frames have small similarity, meaning potential missing frame

# Detect potential missing frames
missing_frames_indices = detect_missing_frames(mse_scores, ssim_scores, mse_threshold, ssim_threshold)

# Prepare the results for Excel output
labels = list(range(1, len(missing_frames_indices) + 1))
positions = [image_files[i - 1] for i in missing_frames_indices]  # Use the first frame of the detected pairs

# Extract the frame numbers from file names (assuming the format 'frameXXXX.bmp')
positions = [int(fname.split('frame')[-1].split('.')[0]) for fname in positions]

# Output the results
results_df = pd.DataFrame({
    'mis_frame #': labels,
    'position': positions
})

# Save the results to an Excel file in the specified format
output_path = "/Users/chensiyu/Desktop/IDETC 2024/Hackathon data/pre_release/sample_data/results.xlsx"
results_df.to_excel(output_path, index=False, sheet_name='Sheet1')
print(f"Results have been saved to {output_path}")
