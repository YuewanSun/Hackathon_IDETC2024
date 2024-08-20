import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
import pandas as pd


def load_images_with_noise_reduction(folder_path, size=(100, 100), blur_type='gaussian'):
    images = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
    for filename in image_files:
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Apply noise reduction
            if blur_type == 'gaussian':
                img = cv2.GaussianBlur(img, (5, 5), 0)
            elif blur_type == 'median':
                img = cv2.medianBlur(img, 5)
            # Resize the image if needed
            if size is not None:
                img = cv2.resize(img, size)
            images.append(img)
    return images, image_files

def load_images_from_folder(folder_path, size=None):
    images = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
    for filename in image_files:
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized_img = cv2.resize(img, size)
            images.append(resized_img)
    return images, image_files

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def ncc(imageA, imageB):
    return np.correlate(imageA.flatten(), imageB.flatten()) / (np.std(imageA) * np.std(imageB) * imageA.size)

def structural_content(imageA, imageB):
    return np.sum(imageA.astype("float") ** 2) / np.sum(imageB.astype("float") ** 2)

def image_gradient(imageA, imageB):
    gradA = cv2.Sobel(imageA, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(imageA, cv2.CV_64F, 0, 1, ksize=3)
    gradB = cv2.Sobel(imageB, cv2.CV_64F, 1, 0, ksize=3) + cv2.Sobel(imageB, cv2.CV_64F, 0, 1, ksize=3)
    return np.sum((gradA - gradB) ** 2) / (imageA.shape[0] * imageA.shape[1])

def entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist /= hist.sum()
    return -np.sum(hist * np.log2(hist + 1e-10))

def histogram_comparison(imageA, imageB):
    histA = cv2.calcHist([imageA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([imageB], [0], None, [256], [0, 256])
    histA = cv2.normalize(histA, histA).flatten()
    histB = cv2.normalize(histB, histB).flatten()
    return cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)

def compare_images(imageA, imageB):
    mse_score = mse(imageA, imageB)
    ssim_score, _ = ssim(imageA, imageB, full=True)
    psnr_score = psnr(imageA, imageB)
    hist_score = histogram_comparison(imageA, imageB)
    ncc_score = ncc(imageA, imageB)
    sc_score = structural_content(imageA, imageB)
    ig_score = image_gradient(imageA, imageB)
    entropy_score = abs(entropy(imageA) - entropy(imageB))
    
    return mse_score, ssim_score, psnr_score, ncc_score, hist_score, sc_score, ig_score, entropy_score

def find_best_fit_location(frames, missing_image):
    mse_scores = []
    ssim_scores = []
    psnr_scores = []
    hist_scores = []
    ncc_scores = []
    sc_scores = []
    ig_scores = []
    entropy_scores = []
    
    # Calculate all metrics for each frame
    for i in range(len(frames)):
        mse_score, ssim_score, psnr_score, hist_score, ncc_score, sc_score, ig_score, entropy_score = compare_images(missing_image, frames[i])
        
        mse_scores.append(mse_score)
        ssim_scores.append(ssim_score)
        psnr_scores.append(psnr_score)
        hist_scores.append(hist_score)
        ncc_scores.append(ncc_score)
        sc_scores.append(sc_score)
        ig_scores.append(ig_score)
        entropy_scores.append(entropy_score)
    
    # Convert lists to 1D NumPy arrays and explicitly flatten
    mse_scores = np.array(mse_scores).flatten()
    ssim_scores = np.array(ssim_scores).flatten()
    psnr_scores = np.array(psnr_scores).flatten()
    hist_scores = np.array(hist_scores).flatten()
    ncc_scores = np.array(ncc_scores).flatten()
    sc_scores = np.array(sc_scores).flatten()
    ig_scores = np.array(ig_scores).flatten()
    entropy_scores = np.array(entropy_scores).flatten()

    # Normalize the scores
    mse_scores = (mse_scores - np.min(mse_scores)) / (np.max(mse_scores) - np.min(mse_scores))
    ssim_scores = (ssim_scores - np.min(ssim_scores)) / (np.max(ssim_scores) - np.min(ssim_scores))
    psnr_scores = (psnr_scores - np.min(psnr_scores)) / (np.max(psnr_scores) - np.min(psnr_scores))
    hist_scores = (hist_scores - np.min(hist_scores)) / (np.max(hist_scores) - np.min(hist_scores))
    ncc_scores = (ncc_scores - np.min(ncc_scores)) / (np.max(ncc_scores) - np.min(ncc_scores))
    sc_scores = (sc_scores - np.min(sc_scores)) / (np.max(sc_scores) - np.min(sc_scores))
    ig_scores = (ig_scores - np.min(ig_scores)) / (np.max(ig_scores) - np.min(ig_scores))
    entropy_scores = (entropy_scores - np.min(entropy_scores)) / (np.max(entropy_scores) - np.min(entropy_scores))

    # Combine the scores
    combined_scores = (0 * mse_scores +
                       0 * (1 - ssim_scores) +
                       0 * (1 - psnr_scores) +
                       1 * hist_scores +
                       0 * (1 - ncc_scores) +
                       0 * (1 - sc_scores) +
                       0 * ig_scores +
                       0 * entropy_scores)

    # Ensure combined_scores is 1D
    combined_scores = combined_scores.flatten()
    print(f"Combined Scores Shape: {combined_scores.shape}")  # Should be (433,)
    print(f"Combined Scores: {combined_scores[:10]}")  # Print first 10 for debugging

    best_fit_index = np.argmin(combined_scores)  # Minimize the combined score
    
    if best_fit_index < 0 or best_fit_index >= len(frames):
        raise IndexError(f"Calculated best_fit_index {best_fit_index} is out of bounds for frames of length {len(frames)}")

    return best_fit_index

def find_missing_frames_locations(given_frames_folder, missing_frames_folder, size=(100, 100)):
    frames, frame_files = load_images_with_noise_reduction(given_frames_folder, size)
    missing_frames, missing_files = load_images_with_noise_reduction(missing_frames_folder, size)
    
    if len(frames) == 0:
        raise ValueError("No frames loaded from given_frames_folder.")
    
    if len(missing_frames) == 0:
        raise ValueError("No missing frames loaded from missing_frames_folder.")
    
    locations = []
    for i, missing_frame in enumerate(missing_frames):
        best_fit_index = find_best_fit_location(frames, missing_frame)
        
        if best_fit_index >= len(frame_files):
            raise IndexError(f"Best fit index {best_fit_index} is out of range for available frames.")
        
        locations.append((missing_files[i], best_fit_index + 1))  # +1 for 1-based indexing
    
    print("Locations:", locations)
    return locations

# Paths to your BMP image folders
given_frames_folder = "/Users/chensiyu/Desktop/IDETC 2024/Hackathon data/pre_release/sample_data/original"
missing_frames_folder = "/Users/chensiyu/Desktop/IDETC 2024/Hackathon data/pre_release/sample_data/missing"

# Generate predicted locations
locations = find_missing_frames_locations(given_frames_folder, missing_frames_folder)

# Save results to an Excel file
locations_df = pd.DataFrame(locations, columns=['mis_frame #', 'position'])
output_path = "/Users/chensiyu/Desktop/IDETC 2024/Hackathon data/pre_release/sample_data/results.xlsx"
locations_df.to_excel(output_path, index=False, sheet_name='Sheet1')
print(f"Generated locations have been saved to {output_path}")
