import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import os
import pandas as pd
import matplotlib.pyplot as plt
import csv

def load_images_with_noise_reduction(folder_path, blur_type='gaussian'):
    images = []
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.bmp')])
    for filename in image_files:
        #print(filename)
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        #print(img.shape)
        low_threshold=30
        
        thresholded_img = np.where(img < low_threshold, 0, img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(thresholded_img)
        # Clip the values to stay within the valid range [0, 255]
        #thresholded_img = np.clip(thresholded_img, 0, 255).astype(np.uint8)
        images.append(clahe_img)
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

def absolute_difference(imageA, imageB):
    # Ensure that both images have the same dimensions

    # Calculate the absolute difference between the two images
    abs_diff = cv2.absdiff(imageA, imageB)
    abs_diff = abs_diff.astype(np.int64)
    
    total_diff = np.sum(abs_diff)
    #print(total_diff)
    return total_diff

def find_window(missingframe, images, win_width):
    record = []
    # Initialize the window value with the sum of MSEs in the first window
    window_value = sum(mse(missingframe, images[i]) for i in range(win_width))
    
    # Record the value for the first window
    record.append([0, window_value])
    
    # Slide the window across the sequence
    for i in range(1, len(images) - win_width + 1):
        # Update the window value by adding the new frame and removing the old one
        window_value += mse(missingframe, images[i + win_width - 1])
        window_value -= mse(missingframe, images[i - 1])
        
        # Record the window position and its corresponding value
        record.append([i, window_value])
    
    return record

def find_window_all(missingimg,missing_files,images,win_width):
    records=[]
    for i in range(len(missingimg)):
        record=find_window(missingimg[i],images,win_width)
        #print(record)
        sorted_data = sorted(record, key=lambda x: x[1])
        #sorted_data = sorted(record, key=lambda x: x[1])
        top_10_smallest = sorted_data[: 10]
        print(missing_files[i])
        print(top_10_smallest)
        frame_set=set()
        for record in top_10_smallest:
            # Add all integers from (integer - 10) to (integer + 10) to the set
            for i in range(record[0] - 5, record[0] + 5):
                if i>=0 and i<len(images):
                    frame_set.add(i)
        records.append(frame_set)
    return records



def show_preprocessing(img_path):
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #print(img.shape)
    low_threshold=50
    
    thresholded_img = np.where(img < low_threshold, 0, img)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(thresholded_img)

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # Equalized Image
    plt.subplot(1, 2, 2)
    plt.title("Equalized Image")
    plt.imshow(clahe_img, cmap='gray')
    plt.axis('off')

    plt.show()


def compare_in_set(missing_path,frame_set,given_frames_folder,image_files):
    result_list=[]
    for index in frame_set:
        #print(index)
        img_path=os.path.join(given_frames_folder,image_files[index])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        missing_image=cv2.imread(missing_path, cv2.IMREAD_GRAYSCALE)
        mse_score=mse(missing_image,img)
        result_list.append([index,mse_score])
    result_list_sorted=sorted(result_list, key=lambda x: x[1])
    frame_name=image_files[result_list_sorted[0][0]]
    answer=frame_name[len("frame"):frame_name.rfind(".")]
    return answer

def cal_score(total_result,correct_answer):
    sum=0
    for i in range(len(total_result)):
        abs_diff=abs(total_result[i]-correct_answer[i])
        if abs_diff==0:
            sum+=10
        elif abs_diff<=10:
            sum-+5
        elif abs_diff<=25:
            sum+=3
        elif abs_diff<=50:
            sum+=1
        
    return sum

def save_list_to_csv(my_list, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(["mis_frame #", "position"])
        
        # Write the data rows
        for i, position in enumerate(my_list, start=1):
            writer.writerow([i, position])


if __name__ == '__main__':

    # given_frames_folder = "pre_release/sample_data/original"
    # missing_frames_folder = "pre_release/sample_data/missing"
    # images, image_files=load_images_with_noise_reduction(given_frames_folder, blur_type='gaussian')
    # missingimg,missing_files=load_images_with_noise_reduction(missing_frames_folder, blur_type='gaussian')
    # print(missing_files)
    # records=find_window_all(missingimg,missing_files,images,3)
    # total_result=[]
    # for i in range(len(missing_files)):
    #     missing_path=os.path.join(missing_frames_folder,missing_files[i])
    #     answer=compare_in_set(missing_path,records[i],given_frames_folder,image_files)
    #     total_result.append(int(answer))

    # print(total_result)
    # correct_answer=[20,61,63,93,175,220,260,338,394,404,426]
    # score=cal_score(total_result,correct_answer)
    # print("total_score: ",score)


    # save_list_to_csv(total_result, 'sample_result.csv')
    

    img_path="pre_release/sample_data/original/frame0019.bmp"
    show_preprocessing(img_path)