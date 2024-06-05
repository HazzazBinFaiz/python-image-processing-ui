import cv2
import os
import numpy as np
import random

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not load image. Please check the file path.")
    return image


def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 5)
    clahe = cv2.createCLAHE(clipLimit=0.2, tileGridSize=(8, 8))
    clahe_image = clahe.apply(blurred_image)
    return clahe_image

def segment_blood_vessels(preprocessed_image):
    thresh_image = cv2.adaptiveThreshold(preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel, iterations=1)
    morph_image = cv2.erode(morph_image, kernel, iterations=1)
    morph_image = cv2.dilate(morph_image, kernel, iterations=1)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel, iterations=1)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    return morph_image

def blood_vessel_extract(image_path):
    image = load_image(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocessed_image = preprocess_image(image)
    segmented_image = segment_blood_vessels(preprocessed_image)
    return segmented_image


def processEye(image_path):
    lesion_image = blood_vessel_extract(image_path)
    os.remove(image_path)
    save_name = f"processed/eye-{random.randint(0, 100000000)}.png"
    cv2.imwrite(save_name, lesion_image)
    return save_name;

