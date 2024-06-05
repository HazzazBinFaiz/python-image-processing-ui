import os
import cv2
import numpy as np
import random


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error: Could not load image. Please check the file path.")
    return image

def extract_lesions(image):

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(10, 10))
    contrasted = clahe.apply(grayscale_image)


    smoothed = cv2.GaussianBlur(contrasted, (9, 9), 0)


    _, binary_image = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    smoothed_lesion = cv2.dilate(closed, kernel, iterations=1)
    smoothed_lesion = cv2.erode(smoothed_lesion, kernel, iterations=1)
    contours, _ = cv2.findContours(smoothed_lesion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            cv2.drawContours(smoothed_lesion, [contour], 0, 0, -1)
    return smoothed_lesion


def process_image(image_path):
    image = load_image(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return extract_lesions(image_rgb)


def processMelanoma(image_path):
    lesion_image = process_image(image_path)
    os.remove(image_path)
    save_name = f"processed/melanoma-{random.randint(0, 100000000)}.png"
    cv2.imwrite(save_name, lesion_image)
    return save_name;
