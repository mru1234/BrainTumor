# # import numpy as np
# # import pandas as pd
# # import os
# # import cv2
# # from skimage.feature import hog
# # from sklearn.decomposition import PCA
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import GradientBoostingClassifier
# # import joblib
# #
# # # Define paths to different folders containing different classes of brain tumor images
# # # folder1 = "glioma_tumor"
# # # folder2 = "meningioma_tumor"
# # # folder3 = "no_tumor"
# # # folder4 = "pituitary"
# # import os
# #
# # # Get the current working directory (your project folder)
# # project_folder = os.getcwd()
# #
# # # Define paths to different folders containing different classes of brain tumor images
# # folder1 = os.path.join(project_folder, "CV_CP_Dataset/glioma_tumor")
# # folder2 = os.path.join(project_folder, "CV_CP_Dataset/meningioma_tumor")
# # folder3 = os.path.join(project_folder, "CV_CP_Dataset/no_tumor")
# # folder4 = os.path.join(project_folder, "CV_CP_Dataset/pituitary")
# #
# #
# # # Function to extract HOG features from images in a folder
# # def extract_hog_features(folder_path, label):
# #     hog_descs = []
# #     for filename in os.listdir(folder_path):
# #         img = cv2.imread(os.path.join(folder_path, filename))
# #         if img is not None:
# #             resize = (200, 200)
# #             img1 = cv2.resize(img, resize)
# #             gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# #             median_img = cv2.medianBlur(gray, 3)
# #             fd, hog_image = hog(median_img, orientations=6, pixels_per_cell=(4, 4),
# #                                 transform_sqrt=True, cells_per_block=(1, 1), visualize=True)
# #             hog_descs.append(fd)
# #     return hog_descs, [label] * len(hog_descs)
# #
# # # Extract HOG features and labels for each class
# # hog_descs1, labels1 = extract_hog_features(folder1, 0)  # Glioma Tumor
# # hog_descs2, labels2 = extract_hog_features(folder2, 1)  # Meningioma Tumor
# # hog_descs3, labels3 = extract_hog_features(folder3, 2)  # No Tumor
# # hog_descs4, labels4 = extract_hog_features(folder4, 3)  # Pituitary
# #
# # # Concatenate features and labels
# # hog_descs = np.concatenate([hog_descs1, hog_descs2, hog_descs3, hog_descs4], axis=0)
# # labels = np.concatenate([labels1, labels2, labels3, labels4], axis=0)
# #
# # # Perform PCA on the HOG features
# # pca = PCA(n_components=100)
# # hog_descs_pca = pca.fit_transform(hog_descs)
# #
# # # Split data into training and testing sets
# # X_train, X_test, y_train, y_test = train_test_split(hog_descs_pca, labels, test_size=0.2, random_state=42)
# #
# # # Train Gradient Boosting Classifier
# # clf = GradientBoostingClassifier()
# # clf.fit(X_train, y_train)
# #
# # # Save the trained model
# # joblib.dump(clf, 'xgb_model.pkl')
# #
# # # Evaluate the model
# # accuracy = clf.score(X_test, y_test)
# # print(f"Accuracy: {accuracy}")
# import numpy as np
# import cv2
# from skimage.feature import hog
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingClassifier
# import joblib
# import os
#
# # Define paths to different folders containing different classes of brain tumor images
# project_folder = os.getcwd()
# folder1 = os.path.join(project_folder, "CV_CP_Dataset/glioma_tumor")
# folder2 = os.path.join(project_folder, "CV_CP_Dataset/meningioma_tumor")
# folder3 = os.path.join(project_folder, "CV_CP_Dataset/no_tumor")
# folder4 = os.path.join(project_folder, "CV_CP_Dataset/pituitary")
#
#
# # Function to extract HOG features from images in a folder
# def extract_hog_features(folder_path, label):
#     hog_descs = []
#     for filename in os.listdir(folder_path):
#         img = cv2.imread(os.path.join(folder_path, filename))
#         if img is not None:
#             resize = (200, 200)
#             img1 = cv2.resize(img, resize)
#             gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#             median_img = cv2.medianBlur(gray, 3)
#             fd, hog_image = hog(median_img, orientations=6, pixels_per_cell=(4, 4),
#                                 transform_sqrt=True, cells_per_block=(1, 1), visualize=True)
#             hog_descs.append(fd)
#     return hog_descs, [label] * len(hog_descs)
#
#
# # Extract HOG features and labels for each class
# hog_descs1, labels1 = extract_hog_features(folder1, 0)  # Glioma Tumor
# hog_descs2, labels2 = extract_hog_features(folder2, 1)  # Meningioma Tumor
# hog_descs3, labels3 = extract_hog_features(folder3, 2)  # No Tumor
# hog_descs4, labels4 = extract_hog_features(folder4, 3)  # Pituitary
#
# # Concatenate features and labels
# hog_descs = np.concatenate([hog_descs1, hog_descs2, hog_descs3, hog_descs4], axis=0)
# labels = np.concatenate([labels1, labels2, labels3, labels4], axis=0)
#
# # Perform PCA on the HOG features
# pca = PCA(n_components=100)
# hog_descs_pca = pca.fit_transform(hog_descs)
#
# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(hog_descs_pca, labels, test_size=0.2, random_state=42)
#
# # Train Gradient Boosting Classifier
# clf = GradientBoostingClassifier()
# clf.fit(X_train, y_train)
#
# # Save the trained model
# joblib.dump(clf, 'xgb_model.pkl')
#
# # Load the pre-trained model
# clf = joblib.load('xgb_model.pkl')
#
#
# # Function to extract HOG features from an image
# def extract_hog_features(img):
#     resize = (200, 200)
#     img1 = cv2.resize(img, resize)
#     gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     median_img = cv2.medianBlur(gray, 3)
#     fd, hog_image = hog(median_img, orientations=6, pixels_per_cell=(4, 4),
#                         transform_sqrt=True, cells_per_block=(1, 1), visualize=True)
#     return fd
#
#
# # Function for tumor detection from an uploaded image
# def detect_tumor(image_path):
#     img = cv2.imread(image_path)
#     if img is not None:
#         # Extract HOG features from the image
#         hog_features = extract_hog_features(img)
#
#         # Perform PCA on the HOG features
#         hog_features_pca = pca.transform(hog_features.reshape(1, -1))
#
#         # Predict the class (tumor type) using the pre-trained model
#         tumor_type = clf.predict(hog_features_pca)
#
#         # Glioma = 0, Meningioma = 1, No tumor = 2, Pituitary = 3
#         if tumor_type == 0:
#             return "Glioma Tumor"
#         elif tumor_type == 1:
#             return "Meningioma Tumor"
#         elif tumor_type == 2:
#             return "No Tumor"
#         elif tumor_type == 3:
#             return "Pituitary Tumor"
#     else:
#         return "Invalid Image"
#
#
# # Example usage:
# image_path = "testing.jpg"
# result = detect_tumor(image_path)
# print("Detected Tumor:", result)
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import os
import streamlit as st

# Define paths to different folders containing different classes of brain tumor images
project_folder = os.getcwd()
folder1 = os.path.join(project_folder, "CV_CP_Dataset/glioma_tumor")
folder2 = os.path.join(project_folder, "CV_CP_Dataset/meningioma_tumor")
folder3 = os.path.join(project_folder, "CV_CP_Dataset/no_tumor")
folder4 = os.path.join(project_folder, "CV_CP_Dataset/pituitary")


# Function to extract HOG features from images in a folder
def extract_hog_features(folder_path, label):
    hog_descs = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            resize = (200, 200)
            img1 = cv2.resize(img, resize)
            gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            median_img = cv2.medianBlur(gray, 3)
            fd, hog_image = hog(median_img, orientations=6, pixels_per_cell=(4, 4),
                                transform_sqrt=True, cells_per_block=(1, 1), visualize=True)
            hog_descs.append(fd)
    return hog_descs, [label] * len(hog_descs)


# Extract HOG features and labels for each class
hog_descs1, labels1 = extract_hog_features(folder1, 0)  # Glioma Tumor
hog_descs2, labels2 = extract_hog_features(folder2, 1)  # Meningioma Tumor
hog_descs3, labels3 = extract_hog_features(folder3, 2)  # No Tumor
hog_descs4, labels4 = extract_hog_features(folder4, 3)  # Pituitary

# Concatenate features and labels
hog_descs = np.concatenate([hog_descs1, hog_descs2, hog_descs3, hog_descs4], axis=0)
labels = np.concatenate([labels1, labels2, labels3, labels4], axis=0)

# Perform PCA on the HOG features
pca = PCA(n_components=100)
hog_descs_pca = pca.fit_transform(hog_descs)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(hog_descs_pca, labels, test_size=0.2, random_state=42)

# Train Gradient Boosting Classifier
clf = GradientBoostingClassifier()
clf.fit(X_train, y_train)

# Save the trained model
joblib.dump(clf, 'xgb_model.pkl')

# Load the pre-trained model
clf = joblib.load('xgb_model.pkl')


# Function to extract HOG features from an image
def extract_hog_features(img):
    resize = (200, 200)
    img1 = cv2.resize(img, resize)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    median_img = cv2.medianBlur(gray, 3)
    fd, hog_image = hog(median_img, orientations=6, pixels_per_cell=(4, 4),
                        transform_sqrt=True, cells_per_block=(1, 1), visualize=True)
    return fd


# Function for tumor detection from an uploaded image
def detect_tumor(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        # Extract HOG features from the image
        hog_features = extract_hog_features(img)

        # Perform PCA on the HOG features
        hog_features_pca = pca.transform(hog_features.reshape(1, -1))

        # Predict the class (tumor type) using the pre-trained model
        tumor_type = clf.predict(hog_features_pca)

        # Glioma = 0, Meningioma = 1, No tumor = 2, Pituitary = 3
        if tumor_type == 0:
            return "Glioma Tumor"
        elif tumor_type == 1:
            return "Meningioma Tumor"
        elif tumor_type == 2:
            return "No Tumor"
        elif tumor_type == 3:
            return "Pituitary Tumor"
    else:
        return "Invalid Image"


# Streamlit App
st.title("Brain Tumor Detection")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Detect Tumor'):
        # Perform tumor detection
        tumor_type = detect_tumor(img)
        st.success(f"Detected Tumor: {tumor_type}")
