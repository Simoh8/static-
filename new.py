import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Define constants
IMG_SIZE = (100, 100)
N_COMPONENTS = 50

# Load and preprocess the face images
def preprocess_images(image_paths):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.normalize(img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        images.append(img)
    return np.array(images)

# Load the face images and labels
face_paths = [...] # list of paths to face images
nonface_paths = [...] # list of paths to non-face images
X_face = preprocess_images(face_paths)
X_nonface = preprocess_images(nonface_paths)
X = np.vstack((X_face, X_nonface))
y = np.hstack((np.ones(len(X_face)), np.zeros(len(X_nonface))))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute the mean face from the training images
mean_face = np.mean(X_train, axis=0)

# Subtract the mean face from each image in the training set to get the mean-centered images
X_train_centered = X_train - mean_face

# Compute the eigenvectors and eigenvalues of the covariance matrix of the mean-centered images
cov_mat = np.cov(X_train_centered.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# Keep only the top k eigenvectors corresponding to the largest eigenvalues
idx = eig_vals.argsort()[::-1]
eig_vecs = eig_vecs[:,idx][:,:N_COMPONENTS]

# Project the mean-centered training images onto the k eigenvectors to obtain the eigenfaces
eigenfaces = np.dot(X_train_centered, eig_vecs)

# Represent each training image as a linear combination of the eigenfaces
weights = np.dot(X_train_centered, eigenfaces)

# To recognize a new face image, first preprocess it and mean-center it
def recognize_face(img_path):
    img = preprocess_images([img_path])[0]
    img_centered = img - mean_face

    # Project the image onto the k eigenvectors to obtain its eigenface representation
    img_eigenface = np.dot(img_centered, eig_vecs)

    # Compare the eigenface representation with those of the training images using a distance metric
    distances = np.linalg.norm(weights - img_eigenface, axis=1)
    min_idx = np.argmin(distances)

    # Return the label and distance of the closest training image
    return y_train[min_idx], distances[min_idx]

# Evaluate the model on the testing set using precision, recall, and F1-score
y_pred = []
distances = []
for img_path in X_test:
    label, dist = recognize_face(img_path)
    y_pred.append(label)
    distances.append(dist)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision: {:.3f}, Recall: {:.3f}, F1-score: {:.3f}".format(precision, recall, f1))
