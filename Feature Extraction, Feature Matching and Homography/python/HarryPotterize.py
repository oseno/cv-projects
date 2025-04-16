import numpy as np
import cv2
import skimage.io
import skimage.color
from planarH import compositeH, computeH_ransac
from matchPics import matchPicsWithSIFT

# Load images
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
hp_cover = cv2.imread('../data/hp_cover.jpg')

# Check if images are loaded correctly
if cv_cover is None:
    print("Error: cv_cover image could not be loaded.")
    exit()

if cv_desk is None:
    print("Error: cv_desk image could not be loaded.")
    exit()

if hp_cover is None:
    print("Error: hp_cover image could not be loaded.")
    exit()

# Resize the Harry Potter image to match the book cover dimensions
if cv_cover.shape[0] != hp_cover.shape[0] or cv_cover.shape[1] != hp_cover.shape[1]:
    print(f"Resizing Harry Potter cover image from {hp_cover.shape[:2]} to {cv_cover.shape[:2]}")
hp_cover_resized = cv2.resize(hp_cover, (cv_cover.shape[1], cv_cover.shape[0]))

# Convert images to grayscale
gray_cv = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)
gray_desk = cv2.cvtColor(cv_desk, cv2.COLOR_BGR2GRAY)

# Compute feature matches between cv_cover and cv_desk
matched_points1, matched_points2 = matchPicsWithSIFT(gray_cv, gray_desk)

# Check if there are enough matched points
if len(matched_points1) < 4:
    print("Error: Not enough matching points found to compute homography.")
    exit()

# Extract matched keypoints
match_indices = np.array([[i, i] for i in range(len(matched_points1))])  
points_from_cover = matched_points1[match_indices[:, 0]]  # Keypoints from cv_cover
points_from_desk = matched_points2[match_indices[:, 1]]  # Keypoints from cv_desk

# Compute homography using RANSAC
homography_matrix, inliers = computeH_ransac(points_from_cover, points_from_desk)

# Check if the homography matrix is valid
if np.any(np.isnan(homography_matrix)) or np.any(np.isinf(homography_matrix)):
    print("Error: Invalid homography matrix.")
    exit()

# Composite the warped HP cover with the desk image
composite_image = compositeH(homography_matrix, hp_cover_resized, cv_desk)

# Check if the composite image has valid dimensions
if composite_image.shape[0] == 0 or composite_image.shape[1] == 0:
    print("Error: Composite image has invalid dimensions.")
    exit()

# Saving the final image as a PNG file
save_status = cv2.imwrite('harry_potter_final.png', composite_image)
if not save_status:
    print("Error: Failed to save the composite image.")
    exit()

# Final message
print(f"Composite image created successfully and saved as 'harry_potter_final.png'.")
