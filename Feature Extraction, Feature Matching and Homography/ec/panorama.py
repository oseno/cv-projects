import numpy as np
import cv2
import matplotlib.pyplot as plt
from matchPics import matchPics, matchPicsWithSIFT
from planarH import computeH_ransac

# First, load the images. These are the left and right images for panorama creation.
left_image = cv2.imread('../data/l.png')
right_image = cv2.imread('../data/r.png')

# Check if both images loaded correctly
if left_image is None:
    print("Error: Left image could not be loaded. Check the file path.")
else:
    print("Left image loaded successfully.")

if right_image is None:
    print("Error: Right image could not be loaded. Check the file path.")
else:
    print("Right image loaded successfully.")

# Convert both images to grayscale for feature matching (as it simplifies the process)
left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
print("Converted images to grayscale.")

# Now, let's find feature matches between the two images
# We use the matchPics function, which will return the best matches and locations
locs_left, locs_right = matchPicsWithSIFT(left_gray, right_gray)
print("Feature matching complete. Best matches found.")

# Create a match array where we can map corresponding points from both images
matches = np.array([[i, i] for i in range(len(locs_left))])
print(f"Created a match array of size {matches.shape}.")

# Get the locations of the matched points
matched_points_left = locs_left[matches[:, 0]]
matched_points_right = locs_right[matches[:, 1]]
print("Extracted matched points.")

# Now, let's compute the homography matrix using RANSAC to handle outliers
H, inliers = computeH_ransac(matched_points_left, matched_points_right)
print("Homography matrix computed using RANSAC.")

# Extract the height and width of both images for later use when warping
h_left, w_left = left_image.shape[:2]
h_right, w_right = right_image.shape[:2]
print(f"Extracted image dimensions: left image ({h_left}, {w_left}), right image ({h_right}, {w_right}).")

# Warp the right image onto the left image using the computed homography matrix
# The size of the output image will be large enough to fit both images side by side
warped_right = cv2.warpPerspective(right_image, H, (w_left + w_right, h_left))
print("Warped the right image onto the left image.")

# Create a blank canvas for the panorama with the appropriate size
# It will hold both the left and warped right images
panorama = np.zeros((h_left, w_left + w_right, 3), dtype=np.uint8)
print("Created a blank canvas for the panorama.")

# Place the left image on the canvas (no transformation needed here)
panorama[:h_left, :w_left] = left_image
print("Placed the left image on the panorama canvas.")

# Now we need to blend the overlapping region of the two images
# We'll use a simple averaging method for the overlap
overlap_region = warped_right[:h_left, :w_left] > 0  # Find where the warped right image is not black
panorama[:h_left, w_left:][overlap_region] = (panorama[:h_left, :w_left][overlap_region] + warped_right[:h_left, :w_left][overlap_region]) // 2
print("Blended the overlapping region of the two images.")

# Place the non-overlapping part of the warped right image onto the panorama
# This part doesn't overlap with the left image and can be placed directly
panorama[:h_right, w_left:] = warped_right[:h_right, w_left:]
print("Placed the non-overlapping part of the right image onto the panorama.")

# Save the resulting panorama to a file
cv2.imwrite('../results/panorama.png', panorama)
print("Panorama saved as 'panorama.png'.")

# Now, let's display the panorama
# Convert the image from BGR to RGB so that it's displayed correctly using matplotlib
plt.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
plt.title('Generated Panorama')
plt.axis('off')  # Hide the axis for a cleaner view
plt.show()
print("Displayed the generated panorama.")