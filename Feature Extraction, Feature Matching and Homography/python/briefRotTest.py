import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from skimage.feature import plot_matched_features
from matchPics import matchPics

# Q3.5: Image processing and feature matching for rotations

# Load and convert image to grayscale if it's in color
img_path = '../data/cv_cover.jpg'
image = cv2.imread(img_path)
if len(image.shape) != 2:  # Check if the image has more than one channel (i.e., color)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

# Initialize histogram to store the number of matches for each rotation angle
match_histogram = np.zeros(36)

# Angles for visualizing results
visualize_angles = [0, 90, 180]

# Loop over 36 rotation angles (from 0 to 350 degrees in steps of 10 degrees)
for angle_idx in range(36):
    rotation_angle = angle_idx * 10
    print(f"Processing rotation: {rotation_angle}°", end=" | ")

    # Rotate the image by the current angle
    rotated_img = rotate(image, rotation_angle, reshape=False)

    # Convert the rotated image back to uint8 (scipy rotate returns float)
    rotated_img = np.uint8(rotated_img)

    # Compute matches, locations of keypoints, and their descriptors
    matches, keypoints1, keypoints2 = matchPics(image, rotated_img)

    # Store the number of matches for the current angle
    match_histogram[angle_idx] = len(matches)

    # Visualize matches for selected angles (0°, 90°, 180°)
    if rotation_angle in visualize_angles:
        # Convert images to RGB for visualization
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        rotated_img_rgb = cv2.cvtColor(rotated_img, cv2.COLOR_GRAY2RGB)

        # Plot the matched features between the original and rotated image
        fig, ax = plt.subplots()
        plot_matched_features(
            ax=ax,
            image0=img_rgb,
            image1=rotated_img_rgb,
            keypoints0=keypoints1,
            keypoints1=keypoints2,
            matches=matches,
            matches_color='r',  # Red lines for matches
            only_matches=True   # Show only matched points
        )
        ax.set_title(f"Matched Features at {rotation_angle}°")
        plt.show()

# Plot the histogram showing the number of matches vs. rotation angle
plt.bar(range(0, 360, 10), match_histogram)
plt.xlabel('Rotation Angle (°)')
plt.ylabel('Number of Matches')
plt.title('Number of Matches vs. Rotation Angle')
plt.xticks(range(0, 360, 30))  # Set x-axis ticks at 30° intervals
plt.show()
