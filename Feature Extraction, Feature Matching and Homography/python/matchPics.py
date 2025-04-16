import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# This is the function to match features between two images using BRIEF descriptors
def matchPics(I1, I2):
    # First, convert the images to grayscale if they are not grayscale (i.e., 2 channels)
    if len(I1.shape) != 2:  # Check if the image has more than one channel (i.e., color)
        image1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    else:
        image1 = I1  # If already grayscale, use the original image
    
    if len(I2.shape) != 2:  # Check if the image has more than one channel (i.e., color)
        image2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    else:
        image2 = I2  # If already grayscale, use the original image

    # Now, we need to detect corners in both images (features to track)
    sigma = 0.35  # This is a threshold for corner detection
    corners1 = corner_detection(image1, sigma)  # Detect corners in image1
    corners2 = corner_detection(image2, sigma)  # Detect corners in image2
    
    # Next, compute the BRIEF descriptors for the detected corners in both images
    descriptors1, corners1 = computeBrief(image1, corners1)  # Get BRIEF descriptors for image1
    descriptors2, corners2 = computeBrief(image2, corners2)  # Get BRIEF descriptors for image2

    # Now, match the BRIEF descriptors between the two images
    # Set match ratio to 0.6, I have also tried 1
    best_matches = briefMatch(descriptors1, descriptors2, 0.6)  # Match descriptors

    # Finally, return the best matches and the corresponding corner locations
    return best_matches, corners1, corners2

# This is a new method for matching features using SIFT
def matchPicsWithSIFT(img1, img2):
    # Convert images to grayscale if they are not already
    if len(img1.shape) != 2:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) != 2:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    # Initialize the SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2_gray, None)

    # Use a Brute-Force Matcher (deterministic) instead of FLANN
    bf_matcher = cv2.BFMatcher()
    matches = bf_matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply the ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Fixed threshold for consistency
            good_matches.append(m)

    # Extract the locations of the keypoints that match
    points1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    points2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    # Print the total number of good matches found
    print("The total good matches found: " + str(len(good_matches)))

    # Return the matched points
    return points1, points2
