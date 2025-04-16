import numpy as np
import cv2
import random

def computeH(points1, points2):
    """
    This function computes the homography between two sets of points.
    It assumes points1 and points2 are corresponding points from two images.
    """

    # Ensure that the points have the correct shape (N, 2)
    if points1.shape[1] != 2 or points2.shape[1] != 2:
        raise ValueError("Each point must have two coordinates (x, y).")

    # Check if there are at least 4 points
    num_points = points1.shape[0]
    if num_points < 4:
        raise ValueError("At least 4 point correspondences are required to compute the homography.")

    # Check if the number of points in both sets match
    if num_points != points2.shape[0]:
        raise ValueError("The number of points in both sets must be the same.")

    # Create the system of equations matrix A
    matrix_A = []
    for i in range(num_points):
        x1, y1 = points1[i]  # Extract x, y coordinates from the first image
        x2, y2 = points2[i]  # Extract x, y coordinates from the second image 

        # Append two rows for each point correspondence
        matrix_A.append([-x2, -y2, -1, 0, 0, 0, x1 * x2, x1 * y2, x1])
        matrix_A.append([0, 0, 0, -x2, -y2, -1, x2 * y1, y1 * y2, y1])

    # Convert matrix_A to a NumPy array
    matrix_A = np.array(matrix_A)

    # Perform Singular Value Decomposition (SVD) on matrix_A
    _, _, Vt = np.linalg.svd(matrix_A)

    # The solution to the homography is the last row of Vt, reshaped into a 3x3 matrix
    homography = Vt[-1, :].reshape(3, 3)

    # Normalize the homography matrix so that the bottom-right value is 1
    homography /= homography[-1, -1]

    # Sanity check: Ensure the homography matrix is valid
    if np.any(np.isnan(homography)) or np.any(np.isinf(homography)):
        raise ValueError("The computed homography contains NaN or Inf values, which indicates an error in computation.")

    # Ensure the homography matrix is 3x3
    if homography.shape != (3, 3):
        raise ValueError("The computed homography should be a 3x3 matrix.")

    return homography

def computeH_norm(x1, x2):
    # Q3.7 Normalize the coordinates
    # Ensure that input points are valid
    if x1.shape[1] != 2 or x2.shape[1] != 2:
        raise ValueError("Input points should be in the form of [x, y] coordinates (2 columns).")
    if x1.shape[0] < 4 or x2.shape[0] < 4:
        raise ValueError("At least 4 points are required to compute a homography.")

    # Check if both point sets have the same number of points
    if x1.shape[0] != x2.shape[0]:
        raise ValueError("The number of points in both sets must be equal.")

    # Normalize x1 (cv_cover points)
    mean_x1 = np.mean(x1, axis=0)
    max_dist_x1 = np.max(np.linalg.norm(x1 - mean_x1, axis=1))
    
    # Check for zero distance (identical points)
    if max_dist_x1 == 0:
        raise ValueError("All points in the first set are identical, normalization cannot proceed.")
    
    scale_x1 = np.sqrt(2) / max_dist_x1
    T1 = np.array([[scale_x1, 0, -scale_x1 * mean_x1[0]], 
                   [0, scale_x1, -scale_x1 * mean_x1[1]], 
                   [0, 0, 1]])

    x1_norm = np.column_stack((x1, np.ones(x1.shape[0]))) @ T1.T  # Add homogeneous coordinate
    x1_norm = x1_norm[:, :2]  # Remove the homogeneous coordinate

    # Normalize x2 (cv_desk points)
    mean_x2 = np.mean(x2, axis=0)
    max_dist_x2 = np.max(np.linalg.norm(x2 - mean_x2, axis=1))
    
    # Check for zero distance (identical points)
    if max_dist_x2 == 0:
        raise ValueError("All points in the second set are identical, normalization cannot proceed.")
    
    scale_x2 = np.sqrt(2) / max_dist_x2
    T2 = np.array([[scale_x2, 0, -scale_x2 * mean_x2[0]], 
                   [0, scale_x2, -scale_x2 * mean_x2[1]], 
                   [0, 0, 1]])

    x2_norm = np.column_stack((x2, np.ones(x2.shape[0]))) @ T2.T  # Add homogeneous coordinate
    x2_norm = x2_norm[:, :2]  # Remove the homogeneous coordinate

    # Compute homography on normalized points
    H_norm = computeH(x1_norm, x2_norm)
    
    # Denormalize the homography matrix
    H2to1 = np.linalg.inv(T1) @ H_norm @ T2
    
    return H2to1

def computeH_ransac(x1, x2):
    # Q3.8 Use RANSAC
    """
    Compute the best fitting homography given a list of matching points using RANSAC.
    """
    max_inliers_count = 0
    optimal_homography = None
    dist_threshold = 4
    num_iterations = 2000

    def select_random_points():
        """
        Select 4 random points for homography estimation.
        """
        random_indices = random.sample(range(x1.shape[0]), 4)
        return x1[random_indices], x2[random_indices]

    def find_inliers(homography_matrix, src_points, dst_points, threshold):
        """
        Compute which points are inliers given a homography and a threshold distance.
        """
        dst_points_homogeneous = np.hstack((dst_points, np.ones((dst_points.shape[0], 1))))
        projected_src_points = (homography_matrix @ dst_points_homogeneous.T).T
        projected_src_points /= projected_src_points[:, 2].reshape(-1, 1)

        distances = np.linalg.norm(src_points - projected_src_points[:, :2], axis=1)
        return distances < threshold

    for _ in range(num_iterations):
        # Randomly select 4 correspondences
        sample_x1, sample_x2 = select_random_points()

        # Compute homography for this sample
        estimated_homography = computeH(sample_x1, sample_x2)

        # Get inliers
        inlier_mask = find_inliers(estimated_homography, x1, x2, dist_threshold)

        # Count inliers and update best homography if needed
        num_inliers = np.sum(inlier_mask)
        if num_inliers > max_inliers_count:
            max_inliers_count = num_inliers
            optimal_homography = estimated_homography
            best_inlier_mask = inlier_mask

    return optimal_homography, best_inlier_mask

def compositeH(H2to1, template, img):
    # Q3.9 Create composite image by warping the template onto img
    """
    Create composite image by warping the template onto img using the homography H2to1.
    """
    # Check if homography is 3x3
    if H2to1.shape != (3, 3):
        raise ValueError("The homography matrix H2to1 must be a 3x3 matrix.")
    
    # Check if template and img have the same number of channels (both should be 3 or both 1)
    if len(template.shape) != len(img.shape):
        raise ValueError("The number of channels of the template and image do not match.")
    
    H_inv = np.linalg.inv(H2to1)  # Inverse homography
    
    # Get the height and width of the template image
    template_height, template_width = template.shape[:2]
    
    # Check if the template has valid dimensions
    if template_height == 0 or template_width == 0:
        raise ValueError("Template image has invalid dimensions (height or width is zero).")
    
    # Check if the target image has valid dimensions
    img_height, img_width = img.shape[:2]
    if img_height == 0 or img_width == 0:
        raise ValueError("Target image has invalid dimensions (height or width is zero).")

    # Create a mask for the template (all pixels set to 255, representing a white mask)
    mask = np.ones((template_height, template_width), dtype=np.uint8) * 255
    
    # Warp the mask and the template image using the inverse homography
    warped_mask = cv2.warpPerspective(mask, H_inv, (img_width, img_height))
    warped_template = cv2.warpPerspective(template, H_inv, (img_width, img_height))
    
    # Composite the template onto the target image wherever the mask is non-zero
    img[warped_mask > 0] = warped_template[warped_mask > 0]
    
    return img
