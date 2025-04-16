import numpy as np
import cv2
import os
from planarH import computeH_ransac, compositeH
from matchPics import matchPicsWithSIFT

# Write script for Q4.1

# Ensure the "results" directory exists
output_dir = "../results"
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
print(f"Output directory created: {output_dir}")

# Paths to the book cover and videos
book_cover_path = '../data/cv_cover.jpg'

# Check if the input files exist
if not os.path.exists(book_cover_path):
    print(f"Error: Book cover image not found at {book_cover_path}")
    exit()
else:
    print(f"Book cover image found at {book_cover_path}")

book_video_path = '../data/book.mov'

if not os.path.exists(book_video_path):
    print(f"Error: Book video not found at {book_video_path}")
    exit()
else:
    print(f"Book video found at {book_video_path}")

ar_video_path = '../data/ar_source.mov'

if not os.path.exists(ar_video_path):
    print(f"Error: AR video not found at {ar_video_path}")
    exit()
else:
    print(f"AR video found at {ar_video_path}")

# Load the book cover image
book_cover_img = cv2.imread(book_cover_path)
print("Book cover image loaded successfully.")

# Load the videos
book_video_cap = cv2.VideoCapture(book_video_path)
# Check if the videos are opened successfully
if not book_video_cap.isOpened():
    print("Error: Failed to load book.mov")
    exit()
else:
    print("book.mov loaded successfully")

ar_video_cap = cv2.VideoCapture(ar_video_path)

if not ar_video_cap.isOpened():
    print("Error: Failed to load ar_source.mov")
    exit()
else:
    print("ar_source.mov loaded successfully")

# Get video properties
fps = int(book_video_cap.get(cv2.CAP_PROP_FPS))
frame_width = int(book_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(book_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video properties: FPS={fps}, Frame width={frame_width}, Frame height={frame_height}")

# Define the output video path
output_video_path = os.path.join(output_dir, "ar_video_output.avi")

# Initialize video writer for the output
output_video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
print(f"Output video writer initialized. Saving to {output_video_path}")

# Convert the book cover to grayscale for feature matching
book_cover_gray = cv2.cvtColor(book_cover_img, cv2.COLOR_BGR2GRAY)
print("Book cover image converted to grayscale.")

# Process each frame of the videos
frame_count = 0
while True:
    # Read frames from both videos
    ret_book_frame, book_frame = book_video_cap.read()
    ret_ar_frame, ar_frame = ar_video_cap.read()

    # Break the loop if either video ends
    if not ret_book_frame or not ret_ar_frame:
        print("End of one or both videos. Exiting loop.")
        break

    # Convert the book frame to grayscale
    book_frame_gray = cv2.cvtColor(book_frame, cv2.COLOR_BGR2GRAY)
    print("Book frame converted to grayscale.")

    # Compute feature matches between the book cover and the current book frame
    locs_book_cover, locs_book_frame = matchPicsWithSIFT(book_cover_gray, book_frame_gray)
    print(f"Found {len(locs_book_cover)} feature matches between book cover and book frame.")

    matches = np.array([[i, i] for i in range(len(locs_book_cover))])

    # Extract matched points
    matched_points_book_cover = locs_book_cover[matches[:, 0]]
    matched_points_book_frame = locs_book_frame[matches[:, 1]]

    # Compute homography using RANSAC
    homography_matrix, inlier_indices = computeH_ransac(matched_points_book_cover, matched_points_book_frame)
    print("Homography matrix computed using RANSAC.")

    # Resize the AR frame to match the book cover dimensions
    resized_ar_frame = cv2.resize(ar_frame, (book_cover_img.shape[1], book_cover_img.shape[0]))
    print("AR frame resized to match book cover dimensions.")

    # Composite the AR frame onto the book frame using the homography matrix
    composite_frame = compositeH(homography_matrix, resized_ar_frame, book_frame)
    print("AR frame composited onto book frame.")

    # Write the composite frame to the output video
    output_video_writer.write(composite_frame)
    print("Composite frame written to output video.")

    # Display the composite frame
    cv2.imshow("AR Output Result", composite_frame)

    # Exit condition (press 'q' to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Pressed 'q'. Exiting.")
        break

# Release video objects and close windows
book_video_cap.release()
ar_video_cap.release()
output_video_writer.release()
cv2.destroyAllWindows()
print(f"AR video result has been saved to: {output_video_path}")

# Function to check if the video has been saved successfully
def check_video_saved(output_video_path):
    """Function to check if the saved video can be opened successfully."""
    video_check_cap = cv2.VideoCapture(output_video_path)

    # Check if the video was opened successfully
    if not video_check_cap.isOpened():
        video_check_cap.release()
        return False

    # Release the video capture object
    video_check_cap.release()
    return True

# Check if the saved video can be opened
if not check_video_saved(output_video_path):
    print(f"Error: Could not open the saved video at {output_video_path}")
    exit()

# Confirm that the video was successfully opened
print("Video saved and can be opened successfully.")

# Release the video capture object and close windows
cv2.destroyAllWindows()
