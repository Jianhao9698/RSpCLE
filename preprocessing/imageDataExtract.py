import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from utils.SSIM_MultiThread import SSIM_parallel

"""
Extract frames from a video file and save them as images.
maybe saved as .npy file or separate images.

The first trial would be save these images as (channels) a .npy file,
to be processed by a simple U-Net model, for channels containing information
could be extracted by the model.

To extract stabilized frames and get rid of noises, 
we could use the optical flow method to stabilize the frames.
"""

# Open the video file
video_path = '../data/MCM_68/STAINED/CONFOCAL/Run_1/2021_07_14_14_56_19.avi'
cap = cv2.VideoCapture(video_path)
save_path = '../data/MCM_68/STAINED/CONFOCAL/Run_1/saved_frames/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Parameters for frame extraction
frame_skip_interval = 4  # Extract every ... frame, 4 by default
input_sequence_length = 24  # Number of frames to extract to stack as .npy file
saved_frame_counter = 0
frame_counter = 0
frames = []

# Loop through the video frames
count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    if count % frame_skip_interval == 0:
        # Convert frame to grayscale and append to the list
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

print("Len frames: ", len(frames), "Total to save: ", input_sequence_length)
# Convert list of frames to a 3D numpy array
video_array = np.array(frames)
num_frames, height, width = video_array.shape
diff_threshold = 10  # Threshold for mean difference to consider frames stable
"""
0.9: assume 1
0.8: 58.2% of 0.9
0.7: 38.2% of 0.9
"""
stable_frames_diff = []

for i in range(1, num_frames):
    diff = cv2.absdiff(video_array[i], video_array[i - 1])
    diff_mean = np.mean(diff)
    if diff_mean < diff_threshold:
        stable_frames_diff.append(i)
print(f"Stable frames based on frame difference: {len(stable_frames_diff)}")
#
# # # Now filter out repeated stabilized frames using similar method
diff_threshold_stabilized = 0.7  # Threshold for SSIM
filtered_stable_frames = []
for i in range(1, len(stable_frames_diff)):
    score, _ = ssim(video_array[stable_frames_diff[i]], video_array[stable_frames_diff[i - 1]], full=True)
    # If SSIM score is below the threshold, consider the frame unique
    if score < diff_threshold_stabilized:
        filtered_stable_frames.append(stable_frames_diff[i])
print(f"\n\nFiltered stable frames: {len(filtered_stable_frames)}")
# % Filter the frames to save, may replace the previous SSIM part
# Calculate SSIM scores between each pair of stable frames
print("Calculating SSIM scores between stable frames...")
#
# ssim_matrix = SSIM_parallel(video_array, stable_frames_diff)
# print("Finished calculating SSIM scores.")
# print("SSIM matrix shape: ", ssim_matrix.shape)
# # Find the max values of rows in the SSIM matrix and their indices
# max_values, max_indices = np.max(ssim_matrix, axis=1), np.argmax(ssim_matrix, axis=1) #Indices of stable_frames_diff
# print("Max values shape: ", max_values.shape, "Max indices shape: ", max_indices.shape)
#
# sorted_indices = np.argsort(max_values)
# sorted_max_values = max_values[sorted_indices]
# sorted_max_indices = max_indices[sorted_indices]
#
# print(f"Original max values: {max_values}")
# print(f"Original indices: {max_indices}")
# print(f"Sorted max values: {sorted_max_values}")
# print(f"Reordered indices: {sorted_max_indices}")
# # Find the indices of the most similar frames based on SSIM scores
# # most_similar_pairs = np.unravel_index(np.argsort(-ssim_matrix, axis=None), ssim_matrix.shape)
# # sorted_pairs = list(zip(most_similar_pairs[0], most_similar_pairs[1]))
#
# # Filter out the most similar frames



#%% Maybe can calculate the CNR to select images with possible information



#%%
# Save the frames as separate images for visualization
for frame_num in filtered_stable_frames:
    frame_counter += 1
    # Save the frame as an image in grayscale
    frame_filename = f"frame_{frame_counter}.png"
    cv2.imwrite(os.path.join(save_path, frame_filename), video_array[frame_num])
    saved_frame_counter += 1
    print('\r', f"Saved {frame_filename}", end='', flush=True)
    if saved_frame_counter == input_sequence_length:
        break

# Save the selected frames as a numpy array (.npy file)
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from utils.SSIM_MultiThread import SSIM_parallel

"""
Extract frames from a video file and save them as images.
maybe saved as .npy file or separate images.

The first trial would be save these images as (channels) a .npy file,
to be processed by a simple U-Net model, for channels containing information
could be extracted by the model.

To extract stabilized frames and get rid of noises, 
we could use the optical flow method to stabilize the frames.
"""

# Open the video file
video_path = '../data/MCM_68/STAINED/CONFOCAL/Run_1/2021_07_14_14_56_19.avi'
cap = cv2.VideoCapture(video_path)
save_path = '../data/MCM_68/STAINED/CONFOCAL/Run_1/saved_frames/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Parameters for frame extraction
frame_skip_interval = 10  # Extract every ... frame, 4 by default
input_sequence_length = 24  # Number of frames to extract to stack as .npy file
saved_frame_counter = 0
frame_counter = 0
frames = []

# Loop through the video frames
count = 0
while True:
    ret, frame = cap.read()
    count += 1
    if not ret:
        break
    if count % frame_skip_interval == 0:
        # Convert frame to grayscale and append to the list
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)

print("Len frames: ", len(frames), "Total to save: ", input_sequence_length)
# Convert list of frames to a 3D numpy array
video_array = np.array(frames)
num_frames, height, width = video_array.shape
diff_threshold = 10  # Threshold for mean difference to consider frames stable
"""
0.9: assume 1
0.8: 58.2% of 0.9
0.7: 38.2% of 0.9
"""
stable_frames_diff = []

for i in range(1, num_frames):
    diff = cv2.absdiff(video_array[i], video_array[i - 1])
    diff_mean = np.mean(diff)
    if diff_mean < diff_threshold:
        stable_frames_diff.append(i)
print(f"Stable frames based on frame difference: {len(stable_frames_diff)}")

# # Now filter out repeated stabilized frames using similar method
# diff_threshold_stabilized = 0.7  # Threshold for SSIM
# filtered_stable_frames = []
# for i in range(1, len(stable_frames_diff)):
#     score, _ = ssim(video_array[stable_frames_diff[i]], video_array[stable_frames_diff[i - 1]], full=True)
#     # If SSIM score is below the threshold, consider the frame unique
#     if score < diff_threshold_stabilized:
#         filtered_stable_frames.append(stable_frames_diff[i])
# print(f"\n\nFiltered stable frames: {len(filtered_stable_frames)}")
#% Filter the frames to save, may replace the previous SSIM part
# Calculate SSIM scores between each pair of stable frames
print("Calculating SSIM scores between stable frames...")

ssim_matrix = SSIM_parallel(video_array, stable_frames_diff)
print("Finished calculating SSIM scores.")
print("SSIM matrix shape: ", ssim_matrix.shape)
# Find the max values of rows in the SSIM matrix and their indices
max_values, max_indices = np.max(ssim_matrix, axis=1), np.argmax(ssim_matrix, axis=1) #Indices of stable_frames_diff
print("Max values shape: ", max_values.shape, "Max indices shape: ", max_indices.shape)

sorted_indices = np.argsort(max_values)
sorted_max_values = max_values[sorted_indices]
sorted_max_indices = max_indices[sorted_indices]

print(f"Original max values: {max_values}")
print(f"Original indices: {max_indices}")
print(f"Sorted max values: {sorted_max_values}")
print(f"Reordered indices: {sorted_max_indices}")
# Find the indices of the most similar frames based on SSIM scores
# most_similar_pairs = np.unravel_index(np.argsort(-ssim_matrix, axis=None), ssim_matrix.shape)
# sorted_pairs = list(zip(most_similar_pairs[0], most_similar_pairs[1]))

# Filter out the most similar frames



#%% Maybe can calculate the CNR to select images with possible information



#%%
# Save the frames as separate images for visualization
for frame_num in filtered_stable_frames:
    frame_counter += 1
    # Save the frame as an image in grayscale
    frame_filename = f"frame_{frame_counter}.png"
    cv2.imwrite(os.path.join(save_path, frame_filename), video_array[frame_num])
    saved_frame_counter += 1
    print('\r', f"Saved {frame_filename}", end='', flush=True)
    if saved_frame_counter == input_sequence_length:
        break

# Save the selected frames as a numpy array (.npy file)
selected_frames = video_array[filtered_stable_frames[:input_sequence_length]]
np.save(os.path.join(save_path, 'MCM_68.npy'), selected_frames)
print(f"Saved {input_sequence_length} frames as {selected_frames}.npy")

# Release the video capture object
cap.release()
cv2.destroyAllWindows()


# Release the video capture object
cap.release()
cv2.destroyAllWindows()
