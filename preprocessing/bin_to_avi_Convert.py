import os
import numpy as np
import cv2


"""
Currently not working properly, using MATLAB to do processing instead
It's thereby deprecated now, will use MATLAB.
"""


# Function to read binary file and convert to video
# def convert_bin_to_avi(folder_path, output_folder, frame_rate=120):
#     # List all .bin files in the folder
#     bin_files = [f for f in os.listdir(folder_path) if f.endswith('.bin')]
#
#     for bin_file in bin_files:
#         print("Processing file:", bin_file)
#         # Set file paths
#         input_file_path = os.path.join(folder_path, bin_file)
#         output_file_name = os.path.splitext(bin_file)[0] + '.avi'
#         output_file_path = os.path.join(output_folder, output_file_name)
#
#         # Open the binary file
#         with open(input_file_path, 'rb') as file:
#             # Read in image dimensions
#             x_size = np.fromfile(file, dtype=np.int32, count=1)[0]
#             y_size = np.fromfile(file, dtype=np.int32, count=1)[0]
#
#             # Read in image type
#             image_type_number = np.fromfile(file, dtype=np.int32, count=1)[0]
#             if image_type_number == 7:  # 16-bit unsigned integer
#                 data_type = np.uint16
#                 byte_depth = 2
#
#             # Read in some other header items (skip the rest of the header)
#             border_size = np.fromfile(file, dtype=np.int32, count=1)[0]
#             line_length = np.fromfile(file, dtype=np.int32, count=1)[0]
#             _ = np.fromfile(file, dtype=np.int32, count=3)  # Skipping unknown fields
#             num_frames = np.fromfile(file, dtype=np.int32, count=1)[0]
#
#             # Calculate size of each frame (they are multiples of 512 bytes for efficient disk access)
#             frame_size = int(np.ceil(line_length * y_size * byte_depth / 512) * 512)
#
#             # Prepare video writer
#             writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (x_size, y_size), isColor=False)
#
#             # Process each frame
#             for i in range(num_frames):
#                 # Skip to frame of interest
#                 file.seek(512 - border_size * byte_depth + frame_size * i, os.SEEK_SET)
#
#                 # Read in the frame data
#                 image_data = np.fromfile(file, dtype=data_type, count=line_length * y_size)
#
#                 # Form a 2D image
#                 image_shaped = np.reshape(image_data, (line_length, y_size))
#
#                 # Remove border and transpose array
#                 image = image_shaped[border_size:x_size + border_size, 0:y_size].T
#
#                 # Convert to 8-bit grayscale
#                 scale = image.max() / 255 if image.max() != 0 else 1
#                 image_8bit = (image / scale).astype(np.uint8)
#
#                 # Write frame to output video
#                 writer.write(image_8bit)
#
#             # Release video writer
#             writer.release()
#
#
# # Set working directory and parameters
# # Now use MCM_68 as an example to run
# root_path = "./data/MCM_68/STAINED/CONFOCAL"
# folders = [
#     f'{root_path}/Run_1',
#     f'{root_path}/Run_2',
#     f'{root_path}/Run_3'
# ]
#
# output_folder = f'{root_path}/output_videos'
# os.makedirs(output_folder, exist_ok=True)
#
# # Convert each folder's .bin files to .avi
# for folder in folders:
#     convert_bin_to_avi(folder, output_folder)
#
# print("Results saved")
