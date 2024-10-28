import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor

# Function to calculate SSIM between two frames

def SSIM_parallel(video_array, stable_frames, set_diag_zero=True):
    num_stable_frames = len(stable_frames)
    ssim_matrix = np.zeros((num_stable_frames, num_stable_frames))
    def calculate_ssim(i, j):
        frame1 = video_array[stable_frames[i]]
        frame2 = video_array[stable_frames[j]]
        score, _ = ssim(frame1, frame2, full=True)
        return i, j, score


    # Use ThreadPoolExecutor to parallelize SSIM computation
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_ssim, i, j)
                   for i in range(num_stable_frames) for j in range(i, num_stable_frames)]

        for future in futures:
            i, j, score = future.result()
            ssim_matrix[i, j] = score
            ssim_matrix[j, i] = score

    if set_diag_zero:
        np.fill_diagonal(ssim_matrix, 0)

    return ssim_matrix