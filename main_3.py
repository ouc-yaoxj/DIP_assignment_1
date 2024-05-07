import cv2
import numpy as np
import os

from Side_Window_Filtering import SW_gaussian_filter
from Noise_Reduction import add_gaussian_noise, NoiseReduction

noise_level_list = [0.001, 0.01, 0.1]
window_size_list = [3, 5, 7, 9]
save_root_path = '/mnt/4T/yxj/digital_image_processing/Assignment_1/image/side_window_filter'

image = cv2.imread('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/RGB.jpg', 0)

for noise_level in noise_level_list:
    noisy_image = add_gaussian_noise(image, mean=0, var=noise_level)
    cv2.imwrite(os.path.join(save_root_path, f'add_noise_image_N{noise_level}.png'), noisy_image)
    for window_size in window_size_list:
        noise_reducer = NoiseReduction(image=noisy_image)
        gaus_output_image = noise_reducer.gaussian_filter(k_size=(window_size, window_size))
        SW_output_image = SW_gaussian_filter(image=noisy_image, r=window_size//2)
        cv2.imwrite(os.path.join(save_root_path, f'gaus_output_image_N{noise_level}_W{window_size}.png'),
                    gaus_output_image)
        cv2.imwrite(os.path.join(save_root_path, f'SW_output_image_N{noise_level}_W{window_size}.png'),
                    SW_output_image)