import cv2
import numpy as np
import scipy
import matplotlib.pyplot as plt
from Noise_Reduction import add_gaussian_noise, NoiseReduction

def SW_gaussian_kernel(gaus_kernel, h_start=0, h_end=1, w_start=0, w_end=1):
    SW_kernel = np.zeros(gaus_kernel.shape)
    SW_kernel[h_start:h_end, w_start:w_end] = gaus_kernel[h_start:h_end, w_start:w_end]
    SW_kernel = SW_kernel / np.sum(SW_kernel)
    return SW_kernel

def SW_gaussian_filter(image, r, sigma=0):
    gaus_kernel = cv2.getGaussianKernel(ksize=2*r+1, sigma=sigma)
    gaus_kernel = gaus_kernel.dot(gaus_kernel.T)
    gaus_kernel = gaus_kernel.astype(np.float32)

    L_kernel = SW_gaussian_kernel(gaus_kernel=gaus_kernel, h_end=2*r+1, w_end=r+1)
    R_kernel = SW_gaussian_kernel(gaus_kernel=gaus_kernel, h_end=2*r+1, w_start=r,
                                  w_end=2*r+1)
    U_kernel = L_kernel.T
    D_kernel = U_kernel[::-1]
    NW_kernel = SW_gaussian_kernel(gaus_kernel=gaus_kernel, h_end=r+1, w_end=r+1)
    NE_kernel = SW_gaussian_kernel(gaus_kernel=gaus_kernel, h_end=r+1, w_start=r,
                                   w_end=2*r+1)
    SW_kernel = NW_kernel[::-1]
    SE_kernel = NE_kernel[::-1]

    all_SW_gaus_kernels = [L_kernel, R_kernel, U_kernel, D_kernel,
                           NW_kernel, NE_kernel, SW_kernel, SE_kernel]

    H = image.shape[0]
    W = image.shape[1]
    dis = np.zeros([len(all_SW_gaus_kernels), image.shape[0], image.shape[1]])
    conv_result = np.zeros([len(all_SW_gaus_kernels), image.shape[0], image.shape[1]])
    image_padding = np.pad(image, (r, r), "edge")

    for i, kernel in enumerate(all_SW_gaus_kernels):
        temp_conv_result = scipy.signal.correlate2d(image_padding, kernel, "valid")
        conv_result[i] = temp_conv_result
        dis[i] = abs(temp_conv_result - image)

    ind_min_dis = np.argmin(dis, axis=0)
    output_image = conv_result[ind_min_dis, np.arange(H)[None, :, None], np.arange(W)[None, None, :]]

    output_image = output_image.astype(np.uint8)

    return output_image.squeeze(0)

if __name__ == "__main__":
    noise_level = {'low': 0.001, 'middle': 0.01, 'high': 0.1}

    select_noise = 'low'

    image = cv2.imread('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/RGB.jpg', 0)
    noisy_image = add_gaussian_noise(image, mean=0, var=noise_level[select_noise])

    noise_reducer = NoiseReduction(image=noisy_image)
    gaus_output_image = noise_reducer.gaussian_filter(k_size=(7, 7))
    output_image = SW_gaussian_filter(noisy_image, 7)

    cv2.imshow("noise", noisy_image)
    cv2.imshow("gaus_output", gaus_output_image)
    cv2.imshow("SW_output", output_image)
    cv2.waitKey(0)