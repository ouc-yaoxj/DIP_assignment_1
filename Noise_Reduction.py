import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import os

def add_gaussian_noise(image, mean, var):
    image = image / 255
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out_image = image + noise
    out_image = np.clip(out_image, 0.0, 1.0) * 255
    out_image = out_image.astype(np.uint8)
    return out_image

# f为相似窗口的半径, t为搜索窗口的半径, h为高斯函数平滑参数(一般取为相似窗口的大小)
def make_kernel(f):
    kernel = np.zeros((2 * f + 1, 2 * f + 1), np.float32)
    for d in range(1, f + 1):
        kernel[f - d:f + d + 1, f - d:f + d + 1] += (1.0 / ((2 * d + 1) ** 2))

    return kernel / kernel.sum()

class NoiseReduction:
    def __init__(self, image):
        self.image = image

    def gaussian_filter(self, k_size, sigmaX=0):
        output_image = cv2.GaussianBlur(self.image, ksize=k_size, sigmaX=sigmaX)
        return output_image

    def bilateral_filter(self, d, sigmaColor, sigmaSpace):
        output_image = cv2.bilateralFilter(self.image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        return output_image

    def wavelet_denoise(self, wavelet='sym4', level=3, threshold=0.05):
        coeffs = pywt.wavedec2(self.image, wavelet=wavelet, level=level)

        coeffs_list = []
        for i in range(1, len(coeffs)):
            coeffs_list.append(list(coeffs[i]))

        for i in range(len(coeffs_list)):
            for j in range(len(coeffs_list[i])):
                coeffs_list[i][j] = pywt.threshold(coeffs_list[i][j], threshold * np.max(coeffs_list[i][j]))

        re_coeffs = [coeffs[0]]
        for i in range(len(coeffs_list)):
            re_coeffs.append(tuple(coeffs_list[i]))

        output_image = pywt.waverec2(coeffs=re_coeffs, wavelet=wavelet)
        output_image = np.clip(output_image, 0.0, 255.0)
        return output_image.astype(np.uint8)

    def NLmeans_filter(self, f, t, h):
        H, W = self.image.shape
        output_image = np.zeros((H, W), np.uint8)
        pad_length = f + t
        src_padding = np.pad(self.image, (pad_length, pad_length), mode='symmetric').astype(np.float32)
        kernel = make_kernel(f)
        h2 = h * h

        for i in range(0, H):
            for j in range(0, W):
                i1 = i + f + t
                j1 = j + f + t
                W1 = src_padding[i1 - f:i1 + f + 1, j1 - f:j1 + f + 1]  # 领域窗口W1
                w_max = 0
                aver = 0
                weight_sum = 0
                # 搜索窗口
                for r in range(i1 - t, i1 + t + 1):
                    for c in range(j1 - t, j1 + t + 1):
                        if (r == i1) and (c == j1):
                            continue
                        else:
                            W2 = src_padding[r - f:r + f + 1, c - f:c + f + 1]  # 搜索区域内的相似窗口
                            Dist2 = (kernel * (W1 - W2) * (W1 - W2)).sum()
                            w = np.exp(-Dist2 / h2)
                            if w > w_max:
                                w_max = w
                            weight_sum += w
                            aver += w * src_padding[r, c]
                aver += w_max * src_padding[i1, j1]  # 自身领域取最大的权重
                weight_sum += w_max
                output_image[i, j] = aver / weight_sum
        return output_image


if __name__ == '__main__':
    save_root_path = '/mnt/4T/yxj/digital_image_processing/Assignment_1/image/reduce_noise_1'
    noise_level = {'low':0.001, 'middle':0.01, 'high':0.1}

    select_noise = 'high'

    image = cv2.imread('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/RGB.jpg', 0)
    noisy_image = add_gaussian_noise(image, mean=0, var=noise_level[select_noise])
    # plt.hist(image.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    # plt.savefig(os.path.join(save_root_path, f'{select_noise}/original_image_hist.png'))
    # plt.close()
    # plt.hist(noisy_image.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    # plt.savefig(os.path.join(save_root_path, f'{select_noise}/{select_noise}_noise_image_hist.png'))
    # plt.close()
    cv2.imwrite(os.path.join(save_root_path, f'{select_noise}/{select_noise}_noise_image.png'),
                noisy_image)

    noise_reducer = NoiseReduction(image=noisy_image)
    output_image_gaussian = noise_reducer.gaussian_filter(k_size=(3, 3))
    output_image_bilateral = noise_reducer.bilateral_filter(d=3, sigmaColor=75, sigmaSpace=75)
    output_image_wavelet = noise_reducer.wavelet_denoise()
    output_image_nlm = noise_reducer.NLmeans_filter(2, 5, 10)
    output_image_dict = {
        'gaussian': output_image_gaussian,
        'bilateral': output_image_bilateral,
        'wavelet': output_image_wavelet,
        'nlm': output_image_nlm,
    }

    for k, v in output_image_dict.items():
        # plt.hist(v.ravel(), bins=255, rwidth=0.8, range=(0, 255))
        # plt.savefig(os.path.join(save_root_path, f'{select_noise}/{k}_output_image_hist.png'))
        # plt.close()
        cv2.imwrite(os.path.join(save_root_path, f'{select_noise}/{k}_output_image.png'), v)