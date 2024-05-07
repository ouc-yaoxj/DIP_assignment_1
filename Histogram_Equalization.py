import cv2
import numpy as np
import matplotlib.pyplot as plt

# 全局直方图均衡化函数
def histogram_equalization(image, L_max=255):
    H, W = image.shape

    N = H * W * 1

    out_image = image.copy()

    sum_pixels = 0.

    for i in range(0, 255):
        index = np.where(image == i)
        sum_pixels += len(image[index])
        # HE算法的公式，L_max代表最大灰度级
        s = L_max * (sum_pixels / N)
        out_image[index] = s

    out_image = out_image.astype(np.uint8)

    return out_image

if __name__ == '__main__':

    image = cv2.imread('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/AHE/RGB_1.jpg',0)
    cv2.imshow('input_gray', image)
    plt.hist(image.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/HE/input_hist_5.png')
    plt.close()

    out_image = histogram_equalization(image)
    # out_image = cv2.equalizeHist(image)
    cv2.imshow('output_result', out_image)
    plt.hist(out_image.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    plt.savefig('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/HE/output_hist_5.png')
    plt.close()
    cv2.waitKey(0)