import cv2
import numpy as np
import matplotlib.pyplot as plt
from Histogram_Equalization import histogram_equalization

def ahe(image, L_max=255, window_size=32, affect_size=16):
    out_image = image.copy()

    H, W = image.shape

    offset = int((window_size - affect_size) / 2)

    # window_size = affect_size + 2 * affect的边界到window的边界的距离
    # 如果window滑动到最后刚好能够与图片的边界重合，则要求 (图片的高 - 2 * affect的边界到window的边界的距离)
    # 能够整除affect区域的高，且这个整除出来的数也是在高的维度上进行AHE的次数
    if (H - 2 * offset) % affect_size == 0:
        rows = int((H - 2 * offset) / affect_size)
    # 如果不能整除，那就在图片的下边界再多进行一次AHE，不过这多出来的一次AHE需要特殊处理
    else:
        rows = int((H - 2 * offset) / affect_size + 1)

    # 在宽维度上与高维度上同理
    if (W - 2 * offset) % affect_size == 0:
        cols = int((W - 2 * offset) / affect_size)
    else:
        cols = int((W - 2 * offset) / affect_size + 1)

    # window在图片上滑动进行AHE
    for i in range(rows):
        for j in range(cols):
            # 第(i, j) 个 window 所对应的 affect 区域坐标
            affect_start_i, affect_end_i = i * affect_size + offset, (i + 1) * affect_size + offset
            affect_start_j, affect_end_j = j * affect_size + offset, (j + 1) * affect_size + offset

            # 第(i, j) 个 window 的区域坐标
            window_start_i, window_end_i = i * affect_size, i * affect_size + window_size
            window_start_j, window_end_j = j * affect_size, j * affect_size + window_size

            # 在window区域进行HE变换
            window_image = image[window_start_i:window_end_i, window_start_j:window_end_j]
            window_he_image = histogram_equalization(window_image, L_max=L_max)

            # 边界的window处理
            if i == 0:
                out_image[
                    window_start_i:affect_start_i, window_start_j:window_end_j
                ] = window_he_image[0:offset, :]
            elif i >= rows - 1:
                out_image[
                    affect_end_i:window_end_i, window_start_j:window_end_j
                ] = window_he_image[affect_end_i-window_start_i:window_end_i-window_start_i, :]

            if j == 0:
                out_image[
                    window_start_i:window_end_i, window_start_j:affect_start_j
                ] = window_he_image[:, 0:offset]
            elif j >= cols - 1:
                out_image[
                    window_start_i:window_end_i, affect_end_j:window_end_j
                ] = window_he_image[:, affect_end_j-window_start_j:window_end_j-window_start_j]

            out_image[
                affect_start_i:affect_end_i, affect_start_j:affect_end_j
            ] = window_he_image[offset:offset+affect_size, offset:offset+affect_size]

    return out_image

if __name__ == '__main__':

    image = cv2.imread('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/AHE/RGB_1.jpg',0)
    cv2.imshow('input_gray', image)
    # plt.hist(image.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    # plt.savefig('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/AHE/input_hist_1.png')
    # plt.close()

    out_image = ahe(image, window_size=32, affect_size=8)
    # cv2.imwrite('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/AHE/output_result_16x16_4.png', out_image)
    cv2.imshow('output_result', out_image)
    # plt.hist(out_image.ravel(), bins=255, rwidth=0.8, range=(0, 255))
    # plt.savefig('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/AHE/output_hist_5.png')
    # plt.close()
    cv2.waitKey(0)