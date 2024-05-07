import cv2
import numpy as np
import scipy
from skimage import color, data, restoration
from Noise_Reduction import add_gaussian_noise, NoiseReduction

noise_level = {'low':0.001, 'middle':0.01, 'high':0.1}
select_noise = 'low'
image = cv2.imread('/mnt/4T/yxj/digital_image_processing/Assignment_1/image/RGB.jpg', 0)
noise_reducer = NoiseReduction(image=image)
blurry_image = noise_reducer.gaussian_filter(k_size=(5, 5))
cv2.imshow('vague', blurry_image)

noisy_image = add_gaussian_noise(blurry_image, mean=0, var=noise_level[select_noise])
cv2.imshow('noise', noisy_image)

# noisy_image = noisy_image.astype(np.float32)
# wiener_image = scipy.signal.wiener(noisy_image, 5)
# wiener_image = wiener_image.astype(np.uint8)
# cv2.imshow('wiener', wiener_image)



cv2.waitKey(0)