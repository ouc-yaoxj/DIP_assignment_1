clc;

I = imread('input_gray_1.png');
image = rgb2gray(I);
figure(1);
imshow(image);
title('Original Image');

image_double = im2double(image);

PSF = fspecial('gaussian',5,5);
Blurred = imfilter(image,PSF,'symmetric','conv');
figure(2);
imshow(Blurred);
title('Gaussian Blurred');

noise_Blurred = imnoise(Blurred,"gaussian",0,0.001);
figure(3);
imshow(noise_Blurred);
title('Noise Gaussian Blurred');

signal_var = var(image_double(:));
NSR = 0.1 / signal_var;
wnr = deconvwnr(Blurred,PSF,0);
figure(4);
imshow(wnr)
title('Restored Blurred Image')
imwrite(wnr,'D:\课程作业\数字图像处理\第一次作业\image\wiener_diffieren_NSR\nsr_0.png');
