clc;

I = imread('input_gray_1.png');
image = rgb2gray(I);
figure(1);
imshow(image);
title('Original Image');

PSF = fspecial('gaussian',5,5);
Blurred = imfilter(image,PSF,'symmetric','conv');
figure(2);
imshow(Blurred);
title('Gaussian Blurred');
%imwrite(Blurred,'D:\课程作业\数字图像处理\第一次作业\image\wiener_lucy\0_001\高斯模糊.png');

noise_Blurred = imnoise(Blurred,"gaussian",0,0.1);
figure(3);
imshow(noise_Blurred);
title('Noise Gaussian Blurred');
imwrite(noise_Blurred,'D:\课程作业\数字图像处理\第一次作业\image\wiener_lucy\0_1\高斯模糊加噪声.png');

lucy = deconvlucy(noise_Blurred,PSF,5);
figure(4);
imshow(lucy);
title('Restored Image, NUMIT = 5');
imwrite(lucy,'D:\课程作业\数字图像处理\第一次作业\image\wiener_lucy\0_1\5次迭代.png');

lucy = deconvlucy(noise_Blurred,PSF,20);
figure(5);
imshow(lucy);
title('Restored Image, NUMIT = 20');
imwrite(lucy,'D:\课程作业\数字图像处理\第一次作业\image\wiener_lucy\0_1\20次迭代.png');