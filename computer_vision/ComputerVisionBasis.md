#Introduction
计算机视觉是目前人工智能应用最为广泛的领域，例如 OCR (Optical Character Recognition), Biometrics 以及正在探索中的 Automated Driving.

#Contrast Enhance 对比增强
我在 traffic sign 的 benchmark 原始数据集处理时，相关论文都采用了类似技术，对原始图片质量进行提升。基于 对抗色觉论（opponent color theory），其主要思路是将原始图片映射到 Lab（L 是 black vs. white 标识光照，a 是 red vs. green，b 是 blue vs. yellow ） 色彩空间中，对 L、a、b 三个维度进行 normalize，然后再映射会到 RGB 色彩空间中，使得原始图像的光照因素影响淡化，色彩对比度也加强。

#Image Filterring
##Background Subtraction
###Optimal Threshold
###Median filter

##Statistical Filters
###Median filter, Mean filter, Mode filter

##Popular Image Process
###Gaussian filter == Isotropic Diffusion
###Anisotropic Diffusion - Image Enhance Method

##Wavelet - extract local patterns
###Haar
###Gabor filter

#Low-level Feature Extraction
##Edge Detection
###Roberts
###Prewitt
###Sobel
###Canny
###Laplacian of Gaussain (LoG)
##Optical Flow

#High-level Feature Extraction
##Hough Transform
##Template Matching
##Active Contours

#Descriptor
原生的图像像素数据存在于非常不稳定的高纬度空间, $1024\times768=7.9\times10^5$, 我们希望把图像信息映射到另一个空间中来解析。这个新的空间最好具有不变形 Invariant
##Desirable Properties
Complete, Congruent, Compact, Invariant
##Invariance
Translation, Rotation, Scale  
###Moments
###SIFT
###HOG（Histogram of Oriented Gradient）
###LBP（Local Binary Pattern）

#Detection

#Recognition 
##inter-distances and intra-distances
##ROC

#Classification



