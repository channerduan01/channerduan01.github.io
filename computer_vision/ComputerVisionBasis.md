#Introduction
计算机视觉是目前人工智能应用最为广泛的领域，例如 OCR (Optical Character Recognition), Biometrics 以及正在探索中的 Automated Driving.

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



