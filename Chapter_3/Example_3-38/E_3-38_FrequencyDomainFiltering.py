# -*- coding: utf-8 -*-
"""
------------------------
版权声明：
本书内部包含的代码示例、算法和技术解释是受到知识产权法律保护的。
这些代码示例和相关内容仅用于学习和教育目的，以帮助读者更好地理解书中的概念和知识。
版权归属于书籍的作者或权利人所有。
    这些代码示例和技术解释的使用受到以下限制：
        代码示例仅用于学习和教育用途。读者可以结合书籍内容，在非商业的环境中使用这些代码示例，进行学习、实验和练习。
        代码示例不得用于商业用途，包括但不限于出售、分发以获取利润、嵌入商业软件或产品中。
        如需使用、修改和分发代码示例，其根据GNU Affero通用公共许可证(AGPL)3.0版本授权，请遵循源代码可用性与网络互动条款。
        任何对代码示例的修改、衍生或重新分发，应该在适当的情况下保留原作者的权利声明，并在代码中进行明确标注。
        代码示例和技术解释的使用不得侵犯任何第三方的知识产权，包括但不限于专利、商标、版权等。
        本书作者和出版社对于读者因使用这些代码示例导致的任何损失或风险概不负责。
请尊重知识产权，遵守以上声明，合理使用本书中的代码示例和相关内容。
------------------------
版权归属于：清华大学出版社 and 《计算机视觉与图像处理》作者
------------------------
【例3-38】在一幅图上进行频域滤波，通过设置不同的理想滤波器的特定频率范围分别进行低通滤波与高通滤波，并显示出来。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 计算图像进行离散傅里叶变换时的最优转换大小
h, w = image.shape[:2]
dft_h = cv2.getOptimalDFTSize(h)
dft_w = cv2.getOptimalDFTSize(w)
# 边界扩充
img_padded = cv2.copyMakeBorder(image,0,dft_h - h,0,dft_w - w, cv2.BORDER_CONSTANT, 0)
# 进行离散傅里叶变换
dft = cv2.dft(np.float32(img_padded), flags = cv2.DFT_COMPLEX_OUTPUT)
# 将离散傅里叶变换结果移到中心
dft_shift = np.fft.fftshift(dft)
# 计算频域图像的幅值谱
magnitude_spectrum = np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
# 确定理想滤波器尺寸
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
# 创建理想低通滤波器
mask_lowpass = np.zeros((rows, cols, 2), np.uint8)
mask_lowpass[crow - 50:crow + 50, ccol - 50:ccol + 50] = 1
# 创建理想高通滤波器（翻转低通滤波器）
mask_highpass = 1 - mask_lowpass
# 应用低通滤波器
filtered_dft_shift_lowpass = dft_shift * mask_lowpass
# 应用高通滤波器
filtered_dft_shift_highpass = dft_shift * mask_highpass
# 计算滤波后的频域图像的幅值谱
filtered_magnitude_spectrum_lowpass = np.log(cv2.magnitude(filtered_dft_shift_lowpass[:, :, 0], filtered_dft_shift_lowpass[:, :, 1]) + 1e-10)
filtered_magnitude_spectrum_highpass = np.log(cv2.magnitude(filtered_dft_shift_highpass[:, :, 0], filtered_dft_shift_highpass[:, :, 1]) + 1e-10)
# 低通滤波后的逆傅立叶变换
filtered_idft_shift_lowpass = np.fft.ifftshift(filtered_dft_shift_lowpass)
filtered_image_lowpass = cv2.idft(filtered_idft_shift_lowpass)
filtered_image_lowpass = cv2.magnitude(filtered_image_lowpass[:, :, 0], filtered_image_lowpass[:, :, 1])
# 高通滤波后的逆傅立叶变换
filtered_idft_shift_highpass = np.fft.ifftshift(filtered_dft_shift_highpass)
filtered_image_highpass = cv2.idft(filtered_idft_shift_highpass)
filtered_image_highpass = cv2.magnitude(filtered_image_highpass[:, :, 0], filtered_image_highpass[:, :, 1])
# 显示图像和幅值谱
plt.subplot(331), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(filtered_image_lowpass, cmap='gray')
plt.title('Lowpass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow((mask_lowpass*255)[:,:,0], cmap='gray')
plt.title('Lowpass Filtered'), plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(filtered_magnitude_spectrum_lowpass, cmap='gray')
plt.title('Lowpass Filtered Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(337), plt.imshow(filtered_image_highpass, cmap='gray')
plt.title('Highpass Filtered Image'), plt.xticks([]), plt.yticks([])
plt.subplot(338), plt.imshow((mask_highpass*255)[:,:,0], cmap='gray')
plt.title('Lowpass Filtered'), plt.xticks([]), plt.yticks([])
plt.subplot(339), plt.imshow(filtered_magnitude_spectrum_highpass, cmap='gray')
plt.title('Highpass Filtered Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()