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
【例3-37】将一幅图通过傅里叶变换，实现卷积，并显示出来。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取图像
img = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 卷积核
kernel = np.ones((5,5),np.float32)/25
# 计算傅里叶变换的最优大小
dft_h = cv2.getOptimalDFTSize(img.shape[0])
dft_w = cv2.getOptimalDFTSize(img.shape[1])
# 扩展图像
img_padded = cv2.copyMakeBorder(img, 0, dft_h - img.shape[0], 0, dft_w - img.shape[1], cv2.BORDER_CONSTANT, 0)
# 扩展核
kernel_padded = cv2.copyMakeBorder(kernel, 0, dft_h - kernel.shape[0], 0, dft_w - kernel.shape[1], cv2.BORDER_CONSTANT, 0)
# 傅里叶变换
dft_img = cv2.dft(np.float32(img_padded), flags = cv2.DFT_COMPLEX_OUTPUT)
dft_kernel = cv2.dft(np.float32(kernel_padded), flags = cv2.DFT_COMPLEX_OUTPUT)
# 频域中点乘
dft_img_conv = cv2.mulSpectrums(dft_img, dft_kernel, cv2.DFT_COMPLEX_OUTPUT)
# 逆变换
img_conv = cv2.idft(dft_img_conv)
img_conv = cv2.magnitude(img_conv[:,:,0],img_conv[:,:,1])
# 将原始图像和卷积后图像都显示出来
plt.figure(figsize=(9, 3))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(img_conv, cmap = 'gray')
plt.title('Convolution Image'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()