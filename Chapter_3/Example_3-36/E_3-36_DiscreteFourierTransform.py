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
【例3-36】将一幅图进行最优离散傅立叶变换，并显示出来。
------------------------
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt


# 读取灰度图像
img = cv2.imread('image/Example-Bridge_gray.jpg', flags=cv2.IMREAD_GRAYSCALE)
# 计算图像进行离散傅里叶变换时的最优转换大小
h, w = img.shape[:2]
dft_h = cv2.getOptimalDFTSize(h)
dft_w = cv2.getOptimalDFTSize(w)
# 边界扩充
img_padded = cv2.copyMakeBorder(img,0,dft_h - h,0,dft_w - w, cv2.BORDER_CONSTANT, 0)
# 进行离散傅里叶变换
dft = cv2.dft(np.float32(img_padded), flags = cv2.DFT_COMPLEX_OUTPUT)
# 将离散傅里叶变换结果移到中心
dft_shift = np.fft.fftshift(dft)
# 计算离散傅里叶变换的幅值谱
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
# 将原始图像和频谱都显示出来
plt.figure(figsize=(9, 3))
plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()