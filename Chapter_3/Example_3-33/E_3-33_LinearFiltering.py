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
【例3-33】在一幅图上产生高斯噪声后，分别用cv2.blur函数进行均值滤波、cv2.boxFilter函数进行方框滤波、
            cv2.GaussianBlur函数进行高斯滤波与sepFilter2D函数进行可分离滤波，然后显示出滤波结果并分析。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


def add_gaussian_noise(image, mean=0, sigma=20):
    """ 添加高斯噪音的函数 """
    noisy_image = image.copy()
    gauss = np.random.normal(mean, sigma, noisy_image.shape)
    noisy_image = noisy_image + gauss
    noisy_image = cv2.convertScaleAbs(noisy_image)
    return noisy_image

# 读取图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 添加高斯噪音
noise_image = add_gaussian_noise(image)
# 均值滤波
blur_image = cv2.blur(noise_image, (5, 5))
# 方框滤波
box_image = cv2.boxFilter(noise_image, -1, (5, 5))
# 高斯滤波
gaussian_image = cv2.GaussianBlur(noise_image, (5, 5), 1)
# 可分离滤波
sep_kernel = cv2.getGaussianKernel(5, 1)
sep_image = cv2.sepFilter2D(noise_image, -1, sep_kernel, sep_kernel)
# 显示图像
plt.figure(num='Filtered Image', figsize=(6, 6))
# 显示原始图像
plt.subplot(231)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
# 显示带有高斯噪音的图像
plt.subplot(232)
plt.imshow(cv2.cvtColor(noise_image, cv2.COLOR_BGR2RGB))
plt.title('Noise Image')
# 显示均值滤波的图像
plt.subplot(233)
plt.imshow(cv2.cvtColor(blur_image, cv2.COLOR_BGR2RGB))
plt.title('Mean Filtered Image')
# 显示方框滤波的图像
plt.subplot(234)
plt.imshow(cv2.cvtColor(box_image, cv2.COLOR_BGR2RGB))
plt.title('Box Filtered Image')
# 显示高斯滤波的图像
plt.subplot(235)
plt.imshow(cv2.cvtColor(gaussian_image, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Filtered Image')
# 显示可分离滤波的图像
plt.subplot(236)
plt.imshow(cv2.cvtColor(sep_image, cv2.COLOR_BGR2RGB))
plt.title('Separable Filtered Image')
plt.show()