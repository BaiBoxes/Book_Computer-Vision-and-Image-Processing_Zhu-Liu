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
【例3-23】将图像用cv2.pyrDown函数与cv2.pyrUp函数构建3层的拉普拉斯金字塔，并将其中的高斯金字塔与拉普拉斯金字塔显示出来。
------------------------
"""


import cv2
import matplotlib.pyplot as plt


# 读取输入图像
input_image = cv2.imread('image/Example-Bridge.jpg')
# 创建高斯金字塔
gaussian_pyramid = [input_image]
for i in range(3):
    downsampled_image = cv2.pyrDown(gaussian_pyramid[-1])
    gaussian_pyramid.append(downsampled_image)
# 创建拉普拉斯金字塔
laplacian_pyramid = [gaussian_pyramid[3]]
for i in range(3, 0, -1):
    expanded_image = cv2.pyrUp(gaussian_pyramid[i])
    laplacian = cv2.subtract(gaussian_pyramid[i - 1], expanded_image)
    laplacian_pyramid.append(laplacian)
# 显示高斯金字塔和拉普拉斯金字塔
original_height, original_width = input_image.shape[:2]
# 创建一个8个子图的图像，两排显示
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# 显示高斯金字塔的各个层
for i in range(4):
    axes[0, i].imshow(cv2.cvtColor(gaussian_pyramid[i], cv2.COLOR_BGR2RGB))
    axes[0, i].set_title(f'Gaussian Layer {i}')
    axes[0, i].set_xticks([])
    axes[0, i].set_yticks([])
    axes[0, i].set_xlim(0, original_width)
    axes[0, i].set_ylim(original_height, 0)
# 显示拉普拉斯金字塔的各个层
for i in range(4):
    axes[1, i].imshow(cv2.cvtColor(laplacian_pyramid[3-i], cv2.COLOR_BGR2RGB))
    axes[1, i].set_title(f'Laplacian Layer {3-i}')
    axes[1, i].set_xticks([])
    axes[1, i].set_yticks([])
    axes[1, i].set_xlim(0, original_width)
    axes[1, i].set_ylim(original_height, 0)
plt.tight_layout()
plt.show()