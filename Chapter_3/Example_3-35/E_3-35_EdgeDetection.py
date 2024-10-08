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
【例3-35】在一幅图上分别用Sobel算子、Scharr算子、Laplacian算子与Canny算法进行边缘检测，然后显示出边缘结果并分析。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 边缘检测使用Sobel算子
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobel_x**2 + sobel_y**2)
sobel = cv2.convertScaleAbs(sobel)  # 转换为uint8
# 边缘检测使用Scharr算子
scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
scharr = np.sqrt(scharr_x**2 + scharr_y**2)
scharr = cv2.convertScaleAbs(scharr)  # 转换为uint8
# 边缘检测使用Laplacian算子
laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
laplacian = cv2.convertScaleAbs(laplacian)  # 转换为uint8
# 边缘检测使用Canny算法
canny = cv2.Canny(image, 100, 200)
# 显示各种边缘检测结果
plt.figure(figsize=(12, 6))
plt.subplot(231)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(232)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel Edge Detection')
plt.subplot(233)
plt.imshow(scharr, cmap='gray')
plt.title('Scharr Edge Detection')
plt.subplot(234)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Edge Detection')
plt.subplot(235)
plt.imshow(canny, cmap='gray')
plt.title('Canny Edge Detection')
plt.tight_layout()
plt.show()