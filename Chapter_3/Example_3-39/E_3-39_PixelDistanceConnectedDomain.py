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
【例3-39】在一幅墙体裂缝图上进行距离变换与连通域分析，并将结果显示出来。
------------------------
"""


import cv2
import matplotlib.pyplot as plt


# 读取图像
image = cv2.imread('image/Example-WallCracks.jpg', cv2.IMREAD_GRAYSCALE)
# 二值化图像
ret, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
# 计算欧几里得距离变换
distance_transform_euclidean = cv2.distanceTransform(binary_image, cv2.DIST_L2, 5)
# 计算曼哈顿距离变换
distance_transform_manhattan = cv2.distanceTransform(binary_image, cv2.DIST_L1, 5)
# 计算切比雪夫距离变换
distance_transform_chebyshev = cv2.distanceTransform(binary_image, cv2.DIST_C, 5)
# 二值化图像并翻转（将白色和黑色像素互换）
labeled_image = cv2.bitwise_not(binary_image)
# 进行连通域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(labeled_image, ltype=cv2.CV_16U)
# 显示结果图像
plt.subplot(231), plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(binary_image, cmap='gray')
plt.title('Binary Image'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(distance_transform_euclidean, cmap='jet')
plt.title('Euclidean Distance'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(distance_transform_manhattan, cmap='jet')
plt.title('Manhattan Distance'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(distance_transform_chebyshev, cmap='jet')
plt.title('Chebyshev Distance'), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(labels, cmap='gray')
plt.title('Connected Components'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()