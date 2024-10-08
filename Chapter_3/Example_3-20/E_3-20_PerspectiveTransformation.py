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
【例3-20】将一幅图通过cv2.warpPerspective函数进行透视变换，只改变三个顶点位置，并显示出原图像与变换后的图像。
------------------------
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 读取原始图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 定义原图中四个角点的坐标
original_points = np.float32([[0, 0], [image.shape[1], 0], [0, image.shape[0]], [image.shape[1], image.shape[0]]])
# 将原图的三个角点分别向内平移一定距离
transformed_points = np.float32([[0, 0], [image.shape[1] - 50, 50], [50, image.shape[0] - 50], [image.shape[1] - 50, image.shape[0] - 50]])
# 计算透视变换矩阵
perspective_matrix = cv2.getPerspectiveTransform(original_points, transformed_points)
# 进行透视变换
warped_image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))
# 显示原图像和透视变换后的图像
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
plt.title('Warped Image'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()