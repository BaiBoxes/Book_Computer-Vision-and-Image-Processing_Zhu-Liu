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
【例3-4】通过利用随机生成20×20的矩阵，分别生成二值图、灰度图、彩色图与带透明通道的彩色图，并打印出数据，观察它们的数据特点。
------------------------
"""

import cv2
import numpy as np

# 随机生成20x20的矩阵,作为灰度图
gray_image = np.random.randint(0, 256, (20, 20), dtype=np.uint8)
# 生成二值图
binary_image = np.where(gray_image > 127, 255, 0)
binary_image = cv2.convertScaleAbs(binary_image, alpha=255.0)
# 生成彩色图
color_image = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
# 打印数据
print("Binary Image:")
print(binary_image)
print("\nGray Image:")
print(gray_image)
print("\nColor Image:")
print(color_image)
# 创建窗口
cv2.namedWindow('Binary Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Color Image', cv2.WINDOW_NORMAL)
# 显示图像
cv2.imshow('Gray Image', gray_image)
cv2.imshow('Binary Image', binary_image)
cv2.imshow('Color Image', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()