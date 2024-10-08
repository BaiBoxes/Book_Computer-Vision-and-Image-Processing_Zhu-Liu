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
【例3-32】随机生成5×5的卷积核，然后用cv2.filter2D函数对图像进行互相关操作与卷积操作，并显示结果。
------------------------
"""


import cv2
import numpy as np


# 创建一个随机5x5的卷积核
random_kernel = np.random.rand(5, 5)
random_kernel = random_kernel / np.sum(random_kernel)
# 打印生成的卷积核
print("随机生成的卷积核：")
print(random_kernel)
# 读取图像
image = cv2.imread('image/Example-Bridge_gray.jpg')
# 对图像执行互相关操作
correlation_result = cv2.filter2D(image, -1, random_kernel)
# 对图像执行卷积操作
convolution_result = cv2.filter2D(image, -1, cv2.flip(random_kernel, -1))
# 显示原始图像、互相关结果和卷积结果
cv2.imshow('Original Image', image)
cv2.imshow('Correlation Result', correlation_result)
cv2.imshow('Convolution Result', convolution_result)
cv2.waitKey(0)
cv2.destroyAllWindows()