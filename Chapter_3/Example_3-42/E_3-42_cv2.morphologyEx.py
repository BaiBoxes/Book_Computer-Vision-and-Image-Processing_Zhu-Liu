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
【例3-42】在一幅墙体裂缝图上用cv2.morphologyEx函数进行腐蚀操作、膨胀操作、形态学开运算、
        形态学闭运算、形态学梯度运算、形态学顶帽运算、形态学黑帽运算与击中击不中变换，并将结果显示出来。
------------------------
"""


import cv2
import matplotlib.pyplot as plt


# 读取图像
image = cv2.imread('image/Example-WallCracks.jpg', cv2.IMREAD_GRAYSCALE)
# 二值化图像并翻转（将白色和黑色像素互换）
ret, binary_image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
binary_image = cv2.bitwise_not(binary_image)
# 生成5x5的矩形核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# 腐蚀
erosion = cv2.morphologyEx(binary_image, cv2.MORPH_ERODE, kernel)
# 膨胀
dilation = cv2.morphologyEx(binary_image, cv2.MORPH_DILATE, kernel)
# 开运算
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
# 闭运算
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
# 梯度运算
gradient = cv2.morphologyEx(binary_image, cv2.MORPH_GRADIENT, kernel)
# 顶帽运算
tophat = cv2.morphologyEx(binary_image, cv2.MORPH_TOPHAT, kernel)
# 黑帽运算
blackhat = cv2.morphologyEx(binary_image, cv2.MORPH_BLACKHAT, kernel)
# 击中击不中变换
hitmiss = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, kernel)
# 显示结果
titles = ['Original Image', 'Binary Image', 'Erosion Image', 'Dilation Image', 'Opening Image',
          'Closing Image', 'Gradient Image', 'TopHat Image', 'BlackHat Image', 'HitMiss Image']
images = [image, binary_image, erosion, dilation, opening,
          closing, gradient, tophat, blackhat, hitmiss]
for i in range(2):
    plt.subplot(3, 4, i + 2), plt.imshow(images[i], 'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])
for i in range(2, 10):
    plt.subplot(3, 4, i + 3), plt.imshow(images[i], 'gray')
    plt.title(titles[i]), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()