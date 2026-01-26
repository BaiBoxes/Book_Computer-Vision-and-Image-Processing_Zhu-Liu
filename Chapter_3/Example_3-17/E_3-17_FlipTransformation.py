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
【例3-17】将一幅图像用cv2.flip函数，进行水平翻转、垂直翻转与双轴同时翻转，并将原图与翻转图显示出来。
------------------------
"""


import cv2
import matplotlib.pyplot as plt


# 读取原始图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 进行水平翻转
horizontal_flip = cv2.flip(image, 1)
# 进行垂直翻转
vertical_flip = cv2.flip(image, 0)
# 进行双轴同时翻转
both_flip = cv2.flip(image, -1)
# 显示原始图像和翻转图像
plt.subplot(141), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142), plt.imshow(cv2.cvtColor(horizontal_flip, cv2.COLOR_BGR2RGB))
plt.title('Horizontal Flip'), plt.xticks([]), plt.yticks([])
plt.subplot(143), plt.imshow(cv2.cvtColor(vertical_flip, cv2.COLOR_BGR2RGB))
plt.title('Vertical Flip'), plt.xticks([]), plt.yticks([])
plt.subplot(144), plt.imshow(cv2.cvtColor(both_flip, cv2.COLOR_BGR2RGB))
plt.title('Both Flips'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()