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
【例3-8】修改一幅图像的像素，把图像的上半部分全变为黑色，把下半部分分成三份，每份只输出一个通道，从左到到右分别为红色、黄色、蓝色。
------------------------
"""

import cv2
import numpy as np

# 读取图像
image = cv2.imread('image/Example-Bridge.jpg')
# 获取图像的高度和宽度
height, width, _ = image.shape
# 将上半部分变为黑色
image[:height // 2, :] = [0, 0, 0]
# 分割下半部分为三份
third_width = width // 3
red_channel = np.zeros_like(image)
red_channel[:, :third_width, 2] = image[:, :third_width, 2]
yellow_channel = np.zeros_like(image)
yellow_channel[:, third_width:2*third_width, 0] = image[:, third_width:2*third_width, 0]
yellow_channel[:, third_width:2*third_width, 1] = image[:, third_width:2*third_width, 1]
blue_channel = np.zeros_like(image)
blue_channel[:, 2*third_width:, 0] = image[:, 2*third_width:, 0]
blue_channel[:, 2*third_width:, 2] = image[:, 2*third_width:, 2]
# 合并三个通道
final_image = red_channel + yellow_channel + blue_channel
# 显示图像
cv2.imshow('Modified Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()