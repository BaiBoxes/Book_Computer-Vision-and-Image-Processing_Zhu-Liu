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
【例3-14】构造一幅上下黑白与一幅左右黑白的图像，进行按位与、或、非与异或运算，并将结果显示处理。
------------------------
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt


# 创建一幅上下黑白的图像
height, width = 200, 300
top_bottom_image = np.zeros((height, width), dtype=np.uint8)
top_bottom_image[:height // 2, :] = 255
# 创建一幅左右黑白的图像
left_right_image = np.zeros((height, width), dtype=np.uint8)
left_right_image[:, :width // 2] = 255
# 按位与运算
bitwise_and_result = cv2.bitwise_and(top_bottom_image, left_right_image)
# 按位或运算
bitwise_or_result = cv2.bitwise_or(top_bottom_image, left_right_image)
# 按位非运算
bitwise_not_result = cv2.bitwise_not(top_bottom_image)
# 按位异或运算
bitwise_xor_result = cv2.bitwise_xor(top_bottom_image, left_right_image)
# 显示结果图像
plt.subplot(231), plt.imshow(cv2.cvtColor(top_bottom_image, cv2.COLOR_BGR2RGB))
plt.title('Top-Bottom Image'), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(cv2.cvtColor(left_right_image, cv2.COLOR_BGR2RGB))
plt.title('Left-Right Image'), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(cv2.cvtColor(bitwise_and_result, cv2.COLOR_BGR2RGB))
plt.title('Bitwise AND'), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(cv2.cvtColor(bitwise_or_result, cv2.COLOR_BGR2RGB))
plt.title('Bitwise OR'), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(cv2.cvtColor(bitwise_not_result, cv2.COLOR_BGR2RGB))
plt.title('Bitwise NOT'), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(cv2.cvtColor(bitwise_xor_result, cv2.COLOR_BGR2RGB))
plt.title('Bitwise XOR'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()