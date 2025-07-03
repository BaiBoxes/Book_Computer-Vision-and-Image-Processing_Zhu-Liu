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
【例3-30】用cv2.matchTemplate函数在输入图像中查找模板的位置，并在匹配位置处绘制一个矩形框来标识匹配结果。然后，显示出输入图像、模板与匹配结果。
------------------------
"""


import cv2
import matplotlib.pyplot as plt


# 读取输入图像和模板图像
input_image = cv2.imread('image/Example-Bridge_gray.jpg')
template_image = cv2.imread('image/target.jpg')
# 应用模板匹配
result = cv2.matchTemplate(input_image, template_image, cv2.TM_CCORR_NORMED)
# 找到最佳匹配位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# 计算模板的宽度和高度
w, h = template_image.shape[1], template_image.shape[0]
bottom_right = (max_loc[0] + w, max_loc[1] + h)
# 在输入图像上绘制矩形框
matched_image = input_image.copy()
cv2.rectangle(matched_image, max_loc, bottom_right, (0, 255, 0), 2)
# 显示输入图像、模板和匹配结果
plt.figure(num='模板匹配', figsize=(12, 6))
plt.subplot(131)
plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')
plt.subplot(132)
plt.imshow(cv2.cvtColor(template_image, cv2.COLOR_BGR2RGB))
plt.title('Template Image')
plt.axis('off')
plt.subplot(133)
plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
plt.title('Matching Result')
plt.axis('off')
plt.tight_layout()
plt.show()