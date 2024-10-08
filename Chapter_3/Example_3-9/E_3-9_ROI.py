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
【例3-9】将一图像横向五分之一到五分之二的区域设为ROI区域，然后显示出原图与ROI区域，并提取出来保存到项目文件夹。
------------------------
"""


import cv2
import matplotlib.pyplot as plt


# 读取图像
image = cv2.imread('image/Example-Bridge_gray.jpg', cv2.IMREAD_GRAYSCALE)
# 获取图像的宽度和高度
height, width= image.shape
# 计算 ROI 区域的范围
start_col = width // 5
end_col = 2 * width // 5
# 提取 ROI 区域
roi = image[:, start_col:end_col]
# 保存 ROI 区域到项目文件夹
cv2.imwrite('image/roi_image.jpg', roi)
# 显示原图像与ROI 区域
plt.subplot(121), plt.imshow(image, 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(roi, 'gray')
plt.title('ROI Region'), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

