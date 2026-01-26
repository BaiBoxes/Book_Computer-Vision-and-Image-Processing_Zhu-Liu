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
【例7-5】使用cv2.add函数将两张图标合成到一起，然后使用图形处理里的算法实现卡通效果图像。
------------------------
"""


import cv2
import matplotlib.pyplot as plt


# 读取图像
bridge_image = cv2.imread('image/Example-Bridge.jpg')
bird_image = cv2.imread('image/bird.png')

# 检查图像是否正确读取
if bridge_image is None or bird_image is None:
    print("无法读取图像，请检查文件路径")
    exit()

# 处理鸟的图像，去除白色背景
gray = cv2.cvtColor(bird_image, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
bird_no_bg = cv2.bitwise_and(bird_image, bird_image, mask=mask)

# 确定鸟图像在桥梁图像中的位置（左上角为起点）
y1, y2 = 10, 10 + bird_no_bg.shape[0]
x1, x2 = 10, 10 + bird_no_bg.shape[1]

# 防止鸟图像超出桥梁图像的边界
if y2 > bridge_image.shape[0] or x2 > bridge_image.shape[1]:
    print("鸟图像超出桥梁图像的范围，请调整位置或图像大小")
    exit()

# 定义 ROI（感兴趣区域）并创建反转的遮罩
roi = bridge_image[y1:y2, x1:x2]
mask_inv = cv2.bitwise_not(mask)

# 通过掩码合并桥梁图像和鸟图像
bridge_image_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
bird_fg = cv2.bitwise_and(bird_no_bg, bird_no_bg, mask=mask)
combined_roi = cv2.add(bridge_image_bg, bird_fg)

# 将合成后的图像放回原图
bridge_image[y1:y2, x1:x2] = combined_roi

# 转换为灰度图并应用中值滤波
gray = cv2.medianBlur(cv2.cvtColor(bridge_image, cv2.COLOR_BGR2GRAY), 5)

# 使用自适应阈值检测边缘
edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

# 使用双边滤波进行模糊处理，保留边缘
color = cv2.bilateralFilter(bridge_image, 9, 300, 300)

# 将边缘和彩色图像结合，生成卡通效果
cartoon_image = cv2.bitwise_and(color, color, mask=edges)


# 使用plt显示原图、鸟图和结果图
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
plt.subplots_adjust(wspace=0.05, hspace=0)

# 显示原桥梁图像
axes[0].imshow(cv2.cvtColor(cv2.imread('image/Example-Bridge.jpg'), cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Bridge Image')
axes[0].axis('off')

# 显示鸟图
axes[1].imshow(cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Bird Image')
axes[1].axis('off')

# 显示处理后的卡通效果图像
axes[2].imshow(cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2RGB))
axes[2].set_title('Cartoon Image Result')
axes[2].axis('off')

plt.show()