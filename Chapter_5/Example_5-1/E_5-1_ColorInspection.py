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
【例5-1】生成一个颜色梯度图，利用鼠标点击和框选进行颜色检查，并显示出检查结果。
"""


import cv2
import numpy as np


# 定义鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global start_point, end_point, drawing, display_image  
    # 左键按下事件
    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        # 重置显示图像 
        display_image = image.copy()
        end_point = None
        drawing = True
    # 鼠标移动事件
    elif event == cv2.EVENT_MOUSEMOVE and drawing:  
        end_point = (x, y)
        img_copy = image.copy()
        cv2.rectangle(img_copy, start_point, end_point, (255, 255, 255), 2)
        cv2.imshow("Image", img_copy)
    # 左键抬起事件
    elif event == cv2.EVENT_LBUTTONUP:  
        drawing = False
        # 点击事件
        if start_point is not None and end_point is None:
            # 计算选定点的颜色范围
            bgr_color = image[start_point[1], start_point[0]]
            lower_bound = np.array([max(0, bgr_color[0] - 10), max(0, bgr_color[1] - 10), max(0, bgr_color[2] - 10)])
            upper_bound = np.array([min(255, bgr_color[0] + 10), min(255, bgr_color[0] + 10), min(255, bgr_color[0] + 10)])
            mask = cv2.inRange(image, lower_bound, upper_bound)
            result = cv2.bitwise_and(image, image, mask=mask)
            # 绘制点并显示检查的颜色
            cv2.circle(display_image, start_point, 5, (255, 255, 255), -1)  
            cv2.imshow("Image", display_image)  
            cv2.imshow('Detected Color', result)
        # 框选事件
        if start_point and end_point:
            # 确保坐标顺序正确
            x1, y1 = start_point
            x2, y2 = end_point
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            # 计算选定区域的颜色范围
            roi = image[y1:y2, x1:x2]
            lower_bound = np.array([np.min(roi[:, :, 0]), np.min(roi[:, :, 1]), np.min(roi[:, :, 2])])
            upper_bound = np.array([np.max(roi[:, :, 0]), np.max(roi[:, :, 1]), np.max(roi[:, :, 2])])
            mask = cv2.inRange(image, lower_bound, upper_bound)
            result = cv2.bitwise_and(image, image, mask=mask)
            # 绘制矩形并显示检查的颜色
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  
            cv2.imshow("Image", display_image)  
            cv2.imshow('Detected Color', result)


# 绘制颜色梯度图
image = np.zeros((500, 500, 3), dtype=np.uint8)
for i in range(500):
    blue_intensity = int(255 * (i / 500))
    image[i, :, 0] = blue_intensity 
# 定义一些全局变量
start_point = None
end_point = None
drawing = False
# 显示图像并设置鼠标回调函数
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", mouse_callback)
# 等待按键事件并关闭窗口
key = cv2.waitKey(0) & 0xFF
cv2.destroyAllWindows()