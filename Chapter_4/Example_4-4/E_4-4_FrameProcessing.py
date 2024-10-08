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
【例4-4】对一个视频逐步进行BGR到RGB转换、帧数绘制、图像翻转、直方图均衡化和均值滤波，然后保存成新视频。接着对新视频进行边缘检测，然后保存成边缘检测的视频。最后对边缘检测的视频进行形态学闭运算，保存成最后的视频。
------------------------
"""


import cv2
import numpy as np


# 读取输入视频
cap = cv2.VideoCapture('video/Example-VBridge.mp4')

# 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out1 = cv2.VideoWriter('video/output_rgb_flip_hist_mean.mp4', fourcc, fps, (width, height))
out2 = cv2.VideoWriter('video/output_edges.mp4', fourcc, fps, (width, height))
out3 = cv2.VideoWriter('video/output_morph_open.mp4', fourcc, fps, (width, height))

# 形态学操作的核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
# 帧数初始化
frame_count = 0

# 处理每一帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 转 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 图像翻转（水平翻转）
    flipped_frame = cv2.flip(rgb_frame, 1)

    # 绘制当前帧数
    cv2.putText(flipped_frame, str(frame_count), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 0), 30)

    # 直方图均衡化
    ycrcb = cv2.cvtColor(flipped_frame, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    equalized_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    # 均值滤波
    smoothed_frame = cv2.blur(equalized_frame, (5, 5))

    # 边缘检测
    edges = cv2.Canny(smoothed_frame, 20, 50)

    # 将边缘检测结果转换为3通道图像以便保存
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # 形态学闭运算
    morph_open = cv2.morphologyEx(edges_color, cv2.MORPH_CLOSE, kernel)


    # 保存处理后的视频帧
    out1.write(smoothed_frame)
    out2.write(edges_color)
    out3.write(morph_open)

    frame_count += 1

# 释放资源
cap.release()
out1.release()
out2.release()
out3.release()