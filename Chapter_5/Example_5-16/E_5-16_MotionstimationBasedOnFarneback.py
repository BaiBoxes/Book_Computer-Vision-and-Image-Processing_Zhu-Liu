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
【例5-16】对视频中梁式桥梁的上下振动，用基于Farneback稠密光流法的运动估计方法对其结构上的红色靶标进行运动估计，并实时显示运动估计结果。
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 导入视频
cap = cv2.VideoCapture('video/Example-VVibration.mp4')

# 读取第一帧
ret, first_frame = cap.read()
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 使用 selectROI 函数框选感兴趣区域
roi = cv2.selectROI('Select ROI', first_frame, showCrosshair=True)
cv2.destroyWindow('Select ROI')

# 获取ROI的坐标
x1, y1, w, h = roi
x2, y2 = x1 + w, y1 + h

# 实时绘图初始化
trajectory = []
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(trajectory)
ax.set_title('Average Optical Flow Magnitude Over Time')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Average Flow Magnitude')

# 视频处理循环
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算稠密光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 可视化光流
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    flow_visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 使用固定的ROI区域
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 提取选取区域中的光流数据
    selected_flow = flow[y1:y2, x1:x2]
    selected_magnitude = magnitude[y1:y2, x1:x2]

    # 计算平均光流大小
    avg_magnitude = np.mean(selected_magnitude)
    trajectory.append(avg_magnitude)

    # 绘制折线图
    line.set_xdata(np.arange(len(trajectory)))
    line.set_ydata(trajectory)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.01)

    # 显示当前帧和光流可视化结果
    cv2.imshow('Frame', frame)
    cv2.imshow('Optical Flow', flow_visualization)

    # 更新前一帧灰度图像
    prev_gray = gray_frame.copy()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()