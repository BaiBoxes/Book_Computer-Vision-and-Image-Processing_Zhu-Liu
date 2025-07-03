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
【例5-15】对视频中梁式桥梁的上下振动，用基于LK稀疏光流法的运动估计方法对其结构上的红色靶标进行运动估计，并实时显示运动估计结果。
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 鼠标点击回调函数，用于获取靶标的初始点
def select_point(event, x, y, flags, param):
    global point_selected, initial_point, old_points
    if event == cv2.EVENT_LBUTTONDOWN:
        initial_point = (x, y)  # 记录初始点
        point_selected = True
        old_points = np.array([[x, y]], dtype=np.float32)


# 初始化
cap = cv2.VideoCapture('video/Example-VVibration.mp4')
point_selected = False
initial_point = ()
old_points = np.array([[]])
trajectory = []

# 初始化实时绘图
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(trajectory)

# 设置图的标题和坐标轴标签
ax.set_title('Vibration Displacement Over Time')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Displacement (pixels)')

# 设置鼠标回调函数
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', select_point)

# 获取视频的第一帧
ret, first_frame = cap.read()
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# 显示第一帧并等待用户点击
while True:
    # 显示第一帧，等待用户选择点
    cv2.imshow('Frame', first_frame)
    
    # 等待按键或鼠标点击，直到选取点
    if point_selected:
        # 将第一帧的位移显式设置为0
        trajectory.append(0)
        break
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# 视频处理循环，开始从第二帧处理
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 将当前帧转换为灰度图像
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected:
        # 计算光流相对于第1帧
        new_points, status, error = cv2.calcOpticalFlowPyrLK(first_gray, gray_frame, old_points, None)

        # 提取新的点坐标
        if new_points is not None:
            new_x, new_y = new_points.ravel()
            initial_x, initial_y = initial_point  # 使用第一帧的初始点

            # 绘制当前选中的点
            cv2.circle(frame, (int(new_x), int(new_y)), 5, (0, 255, 0), -1)

            # 计算相对于第1帧的垂直位移
            dy = new_y - initial_y
            trajectory.append(dy)

            # 更新实时绘图
            line.set_xdata(np.arange(len(trajectory)))
            line.set_ydata(trajectory)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    # 显示当前帧
    cv2.imshow('Frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()