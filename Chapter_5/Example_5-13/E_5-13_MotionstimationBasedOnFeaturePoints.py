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
【例5-13】对视频中梁式桥梁的上下振动，用基于特征点的运动估计方法对其结构上的红色靶标进行运动估计，并实时显示运动估计结果。
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 初始化视频捕获、ORB特征检测器和BFMatcher对象
cap = cv2.VideoCapture('video/Example-VVibration.mp4')
orb = cv2.ORB_create(nfeatures=1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 初始化图形界面和轨迹数据
trajectory = []
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot(trajectory, 'b-')
ax.set_title('Vibration Displacement Over Time')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Displacement (pixels)')

# 读取第一帧并手动选择ROI
ret, first_frame = cap.read()
if not ret:
    print("Failed to read video")
    exit()

# 选择ROI
roi = cv2.selectROI("Select ROI", first_frame, fromCenter=False, showCrosshair=True)
cv2.destroyWindow("Select ROI")

# 将选择的区域转换为灰度图像，用于后续处理
x, y, w, h = roi
gray1 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
roi_mask = np.zeros(first_frame.shape[:2], dtype=np.uint8)
roi_mask[int(y):int(y + h), int(x):int(x + w)] = 255

# 使用ORB提取特征点和描述符
kp1, des1 = orb.detectAndCompute(gray1, roi_mask)

# 处理视频帧，计算位移并更新图表
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    roi_mask[int(y):int(y + h), int(x):int(x + w)] = 255
    roi_masked = cv2.bitwise_and(gray2, gray2, mask=roi_mask)

    # 检测当前帧的特征点
    kp2, des2 = orb.detectAndCompute(roi_masked, None)

    if des1 is not None and des2 is not None:
        # 使用BFMatcher匹配特征点
        matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        if pts1.size and pts2.size:
            # 使用cornerSubPix函数进行亚像素级别的优化
            pts1_subpix = cv2.cornerSubPix(gray1, pts1, (5, 5), (-1, -1),
                                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            pts2_subpix = cv2.cornerSubPix(gray2, pts2, (5, 5), (-1, -1),
                                           criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            
            # 计算位移：这里仅考虑y轴方向上的变化
            displacement = np.mean(pts2_subpix[:, 1] - pts1_subpix[:, 1])
            trajectory.append(displacement)

            # 更新折线图数据
            line.set_xdata(np.arange(len(trajectory)))
            line.set_ydata(trajectory)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)

    # 显示特征点匹配结果
    img_matches = cv2.drawMatches(first_frame, kp1, frame, kp2, matches, None,
                                  matchColor=(0, 255, 0), singlePointColor=None,
                                  matchesMask=[1] * len(matches), flags=2)
    cv2.imshow('Feature Matching', img_matches)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()