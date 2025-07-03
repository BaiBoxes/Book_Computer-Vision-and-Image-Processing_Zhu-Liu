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
【例5-12】对视频中梁式桥梁的上下振动，用基于特征颜色的运动估方法对其结构上的红色靶标进行运动估计，并实时显示运动估计结果。
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt


# 导入视频
cap = cv2.VideoCapture('video/Example-VVibration.mp4')

# 初始化，开启交互模式
first_frame = None
trajectory = []
plt.ion()  
fig, ax = plt.subplots()
line, = ax.plot(trajectory)  

# 设置图的标题和坐标轴标签
ax.set_title('Vibration Displacement Over Time')
ax.set_xlabel('Frame Number')
ax.set_ylabel('Displacement (pixels)')

# 视频循环
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 创建红色掩码
    mask = cv2.inRange(frame, np.array([0, 0, 180]), np.array([100, 100, 255]))
    red_only = cv2.bitwise_and(frame, frame, mask=mask)
    
    # 查找红色区域并计算质心
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            if first_frame is None:
                first_frame = (cx, cy)
            
            # 计算相对于第一帧的位移
            dy = cy - first_frame[1]
            trajectory.append(dy)
            
            # 绘制当前帧与红色靶标位置
            cv2.circle(red_only, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(red_only, f"Position: {cy}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow('Frame', red_only)
        
            # 实时更新振动位移的折线图
            line.set_xdata(np.arange(len(trajectory)))
            line.set_ydata(trajectory)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()