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
【例5-14】对视频中梁式桥梁的上下振动，用基于模板匹配的运动估计方法对其结构上的红色靶标进行运动估计，并实时显示运动估计结果。
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

# 初始化一些变量
first_frame_loc = None
displacements = []
template = None
selecting_template = False
roi = [0, 0, 0, 0]

# 鼠标回调函数，用于选择模板
def select_template(event, x, y, flags, param):
    global template, selecting_template, roi
    if event == cv2.EVENT_LBUTTONDOWN:
        roi = [x, y, x, y]
        selecting_template = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting_template:
            roi[2] = x
            roi[3] = y
            img_copy = first_frame.copy()
            cv2.rectangle(img_copy, (roi[0],roi[1]), (roi[2],roi[3]), (255, 255, 255), 2)
            cv2.imshow('Video', img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        selecting_template = False
        roi[2] = x
        roi[3] = y
        template = first_frame[roi[1]:roi[3], roi[0]:roi[2]]

# 打开视频文件
cap = cv2.VideoCapture('video/Example-VVibration.mp4')

# 读取第一帧并显示，供选择模板
ret, first_frame = cap.read()
cv2.imshow('Video', first_frame)
print('选择模板，按任意键确定')
cv2.setMouseCallback('Video', select_template)
cv2.waitKey(0)

# 初始化图形，开启交互模式
plt.ion()  
fig, ax = plt.subplots()
line, = ax.plot(displacements, 'b-')
ax.set_ylim(-2, 2)
ax.set_xlabel('Frame')
ax.set_ylabel('Displacement (pixels)')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if template is not None:
        # 使用模板匹配
        result = cv2.matchTemplate(frame_gray, cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # 提取最大值周围的邻域，用于二次插值
        x, y = max_loc
        h_res, w_res = result.shape
        # 确保不越界
        if 1 <= x < w_res - 1 and 1 <= y < h_res - 1:
            # 获取x方向上的三个值
            dx = np.array([-1, 0, 1])
            val_x = np.array([result[y, x - 1], result[y, x], result[y, x + 1]])
            # 在x方向进行二次拟合
            coeff_x = np.polyfit(dx, val_x, 2)
            if coeff_x[0] != 0:
                peak_offset_x = -coeff_x[1] / (2 * coeff_x[0])
            else:
                peak_offset_x = 0

            # 获取y方向上的三个值
            dy = np.array([-1, 0, 1])
            val_y = np.array([result[y - 1, x], result[y, x], result[y + 1, x]])
            # 在y方向进行二次拟合
            coeff_y = np.polyfit(dy, val_y, 2)
            if coeff_y[0] != 0:
                peak_offset_y = -coeff_y[1] / (2 * coeff_y[0])
            else:
                peak_offset_y = 0

            # 计算亚像素级别的峰值位置
            subpixel_x = x + peak_offset_x
            subpixel_y = y + peak_offset_y
        else:
            # 边界情况，无法进行二次插值，使用整数像素位置
            subpixel_x = x
            subpixel_y = y

        # 绘制匹配结果
        top_left = (subpixel_x, subpixel_y)
        h, w = template.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        # 将浮点数坐标转换为整数用于绘制
        cv2.rectangle(frame, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])), (0, 255, 0), 2)

        # 记录初始位置，用于计算位移
        if first_frame_loc is None:
            first_frame_loc = (subpixel_x, subpixel_y)

        # 计算上下位移（仅考虑y轴方向的变化）
        vertical_displacement = subpixel_y - first_frame_loc[1]
        displacements.append(vertical_displacement)

        # 实时更新折线图
        line.set_xdata(np.arange(len(displacements)))
        line.set_ydata(displacements)
        ax.set_xlim(0, len(displacements))
        ax.set_ylim(min(displacements)-1, max(displacements)+1)
        plt.draw()
        plt.pause(0.01)

        # 在视频图像上显示位移
        cv2.putText(frame, f"Displacement: {vertical_displacement:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()