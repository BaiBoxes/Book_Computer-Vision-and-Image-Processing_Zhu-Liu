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
【例4-1】使用cv2.VideoCapture函数打开视频文件后，使用isOpened属性检测视频文件是否成功打开，然后使用get属性获取视频帧率、宽度、高度和总帧数并打印出来，最后使用read属性逐帧读取视频并创建窗口来显示视频。
------------------------
"""


import cv2


# 打开视频文件
cap = cv2.VideoCapture('video/Example-VBridge.mp4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("无法打开视频文件")
    exit()

# 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 输出视频属性
print("视频帧率: ", fps)
print("视频宽度: ", width)
print("视频高度: ", height)
print("视频总帧数: ", total_frames)


# 逐帧读取视频流
while cap.isOpened():
    ret, frame = cap.read()
    # 检查是否成功读取帧
    if not ret:
        break
    # 显示当前帧
    cv2.imshow('Video', frame)
    # 等待按键事件，如果按下 'q' 键则退出循环
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 释放视频对象和关闭窗口
cap.release()
cv2.destroyAllWindows()