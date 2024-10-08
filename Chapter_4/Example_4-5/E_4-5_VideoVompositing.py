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
【例4-5】读取两个视频，将两个视频分别分割为两段再合成一个视频，并对新视频进行高斯滤波，然后在视频帧中添加视频段信息水印。
------------------------
"""


import cv2


# 打开视频文件
cap1 = cv2.VideoCapture('video/Example-VBridge.mp4')
cap2 = cv2.VideoCapture('video/Example-VBridge_gray.mp4')

# 获取视频信息
fps = cap1.get(cv2.CAP_PROP_FPS)
width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video/output_Vompositing.mp4', fourcc, fps, (width, height))

# 处理视频段并添加水印和滤波操作
def process_and_merge_segment(cap, start_frame, end_frame, label):
    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        # 高斯模糊
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # 添加水印
        watermarked_frame = cv2.putText(blurred_frame, label, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 0), 15)
        # 保存处理后的视频帧
        out.write(watermarked_frame)

# 合并视频
process_and_merge_segment(cap1, 0, total_frames1 // 2, 'Video1-Part1')
process_and_merge_segment(cap2, 0, total_frames2 // 2, 'Video2-Part1')
process_and_merge_segment(cap1, total_frames1 // 2, total_frames1, 'Video1-Part2')
process_and_merge_segment(cap2, total_frames2 // 2, total_frames2, 'Video2-Part2')

# 释放资源
cap1.release()
cap2.release()
out.release()