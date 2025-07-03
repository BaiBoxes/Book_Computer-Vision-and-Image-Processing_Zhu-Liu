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
【例5-9】根据所给图像，分别使用OpenCV中的cv2.ORB_create函数和cv2.SIFT_create函数创建ORB对象和SIFT对象，然后计算图像的关键点和描述子。之后，分别在图像上绘制ORB和SIFT的特征点，并显示结果。最后，根据特征点数量和检测精度来比较两种算法的效果。
"""


import cv2
import matplotlib.pyplot as plt


# 读取图像
# image_color = cv2.imread('image/Example-Bridge.jpg', cv2.IMREAD_COLOR)
image_color = cv2.imread('image/Example-Vibration.jpg', cv2.IMREAD_COLOR)

# 转换为灰度图像
image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# 创建SIFT对象并检测
sift = cv2.SIFT_create()
keypoints_sift, descriptors_sift = sift.detectAndCompute(image_gray, None)

# 创建ORB对象并检测
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(image_gray, None)

# 过滤关键点
filtered_keypoints_sift = [kp for kp in keypoints_sift if kp.response > 0.0002]
filtered_keypoints_orb = [kp for kp in keypoints_orb if kp.response > 0.0002]

# 绘制关键点
image_color_sift = cv2.drawKeypoints(image_color, filtered_keypoints_sift, outImage=None)
image_color_orb = cv2.drawKeypoints(image_color, filtered_keypoints_orb, outImage=None)

# 打印SIFT关键点数量和响应值范围
if keypoints_sift:
    responses_sift = [kp.response for kp in keypoints_sift]
    print(f"SIFT检测到的原始关键点数量: {len(keypoints_sift)}")
    print(f"SIFT过滤后的关键点数量: {len(filtered_keypoints_sift)}")
    print(f"SIFT关键点的最小响应值: {min(responses_sift):.6f}")
    print(f"SIFT关键点的最大响应值: {max(responses_sift):.6f}")

# 打印ORB关键点数量和响应值范围
if keypoints_orb:
    responses_orb = [kp.response for kp in keypoints_orb]
    print(f"ORB检测到的原始关键点数量: {len(keypoints_orb)}")
    print(f"ORB过滤后的关键点数量: {len(filtered_keypoints_orb)}")
    print(f"ORB关键点的最小响应值: {min(responses_orb):.6f}")
    print(f"ORB关键点的最大响应值: {max(responses_orb):.6f}")


# 显示SIFT和ORB结果
plt.figure(figsize=(8, 4))
plt.subplots_adjust(wspace=0.05, hspace=0)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image_color_sift, cv2.COLOR_BGR2RGB))
plt.title('SIFT Keypoints (Filtered)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_color_orb, cv2.COLOR_BGR2RGB))
plt.title('ORB Keypoints (Filtered)')
plt.axis('off')

plt.show()




