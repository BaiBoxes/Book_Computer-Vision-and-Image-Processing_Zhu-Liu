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
【例5-10】使用cv2.BFMatcher函数创建BFMatcher对象，对两幅图进行特征点匹配，然后cv2.drawMatches函数绘制两幅图像的特征点匹配的结果，最后显示其结果。
"""


import cv2


# 读取两幅图像
img1 = cv2.imread('image/Example-Bridge_part.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image/Example-Bridge.jpg', cv2.IMREAD_GRAYSCALE)

# 创建SIFT特征检测器
sift = cv2.SIFT_create()

# 在两幅图像上检测特征点和计算描述子
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

# 创建暴力匹配器
bf = cv2.BFMatcher()

# 使用匹配器进行特征点匹配
matches = bf.knnMatch(descriptors1, descriptors2, k=2)  # k=2表示获取两个最佳匹配

# 应用比例测试，以确保匹配的质量
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 绘制匹配结果
output_img = cv2.drawMatches(
    img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 显示结果图像
cv2.imshow('Matches', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
