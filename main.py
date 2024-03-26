# outline.py
import cv2
import numpy as np

# 이미지 불러오기
image_path = './3dog.png'
image = cv2.imread(image_path)

# 그레이스케일링
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러(노이즈 제거)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 케니 엣지 검출
edges = cv2.Canny(blurred, 50, 150)

# 컨투어 찾기
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 초록색 외곽선 그리기
green_contours_image = image.copy()
cv2.drawContours(green_contours_image, contours, -1, (0, 255, 0), 2)

# 이미지 저장
cv2.imwrite('green_contours_image.png', green_contours_image)
