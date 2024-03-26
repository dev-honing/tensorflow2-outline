# outline.py
import cv2
import numpy as np

# 이미지 불러오기
path = './3dog.png'
image = cv2.imread(path)

# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 가우시안 블러(노이즈 제거)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 케니 엣지 검출
edges = cv2.Canny(blurred, 50, 150)