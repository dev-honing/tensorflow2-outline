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

# 컨투어 찾기
contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 컨투어 그리기
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# 결과 이미지 출력
cv2.imshow('Outline Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()