# try.py
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

# 마스크 이미지 생성 (흰색 배경, 초록색 외곽선)
mask = np.zeros_like(image)
cv2.drawContours(mask, contours, -1, (0, 255, 0), -1)

# 안쪽 이미지 생성
inner_image = cv2.bitwise_and(image, mask)

# 외곽선 개수 출력
num_contours = len(contours)
print("외곽선 개수:", num_contours)

# 결과 이미지 출력
cv2.imshow('Result', inner_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
