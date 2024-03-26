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

# 초록색 외곽선 이미지 불러오기
green_contours_image_path = 'green_contours_image.png'
green_contours_image = cv2.imread(green_contours_image_path)

# 배경색 변경
background_color = [0, 0, 0]  # 변경하고자 하는 배경색: 검은색
green_contours_image[np.where((green_contours_image != [0, 255, 0]).all(axis=-1))] = background_color # 초록색 외곽선만 남기고 나머지 배경색 변경

# 변경된 이미지 저장
cv2.imwrite('green_contours_with_background.png', green_contours_image)

# 변경된 이미지 불러오기
new_image_path = './green_contours_with_background.png'
image = cv2.imread(new_image_path)

# 초록색 영역 추출
lower_green = np.array([35, 50, 50])
upper_green = np.array([85, 255, 255])
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

# 초록색 영역의 면적과 인덱스 계산
contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
areas = []
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    areas.append(area)
    print(f"초록색 영역 {i+1}의 면적:", area)
    # 초록색 영역의 인덱스 계산 및 표시
    for point in contour:
        cv2.circle(image, tuple(point[0]), 3, (0, 0, 255), -1)  # 빨간색 점으로 표시

    # 초록색 영역 중 가장 큰 면적을 타겟으로 설정
    target_area = max(areas)
