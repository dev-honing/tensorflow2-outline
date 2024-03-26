# outline_functions.py

import cv2
import numpy as np

def add_image(image_path):
    """
    이미지를 추가하는 함수
    :param image_path: 이미지 파일의 경로
    :return: 추가된 이미지
    """
    image = cv2.imread(image_path)
    return image

def preprocess_image(image):
    """
    이미지를 전처리하는 함수
    :param image: 전처리할 이미지
    :return: 전처리된 이미지, 초록색으로 찾은 영역의 외곽선 정보
    """
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러 적용 (노이즈 제거)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 케니 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)

    # 컨투어 찾기
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 초록색 외곽선 그리기
    green_contours_image = image.copy()
    cv2.drawContours(green_contours_image, contours, -1, (0, 255, 0), 2)

    # 초록색으로 찾은 영역의 외곽선 정보를 리스트에 저장
    green_contours = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        green_contours.append(approx)

    # 초록색으로 찾은 영역 주위에 사각형 또는 다각형 그리기
    for contour in green_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(green_contours_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 배경색 변경
    background_color = [0, 0, 0]  # 변경하고자 하는 배경색: 검은색
    green_contours_image[np.where((green_contours_image != [0, 255, 0]).all(axis=-1))] = background_color

    return green_contours_image, contours

def count_objects(image):
    """
    이미지 개체수를 판단하는 함수
    :param image: 이미지 파일
    :return: 개체수와 면적 정보
    """
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
        print(f"초록색 영역 {i+1}을 찾았습니다.")

    # 개체수 계산
    num_objects = len(areas)
    return num_objects, areas, contours

def display_result(image, num_objects, areas, contours):
    """
    결과를 출력하는 함수
    :param image: 결과 이미지
    :param num_objects: 개체수
    :param areas: 면적 정보 리스트
    :param contours: 컨투어 정보
    """
    # 결과 이미지에 초록색 외곽선 그리기
    for contour in contours:
        for point in contour:
            cv2.circle(image, tuple(point[0]), 3, (0, 0, 255), -1)  # 빨간색 점으로 표시

    # 개체수 출력
    print(f"개체수는 {num_objects}인 것으로 예측합니다. 이는 초록색 영역 {num_objects}개를 찾았기 때문입니다.")

    # 결과 이미지 출력
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 각 함수를 사용
if __name__ == "__main__":
    # 이미지 추가 및 전처리
    image_path = './2kitten.jpg' # fixme: 이미지 경로를 입력하세요.
    original_image = add_image(image_path)
    processed_image, contours = preprocess_image(original_image)

    # 이미지 개체수 판단
    num_objects, areas, contours = count_objects(processed_image)

    # 결과 출력
    display_result(processed_image, num_objects, areas, contours)
