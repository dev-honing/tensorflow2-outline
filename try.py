# try.py
import cv2
import numpy as np

# 이미지를 추가하는 함수
def put_image(image_path):
    """
    이미지 경로를 받아서 해당 이미지를 불러와 반환하는 함수
    """
    image = cv2.imread(image_path)
    if image is None: 
        print("이미지를 불러오지 못했습니다.")
        return None
    return image

# 이미지를 전처리하는 함수
def preprocess_image(image_path):
    """
    이미지를 전처리하여 내부 이미지와 컨투어 정보를 반환하는 함수
    """
    # 이미지를 불러오기
    image = put_image(image_path)
    if image is None:
        return None
    
    # 그레이스케일링
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러(노이즈 제거)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 캐니 엣지 검출
    edges = cv2.Canny(blurred, 50, 150)
    
    # 컨투어 찾기
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 마스크 이미지 생성 - 초록색 외곽선
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (0, 255, 0), -1)
    
    # 안쪽 이미지 생성
    inner_image = cv2.bitwise_and(image, mask)
    
    # 컨투어의 최소 길이 동적으로 설정하기
    original_size = max(image.shape[:2])
    min_contour_length = original_size * 0.4
    
    # 컨투어의 길이가 최소 길이보다 큰 경우만 유지
    filtered_contours = []
    for contour in contours:
        contour_length = cv2.arcLength(contour, True)
        if contour_length > min_contour_length:
            filtered_contours.append(contour)
    
    # 빨간색 외곽선 추가
    cv2.drawContours(inner_image, filtered_contours, -1, (0, 0, 255), 2)
    
    # 빨간색 컨투어 내부의 면적 계산
    red_contour_area = 0
    for contour in filtered_contours:
        red_contour_area += cv2.contourArea(contour)
    
    return inner_image, red_contour_area

# 결과 이미지를 출력하는 함수
def show_image(image, window_name='Image'):
    """
    결과 이미지를 출력하는 함수
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

# 각 함수를 사용
if __name__ == "__main__":
    # 이미지 경로 설정
    image_path = input("이미지 경로를 입력하세요: ")
    
    # 전처리된 이미지와 빨간색 컨투어 내부의 면적
    result_image, red_contour_area = preprocess_image(image_path)
    
    # 빨간색 컨투어 내부의 면적 출력
    print("빨간색 컨투어 내부의 면적:", red_contour_area)

    # 결과 이미지 출력
    show_image(result_image, window_name='Preprocessed Image')    
    