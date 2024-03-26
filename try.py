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
    
    return inner_image, contours

# 확인 콘솔을 출력하는 함수
def print_console(image, contours):
    """
    확인 콘솔을 출력하는 함수
    """
    # 이미지 정보 출력
    print("이미지 정보:")
    print(" - 이미지 크기:", image.shape)
    print(" - 이미지 타입:", type(image))
    
    # 컨투어 정보 출력
    print("\n컨투어 정보:")
    print(" - 컨투어 개수:", len(contours))
    for i, contour in enumerate(contours):
        print(f" - 컨투어 {i+1}의 포인트 개수:", len(contour))
