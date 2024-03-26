# outline.py
import cv2
import numpy as np

# 이미지 불러오기
path = './3dog.png'
image = cv2.imread(path)

# 이미지를 그레이스케일로 변환
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
