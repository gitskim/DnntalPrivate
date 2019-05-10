import cv2

clahe_resolution = 2.0

def clahe(path):
    clahe = cv2.createCLAHE(clipLimit=clahe_resolution, tileGridSize=(16, 16))
    img = cv2.imread(path, 0)
    cv2.imwrite(path, clahe.apply(img))