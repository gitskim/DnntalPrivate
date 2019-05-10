import cv2

def clahe(path):
    clahe = cv2.createCLAHE(clipLimit=resolution, tileGridSize=(16, 16))
    img = cv2.imread(path, 0)
    cv2.imwrite(path, clahe.apply(img))