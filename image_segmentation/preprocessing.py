import cv2

clahe_resolution = 2.0

def clahe(path):
    clahe = cv2.createCLAHE(clipLimit=clahe_resolution, tileGridSize=(16, 16))
    img = cv2.imread(path, 0)
    cv2.imwrite(path, clahe.apply(img))


def center_crop(filelist_original):
    for i, img in enumerate(filelist_original):
        im = Image.open(img)
        width, height = im.size

        # print(f"width: {width}, {width2} height: {height}, {height2}")

        w_after_crop = 1300
        h_after_crop = 590

        left = (width - w_after_crop) // 2
        top = (height - h_after_crop) // 2
        bottom = top + h_after_crop
        right = left + w_after_crop
        crop_rectangle = (left, top, right, bottom)

        cropped_im = im.crop(crop_rectangle)
        plt.imshow(cropped_im, cmap='gray', vmin=0, vmax=255)
        width, height = cropped_im.size
        # print(f"width: {width}, height: {height}")