import cv2

def select_roi(img):
    roi = cv2.selectROI('roi', img)
    print(roi)
    cv2.destroyAllWindows()
    return img, roi

def save_roi(img, roi):
    # Save selected region
    cv2.imwrite('roi.png', img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])

if __name__ == '__main__':
    img = cv2.imread('.logs/champ.png')
    # img, roi = select_roi(img)
    save_roi(img, (1200, 175, 64, 64))