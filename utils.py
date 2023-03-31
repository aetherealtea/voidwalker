import cv2
import numpy as np


def select_roi(img):
    # Create window of selected size
    cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('roi', 1960, 1080)

    roi = cv2.selectROI('roi', img)
    print(roi)
    cv2.destroyAllWindows()
    return img, roi

def save_roi(img, roi):
    # Save selected region
    # processed_region = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    # _, processed_region = cv2.threshold(processed_region, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite('roi.png', img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])

def find_marker_coords(marker, source, roi):
    "Find marker in the region"
    # Convert to grayscale
    source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # Threshold
    _, source = cv2.threshold(source, 130, 255, cv2.THRESH_BINARY)
    # Crop region of interest
    source = source[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

    # Find marker
    # fix for (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() && _img.dims() <= 2 in function 'cv::matchTemplate'
    marker = cv2.cvtColor(marker, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(source, marker, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # Reset coordinates to the original image
    coords = (max_loc[0] + roi[0], max_loc[1] + roi[1])

    return coords
    
def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_images(imgs):
    for i in range(len(imgs)):
        cv2.imshow(f'image {i}', imgs[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread('.logs/2023-02-23_20-16/summary.png')
    # img, roi = img, (2544, 1421, 16, 16)

    # processed_region = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

    # processed_region = cv2.cvtColor(np.array(processed_region), cv2.COLOR_BGR2GRAY)
    # _, processed_region = cv2.threshold(processed_region, 150, 255, cv2.THRESH_BINARY)

    # # crop excessive borders
    # crop_coords = cv2.boundingRect(processed_region)
    # processed_region = processed_region[(crop_coords[1]):crop_coords[1]+crop_coords[3], crop_coords[0]:crop_coords[0]+crop_coords[2]]
    # processed_region = cv2.copyMakeBorder(processed_region, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0,0,0])

    # cv2.imwrite('resources/markers/start/mode_limiter.png', processed_region)

    # img, roi = select_roi(img)
    save_roi(img, (1492, 624, 228, 135))

    # marker = cv2.imread('resources/markers/summary/skins/wattson/lightning_spirit.png')
    # source = cv2.imread('.logs/2023-02-23_20-16/summary.png')
    # roi = (0, 0, 2560, 1440)

    # coords = find_marker_coords(marker, source, roi)
    # print(coords)
    # save_roi(img, list(coords) + [51, 44])