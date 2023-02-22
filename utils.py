import cv2
import numpy as np


def select_roi(img):
    # Create window of selected size
    cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('roi', 1366, 768)

    roi = cv2.selectROI('roi', img)
    print(roi)
    cv2.destroyAllWindows()
    return img, roi

def save_roi(img, roi):
    # Save selected region
    processed_region = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    _, processed_region = cv2.threshold(processed_region, 230, 255, cv2.THRESH_BINARY)
    cv2.imwrite('roi.png', processed_region[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]])

def find_marker_coords(marker, source, roi):
    "Find marker in the region"
    # Convert to grayscale
    source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # Threshold
    _, source = cv2.threshold(source, 150, 255, cv2.THRESH_BINARY)
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
    

if __name__ == '__main__':
    img = cv2.imread('.logs/2023-02-19_14_28_15.png')
    # img, roi = select_roi(img)

    # processed_region = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

    # processed_region = cv2.cvtColor(np.array(processed_region), cv2.COLOR_BGR2GRAY)
    # _, processed_region = cv2.threshold(processed_region, 95, 255, cv2.THRESH_BINARY)

    # # crop excessive borders
    # crop_coords = cv2.boundingRect(processed_region)
    # processed_region = processed_region[(crop_coords[1]):crop_coords[1]+crop_coords[3], crop_coords[0]:crop_coords[0]+crop_coords[2]]
    # processed_region = cv2.copyMakeBorder(processed_region, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0,0,0])

    # cv2.imwrite('resources/markers/summary/k_a_k_separator.png', processed_region)

    save_roi(img, (1200, 175, 64, 64))

    # marker = cv2.imread('resources/markers/summary/banner_corner_hint.png')
    # source = cv2.imread('.logs/2023-02-19_14_32_10.png')
    # roi = (309, 309, 148, 113)

    # coords = find_marker_coords(marker, source, roi)
    # print(coords)