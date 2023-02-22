from pynput import keyboard

import numpy as np
import cv2
from mss import mss
from PIL import Image
from pytesseract import image_to_string

from datetime import datetime
import time

import json

import keyboard  # ! delete



logs_dir = '.logs/'


class ApexLegendsAnalyser:

    monitor = None
    settings = {}

    def __init__(self):

        with open("resources/configs/global.json") as f:
            self.settings['global'] = json.load(f)

        with open("resources/configs/user.json") as f:
            self.settings['user'] = json.load(f)

        if self.settings['user']['name'] == "":
            self.settings['user']['name'] = input('Enter your nickname: ')
            self.save_settings()

        if self.settings['user']['monitor'] == {}:
            self.settings['user']['monitor'] = self.set_monitor()
            # Ask if user wants to save monitor settings
            if input('Save monitor settings? (y/n) ') == 'y':
                self.save_settings()
        
        self.monitor = self.settings['user']['monitor']

        resolution = str(self.monitor['width']) + 'x' + str(self.monitor['height'])
        path_to_settings = 'resources/configs/rois/' + resolution + '.json'
        with open(path_to_settings) as f:
            self.settings['rois'] = json.load(f)

        # Unpack markers
        self.settings['markers'] = {}
        for marker in self.settings['rois']['markers']:
            self.settings['markers'][marker['name']] = marker


    def save_settings(self):
        with open("resources/configs/user.json", "w") as f:
            json.dump(self.settings['user'], f, indent=4)

    def set_monitor(self):

        "Chooses the monitor to capture from."

        monitors = mss().monitors
        print('Monitors detected: {}'.format(len(monitors)))
        for i, monitor in enumerate(monitors):
            print('{}. {}x{}@{};{}'.format(i, monitor['width'], monitor['height'], monitor['left'], monitor['top']))
        monitor = int(input('Choose monitor: '))
        print('Monitor {} chosen'.format(monitor))
        return monitors[monitor]


    def capture(self):

        "Captures the screen."

        scrennshot = mss().grab(self.monitor)
        img = Image.frombytes('RGB', scrennshot.size, scrennshot.bgra, 'raw', 'BGRX')
        return img

    def observe(self):

        "Observs the screen and detects various in-game screen types for further analysis."

        frame_type_buffer = "misc"
        img_buffer = None

        while True:
            # Capture screenshot and detect screen type
            img = self.capture()
            frame_type = self.frame_type(img)
            # print(frame_type)

            # If screen type is not misc
            if frame_type != "misc":
                # if equal to the previous screen type, find out which one has more info
                if frame_type == frame_type_buffer:
                    # if the current one has more info, update the buffer
                    if self.compare(img, img_buffer):
                        img_buffer = img
                else:
                    # update the buffer, beggining a new comparison chain
                    frame_type_buffer = frame_type
                    img_buffer = img

            else:
                # if the screen type was not misc, send image for data extraction; ideally the image should be the one with the most info
                if frame_type_buffer != "misc":
                    # data = self.extract(img_buffer)
                    # Save image and data for processing
                    img_buffer.save(logs_dir + datetime.now().strftime('%Y-%m-%d_%H_%M_%S') + '.png')
                    print('Saved image to {}'.format(logs_dir + datetime.now().strftime('%Y-%m-%d_%H_%M') + '.png'))
                    

                    frame_type_buffer = "misc"
                    img_buffer = None

            # Chill for a bit
            time.sleep(0.1)

            
    def frame_type(self, img) -> str:
        
        """
        Detects the frame type.

        :param img: Image to detect the frame type of.
        :type img: PIL.Image
        :return: Frame type.
        :rtype: str
        """

        detected_frame_type = "misc"

        for frame_type in self.settings['global']['markers']['frame_type'].keys():
            # Compare markers, if there is a match, return the screen type
            marker_matched = False

            for marker in self.settings['global']['markers']['frame_type'][frame_type]:
                if self.check_marker(marker, img):
                    marker_matched = True
                    break

            if marker_matched:
                detected_frame_type = frame_type
                break
        
        return detected_frame_type


    def mse(self, img1, img2) -> float:

        "Calculates the mean squared error between two images. Sizes must match."

        # Assert that the images are of the same size
        assert img1.shape == img2.shape

        err = np.sum((img1 - img2) ** 2)
        err /= float(np.prod(img1.shape))
        print(err)
        return err


    def compare(self, img1, img2) -> bool:

        """
        Compares two images and returns True if img1 has more info than img2.
        Currently compares the variance of the images.
        """

        # If the variance of img1 is greater than img2, return True
        return np.var(img1) > np.var(img2)


    def check_marker(self, marker, img) -> bool:
        """
        Searches for the given marker and returns the detection status.
        
        :param marker: Marker to search for (refer to the rois config file for names)
        :type marker: str
        :param img: Image to search for markers in
        :type img: np.array
        :return: Detection status
        :rtype: bool
        """

        marker_info = self.settings['markers'][marker]
        cropped_roi = np.array(img.crop([marker_info['roi'][0], marker_info['roi'][1], marker_info['roi'][0]+marker_info['roi'][2], marker_info['roi'][1]+marker_info['roi'][3]]))
        
        marker_img = cv2.imread(marker_info['path'])
        marker_img = cv2.cvtColor(marker_img, cv2.COLOR_BGR2GRAY)

        if marker_info.get('processing', {}).get('grayscale', False):
            cropped_roi = cv2.cvtColor(cropped_roi, cv2.COLOR_BGR2GRAY)
        if marker_info.get('processing', {}).get('threshold') is not None:
            _, cropped_roi = cv2.threshold(cropped_roi, marker_info['processing']['threshold'], 255, cv2.THRESH_BINARY)
        if marker_info.get('processing', {}).get('blur') is not None:
            cropped_roi = cv2.blur(cropped_roi, (marker_info['processing']['blur'], marker_info['processing']['blur']))
            marker_img = cv2.blur(marker_img, (marker_info['processing']['blur'], marker_info['processing']['blur']))

        return self.mse(cropped_roi, marker_img) < marker_info['match_threshold']


    def find_marker(self, marker, img) -> list:
            
            """
            Searches for the given marker and returns the coordinates of all the places where the marker is found.
            
            :param marker: Marker to search for (refer to the rois config file for names)
            :type marker: str
            :param img: Image to search for markers in
            :type img: np.array
            :return: Coordinates of all the places where the marker is found (list of tuples)
            :rtype: list
            """


            marker_info = self.settings['markers'][marker]
            
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except:
                pass

            if marker_info.get('processing', {}).get('threshold') is not None:
                _, img = cv2.threshold(img, marker_info['processing']['threshold'], 255, cv2.THRESH_BINARY)
    
            marker_img = cv2.imread(marker_info['path'])
            marker_img = cv2.cvtColor(marker_img, cv2.COLOR_BGR2GRAY)
    
            result = cv2.matchTemplate(img, marker_img, cv2.TM_CCOEFF_NORMED)

            # Get the coordinates of all the places where the marker is found
            loc = np.where(result >= marker_info['match_threshold'])
            coords = list(zip(*loc[::-1]))

            # Remove duplicates and close coordinates
            coords = list(set(coords))
            for coord in coords:
                for coord2 in coords:
                    if coord != coord2 and abs(coord[0] - coord2[0]) < 5 and abs(coord[1] - coord2[1]) < 5:
                        coords.remove(coord2)

            return coords



    def extract(self, img, frame_type) -> dict:

        """
        Extracts data from the frame.
        Currently only extracts the metrics from the summary screen.
        
        :param img: Image to extract data from
        :type img: PIL.Image
        :param frame_type: Type of screen
        :type frame_type: str
        :return: Dictionary of extracted data
        :rtype: dict
        """

        # Identify the layout
        layout = None
        for layout_option, layout_info in self.settings['global']['markers']['layout'][frame_type].items():
            for marker in layout_info:
                if self.check_marker(marker, img):
                    layout = layout_option
                    break
            if layout is not None:
                break

        if layout is None:
            raise Exception('Could not identify layout')


        measured_entities = self.settings['rois']['metrics'][frame_type][layout]

        def __bound(img):
            # find text and crop excessive borders
            crop_coords = cv2.boundingRect(img)
            img = img[(crop_coords[1]):crop_coords[1]+crop_coords[3], crop_coords[0]:crop_coords[0]+crop_coords[2]]
            # add a bit of padding
            img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0])
            return img

        # Extract metrics
        extracted_metrics = {}
        for entity in measured_entities:

            extracted_metrics[entity] = {}

            # Check if the entity is present, continue if not. Player is always present
            if entity != 'player':
                if not any([self.check_marker(marker, img) for marker in self.settings['global']['markers']['presence'][frame_type][entity]]):
                    continue
            
            for metric in measured_entities[entity]:
                # Crop the region of interest
                roi = self.settings['rois']['metrics'][frame_type][layout][entity][metric]
                processed_region = img.crop([roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]])

                # preprocess image for better OCR
                processed_region = cv2.cvtColor(np.array(processed_region), cv2.COLOR_BGR2GRAY)
                _, processed_region = cv2.threshold(processed_region, 100, 255, cv2.THRESH_BINARY)
                
                processed_region = __bound(processed_region)
                processed_region = cv2.blur(processed_region, (2,2))

                if metric == 'kills-assists-knocks':

                    # Cut the image in three pieces separated by the slashes (k_a_k_separator marker)
                    separator_coords = self.find_marker('kills-assists-separator', processed_region)
                    if len(separator_coords) < 2:
                        raise Exception('Could not find the separators for kills-assists-knocks')
                    separator_coords = sorted(separator_coords, key=lambda x: x[0])
                    separator_coords = [separator_coords[0][0], separator_coords[1][0]]

                    # Extract the three pieces
                    k_a_k = []
                    margin = 15 # equal to width of the separator
                    k_a_k.append(processed_region[:, :separator_coords[0]])
                    k_a_k.append(processed_region[:, separator_coords[0]+margin:separator_coords[1]])
                    k_a_k.append(processed_region[:, separator_coords[1]+margin:])

                    # Extract the text from each piece
                    k_a_k = [int(image_to_string(k_a_k[i], config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789Oo').replace('\n','').replace('O','0').replace('o','0')) for i in range(3)]

                    extracted_metrics[entity]['kills'] = k_a_k[0]
                    extracted_metrics[entity]['assists'] = k_a_k[1]
                    extracted_metrics[entity]['knocks'] = k_a_k[2]

                else:

                    parsed_text = image_to_string(processed_region, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789/:Oo').replace('\n','').replace('O','0').replace('o','0')

                    if metric == 'time_survived':
                        parsed_text = parsed_text.split(':')
                        extracted_metrics[entity][metric] = int(parsed_text[0])*60 + int(parsed_text[1])

                    # Overriding text extraction for the nickname
                    elif metric == 'nickname':
                        parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3').replace('\n','')
                        extracted_metrics[entity][metric] = parsed_text

                    else:
                        extracted_metrics[entity][metric] = int(''.join([num for num in parsed_text if num.isdigit()]))

                cv2.imshow('im', processed_region)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

                print()

        return extracted_metrics

        

if __name__ == '__main__':

    screen_analyser = ApexLegendsAnalyser()
    # screen_analyser.observe()
    image = Image.open('.logs/2023-02-20_10_24_34.png')
    frame_type = screen_analyser.frame_type(image)
    print(frame_type)
    data = screen_analyser.extract(image, frame_type)
    # print(screen_analyser.extract(Image.open('.logs/2023-02-20_12_31_50.png')))

    # img = cv2.imread('roi.png')
    # marker = "kills-assists-separator"

    # print(screen_analyser.find_marker(marker, img))