from pynput import keyboard

import os
import numpy as np
import pandas as pd
import cv2
from mss import mss
from PIL import Image
from pytesseract import image_to_string

from Levenshtein import distance as levenshtein_distance

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

        "Observs the screen and tracks various in-game states for further analysis."

        ongoing_match = False
        match_data = {}
        log_folder = None

        buffer = {
            'frame_type': "misc",
            'img': None
        }

        while True:
            # Capture screenshot and detect screen type
            img = self.capture()
            frame_info = self.frame_info(img)
            frame_type = frame_info['type']
            # print(frame_type)
            
            # Compare screen types and define what to do
            if frame_type == "start":
                if not ongoing_match:
                    # Start a new match
                    ongoing_match = True
                    match_data = {
                        'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M'),
                        'processed': False
                    }
                    log_folder = logs_dir + datetime.now().strftime('%Y-%m-%d_%H-%M') + '/'
                    os.mkdir(log_folder)

                if frame_type != buffer['frame_type']: # if starting screen is occuring for the first time
                    # Save frame and send for data extraction
                    img.save(log_folder + 'start.png')
                    data = self.extract(img, frame_type)
                    match_data.update(data)

            elif frame_type == "summary":
                if not ongoing_match:
                    # Start a new match, however late
                    ongoing_match = True
                    match_data = {
                        'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M'),
                        'processed': False
                    }
                    log_folder = logs_dir + datetime.now().strftime('%Y-%m-%d_%H-%M') + '/'
                    os.mkdir(log_folder)

                if frame_type == buffer['frame_type']: # if summary screen is reoccuring
                    # if the current one has more info, update the buffer
                    if self.compare(img, buffer['img']):
                        buffer['img'] = img
                else:
                    # start the buffer, beggining a new comparison chain
                    buffer['frame_type'] = frame_type
                    buffer['img'] = img

            elif frame_type == "gameplay":
                if not ongoing_match:
                    # Start a new match, however late
                    ongoing_match = True
                    match_data = {
                        'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M'),
                        'processed': False
                    }
                    log_folder = logs_dir + datetime.now().strftime('%Y-%m-%d_%H-%M') + '/'
                    os.mkdir(log_folder)

            elif frame_type == "legend_select":

                if not ongoing_match:
                    # Start a new match, however late
                    ongoing_match = True
                    match_data = {
                        'start_time': datetime.now().strftime('%Y-%m-%d_%H-%M'),
                        'processed': False
                    }
                    log_folder = logs_dir + datetime.now().strftime('%Y-%m-%d_%H-%M') + '/'
                    os.mkdir(log_folder)

                buffer['img'] = img
                        

            else:
                if buffer['frame_type'] == "summary":
                    # Save frame and send for data extraction
                    buffer['img'].save(log_folder + 'summary.png')
                    data = self.extract(buffer['img'], buffer['frame_type'])
                    match_data.update(data)

                    # ! Specific analysis for summary screen. This should be removed or replaced with a more general solution
                    # If player played wattson, try recognizing and saving the skin
                    username = self.settings['user']['name']
                    for entry in match_data['legends']:
                        if entry['nickname'] == username and entry['legend'] == 'wattson':
                            skin = self.__skin_recog(buffer['img'])
                            if skin != None:
                                match_data['skin'] = skin


                    # End match
                    # ! WARNING: This is a temporary solution, it will fail if 
                    #   - the game is closed before the summary screen is shown
                    #   - summary screen is shown again (e.g., "exit to lobby" prompt is called multiple times)
                    # Soltution:
                    #   - save data on each screen change or frame update
                    #   - stop match on frames further down the pipeline - breakdown, lobby, etc.
                    ongoing_match = False
                    match_data['end_time'] = datetime.now().strftime('%Y-%m-%d_%H-%M')

                    # Join summary data with legends data if present
                    if 'legends' in match_data.keys():
                        if match_data['player']['nickname'] != self.settings['user']['name']:
                            print('Nickname mismatch! ({} != {})'.format(match_data['player']['nickname'], self.settings['user']['name']))
                        nickname2player = {}
                        nickname2player[match_data['player']['nickname']] = 'player'
                        try:
                            nickname2player[match_data['teammate_a']['nickname']] = 'teammate_a'
                        except:
                            pass
                        try:
                            nickname2player[match_data['teammate_b']['nickname']] = 'teammate_b'
                        except:
                            pass
                        for entry in match_data['legends']:
                            # Define what nickname corresponds to what teammate using levenshtein distance
                            # This is done due to possible differences in nickname recognition between summary and legend_select screens

                            # Get the nickname with the smallest levenshtein distance
                            nickname = min(nickname2player.keys(), key=lambda x: levenshtein_distance(x, entry['nickname']))

                            # Write legend data to the corresponding teammate
                            match_data[nickname2player[nickname]]['legend'] = entry['legend']
                    
                    # Add skin info to player if present
                    if 'skin' in match_data:
                        match_data['player']['skin'] = match_data['skin']
                        del match_data['skin']

                    # Remove legends data
                    del match_data['legends']

                    match_data['processed'] = True



                    # Save match data
                    with open(log_folder + 'data.json', 'w') as f:
                        json.dump(match_data, f, indent=4)

                elif buffer['frame_type'] == "legend_select":
                    # Save frame and send for data extraction
                    buffer['img'].save(log_folder + 'legend_select.png')
                    data = self.extract(buffer['img'], buffer['frame_type'])
                    match_data.update(data)

            # Update buffer
            buffer['frame_type'] = frame_type

            # Save match data
            if ongoing_match:
                with open(log_folder + 'data.json', 'w') as f:
                    json.dump(match_data, f, indent=4)

            # Chill for a bit
            time.sleep(0.1)

            
    def frame_info(self, img) -> dict:
        
        """
        Detects the frame type and return type with metadata.

        :param img: Image to detect the frame type of.
        :type img: PIL.Image
        :return: Frame type and metadata.
        :rtype: dict
        """

        detected_frame_type = "misc"

        for frame_type in self.settings['global']['markers'].keys():
            # Compare markers, if there is a match, return the screen type
            marker_matched = False

            for marker in self.settings['global']['markers'][frame_type]['keys']:
                if self.check_marker(marker, img):
                    marker_matched = True
                    break

            if marker_matched:
                detected_frame_type = frame_type
                break
        
        return {"type": detected_frame_type, "metadata": {}}


    def mse(self, img1, img2) -> float:

        "Calculates the mean squared error between two images. Sizes must match."

        # Assert that the images are of the same size
        assert img1.shape == img2.shape

        err = np.sum((img1 - img2) ** 2)
        err /= float(np.prod(img1.shape))
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



    def extract(self, img, frame_type, **kwargs) -> dict:

        """
        Extracts data from the frame.
        
        :param img: Image to extract data from
        :type img: PIL.Image
        :param frame_type: Type of frame to extract data from
        :type frame_type: str
        :param kwargs: Additional arguments
        :type kwargs: dict
        :return: Dictionary of extracted data
        :rtype: dict
        """

        def identify_layout(__img, __frame_type):
            # Identify the layout
            layout = None
            for layout_option, layout_info in self.settings['global']['markers'][__frame_type]['layout'].items():
                for marker in layout_info:
                    if self.check_marker(marker, __img):
                        layout = layout_option
                        break
                if layout is not None:
                    break

            if layout is None:
                raise Exception('Could not identify layout')
            return layout

        def __bound(__img):
            # find text and crop excessive borders
            crop_coords = cv2.boundingRect(__img)
            __img = __img[(crop_coords[1]):crop_coords[1]+crop_coords[3], crop_coords[0]:crop_coords[0]+crop_coords[2]]
            # add a bit of padding
            __img = cv2.copyMakeBorder(__img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0])
            return __img


        # Extract metrics
        extracted_data = {}
        if frame_type == 'summary':
            # Identify the layout
            layout = identify_layout(img, frame_type)
            measured_entities = self.settings['rois']['data'][frame_type][layout]

            for entity in measured_entities:

                extracted_data[entity] = {}

                # Check if the entity is present, continue if not. Player and global are always present
                if entity != 'player' and entity != 'global':
                    if not any([self.check_marker(marker, img) for marker in self.settings['global']['markers'][frame_type]['presence'][entity]]):
                        continue
                
                for metric in measured_entities[entity]:
                    try:
                        # Crop the region of interest
                        roi = self.settings['rois']['data'][frame_type][layout][entity][metric]
                        processed_region = img.crop([roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]])

                        # Process the image
                        if metric == "place":
                            # Filter out the place only, which is written in a specific color
                            processed_region = np.array(processed_region)
                            processed_region = cv2.cvtColor(processed_region, cv2.COLOR_RGB2HSV)
                            filtered_color = np.array([22, 254, 251])
                            # Create and show a square of the filtered color for debugging
                            _square = np.zeros((100, 100, 3), np.uint8)
                            _square[:] = filtered_color

                            processed_region = cv2.inRange(processed_region, filtered_color - 20, filtered_color + 20)


                        else:
                            processed_region = cv2.cvtColor(np.array(processed_region), cv2.COLOR_BGR2GRAY)
                            _, processed_region = cv2.threshold(processed_region, 105, 255, cv2.THRESH_BINARY)
                            
                        processed_region = __bound(processed_region)

                        

                        if metric == 'kills-assists-knocks':
                            extracted_data[entity]['kills'] = None
                            extracted_data[entity]['assists'] = None
                            extracted_data[entity]['knocks'] = None

                            # Try processing metrics separatly first
                            # Cut the image in three pieces separated by the slashes (k_a_k_separator marker)
                            separator_coords = self.find_marker('kills-assists-separator', processed_region)
                            if len(separator_coords) >= 2:
                                
                                separator_coords = sorted(separator_coords, key=lambda x: x[0])
                                separator_coords = [separator_coords[0][0], separator_coords[1][0]]
                                processed_region = cv2.blur(processed_region, (4,4))

                                # Extract the three pieces
                                k_a_k = []
                                margin = 15 # equal to width of the separator
                                k_a_k.append(processed_region[:, :separator_coords[0]])
                                k_a_k.append(processed_region[:, separator_coords[0]+margin:separator_coords[1]])
                                k_a_k.append(processed_region[:, separator_coords[1]+margin:])

                                # Extract the text from each piece
                                k_a_k_metrics = []
                                for piece in k_a_k:
                                    try:
                                        k_a_k_metric = image_to_string(piece, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789Oo').replace('\n','').replace('O','0').replace('o','0')
                                        if k_a_k_metric == '':
                                            k_a_k_metric = image_to_string(piece, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789Oo').replace('\n','').replace('O','0').replace('o','0')
                                        k_a_k_metrics.append(int(k_a_k_metric))
                                    except:
                                        k_a_k_metrics.append(None)

                                extracted_data[entity]['kills'] = k_a_k_metrics[0]
                                extracted_data[entity]['assists'] = k_a_k_metrics[1]
                                extracted_data[entity]['knocks'] = k_a_k_metrics[2]

                            else:
                                # If the separator is not detected, try parsing the whole text
                                processed_region = cv2.blur(processed_region, (2,2))
                                parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789/Oo').replace('\n','').replace('O','0').replace('o','0').replace(' ','')
                                parsed_text = parsed_text.split('/')
                                if len(parsed_text) == 3:
                                    extracted_data[entity]['kills'] = int(parsed_text[0])
                                    extracted_data[entity]['assists'] = int(parsed_text[1])
                                    extracted_data[entity]['knocks'] = int(parsed_text[2])
                                else:
                                    raise Exception('Could not parse kills-assists-knocks')

                        else:

                            processed_region = cv2.blur(processed_region, (2,2))
                            parsed_text = image_to_string(processed_region, config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789:Oo#').replace('\n','').replace('O','0').replace('o','0')

                            if metric == 'time_survived':
                                
                                # If no text is detected, try again with a different psm
                                if parsed_text == '':
                                    parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789:Oo').replace('\n','').replace('O','0').replace('o','0')

                                parsed_text = parsed_text.split(':')
                                # Fix for when semicolon is not detected
                                if len(parsed_text) == 1:
                                    parsed_text = [parsed_text[0][:-2], parsed_text[0][-2:]]
                                extracted_data[entity][metric] = int(parsed_text[0])*60 + int(parsed_text[1])

                                # Sanity check: survived time should not be more than 30 minutes
                                if extracted_data[entity][metric] > 1800:
                                    # Leave blank, will be filled with NaN during analysis
                                    pass

                            # Overriding text extraction for the nickname
                            elif metric == 'nickname':
                                parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3').replace('\n','')
                                extracted_data[entity][metric] = parsed_text

                            else:
                                if parsed_text == '':
                                    print()
                                extracted_data[entity][metric] = int(''.join([num for num in parsed_text if num.isdigit()]))
                    except Exception as e:
                        print(e)
                        extracted_data[entity][metric] = None

        elif frame_type == 'start':
            data_to_process = self.settings['rois']['data'][frame_type]
            for parameter, roi in data_to_process.items():
                # Crop the region of interest
                processed_region = img.crop([roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]])
                processed_region = cv2.cvtColor(np.array(processed_region), cv2.COLOR_BGR2GRAY)
                _, processed_region = cv2.threshold(processed_region, 200, 255, cv2.THRESH_BINARY)

                processed_region = __bound(processed_region)
                processed_region = cv2.blur(processed_region, (2,2))
                parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3').replace('\n','').replace(' ', '_').lower()

                parsed_match = None
                if parameter == 'mode':
                    available_modes = self.settings['global']['modes']
                    for mode in available_modes:
                        if mode[:3] == parsed_text[:3]:
                            parsed_match = mode
                            break

                elif parameter == 'map':
                    available_maps = self.settings['global']['maps']
                    for _map in available_maps:
                        if _map[:3] == parsed_text[:3]:
                            parsed_match = _map
                            break


                extracted_data[parameter] = parsed_match

        elif frame_type == 'legend_select':
            extracted_data['legends'] = []

            # Detect layout by looking at the first color change in the bottom banner.
            # If it occurs roughly at the 0.33 mark, it's trios, if it's at 0.5 it's duos
            # Note: detecting the number of color changes is not reliable, as missing players are grayed out with the same color

            # Crop the region of interest
            layout_line_roi = self.settings['markers'][self.settings['global']['markers'][frame_type]['layout']]['roi']
            processed_region = img.crop([layout_line_roi[0], layout_line_roi[1], layout_line_roi[0]+layout_line_roi[2], layout_line_roi[1]+layout_line_roi[3]])
            processed_region = cv2.cvtColor(np.array(processed_region), cv2.COLOR_BGR2RGB)

            # Select 2 pixels to the right and left of the mark, check if they are the same color
            if not np.allclose(processed_region[0, int(processed_region.shape[1]/3.)-50], processed_region[0, int(processed_region.shape[1]/3.)+50], atol=10):
                layout = 'trios'
            elif not np.allclose(processed_region[0, int(processed_region.shape[1]/2.)-50], processed_region[0, int(processed_region.shape[1]/2.)+50], atol=10):
                layout = 'duos'
            else: # Layout unknown, further processing is not possible
                return extracted_data

            # For each player in the layout, attempt extractin nickname and legend
            for player, rois in self.settings['rois']['data'][frame_type][layout].items():
                player_data = {'nickname': None, 'legend': None}
                for parameter, roi in rois.items():
                    processed_region = img.crop([roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]])
                    processed_region = cv2.cvtColor(np.array(processed_region), cv2.COLOR_BGR2GRAY)

                    if parameter == 'nickname':
                        _, processed_region = cv2.threshold(processed_region, 115, 255, cv2.THRESH_BINARY)
                    else:
                        _, processed_region = cv2.threshold(processed_region, 180, 255, cv2.THRESH_BINARY)

                    processed_region = __bound(processed_region)
                    processed_region = cv2.blur(processed_region, (2,2))

                    if parameter == 'nickname':
                        parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3').replace('\n','')
                        if parsed_text == '':
                            continue # If no nickname is detected, skip this player
                        player_data[parameter] = parsed_text
                    else:
                        parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3').replace('\n','').replace(' ', '_').lower()
                        if parsed_text in self.settings['global']['legends']:
                            player_data[parameter] = parsed_text
                        else:
                            continue
                
                extracted_data['legends'].append(player_data)
                    

                

        return extracted_data

    
    def __skin_recog(self, img):
        """Returns the skin of the legend in the image"""

        # Detect layout (same way as in data extraction for summary)
        def identify_layout(__img, __frame_type):
            # Identify the layout
            layout = None
            for layout_option, layout_info in self.settings['global']['markers'][__frame_type]['layout'].items():
                for marker in layout_info:
                    if self.check_marker(marker, __img):
                        layout = layout_option
                        break
                if layout is not None:
                    break

            if layout is None:
                raise Exception('Could not identify layout')
            return layout
        layout = identify_layout(img, 'summary')

        # Load files with banner patches
        banner_patch_files = os.listdir('resources/markers/summary/skins/wattson')

        # Crop the region of interest
        roi = self.settings['rois']['special']['player_skin_patches'][layout]
        patch = img.crop([roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]])
        patch = cv2.cvtColor(np.array(patch), cv2.COLOR_BGR2RGB)
        
        # Compare the patch to the banner patches
        skin = None
        for banner_patch_file in banner_patch_files:
            banner_patch = cv2.imread(f'resources/markers/summary/skins/wattson/{banner_patch_file}')
            res = cv2.matchTemplate(patch, banner_patch, cv2.TM_CCOEFF_NORMED)
            if res > 0.9:
                skin = banner_patch_file.replace('.png', '')
                break
        
        return skin



        



if __name__ == '__main__':

    screen_analyser = ApexLegendsAnalyser()
    # screen_analyser.observe()
    image = Image.open('.logs/2023-02-23_23-46/summary.png')
    frame_info = screen_analyser.frame_info(image)
    # print(frame_info['type'])
    data = screen_analyser.extract(image, frame_info['type'])
    print(data)

    # skin = screen_analyser.get_skin(image)
    # print(skin)
