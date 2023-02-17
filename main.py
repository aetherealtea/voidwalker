from pynput import keyboard

import numpy as np
import cv2
from mss import mss
from PIL import Image
from pytesseract import image_to_string

from datetime import datetime

import json


logs_dir = '.logs/'
monitor_size = {'top': 0, 'left': 0, 'width': 0, 'height': 0}
with open('settings.json') as f:
    settings = json.load(f)


def set_monitor():
    "Chooses the monitor to capture from."
    monitors = mss().monitors
    print('Monitors detected: {}'.format(len(monitors)))
    for i, monitor in enumerate(monitors):
        print('{}. {}x{}@{}:{}'.format(i, monitor['width'], monitor['height'], monitor['left'], monitor['top']))
    monitor = int(input('Choose monitor: '))
    return monitors[monitor]


def capture():
    "Captures the screen."
    scrennshot = mss().grab(monitor_size)
    img = Image.frombytes('RGB', scrennshot.size, scrennshot.bgra, 'raw', 'BGRX')
    img.save(logs_dir+datetime.now().strftime("%Y-%m-%d_%H_%M_%S")+'.png') # Save to logs with timestamp
    print('Screen captured')


def analyse(img):

    "Analyses the screen."
    
    metrics_rois = settings['global']['roi']['summary']['trios'] # Only available for a specific mode rn, hence the hard coded values
    extracted_metrics = {
        'kills': -1,
        'assists': -1,
        'knocks': -1,
        'damage': -1,
        'time_survived': -1,
        'place': -1,
        'team_kills': -1,
        'team_damage': -1,
        'team_performance': 'unknown'
    }

    preproc_metrics = {}
    for metric, roi in metrics_rois.items():
        processed_region = img.crop([roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]])

        # preprocess image for better OCR
        processed_region = cv2.cvtColor(np.array(processed_region), cv2.COLOR_BGR2GRAY)
        # processed_region = cv2.adaptiveThreshold(processed_region, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        #                         cv2.THRESH_BINARY_INV, 47, 2)
        _, processed_region = cv2.threshold(processed_region, 95, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        # processed_region = cv2.morphologyEx(processed_region, cv2.MORPH_OPEN, kernel)
        processed_region = cv2.blur(processed_region, (2,2))
        # processed_region = cv2.Canny(processed_region, 100, 200)

        # cv2.imshow('im', processed_region)
        # cv2.waitKey(3000)
        # cv2.destroyAllWindows()
        parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789/:Oo').replace('\n','').replace('O','0').replace('o','0')

        
        if metric == 'kills-assists-knocks':
            parsed_text = [num.replace(' ', '') for num in parsed_text.split('/')]
            preproc_metrics['kills'] = int(parsed_text[0])
            preproc_metrics['assists'] = int(parsed_text[1])
            preproc_metrics['knocks'] = int(parsed_text[2])
        
        elif metric == 'time_survived':
            parsed_text = parsed_text.split(':')
            preproc_metrics[metric] = int(parsed_text[0])*60 + int(parsed_text[1])

        elif metric == 'team_kills':
            # processed_region = img.crop([roi[0], roi[1], roi[0]+roi[2], roi[1]+roi[3]])
            _, processed_region = cv2.threshold(processed_region, 250, 255, cv2.THRESH_BINARY)
            parsed_text = image_to_string(processed_region, config='--psm 6 --oem 3 outputbase digits').replace('\n','')
            # cv2.imshow('im', processed_region)
            # cv2.waitKey(2000)
            # cv2.destroyAllWindows()
            preproc_metrics[metric] = int(''.join([num for num in parsed_text if num.isdigit()]))
        
        else:
            if metric == 'teammate_2_damage':
                print()
            preproc_metrics[metric] = int(''.join([num for num in parsed_text if num.isdigit()]))

    # evaluate team performance based on teammates' damage relative to player's damage
    teammate_damages = (
        'player', preproc_metrics['damage'],
        'teammate1', preproc_metrics['teammate_1_damage'],
        'teammate2', preproc_metrics['teammate_2_damage']
    ).sort(key = lambda x: x[1])
    if teammate_damages[0][0] == 'player':
        extracted_metrics['team_performance'] = 'lower'
    elif teammate_damages[1][0] == 'player':
        extracted_metrics['team_performance'] = 'same'
    else:
        extracted_metrics['team_performance'] = 'higher'

    for key in extracted_metrics.keys():
        if key in preproc_metrics.keys():
            extracted_metrics[key] = preproc_metrics[key]
        else:
            raise Exception('Metric {} not found in preprocessed metrics'.format(key))

    return extracted_metrics

        
                

            


def exit():
    "Terminates the program."
    print('Exiting')
    raise SystemExit(0)

if __name__ == '__main__':
    # monitor_size = set_monitor()
    # with keyboard.GlobalHotKeys({
    #         '<alt>+<ctrl>+p': capture,
    #         '<alt>+<ctrl>+e': exit}) as h:
    #     h.join()
    # print(analyse(Image.open('.logs/2023-02-16_17_46.png')))
    # print(analyse(Image.open('.logs/2023-02-16_21_16.png')))
    print(analyse(Image.open('.logs/2023-02-16_21_16.png')))