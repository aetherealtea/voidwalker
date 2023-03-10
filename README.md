# Project Voidwalker

This is a project to track the player performance in the Apex Legends videogame. The project is under development and as of now ready for limited use.

## Features

- Track game modes (limited to main ones: trios, duos, ranked)
- Track legends picked by each teammate
- Track end-of-game stats (kills, damage, etc.) for each teammate
- [Temporary] Recognize the skin for the legend Wattson if picked by the player (used for a one-time research with pre-defined skins and pre-defined in-game settings)

## Usage

Just run the game and the `main.py` script. The script will capture the screen and extract all the information needed. The script is set to track the matches and store the data in a `.logs` folder inside the corresponding subdirectory, marking the match with a timestamp. The data is stored in a `.json` file alongside with the screenshots of captured screens for debugging purposes.

The first run will invoke the user settings setup. The settings are stored in a `/configs/user.json` folder and can be edited manually.

To stop the script, press `Ctrl+C` in the terminal.

Limitations:
 - The game must be played in fullscreen mode with 2560x1440 resolution
 - Only English language is supported
 - The game can't be minimized during the legend selection or match summary screens.

 ## How it works

The script uses the `pyautogui` library to capture the screen, `opencv` library to do the screen processing (detecting screen types and selecting rois) and the `pytesseract` library to extract the text from the screen. The screens are detected by comparing 'markers' (certain cropped regions of the screen that are unique for each screen type) with the current screen. The markers are stored in the `/resources/markers` folder.


## Future plans

- Fine-tune the tessaract OCR to raise the accuracy of the text extraction
- Link the screen capture to the particular window
- Track the entire match and extract metrics such as incoming damage, weapons used, accuracy, ping, etc.
- Make the script resolition-independent
- Add GUI

## Known issues

- "Legends" tab in the main menu is recognised as the legend selection screen
- Script crashes if one or more teammates are missing while trying to match the data from legend selection screen with the data from the match summary screen
- Some metrics are not extracted correctly (e.g. damage dealt) with no way to detect the issue (e.g. 779 damage dealt is extracted as 7719)