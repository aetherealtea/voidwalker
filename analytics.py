import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
import os
import json


def data_to_table(ts_start, ts_end, match_data_path, csv_path=None):
    """
    Converts the json data entries for each match into a Pandas dataframe and saves it to a csv file.

    :param ts_start: Timestamp of the first match
    :param ts_end: Timestamp of the last match
    :param match_data_path: Path to the folder containing the match data
    :param csv_path: Path to the csv file to be created
    """

    def validate_timestamp(ts):
        try:
            datetime.strptime(ts, '%Y-%m-%d_%H-%M')
            return True
        except:
            return False

    # List subfolders in the match data folder, filter out dirs with timestamp-like name
    match_dirs = [f for f in os.listdir(match_data_path) if os.path.isdir(os.path.join(match_data_path, f)) and validate_timestamp(f)]

    # Filter dirs to match range
    match_dirs = [f for f in match_dirs if ts_start <= f <= ts_end]

    # Gather all data from the match dirs
    data = []
    for match_dir in match_dirs:
        with open(os.path.join(match_data_path, match_dir, 'data.json')) as f:
            data.append(json.load(f))

    # Create a dataframe from the data, unpacking players' data
    output = []
    ix_counter = 0
    for match in data:
        for player in ['player', 'teammate_a', 'teammate_b']:
            if player not in match.keys():
                continue
            entry = {}
            entry['index'] = ix_counter
            entry['start_time'] = datetime.strptime(match['start_time'], '%Y-%m-%d_%H-%M')
            entry['mode'] = match.get('mode', None)
            entry['map'] = match.get('map', None)
            entry['place'] = match['global']['place']
            entry['duration'] = (datetime.strptime(match['end_time'], '%Y-%m-%d_%H-%M') - datetime.strptime(match['start_time'], '%Y-%m-%d_%H-%M')).total_seconds()/60
            entry['is_player'] = player == 'player'
            for k, v in match[player].items():
                entry[k] = v 
            output.append(entry)
        ix_counter += 1

    df = pd.DataFrame(output)

    # Save the dataframe to csv
    if csv_path is not None:
        df.to_csv(csv_path, index=False)

    return df


def get_skin_data():
    """
    Filter out the skin data from the match data and plot metrics.
    """

    # Load the match data
    df = data_to_table('2020-01-01_00-00', '2024-12-31_23-59', '.logs/')

    # Filter out the skin data
    df = df[df['skin'].notna()]
    df = df.dropna()
    # add DPS column
    df['DPS'] = df['damage'] / df['time_survived']
    print (df.head())

    # Aggregate the data and find metrics averages for each skin
    data = {
        'matches': df.groupby('skin')['index'].count(),
        'kills': df.groupby('skin')['kills'].mean(),
        'assists': df.groupby('skin')['assists'].mean(),
        'knocks': df.groupby('skin')['knocks'].mean(),
        'damage': df.groupby('skin')['damage'].mean(),
        'time_survived': df.groupby('skin')['time_survived'].mean(),
        'DPS': df.groupby('skin')['DPS'].mean()
    }

    
    # Plot the metrics
    for k, v in data.items():
        sns.barplot(x=v.index, y=v)
        plt.title(k)
        plt.show()
        
    print(data)


def detect_anomalies():
    """
    Plot the metrics to detect anomalies.
    """

    # Load the match data
    df = data_to_table('2020-01-01_00-00', '2024-12-31_23-59', '.logs/')

    # Filter out the player data
    df = df[df['is_player']]

    # Plot the metrics as time series
    for k in ['damage', 'time_survived']:
        sns.lineplot(x=df['start_time'], y=df[k])
        plt.show()


    

if __name__ == '__main__':
    get_skin_data()
    # detect_anomalies()
