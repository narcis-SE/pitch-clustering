import requests
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from pybaseball import statcast, statcast_pitcher, playerid_lookup, pitching_stats

def get_player_id(first_name: str, last_name: str) -> int:
    '''Get the MLBAM player ID for a given player name. Returns the player ID as an int.'''
    player = playerid_lookup(last_name, first_name)
    if not player.empty:
        return player['key_mlbam'].values[0]
    else:
        raise ValueError(f'Player {first_name} {last_name} not found.')
    
def get_pitcher_data(first_name: str , last_name: str, start_dt: str, end_dt: str) -> 'pd.DataFrame':
    '''Get the statcast data for a given pitcher and date range. Returns a Dataframe with statcast data.'''
    player_id = get_player_id(first_name, last_name)
    data = statcast_pitcher(start_dt, end_dt, player_id)
    return data

def main():
    #some pitchers with longer careers that took place at least partially within the statcast era. We could show 10 year evolutions for each of these potentially?
    pitchers = ['justin verlander', 'max scherzer', 'chris sale', 'gerrit cole', 'corbin burnes', 'clayton kershaw']

    #one example with kershaw using code that was here before
    data = get_pitcher_data('clayton', 'kershaw', '2015-01-01', '2023-12-31')
    keepCols = ["pitch_type", "pitch_name", "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension", "spin_axis"]
    data = data.filter(keepCols).dropna()
    print(data.head())
    print(data.shape[0])
    # response = requests.get("https://httpbin.org/get")
    # print(response.status_code)
    # print(response.json())


if __name__ == "__main__":
    main()
