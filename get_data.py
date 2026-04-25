import pandas as pd
from pybaseball import statcast, statcast_pitcher, playerid_lookup, pitching_stats

def get_pitcher_data(first_name: str, last_name: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    '''Get the statcast data for a given pitcher and date range. Returns a Dataframe with statcast data.'''
    player_id = get_player_id(first_name, last_name)
    data = statcast_pitcher(start_dt, end_dt, player_id)
    return data

def get_player_id(first_name: str, last_name: str) -> int:
    '''Get the MLBAM player ID for a given player name. Returns the player ID as an int.'''
    player = playerid_lookup(last_name, first_name)
    if not player.empty:
        return player['key_mlbam'].values[0]
    else:
        raise ValueError(f'Player {first_name} {last_name} not found.')

keepCols = ["game_date", "player_name", "pitcher", "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension", "spin_axis"]
pitchers = [
            "Logan Gilbert", "Seth Lugo", "Logan Webb", "Zack Wheeler", "Aaron Nola",
            "Corbin Burnes", "Clayton Kershaw", "Tarik Skubal", "George Kirby", "Dylan Cease",
            "Sean Manaea", "Cristopher Sánchez", "Brandon Pfaadt", "Kevin Gausman", "JP Sears",
            "Bryce Miller", "Brady Singer", "Tyler Anderson", "Tanner Houck", "Bailey Ober",
            "Garrett Crochet", "Yu Darvish", "mason miller", "Ryan Pepiot", "Chris Sale",
            "Reid Detmers", "Tyler Glasnow", "Ranger Suárez", "Paul Skenes", "Yency Almonte",
            "Taijuan Walker", "Gerrit Cole", "Triston McKenzie", "Walker Buehler", "Jacob deGrom",
            "Justin Verlander", "Max Scherzer", "Jacob Bird", "Kyle Gibson", "Miles Mikolas"
        ]
keepCols = [
            "pitch_type", "game_date", "release_speed", "release_pos_x", "release_pos_z",
            "player_name", "pitcher", "events", "description", "spin_dir",
            "zone", "des", "p_throws", "type", "game_year",
            "release_spin_rate", "release_extension", "release_pos_y", "babip_value", "iso_value",
            "launch_speed", "pitch_number", "pitch_name", "spin_axis", "hyper_speed",
            "age_pit", "n_thruorder_pitcher", "pitcher_days_since_prev_game", "arm_angle"
        ]
tempData = []
for pitcherName in pitchers:
    print(f'working on pitcher {pitcherName}')
    fullName = pitcherName.split(' ')
    try:
        temp = get_pitcher_data(fullName[0], fullName[1], '2015-01-01', '2025-12-31').filter(keepCols)#.dropna()
        tempData.append(temp)
        # print(len(temp))
    except ValueError as e:
        print(e)
print('data retrieval complete, concatenating and writing to csv')
data = pd.concat(tempData, ignore_index = True)
data.to_csv('pitcher_data_detailed.csv', index = False)