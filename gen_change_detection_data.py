import pandas as pd

data = pd.read_csv('pitcher_data_detailed_cleaned.csv').reset_index()
data['game_date'] = pd.to_datetime(data['game_date'])
cols = ['game_date', 'release_speed', 'release_pos_x',
       'release_pos_z', 'player_name', 'zone', 'release_spin_rate', 'release_extension', 'release_pos_y',
       'pitch_name', 'spin_axis']
grouped = data[cols].groupby(['player_name', 'game_date', 'pitch_name']).agg(
                    avg_release_speed = ('release_speed', 'mean'),
                    avg_release_pos_x = ('release_pos_x', 'mean'),
                    avg_release_pos_z = ('release_pos_z', 'mean'),
                    avg_zone = ('zone', 'mean'),
                    avg_release_spin_rate = ('release_spin_rate', 'mean'),
                    avg_release_extension = ('release_extension', 'mean'),
                    avg_release_pos_y = ('release_pos_y', 'mean'),
                    avg_spin_axis = ('spin_axis', 'mean'),
                    num_pitches = ('game_date', 'count')
                    ).sort_values(by = ['player_name', 'game_date', 'pitch_name']).reset_index()
grouped.to_csv('change_detection.csv', index=False)