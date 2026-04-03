import requests
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from pybaseball import statcast, statcast_pitcher, playerid_lookup, pitching_stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def get_player_id(first_name: str, last_name: str) -> int:
    '''Get the MLBAM player ID for a given player name. Returns the player ID as an int.'''
    player = playerid_lookup(last_name, first_name)
    if not player.empty:
        return player['key_mlbam'].values[0]
    else:
        raise ValueError(f'Player {first_name} {last_name} not found.')
    
def get_pitcher_data(first_name: str , last_name: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    '''Get the statcast data for a given pitcher and date range. Returns a Dataframe with statcast data.'''
    player_id = get_player_id(first_name, last_name)
    data = statcast_pitcher(start_dt, end_dt, player_id)
    return data

def find_similar_pitchers(data, target_pitcher, target_year, n_neighbors=3):
    ''''Find similar pitchers based on pitch features.'''

    features = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'release_extension']
    pitchers = data.groupby(['player_name'], data['game_date'].dt.year)[features].mean().reset_index()
    pitchers.column = ['player_name', 'year'] + features
    pitchers['year'] = pitchers['year'].astype(int)
    pitchers = pitchers.dropna().reset_index(drop=True)

    if pitchers.empty: return None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pitchers[features])

    


def main():

    ############# code to retrieve and write data to csv for use in the visualization app since we don't want to have to fetch it each time. commenting this out but leaving it here in case we want to add more #############

    # we could show 3, 5, 10 year evolutions where possible?
    # pitchers = ['justin verlander', 'max scherzer', 'chris sale', 'gerrit cole', 'corbin burnes', 'clayton kershaw', 'yu darvish', 'garrett crochet', 'mason miller']
    # keepCols = ["game_date", "player_name", "pitcher", "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension", "spin_axis"]

    # tempData = []
    # for pitcherName in pitchers:
    #     print(f'working on pitcher {pitcherName}')
    #     fullName = pitcherName.split(' ')
    #     temp = get_pitcher_data(fullName[0], fullName[1], '2015-01-01', '2025-12-31').filter(keepCols).dropna()
    #     tempData.append(temp)

    # data = pd.concat(tempData, ignore_index = True)
    # data.to_csv('pitcher_data.csv', index = False)
    # print(data.head())
    # print(data.shape[0])

    # response = requests.get("https://httpbin.org/get")
    # print(response.status_code)
    # print(response.json())

    # pass

    @st.cache_data
    def getData():
        return pd.read_csv('pitcher_data.csv', parse_dates = ['game_date'])
    
    appData = getData()
    st.title('MLB Pitch Clustering')
    selectPitcher = st.selectbox('Select Pitcher', sorted(appData['player_name'].unique()), index = 4)
    selectYear = st.slider('Select Year', min_value = appData['game_date'].dt.year.min(), max_value = appData['game_date'].dt.year.max())

    dfFilter = appData.loc[(appData['player_name'] == selectPitcher) & (appData['game_date'].dt.year == selectYear), :]

    st.subheader('Basic Scatter Plot Test')
    basicPlot = px.scatter(dfFilter, x = 'pfx_x', y = 'pfx_z', color = 'pitch_name', hover_data = ['release_speed'], labels = {'pfx_x': 'Horizontal Break (in)', 'pfx_z': 'Vertical Break (in)'})
    st.plotly_chart(basicPlot, width = 'stretch')

    st.subheader('Feature Distributions by Pitch Type')
    features = ['release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'spin_axis']
    col1, col2 = st.columns(2)
    cols = [col1, col2]
    for i, feature in enumerate(features):
        hist = px.histogram(
            dfFilter,
            x=feature,
            color='pitch_name',
            barmode='overlay',
            opacity=0.7,
            title=f'{feature} of {selectPitcher} in {selectYear}',
        )
        cols[i%2].plotly_chart(hist)

if __name__ == "__main__":
    main()
