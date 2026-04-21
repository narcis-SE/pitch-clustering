from distro import name
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

def find_similar_pitchers(data, target_pitcher, target_year, n_components=3):
    ''''Find similar pitchers based on pitch features.'''

    features = ['release_speed', 'release_spin_rate', 'release_pos_x', 'release_pos_z', 'release_extension']
    pitchers = data.groupby(['player_name', data['game_date'].dt.year])[features].mean().reset_index()
    pitchers.columns = ['player_name', 'year'] + features
    pitchers['year'] = pitchers['year'].astype(int)
    pitchers = pitchers.dropna().reset_index(drop=True)

    if pitchers.empty: return None

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(pitchers[features])

    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)

    pca_full = PCA(n_components=len(features))
    pca_full.fit(scaled_data)
    # print(pca_full.explained_variance_ratio_)

    pitcher_match = pitchers[
        (pitchers['player_name'] == target_pitcher) & (pitchers['year'] == int(target_year))
    ]

    if pitcher_match.empty: return None

    match_idx = pitcher_match.index[0]
    match_pca = pca_data[match_idx].reshape(1, -1)

    knn = NearestNeighbors(n_neighbors=min(10, len(pitchers)))
    knn.fit(pca_data)
    dist, idx = knn.kneighbors(match_pca)

    res = pitchers.iloc[idx[0]].copy()
    res['sim_score'] = 1/(1 + dist[0])

    final = res[~(
        (res['player_name'] == target_pitcher) &
        (res['year'] == int(target_year))
    )]

    final = final.drop_duplicates(subset='player_name', keep='first').head(3).reset_index(drop=True)

    return final 

def display_knn_experiment(pitcher):
    st.header('How Consistent Do Pitchers Pitch Over Time?')
    st.info("""
    This experiment tests whether we can distinguish pitch types based on physical metrics. 
    A high weighted F1 score indicates that a pitcher's range of pitches are physically discernible and longitudinally consistent and reliable.
    """)

    @st.cache_data
    def getKNNResults():
        return pd.read_csv('knn_results.csv')
    
    @st.cache_data
    def getKNNFolds():
        return pd.read_csv('knn_folds.csv')
    
    knn_results = getKNNResults()
    knn_folds = getKNNFolds()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg. Weighted F1", f"{knn_results['weighted_f1'].mean():.3f}")
    col2.metric("Top Performer", knn_results.iloc[0]['pitcher'], f"{knn_results.iloc[0]['weighted_f1']:.3f}")
    col3.metric("Total Pitchers Analyzed", len(knn_results))

    # st.divider()

    st.subheader(f'Pitcher Drilldown for {pitcher}')
    selected_pitcher = pitcher 

    pitcher_data = knn_results[knn_results['pitcher'] == selected_pitcher]

    if pitcher_data.empty:
        st.warning("No KNN results available for this pitcher.")
    else:
        pitcher_data = pitcher_data.iloc[0]

        def f1_scores(score):
            if score >= 0.9:
                return 'Highly consistent pitches'
            
            elif score >= 0.75:
                return 'Moderately consistent pitches'
            
            elif score >= 0.5:
                return 'Somewhat consistent pitches'
            
            else: 
                return 'Inconsistent pitches'
        
        c1, c2, c3 = st.columns(3)
        with c1:
            st.write("**Model Parameters**")
            st.write(f"**Best $k$:** {pitcher_data['best_k']}")
            st.write(f"**Distance Metric:** {pitcher_data['best_metric'].capitalize()}")
            st.write(f"**Pitch Types:** {pitcher_data['pitch_types'].replace(',', ', ')}")
            st.write(f"**Data Span:** {pitcher_data['n_years']} Years")
            st.write(f"**Total Pitches:** {pitcher_data['n_pitches']:,}")
        with c2:
            st.metric("Weighted F1 Score", f"{pitcher_data['weighted_f1']:.3f}")
            st.info(f1_scores(pitcher_data['weighted_f1']))
        with c3:
            rank = knn_results['weighted_f1'].rank(ascending=False).loc[knn_results['pitcher'] == selected_pitcher].iloc[0]
            st.metric("Leaderboard Rank", f"{int(rank)} / {len(knn_results)}")
            n_pitch_types = len(pitcher_data['pitch_types'].split(','))
            st.metric("# Pitch Types", n_pitch_types)

        st.divider()
        pitcher_folds = knn_folds[knn_folds['pitcher'] == selected_pitcher]

        if pitcher_folds.empty or (
            len(pitcher_folds) == 1 and
            str(pitcher_folds['test_year'].iloc[0]) == 'holdout'
        ):
            st.info("Insufficient years of data to show classification trend for this pitcher.")
        else:
            fold_fig = px.line(
                pitcher_folds,
                x='test_year',
                y='weighted_f1',
                markers=True,
                title=f'Classification Stability Over Time: {selected_pitcher}',
                labels={'test_year': 'Test Year', 'weighted_f1': 'Weighted F1 Score'},
                range_y=[0, 1]
            )
            fold_fig.add_hline(
                y=0.8,
                line_dash='dash',
                line_color='red',
                annotation_text='0.8 threshold'
            )
            fold_fig.update_layout(title_x=0.5, title_xanchor='center')
            st.plotly_chart(fold_fig, use_container_width=True)
            st.caption(
                "Each point shows how well the model classified pitch types in that year, "
                "trained on all prior years. A declining trend may indicate meaningful "
                "changes in the pitcher's mechanics over time."
            )

    st.divider()
    st.subheader("Reliability Leaderboard Among Selected Pitchers")

    fig_bar = px.bar(
        knn_results.sort_values('weighted_f1'),
        x='weighted_f1',
        y='pitcher',
        orientation='h',
        color='weighted_f1',
        color_continuous_scale='RdYlGn',
        range_color=[0.3, 1.0],
        hover_data=['pitch_types', 'best_k', 'best_metric', 'n_pitches', 'n_years'],
        labels={
            'weighted_f1': 'Weighted F1 Score',
            'pitcher': 'Pitcher',
            'pitch_types': 'Pitch Types',
            'best_k': 'Best k',
            'best_metric': 'Metric',
            'n_pitches': '# Pitches',
            'n_years': 'Years of Data'
        },
        title='Pitch Classification Reliability by Pitcher',
        height=900
    )
    fig_bar.add_vline(
        x=0.8,
        line_dash='dash',
        line_color='white',
        annotation_text='0.8 threshold'
    )
    fig_bar.update_layout(title_x=0.5, title_xanchor='center')
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption(
        "Pitchers with higher F1 scores have more physically consistent and distinguishable pitch types. "
        "Lower scores may reflect pitch type changes over time or overlapping pitch characteristics. "
    )


def main():

    # pitchers = ['justin verlander', 'max scherzer', 'chris sale', 'gerrit cole', 'corbin burnes', 'clayton kershaw', 'yu darvish', 'lance lynn', 'sonny gray', 'aroldis chapman']
    # keepCols = ["game_date", "player_name", "pitcher", "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension", "spin_axis"]
    # pitchers = [
    #             "Logan Gilbert", "Seth Lugo", "Logan Webb", "Zack Wheeler", "Aaron Nola",
    #             "Corbin Burnes", "Clayton Kershaw", "Tarik Skubal", "George Kirby", "Dylan Cease",
    #             "Sean Manaea", "Cristopher Sánchez", "Brandon Pfaadt", "Kevin Gausman", "JP Sears",
    #             "Bryce Miller", "Brady Singer", "Tyler Anderson", "Tanner Houck", "Bailey Ober",
    #             "Garrett Crochet", "Yu Darvish", "mason miller", "Ryan Pepiot", "Chris Sale",
    #             "Reid Detmers", "Tyler Glasnow", "Ranger Suárez", "Paul Skenes", "Yency Almonte",
    #             "Taijuan Walker", "Gerrit Cole", "Triston McKenzie", "Walker Buehler", "Jacob deGrom",
    #             "Justin Verlander", "Max Scherzer", "Jacob Bird", "Kyle Gibson", "Miles Mikolas"
    #         ]
    # keepCols = [
    #             "pitch_type", "game_date", "release_speed", "release_pos_x", "release_pos_z",
    #             "player_name", "pitcher", "events", "description", "spin_dir",
    #             "zone", "des", "p_throws", "type", "game_year",
    #             "release_spin_rate", "release_extension", "release_pos_y", "babip_value", "iso_value",
    #             "launch_speed", "pitch_number", "pitch_name", "spin_axis", "hyper_speed",
    #             "age_pit", "n_thruorder_pitcher", "pitcher_days_since_prev_game", "arm_angle"
    #         ]
    # tempData = []
    # for pitcherName in pitchers:
    #     print(f'working on pitcher {pitcherName}')
    #     fullName = pitcherName.split(' ')
    #     try:
    #         temp = get_pitcher_data(fullName[0], fullName[1], '2015-01-01', '2025-12-31').filter(keepCols)#.dropna()
    #         tempData.append(temp)
    #         # print(len(temp))
    #     except ValueError as e:
    #         print(e)
    # print('data retrieval complete, concatenating and writing to csv')
    # data = pd.concat(tempData, ignore_index = True)
    # data.to_csv('pitcher_data_detailed.csv', index = False)

    @st.cache_data
    def getPitcherData():
        return pd.read_csv('pitcher_data_detailed_cleaned.csv', parse_dates = ['game_date'])
    
    @st.cache_data
    def getRFPred():
        return pd.read_csv('random_forest_predictions.csv', parse_dates = ['period'])
    
    @st.cache_data
    def getKNNResults():
        return pd.read_csv('knn_results.csv')
    
    @st.cache_data
    def getKNNFolds():
        return pd.read_csv('knn_folds.csv')

    appData = getPitcherData()
    st.title('MLB Pitch Clustering')
    selectPitcher = st.selectbox('Select Pitcher', sorted(appData['player_name'].unique()), index = 4)
    selectYear = st.slider('Select Year', min_value = appData['game_date'].dt.year.min(), max_value = appData['game_date'].dt.year.max(), value = 2022)

    dfFilter = appData.loc[(appData['player_name'] == selectPitcher) & (appData['game_date'].dt.year == selectYear), :]

    # st.subheader('EDA Visuals')
    # basicPlot = px.scatter(dfFilter, x = 'pfx_x', y = 'pfx_z', color = 'pitch_name', hover_data = ['release_speed'], labels = {'pfx_x': 'Horizontal Break (in)', 'pfx_z': 'Vertical Break (in)'}, title = f'{selectPitcher} {selectYear} Pitches by Type')
    # basicPlot.update_layout(title_x = 0.5, title_xanchor = 'center')
    # st.plotly_chart(basicPlot, width = 'stretch')
    
    # st.divider()

    # st.subheader('Feature Distributions by Pitch Type')
    features = ['release_speed', 'release_spin_rate', 'release_pos_x', 'release_pos_z']
    titleFeatures = ['Release Speed', 'Spin Rate', 'Horizontal Release', 'Vertical Release']
    col1, col2 = st.columns(2)
    cols = [col1, col2]
    for i, feature in enumerate(features):
        hist = px.histogram(
            dfFilter,
            x=feature,
            color='pitch_name',
            barmode='overlay',
            opacity=0.7,
            title=f'{selectPitcher} {titleFeatures[i]} in {selectYear}',
        )
        cols[i%2].plotly_chart(hist)

    st.divider()

    display_knn_experiment(selectPitcher)

    st.divider()

    st.subheader("Random Forest Residual Performance Indicator")
    st.info('This section shows residual plots for the random forest test set predictions of a given pitcher. The random forest model uses data from January 2015 through July 2022 (approximately 70 percent of the available data from the StatCast era) to make predictions about a pitcher\'s monthly average spin rate for August 2022 onwards. ')
    predResults = getRFPred()
    if selectYear >= 2022:
        rfFilter = predResults.loc[(predResults['player_name'] == selectPitcher) & (predResults['period'] >= f'{selectYear}-01-01'), :]
        rfPlot = px.scatter(rfFilter, x = 'period', y = 'residual', labels = {'period': 'Date', 'residual': 'Random Forest Residual'}, title = f'{selectPitcher} Random Forest Residuals for the Period {selectYear} through 2025')
        rfPlot.update_layout(title_x = 0.5, title_xanchor = 'center')
        st.plotly_chart(rfPlot, width = 'stretch')

        rfHist = px.box(rfFilter, x = 'residual', title = ' ')#, title = f'{selectPitcher} Random Forest Residuals for the Period {selectYear} through 2025')
        # rfHist.update_layout(title_x = 0.5, title_xanchor = 'center')
        st.plotly_chart(rfHist, width = 'stretch')

    else:
        st.warning('Test set prediction results only available for August 2022 and on.')

    st.caption('An outlier residual indicates that a significant difference exists between the best model\'s prediction and the actual test set data. Unexpected high residuals indicate a significant gap between the pitcher\'s expected average spin rate and actual results, which can indicate injury or other issues.')

    st.divider()

    st.header("Pitcher Similarity")
    st.info("This section uses the K-nearest neighbors algorithm to find which pitcher-year most closely resembles the selected pitcher's pitch mechanics. " \
    "The similarity score is based on the distance in PCA space, with a higher score indicating a closer match. " \
    "Note that this is based on average pitch characteristics for the season, so it may not capture in-season changes or specific pitch types. " \
    "Results may also include different seasons of the selected pitcher, since their own career trajectory can be the closest physical match.")

    sim_results = find_similar_pitchers(appData, selectPitcher, selectYear)

    if sim_results is not None:
        cols = st.columns(3)
        for i, (_, row) in enumerate(sim_results.iterrows()):
            if i >= 3:
                break
            with cols[i]:
                name = ' '.join(reversed(row['player_name'].split(', ')))
                st.markdown(f"### {name}")
                st.markdown(f"**Season:** {int(row['year'])}")
                st.write(f"Similarity Score: {row['sim_score']:.2%}")                
                st.caption(f"Velocity: {row['release_speed']:.1f} mph")
                st.caption(f"Spin: {int(row['release_spin_rate'])} rpm")
    else:
        st.warning("Select a different year or pitcher to generate similarity matches.")

    # st.divider()


if __name__ == "__main__":
    main()
