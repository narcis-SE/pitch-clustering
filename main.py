from distro import name
# import requests
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
from changepoint_online import MDFocus, MDGaussian
# from plotly.subplots import make_subplots
import plotly.colors as pc
import matplotlib.pyplot as plt



def analyze_pitcher_stability(df, pitcher_id, baseline_size=500, window_size=100):
    pitcher_df = df[df['pitcher'] == pitcher_id].copy()
    pitcher_name = pitcher_df['player_name'].iloc[0]
    pitcher_df['game_date'] = pd.to_datetime(pitcher_df['game_date'])
    pitcher_df = pitcher_df.sort_values('game_date', kind='mergesort').reset_index(drop=True)
    features = ['release_speed', 'release_spin_rate', 'release_pos_x', 'release_pos_z', 'release_extension']
    clean_df = pitcher_df.dropna(subset=features).copy().reset_index(drop=True)

    #Scale and baseline
    scaler = StandardScaler()
    X = scaler.fit_transform(clean_df[features])
    gmm_ref = GMM(n_components=3, covariance_type='full', random_state=42)

    #train gm mode on baseline size
    gmm_ref.fit(X[:baseline_size])

    #Calculating log probability of the current window size
    #higher scores = pitcher looks close to baseline
    #lower scores = physical profile of pitches is drifting away
    results = []
    for i in range(len(X) - window_size):
        score = gmm_ref.score(X[i: i + window_size])
        current_date = clean_df.iloc[i + window_size]['game_date']
        results.append({'game_date': current_date, 'drift_score': score})

    #Get averages
    drift_df = pd.DataFrame(results)
    game_stability = drift_df.groupby('game_date')['drift_score'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(game_stability['game_date'], game_stability['drift_score'], marker='o', linestyle='-', color='darkorange',
            alpha=0.6, label='Game Avg')
    ax.plot(game_stability['game_date'], game_stability['drift_score'].rolling(5).mean(), color='lime', linewidth=2.5,
            label='5-Game Trend')

    baseline_metrics = game_stability['drift_score'].iloc[:5]
    threshold = baseline_metrics.mean() - (2 * baseline_metrics.std())

    ax.axhline(y=threshold, color='red', linestyle='--', label='Physical Signature Limit')
    ax.set_title(f"Physical Profile Stability: {pitcher_name}")
    ax.set_ylabel("GMM Log Probability (Stability)")
    ax.set_xlabel("Year")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def get_player_id(first_name: str, last_name: str) -> int:
    '''Get the MLBAM player ID for a given player name. Returns the player ID as an int.'''
    player = playerid_lookup(last_name, first_name)
    if not player.empty:
        return player['key_mlbam'].values[0]
    else:
        raise ValueError(f'Player {first_name} {last_name} not found.')

def get_pitcher_data(first_name: str, last_name: str, start_dt: str, end_dt: str) -> pd.DataFrame:
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

def get_rdylgn_color(value, min_val=0.3, max_val=1.0):
    normalized = (value - min_val) / (max_val - min_val)
    normalized = max(0, min(1, normalized))
    colorscale = pc.get_colorscale('RdYlGn')
    return pc.sample_colorscale(colorscale, normalized)[0]

def display_knn_experiment(pitcher):
    st.header('How Consistent Do Pitchers Pitch Over Time?', text_alignment = 'center')
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

    st.divider()

    st.subheader(f'Drilldown for {pitcher}')
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
            st.plotly_chart(fold_fig, width='stretch')
            st.caption(
                "Each point shows how well the model classified pitch types in that year, "
                "trained on all prior years. A declining trend may indicate meaningful "
                "changes in the pitcher's mechanics over time."
            )

    st.divider()
    st.subheader("Reliability Leaderboard Among Selected Pitchers", text_alignment = 'center')

    sorted_results = knn_results.sort_values('weighted_f1')

    fig_bar = px.bar(
        sorted_results,
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
        title='KNN F1 Score Experiment - Pitcher Reliability',
        height=900
    )

    colors = [
        'dodgerblue' if p == selected_pitcher 
        else get_rdylgn_color(v) 
        for p, v in zip(sorted_results['pitcher'], sorted_results['weighted_f1'])
    ]
    fig_bar['data'][0]['marker']['color'] = colors
    
    fig_bar.add_vline(
        x=0.8,
        line_dash='dash',
        line_color='grey',
        annotation_text='0.8 threshold'
    )
    fig_bar.update_layout(title_x=0.5, title_xanchor='center')
    st.plotly_chart(fig_bar, width = 'stretch')
    st.caption(
        "Pitchers with higher F1 scores have more physically consistent and distinguishable pitch types. "
        "Lower scores may reflect pitch type changes over time or overlapping pitch characteristics. "
    )

def plot_kmeans_experiment(pitcher):
    @st.cache_data
    def getKMeansResults():
        return pd.read_csv('kmeans_results.csv')

    kmeans_results = getKMeansResults()

    selected_pitcher = pitcher

    sorted_results = kmeans_results.sort_values('best_adj_rand')
    fig_bar = px.bar(
    sorted_results,
    x='best_adj_rand',
    y='pitcher',
    orientation='h',
    color='best_adj_rand',
    color_continuous_scale='RdYlGn',
    range_color=[0.3, 1.0],
    hover_data=['pitch_types', 'best_k', 'best_adj_rand', 'n_pitches', 'n_years'],
    labels={
        'best_adj_rand': 'Best Adjusted Rand Index',
        'pitcher': 'Pitcher',
        'pitch_types': 'Pitch Types',
        'best_k': 'Best k',
        'n_pitches': '# Pitches',
        'n_years': 'Years of Data'
    },
    title='KMeans Adjusted Rand Index Experiment - Pitch Separation',
    height=900
    )

    colors = [
        'dodgerblue' if p == selected_pitcher 
        else get_rdylgn_color(v) 
        for p, v in zip(sorted_results['pitcher'], sorted_results['best_adj_rand'])
    ]
    fig_bar['data'][0]['marker']['color'] = colors

    fig_bar.add_vline(
        x=0.8,
        line_dash='dash',
        line_color='grey',
        annotation_text='0.8 threshold'
    )
    fig_bar.update_layout(title_x=0.5, title_xanchor='center')
    st.subheader("Pitch Separation Leaderboard Among Selected Pitchers", text_alignment = 'center')
    st.plotly_chart(fig_bar, width = 'stretch')
    st.caption('Pitchers with higher adjusted rand index scores have more distinct and separable pitches than those with lower ones. Lower scores can indicate blending of pitch metrics over time or overlapping characteristics. This indistinctness can be unfavorable for pitchers.')

def plot_pitcher_trends(df, pitcher_name, pitch_type, column):
    filtered_df = df[(df['player_name'] == pitcher_name) & (df['pitch_name'] == pitch_type)]
    filtered_df['moving_avg'] = filtered_df[column].rolling(window=10, center = True).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(1, len(filtered_df) + 1),
        y=filtered_df[column],
        mode='lines+markers',
        name=column,
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=np.arange(1, len(filtered_df) + 1),
        y=filtered_df['moving_avg'],
        mode='lines',
        name='10-Game Moving Avg',
        line=dict(color='red', dash='dash')
    ))

    fig.update_layout(
        title=f"{column} Trend: {pitcher_name} - {pitch_type}",
        xaxis_title="Game Sequence",
        yaxis_title=column,
        hovermode="x unified",
        template="plotly_white"
    )

    st.plotly_chart(fig, width = 'stretch')

def detect_changepoints(df, pitcher_name, pitch_type, threshold=25, use_game_sequence=True):
    filtered_df = df[(df['player_name'] == pitcher_name) & (df['pitch_name'] == pitch_type)].select_dtypes(include='number').copy()
    game_dates = df[(df['player_name'] == pitcher_name) & (df['pitch_name'] == pitch_type)]['game_date']
    #standardize the data
    scaler = StandardScaler()
    filtered_df = pd.DataFrame(scaler.fit_transform(filtered_df), columns=filtered_df.columns, index=filtered_df.index)
    data_series = filtered_df.values
    #iterate through the data and update the change detection model with each new data point, plotting the changepoints as they are detected
    detector = MDFocus(MDGaussian(), pruning_params=(2, 1))
    detect_list = []
    #run the change detection model, each time the threshold is breached, reset the model and continue iterating through the data
    for x in data_series:
        detector.update(x)
        detect_list.append(detector.statistic())
        if detector.statistic() >= threshold:
            detector = MDFocus(MDGaussian(), pruning_params=(2, 1))
    #conduct pca on the filtered data to reduce it to one dimension for easier visualization of the change detection statistic
    pca = PCA(n_components=9)
    pca_data = pca.fit_transform(filtered_df)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=np.arange(1, len(filtered_df) + 1) if use_game_sequence else game_dates,
        y=detect_list, 
        mode='lines+markers',
        name='Detection Statistic',
        line=dict(color='blue')
    ))
    fig1.add_hline(y=threshold, line_dash="dash", line_color="red", 
                annotation_text="Threshold", annotation_position="top left")

    fig1.update_layout(
        title=f"Change Detection: {pitcher_name}, {pitch_type}",
        xaxis_title="Game Sequence" if use_game_sequence else "Game Date",
        yaxis_title="Statistic",
        hovermode="x unified"
    )

    st.plotly_chart(fig1, width = 'stretch')

    st.subheader("Reduced-Dimension Visualization of Pitch Characteristics with Detected Changepoints")
    st.write("Click and drag to rotate the view. Use the scroll wheel to zoom.")
    st.write(f"The first 2 components represent {np.sum(pca.explained_variance_ratio_[:2]) * 100:.2f}% of the variance in the original data.")
    colors = ['red' if val > threshold else 'darkblue' for val in detect_list]
    breach_indices = [i for i, val in enumerate(detect_list) if val > threshold]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter3d(
        x=np.arange(1, len(filtered_df) + 1) if use_game_sequence else game_dates,
        y=pca_data[:, 0], 
        z=pca_data[:, 1],
        mode='lines+markers',
        name='Pitch Characteristics',
        line=dict(
            color=colors, 
            width=2
        ),
        marker=dict(
            size=2, 
            color=colors,
            opacity=0.7
        ),
        hovertemplate="Game: %{x}<br>1st PC: %{y:.2f}<br>2nd PC: %{z:.2f}<extra></extra>"
    ))

    #add highlight when threshold is breached
    if breach_indices:
        fig2.add_trace(go.Scatter3d(
            x=game_dates.iloc[breach_indices],
            y=pca_data[breach_indices, 0],
            z=pca_data[breach_indices, 1],
            mode='markers',
            name='Threshold Breached',
            marker=dict(
                size=4,
                color='red',
                line=dict(color='white', width=1)
            )
        ))

    fig2.update_layout(
        title="Reduced Dimension Visualization",
        scene=dict(
            xaxis_title='Game Sequence' if use_game_sequence else 'Game Date',
            yaxis_title=f'1st PC: {pca.explained_variance_ratio_[0] * 100:.2f}% of variance',
            zaxis_title=f'2nd PC: {pca.explained_variance_ratio_[1] * 100:.2f}% of variance'
        ),
        height=700,
        showlegend=True
    )

    st.plotly_chart(fig2, width = 'stretch')
    st.write("Game sequence is plotted on the x-axis, with the first two principal components of the pitch "
            "characteristics on the y and z axes. Points highlighted in red represent a statistically significant change "
             "in the pitch characteristics. Look for clusters "
             "or shifts in the trajectory that may correspond to detected changepoints or notable trends in the data."
             "\n\n Note: the principal components are simply shown to represent the trend of the multi-dimensional "
             "pitching characteristics over time, they are not used to calculate the change statistic.")
    #return detect_list, pca_data, pca.explained_variance_ratio_

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

    @st.cache_data
    def getCDData():
        return pd.read_csv('change_detection.csv')

    appData = getPitcherData()
    st.title('MLB Pitch Clustering', text_alignment = 'center')
    selectPitcher = st.selectbox('Select Pitcher', sorted(appData['player_name'].unique()), index = 4)
    selectYear = st.slider('Select Year', min_value = appData['game_date'].dt.year.min(), max_value = appData['game_date'].dt.year.max(), value = 2022)

    dfFilter = appData.loc[(appData['player_name'] == selectPitcher) & (appData['game_date'].dt.year == selectYear), :]

    st.subheader('EDA Visuals')

    st.subheader('Feature Distributions by Pitch Type', text_alignment = 'center')
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

    basicPlot = px.scatter(dfFilter, x = 'release_pos_x', y = 'release_pos_z', color = 'pitch_name', hover_data = ['release_speed'], labels = {'release_pos_x': 'Horizontal Release (in)', 'release_pos_z': 'Vertical Release (in)'}, title = f'{selectPitcher} {selectYear} Pitches by Type')
    basicPlot.update_layout(title_x = 0.5, title_xanchor = 'center')
    st.plotly_chart(basicPlot, width = 'stretch')

    st.divider()

    st.subheader("Random Forest Residual Performance Indicator", text_alignment = 'center')
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
        st.warning('Test set prediction results only available for 2022 and on.')

    st.caption('An outlier residual indicates that a significant difference exists between the best model\'s prediction and the actual test set data. Unexpected high residuals indicate a significant gap between the pitcher\'s expected average spin rate and actual results, which can indicate injury or other issues.')

    st.divider()

    display_knn_experiment(selectPitcher)

    st.divider()

    plot_kmeans_experiment(selectPitcher)

    st.divider()

    st.header("Pitcher Similarity", text_alignment = 'center')
    st.info("This section uses the K-nearest neighbors algorithm to find which pitcher-year most closely resembles the selected pitcher's pitch mechanics. " \
    "The similarity score is based on the distance in PCA space, with a higher score indicating a closer match. " \
    "Note that this is based on average pitch characteristics for the season, so it may not capture in-season changes or specific pitch types. " \
    "Results may also include different seasons of the selected pitcher, since their own career trajectory can be the closest physical match.")
    st.markdown(f'Current Pitcher: {selectYear} {selectPitcher}')

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

        radar_features = ['release_speed', 'release_spin_rate', 'release_pos_x', 'release_pos_z', 'release_extension']
        radar_labels = ['Release Speed', 'Spin Rate', 'Horizontal Release', 'Vertical Release', 'Extension']

        selected_avg = appData[
            (appData['player_name'] == selectPitcher) & 
            (appData['game_date'].dt.year == selectYear)
        ][radar_features].mean()

        if not selected_avg.isna().all():
            all_values = pd.concat([
                selected_avg.to_frame().T,
                sim_results[radar_features]
            ])
            normalized = (all_values - all_values.min()) / (all_values.max() - all_values.min())

            fig_radar = go.Figure()

            fig_radar.add_trace(go.Scatterpolar(
                r=normalized.iloc[0].tolist() + [normalized.iloc[0].tolist()[0]],
                theta=radar_labels + [radar_labels[0]],
                fill='toself',
                name=f'{selectPitcher} ({selectYear})',
                line=dict(color='#00CC96', width=2)
            ))

            colors = ['#636EFA', '#EF553B', '#FFA15A']
            for i, (_, row) in enumerate(sim_results.iterrows()):
                name = ' '.join(reversed(row['player_name'].split(', ')))
                fig_radar.add_trace(go.Scatterpolar(
                    r=normalized.iloc[i+1].tolist() + [normalized.iloc[i+1].tolist()[0]],
                    theta=radar_labels + [radar_labels[0]],
                    fill='toself',
                    name=f"{name} ({int(row['year'])})",
                    line=dict(color=colors[i], width=2),
                    opacity=0.4
                ))

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(color='black', size=10))),
                title=f'Physical Profile Comparison: {selectPitcher} ({selectYear}) vs Similar Pitchers',
                title_x=0.5,
                title_xanchor='center',
                showlegend=True,
                height=500
            )
            st.plotly_chart(fig_radar, width = 'stretch')
            st.caption(
                "Radar chart shows physical metrics for the selected pitcher versus their most similar matches. "
                "Overlapping profiles may indicate greater mechanical similarity."
            )
    else:
        st.warning("Select a different year or pitcher to generate similarity matches.")

    st.divider()

        #change detection
    CDData = getCDData()
    st.header("Change Detection", text_alignment = 'center')
    st.info("This section applies an online multivariate change detection algorithm to identify potential shifts in a pitcher's mechanics over time " \
            "to assess when a pitcher may have changed their pitching style or experienced a significant event (e.g. injury). " \
            "The first plot is exploratory, and shows the selected pitch characteristic over time, with a moving average to help identify trends. " \
            "The second plot shows the change detection statistic over time, with points above the threshold indicating a detected change. "
            "The third plot shows the trajectory of the pitch characteristics in PCA space, with points colored by the change detection statistic "
            "to visually identify when significant shifts in mechanics may have occurred.")
    
    CDPitcher = st.selectbox('Select Pitcher', sorted(CDData['player_name'].unique()), index = 6)
    CDPitchType = st.selectbox('Select Pitch Type', sorted(CDData[CDData['player_name'] == CDPitcher]['pitch_name'].unique()))
    exploreColumn = st.selectbox('Select Column to Explore', CDData.select_dtypes(include='number').columns.tolist(), index = 0)
    CDThreshold = st.slider('Select Change Detection Threshold', min_value = 5, max_value = 100, value = 25)
    #add drop down yes or no option to use the game sequence or actual date for the x axis of the plots
    use_game_sequence = st.selectbox('Use Game Sequence or Game Date for X-axis', ['Game Sequence', 'Game Date'], index = 1)
    
    try:
        st.subheader('Exploratory Plot')
        plot_pitcher_trends(CDData, CDPitcher, CDPitchType, exploreColumn)
        st.subheader('Change Detection Analysis and PCA Visualization')
        if use_game_sequence == 'Game Sequence':
            detect_changepoints(CDData, CDPitcher, CDPitchType, threshold=CDThreshold, use_game_sequence=True)
        else:
            detect_changepoints(CDData, CDPitcher, CDPitchType, threshold=CDThreshold, use_game_sequence=False)
    except Exception as e:
        st.error(f"An error occurred during change detection")
        st.info("This can happen if there are insufficient data points for the selected pitcher and pitch type. Try selecting a different pitcher, pitch type, or adjusting the threshold.")
    
    # st.divider()

    # commenting this out as it uses the small test pitcher dataset. used getPitcherData() on line 693 instead.
    # @st.cache_data
    # def getStabilityData():
    #     return pd.read_csv('pitcher_data.csv')

    st.header("Pitcher Stability", text_alignment='center')
    st.info(
        "This section models a pitcher's physical profile over time using a Gaussian Mixture Model trained on their baseline pitches. "
        "A declining drift score indicates the pitcher's mechanics are moving away from their established baseline, implying a decreased in performance.")

    stability_data = getPitcherData()
    pitcher_map = stability_data.drop_duplicates(subset='pitcher')[['player_name', 'pitcher']].sort_values(
        'player_name')
    # pitcher_names = pitcher_map['player_name'].tolist()
    # selected_stability_pitcher = st.selectbox('Select Pitcher', pitcher_names, key='stability_pitcher')
    selected_pitcher_id = pitcher_map[pitcher_map['player_name'] == selectPitcher]['pitcher'].iloc[0]
    analyze_pitcher_stability(stability_data, selected_pitcher_id)


if __name__ == "__main__":
    main()