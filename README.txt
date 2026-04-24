===========================================================================
MLB PITCH CLUSTERING — USER GUIDE
===========================================================================

---------------------------------------------------------------------------
1. DESCRIPTION
---------------------------------------------------------------------------

This package provides an interactive Streamlit application for monitoring
pitcher health and pitcher consistency using MLB Statcast data fetched from 
the pybaseball Python library. The app integrates clustering, classification, 
and predictive modeling to preemptively detect physical decline in pitchers 
over time, which fills a gap in existing baseball analytics tools. 

The app includes the following components:

  - Exploratory Data Analysis: Feature distributions and release
    position scatter plots filterable by pitcher and season.

  - Random Forest Residual Performance: This section shows residual plots for 
    the random forest test set predictions of a given pitcher. The random forest 
    model uses data from January 2015 through December 2025 (approximately 70 percent 
    of the available data from the StatCast era) to make predictions about a 
    pitcher's monthly average spin rate for August 2022 onwards.

  - KNN Pitch Classification Experiment: Per-pitcher K-Nearest Neighbors
    classifiers trained on physical metrics (release speed, spin rate,
    break, release position, extension, spin axis). Temporal forward-chain
    cross-validation is used to select the optimal number of neighbors k
    and distance metric and prevent data leakage. Classification accuracy over 
    time serves as a proxy for pitcher consistency over time. Results are 
    evaluated using weighted F1 score to account for class imbalance across pitch types.

  - K-Means Clustering Validation: Adjusted Rand Index (ARI) measures
    how well K-Means derived pitch clusters correspond to actual
    pitch type labels, evaluating the validity of our unsupervised
    clustering approach.

  - Pitcher Similarity Tool: Uses PCA and KNN to identify pitchers with 
    similar physical profiles.

  - Online Change Detection: Applies a sequential multivariate change detection 
    algorithm to identify potential shifts in a pitcher's mechanics over time to assess
    when a pitcher may have changed their pitching style or experienced a significant 
    event (e.g. injury), visualized alongside PCA-reduced pitch trajectories in an 
    interactive 3D plot.

  - Pitcher Stability: Uses a Gaussian Mixture Model to model a pitcher's physical profile
    over time, which has been trained on their baseline pitches. A declining drift score 
    indicates the pitcher's mechanics are moving away from their established baseline, 
    implying a decreased in performance.

---------------------------------------------------------------------------
1. INSTALLATION
---------------------------------------------------------------------------

Requirements:
  - Python 3.9 or higher
  - Anaconda

Step 1: Clone or download the project folder.

Step 2: Install required dependencies:

    pip install distro pandas numpy scikit-learn pybaseball streamlit plotly changepoint_online matplotlib

Step 3: Ensure the following pre-generated CSV files are present in the
project root directory:

    pitcher_data_detailed_cleaned.csv  — Cleaned Statcast pitch data
    random_forest_predictions.csv      — RF model predictions and residuals
    knn_results.csv                    — KNN experiment summary results
    knn_folds.csv                      — KNN forward-chain fold results
    kmeans_results.csv                 — K-Means ARI experiment results
    kmeans_folds.csv                   — K-Means forward-chain results
    change_detection.csv               — Data for change detection analysis

Note: These CSV files are included in the submission package. To refresh them 
or fetch data for additional pitchers, uncomment the data retrieval block at
the top of main() in main.py and edit the pitchers list accordingly.

---------------------------------------------------------------------------
3. EXECUTION
---------------------------------------------------------------------------

Step 1: Navigate to the project directory:

    cd pitch-clustering

Step 2: Launch the Streamlit app:

    streamlit run main.py

Step 3: The app will open automatically in your default web browser.
Use the pitcher selector and year slider at the top of the page to
filter all visualizations by pitcher and season.

Step 4: Scroll through the app to explore each section:

    - EDA Visuals: Feature distributions and release position plots
    - Random Forest Residuals: Select a year >= 2022 to view test set results
    - KNN Experiment: View per-pitcher classification accuracy and leaderboard
    - K-Means Experiment: View pitch separability leaderboard
    - Pitcher Similarity: View the three most physically similar pitcher-seasons
    - Change Detection: Select a pitcher, pitch type, and threshold to run
      the online change detection algorithm
    - Pitcher Stability: Select a pitcher to display a pitcher's physical profile

---------------------------------------------------------------------------
4. DEMO VIDEO
---------------------------------------------------------------------------


===========================================================================