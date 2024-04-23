import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from memory_profiler import memory_usage

def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    
    numeric_columns = ['streams', 'in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts', 
                       'in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    df.dropna(subset=numeric_columns, inplace=True)
    
    df['total_charts'] = df[['in_spotify_charts', 'in_apple_charts', 'in_deezer_charts', 'in_shazam_charts']].sum(axis=1)
    df['total_playlists'] = df[['in_spotify_playlists', 'in_apple_playlists', 'in_deezer_playlists']].sum(axis=1)
    
    df['is_hit'] = ((df['streams'] >= df['streams'].quantile(0.9)) |
                    ((df['total_charts'] >= df['total_charts'].quantile(0.90)) &
                     (df['total_playlists'] >= df['total_playlists'].quantile(0.90)))).astype(int)
    
    features = df[['artist_count', 'released_year', 'released_month', 'released_day',
                   'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
                   'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
                   'instrumentalness_%', 'liveness_%', 'speechiness_%', 'key', 'mode']]
    if 'mode' in features:
        df.loc[:, 'mode'] = df['mode'].map({'Major': 1, 'Minor': 0}, na_action='ignore')
        features = df[['artist_count', 'released_year', 'released_month', 'released_day',
                       'in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists',
                       'danceability_%', 'valence_%', 'energy_%', 'acousticness_%',
                       'instrumentalness_%', 'liveness_%', 'speechiness_%', 'key', 'mode']]
    
    return features, df['is_hit']

file_path = 'Popular_Spotify_Songs.csv'

X, y = load_and_prepare_data(file_path)

numeric_features = ['artist_count', 'released_year', 'released_month', 'released_day', 'in_spotify_playlists', 
                    'in_spotify_charts', 'in_apple_playlists', 'danceability_%', 'valence_%', 'energy_%', 
                    'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']
categorical_features = ['key', 'mode']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Neural Network': MLPClassifier(max_iter=1000, random_state=42),
    'SVM': SVC(kernel='linear'),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model_metrics = {}
for name, model in models.items():
    pipeline = make_pipeline(preprocessor, model)
    
    start_time = time.time()
    
    mem_usage_before = memory_usage(-1, interval=0.1, timeout=1)
    
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='accuracy')
    
    elapsed_time = time.time() - start_time
    
    mem_usage_after = memory_usage(-1, interval=0.1, timeout=1)
    memory_used = max(mem_usage_after) - mem_usage_before[0] 
    
    print(f"\n{name} Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print(f"{name} Training Time: {elapsed_time:.4f} seconds")
    print(f"{name} Memory Consumption: {memory_used:.4f} MiB")
    
    model_metrics[name] = {
        'Accuracy': np.mean(cv_scores),
        'Training Time': elapsed_time,
        'Memory Consumption': memory_used
    }
