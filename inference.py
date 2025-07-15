import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

import os

# Page config
st.set_page_config(page_title="Spotify Popularity Predictor", layout="wide")

class SpotifyPopularityPredictor:
    def __init__(self, model_path, feature_info_path, categorical_mappings_path, scaler_path=None, encoder_path=None):
        """Load the trained model and preprocessing components"""
        self.model = joblib.load(model_path)
        
        with open(feature_info_path, 'rb') as f:
            self.feature_info = pickle.load(f)
        
        with open(categorical_mappings_path, 'rb') as f:
            self.categorical_mappings = pickle.load(f)
        
        # Load preprocessors if available
        self.scaler = None
        self.encoder = None
        
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        if encoder_path and os.path.exists(encoder_path):
            self.encoder = joblib.load(encoder_path)
        
        print(f"✅ Model loaded: {self.feature_info['model_name']}")
        print(f"   Test R²: {self.feature_info['model_metrics']['Test_R2']:.4f}")

    def prepare_features(self, data):
        """Prepare features for prediction with proper preprocessing"""
        df = data.copy()

        # Feature engineering
        df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date'], errors='coerce')
        df['release_year'] = df['track_album_release_date'].dt.year

        current_year = datetime.now().year
        df['track_age'] = current_year - df['release_year']
        df['duration_minutes'] = df['duration_ms'] / 60000

        # Audio feature combinations
        df['energy_valence'] = df['energy'] * df['valence']
        df['danceability_energy'] = df['danceability'] * df['energy']
        df['mood_score'] = (df['energy'] + df['valence'] + df['danceability']) / 3
        df['audio_complexity'] = (df['speechiness'] + df['instrumentalness'] +
                                 df['acousticness'] + df['liveness']) / 4
        df['mainstream_score'] = (df['danceability'] + df['energy'] + df['valence'] -
                                 df['audio_complexity']) / 3

        # Categorical features
        df['duration_category'] = pd.cut(df['duration_minutes'],
                                       bins=[0, 2.5, 3.5, 4.5, float('inf')],
                                       labels=['Short', 'Medium', 'Long', 'Very_Long'])

        df['loudness_category'] = pd.cut(df['loudness'],
                                       bins=[-float('inf'), -10, -5, 0, float('inf')],
                                       labels=['Quiet', 'Medium', 'Loud', 'Very_Loud'])

        df['tempo_category'] = pd.cut(df['tempo'],
                                    bins=[0, 90, 120, 140, 180, float('inf')],
                                    labels=['Slow', 'Medium', 'Upbeat', 'Fast', 'Very_Fast'])

        # Convert categorical columns to string to match training
        df['duration_category'] = df['duration_category'].astype(str)
        df['loudness_category'] = df['loudness_category'].astype(str)
        df['tempo_category'] = df['tempo_category'].astype(str)

        # Get feature categories
        numerical_features = self.feature_info['numerical_features']
        categorical_features = self.feature_info['categorical_features']
        
        # Handle numerical features
        df_num = df[numerical_features].fillna(df[numerical_features].median())
        
        # Handle categorical features
        df_cat = df[categorical_features].fillna(df[categorical_features].mode().iloc[0] if not df[categorical_features].empty else 'Unknown')
        
        # Apply preprocessing if available
        if self.scaler is not None:
            df_num_scaled = self.scaler.transform(df_num)
        else:
            # Fallback: simple standardization (not recommended for production)
            df_num_scaled = (df_num - df_num.mean()) / df_num.std()
        
        if self.encoder is not None:
            df_cat_encoded = self.encoder.transform(df_cat)
        else:
            # Fallback: simple one-hot encoding (not recommended for production)
            encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
            df_cat_encoded = encoder.fit_transform(df_cat)
        
        # Combine features
        if isinstance(df_num_scaled, np.ndarray) and isinstance(df_cat_encoded, np.ndarray):
            X_processed = np.hstack([df_num_scaled, df_cat_encoded])
        else:
            X_processed = np.hstack([df_num_scaled, df_cat_encoded])
        
        return X_processed

    def predict(self, data):
        """Make predictions on new data"""
        X = self.prepare_features(data)
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, 100)

# Load models
@st.cache_resource
def load_models():
    model_dir = 'trained_models'
    
    models = {}
    model_names = ['ridge', 'linear', 'rf']  # Update with your actual model names
    
    for model_name in model_names:
        try:
            predictor = SpotifyPopularityPredictor(
                model_path=f'{model_dir}/{model_name}_model.pkl',
                feature_info_path=f'{model_dir}/{model_name}_feature_info.pkl',
                categorical_mappings_path=f'{model_dir}/categorical_mappings.pkl',
                scaler_path=f'{model_dir}/scaler.pkl',
                encoder_path=f'{model_dir}/encoder.pkl'
            )
            models[model_name] = predictor
        except Exception as e:
            st.error(f"Error loading {model_name} model: {e}")
    
    return models

# Main app
def main():
    st.title("Spotify Popularity Predictor")
    st.write("Predict track popularity using audio features")
    
    # Load models
    models = load_models()
    
    if not models:
        st.error("No models loaded. Please check your model files.")
        return
    
    # Model selection
    model_choice = st.selectbox("Select Model:", list(models.keys()))
    
    # Input form
    st.subheader("Track Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        energy = st.slider("Energy", 0.0, 1.0, 0.8)
        danceability = st.slider("Danceability", 0.0, 1.0, 0.7)
        valence = st.slider("Valence", 0.0, 1.0, 0.6)
        acousticness = st.slider("Acousticness", 0.0, 1.0, 0.1)
    
    with col2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
        speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
        liveness = st.slider("Liveness", 0.0, 1.0, 0.2)
        loudness = st.slider("Loudness", -30.0, 0.0, -5.0)
    
    with col3:
        tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
        duration_ms = st.slider("Duration (ms)", 30000, 300000, 210000)
        
        # Get unique values from categorical mappings
        if models[model_choice].categorical_mappings:
            playlist_genre = st.selectbox("Genre", 
                                        models[model_choice].categorical_mappings.get('playlist_genre', ['pop']))
            playlist_subgenre = st.selectbox("Subgenre", 
                                           models[model_choice].categorical_mappings.get('playlist_subgenre', ['dance pop']))
        else:
            playlist_genre = st.text_input("Genre", "pop")
            playlist_subgenre = st.text_input("Subgenre", "dance pop")
        
        release_date = st.date_input("Release Date", datetime(2023, 1, 15))
    
    # Predict button
    if st.button("Predict Popularity", type="primary"):
        # Create input data
        input_data = pd.DataFrame([{
            'energy': energy,
            'danceability': danceability,
            'valence': valence,
            'acousticness': acousticness,
            'instrumentalness': instrumentalness,
            'speechiness': speechiness,
            'liveness': liveness,
            'loudness': loudness,
            'tempo': tempo,
            'playlist_genre': playlist_genre,
            'playlist_subgenre': playlist_subgenre,
            'duration_ms': duration_ms,
            'track_album_release_date': release_date.strftime('%Y-%m-%d')
        }])
        
        try:
            # Make prediction
            predictor = models[model_choice]
            prediction = predictor.predict(input_data)[0]
            
            # Display result
            st.success(f"Predicted Popularity: {prediction:.1f}/100")
            
            # Show model info
            st.info(f"Model: {predictor.feature_info['model_name']}")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

if __name__ == "__main__":
    main()