import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import glob
from io import StringIO

class ModelPredictor:
    """Load and use saved ML models for predictions"""
    
    def __init__(self, model_path='.'):
        self.model_path = model_path
        self.models = {}
        self.scaler = None
        self.label_encoders = {}
        self.load_all_models()
    
    def load_model_file(self, file_path):
        """Load a model file using multiple methods"""
        try:
            # Try joblib first
            import joblib
            return joblib.load(file_path)
        except ImportError:
            try:
                # Try pickle as fallback
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                st.error(f"Could not load {file_path}: {e}")
                return None
        except Exception as e:
            try:
                # Try pickle as fallback
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e2:
                st.error(f"Could not load {file_path} with joblib or pickle: {e}, {e2}")
                return None
    
    def load_all_models(self):
        """Load all saved models and preprocessing objects from current directory"""
        
        # Load models - look for .pkl files that are models (not scaler or encoder)
        model_files = []
        for file in glob.glob(os.path.join(self.model_path, "*.pkl")):
            filename = os.path.basename(file)
            if not filename.startswith(('scaler_', 'encoder_')):
                model_files.append(filename)
        
        if not model_files:
            st.warning("No model files found in the current directory")
            return
        
        for file in model_files:
            try:
                model_name = file.replace('.pkl', '')
                model_path = os.path.join(self.model_path, file)
                loaded_model = self.load_model_file(model_path)
                if loaded_model is not None:
                    self.models[model_name] = loaded_model
                    st.success(f"✅ Loaded model: {model_name}")
            except Exception as e:
                st.error(f"❌ Failed to load {file}: {e}")
        
        # Load scaler - look for scaler files
        scaler_files = glob.glob(os.path.join(self.model_path, "scaler_*.pkl"))
        if scaler_files:
            try:
                loaded_scaler = self.load_model_file(scaler_files[0])
                if loaded_scaler is not None:
                    self.scaler = loaded_scaler
                    st.success(f"✅ Loaded scaler: {os.path.basename(scaler_files[0])}")
            except Exception as e:
                st.warning(f"⚠️ Could not load scaler: {e}")
        else:
            st.info("ℹ️ No scaler found (this is optional)")
        
        # Load label encoders
        encoder_files = glob.glob(os.path.join(self.model_path, "encoder_*.pkl"))
        for file in encoder_files:
            try:
                filename = os.path.basename(file)
                encoder_name = filename.replace('encoder_', '').replace('.pkl', '')
                loaded_encoder = self.load_model_file(file)
                if loaded_encoder is not None:
                    self.label_encoders[encoder_name] = loaded_encoder
                    st.success(f"✅ Loaded encoder: {encoder_name}")
            except Exception as e:
                st.warning(f"⚠️ Could not load encoder {file}: {e}")
        
        if not encoder_files:
            st.info("ℹ️ No encoders found (this is optional)")
    
    def engineer_features(self, h_odd, d_odd, a_odd, league='Premier League'):
        """Create feature vector for single match"""
        
        # Extract decimals
        h_decimal = int((h_odd % 1) * 100) if (h_odd % 1) > 0 else int(h_odd * 100) % 100
        d_decimal = int((d_odd % 1) * 100) if (d_odd % 1) > 0 else int(d_odd * 100) % 100
        a_decimal = int((a_odd % 1) * 100) if (a_odd % 1) > 0 else int(a_odd * 100) % 100
        
        sum_decimals = h_decimal + d_decimal + a_decimal
        
        # Calculate all features (must match training features exactly)
        features = {
            'h_odd': h_odd,
            'd_odd': d_odd,
            'a_odd': a_odd,
            'h_decimal': h_decimal,
            'd_decimal': d_decimal,
            'a_decimal': a_decimal,
            'sum_decimals': sum_decimals,
            'home_draw_ratio': h_odd / d_odd,
            'home_away_ratio': h_odd / a_odd,
            'draw_away_ratio': d_odd / a_odd,
            'draw_div4': d_decimal / 4,
            'sum_div10': sum_decimals / 10,
            'sum_div100': sum_decimals / 100,
        }
        
        # Pattern decimals
        features['draw_div4_decimal'] = int((features['draw_div4'] % 1) * 100) if features['draw_div4'] % 1 else int(features['draw_div4']) % 100
        features['sum_div10_decimal'] = int((features['sum_div10'] % 1) * 100) if features['sum_div10'] % 1 else int(features['sum_div10']) % 100
        features['sum_div100_decimal'] = int((features['sum_div100'] % 1) * 100) if features['sum_div100'] % 1 else int(features['sum_div100'] * 100) % 100
        
        # Pattern matches (binary)
        features['draw_div4_matches_h'] = 1 if str(int(features['draw_div4_decimal']))[0] == str(h_decimal)[0] else 0
        features['draw_div4_matches_d'] = 1 if str(int(features['draw_div4_decimal']))[0] == str(d_decimal)[0] else 0
        features['draw_div4_matches_a'] = 1 if str(int(features['draw_div4_decimal']))[0] == str(a_decimal)[0] else 0
        
        # Ratio signals
        features['ratio_away_signal'] = 1 if features['home_away_ratio'] > 1.5 else 0
        features['ratio_draw_signal'] = 1 if 0.8 <= features['home_draw_ratio'] <= 1.2 else 0
        features['ratio_home_signal'] = 1 if features['home_away_ratio'] < 0.7 else 0
        
        # League encoding
        if 'league' in self.label_encoders:
            try:
                features['league_encoded'] = self.label_encoders['league'].transform([league])[0]
            except:
                features['league_encoded'] = 0  # Default for unknown leagues
        else:
            features['league_encoded'] = 0
        
        # Statistical features
        odds_list = [h_odd, d_odd, a_odd]
        features['odds_variance'] = np.var(odds_list)
        features['odds_mean'] = np.mean(odds_list)
        features['favorite_odds'] = min(odds_list)
        
        return features
    
    def predict_single_match(self, h_odd, d_odd, a_odd, league='Premier League'):
        """Predict outcomes for a single match"""
        
        if not self.models:
            st.error("No models loaded! Please make sure your .pkl model files are in the same directory as this app.")
            return {}, {}
        
        # Engineer features
        features_dict = self.engineer_features(h_odd, d_odd, a_odd, league)
        
        # Convert to DataFrame with correct column order
        feature_names = list(features_dict.keys())
        features_df = pd.DataFrame([list(features_dict.values())], columns=feature_names)
        
        # Scale features if scaler available
        if self.scaler:
            features_scaled = self.scaler.transform(features_df)
        else:
            features_scaled = features_df.values
        
        results = {}
        
        # Make predictions with all available models
        for model_name, model in self.models.items():
            target = 'FTR' if '_ftr' in model_name else 'HTR'
            algorithm = model_name.replace('_ftr', '').replace('_htr', '')
            
            # Choose appropriate input data
            if algorithm in ['logistic_regression', 'neural_network']:
                X_input = features_scaled
            else:
                X_input = features_df
            
            try:
                prediction = model.predict(X_input)[0]
                probabilities = model.predict_proba(X_input)[0]
                confidence = max(probabilities)
                
                prob_dict = dict(zip(model.classes_, probabilities))
                
                if target not in results:
                    results[target] = {}
                
                results[target][algorithm] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'probabilities': prob_dict
                }
                
            except Exception as e:
                st.error(f"Error with {model_name}: {e}")
        
        return results, features_dict

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.models_loaded = False

def check_available_files():
    """Check what model files are available in current directory"""
    pkl_files = glob.glob("*.pkl")
    
    model_files = [f for f in pkl_files if not f.startswith(('scaler_', 'encoder_'))]
    scaler_files = [f for f in pkl_files if f.startswith('scaler_')]
    encoder_files = [f for f in pkl_files if f.startswith('encoder_')]
    
    return model_files, scaler_files, encoder_files

def main():
    st.set_page_config(
        page_title="Football Match Predictor",
        page_icon="⚽",
        layout="wide"
    )
    
    st.title("⚽ Football Match Prediction App")
    st.markdown("---")
    
    # Show dependency info
    with st.expander("ℹ️ Dependency Information"):
        try:
            import joblib
            st.success("✅ joblib is available")
        except ImportError:
            st.warning("⚠️ joblib not available, using pickle as fallback")
        
        st.info("Required packages: streamlit, pandas, numpy, scikit-learn")
    
    # Check available files
    model_files, scaler_files, encoder_files = check_available_files()
    
    # Sidebar for model loading and file info
    with st.sidebar:
        st.header("🔧 Model Management")
        
        # Show available files
        st.subheader("📁 Available Files")
        
        if model_files:
            st.write("**Model Files Found:**")
            for file in model_files:
                st.write(f"🤖 {file}")
        else:
            st.warning("❌ No model files (.pkl) found in current directory")
        
        if scaler_files:
            st.write("**Scaler Files:**")
            for file in scaler_files:
                st.write(f"📊 {file}")
        
        if encoder_files:
            st.write("**Encoder Files:**")
            for file in encoder_files:
                st.write(f"🏷️ {file}")
        
        st.markdown("---")
        
        # Auto-load or manual load button
        if model_files and not st.session_state.models_loaded:
            if st.button("🚀 Load All Models", type="primary"):
                try:
                    with st.spinner("Loading models from current directory..."):
                        st.session_state.predictor = ModelPredictor('.')
                        st.session_state.models_loaded = True
                        
                    if st.session_state.predictor.models:
                        st.balloons()
                        st.success(f"🎉 Successfully loaded {len(st.session_state.predictor.models)} models!")
                    else:
                        st.error("No models could be loaded")
                        st.session_state.models_loaded = False
                        
                except Exception as e:
                    st.error(f"Error loading models: {e}")
                    st.session_state.models_loaded = False
        
        elif st.session_state.models_loaded:
            st.success(f"✅ {len(st.session_state.predictor.models)} models loaded")
            if st.button("🔄 Reload Models"):
                st.session_state.models_loaded = False
                st.rerun()
    
    # Main content remains the same as before...
    # [Rest of the main() function stays identical to the previous version]

if __name__ == "__main__":
    main()