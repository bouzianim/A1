your-folder/
    ├── streamlit_app.py (this file)
    ├── neural_network_ftr.pkl
    ├── gradient_boosting_ftr.pkl
    ├── logistic_regression_ftr.pkl
    ├── scaler_main.pkl (optional)
    └── encoder_league.pkl (optional)
    ```
    """)
    return

if not st.session_state.models_loaded:
    st.warning("⚠️ Please load the models first using the 'Load All Models' button in the sidebar.")
    st.info("Click the red button in the sidebar to load your models!")
    return

# Create tabs - NOW THIS WILL SHOW!
tab1, tab2, tab3 = st.tabs(["🎯 Single Match Prediction", "📊 Batch Predictions", "ℹ️ Model Info"])

with tab1:
    st.header("Single Match Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚽ Match Details")
        
        # Input form
        with st.form("prediction_form"):
            h_odd = st.number_input("🏠 Home Team Odds", min_value=1.01, max_value=50.0, value=2.0, step=0.01, help="Enter the betting odds for home team win")
            d_odd = st.number_input("🤝 Draw Odds", min_value=1.01, max_value=50.0, value=3.5, step=0.01, help="Enter the betting odds for draw")
            a_odd = st.number_input("✈️ Away Team Odds", min_value=1.01, max_value=50.0, value=3.0, step=0.01, help="Enter the betting odds for away team win")
            league = st.selectbox("🏆 League", 
                                ["Premier League", "La Liga", "Serie A", "Bundesliga", "Ligue 1", "Other"], 
                                help="Select the league")
            
            if league == "Other":
                league = st.text_input("Enter League Name", value="Premier League")
            
            submit_button = st.form_submit_button("🔮 Make Prediction", type="primary")
        
        if submit_button:
            try:
                with st.spinner("Making predictions..."):
                    predictions, features = st.session_state.predictor.predict_single_match(
                        h_odd, d_odd, a_odd, league
                    )
                if predictions:
                    st.session_state.current_predictions = predictions
                    st.session_state.current_features = features
                    st.session_state.current_odds = (h_odd, d_odd, a_odd)
                    st.success("✅ Predictions completed!")
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")
    
    with col2:
        if hasattr(st.session_state, 'current_predictions') and st.session_state.current_predictions:
            st.subheader("🎯 Prediction Results")
            
            # Show match info
            if hasattr(st.session_state, 'current_odds'):
                h, d, a = st.session_state.current_odds
                st.info(f"📊 Match Odds: {h:.2f} / {d:.2f} / {a:.2f}")
            
            predictions = st.session_state.current_predictions
            
            # Display FTR predictions
            if 'FTR' in predictions:
                st.write("**⏰ Full Time Result (FTR):**")
                ftr_df_data = []
                for model_name, pred_data in predictions['FTR'].items():
                    ftr_df_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Prediction': pred_data['prediction'],
                        'Confidence': f"{pred_data['confidence']:.1%}",
                        'H': f"{pred_data['probabilities'].get('H', 0):.2f}",
                        'D': f"{pred_data['probabilities'].get('D', 0):.2f}",
                        'A': f"{pred_data['probabilities'].get('A', 0):.2f}"
                    })
                
                if ftr_df_data:
                    ftr_df = pd.DataFrame(ftr_df_data)
                    st.dataframe(ftr_df, use_container_width=True, hide_index=True)
            
            # Display HTR predictions
            if 'HTR' in predictions:
                st.write("**⏱️ Half Time Result (HTR):**")
                htr_df_data = []
                for model_name, pred_data in predictions['HTR'].items():
                    htr_df_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Prediction': pred_data['prediction'],
                        'Confidence': f"{pred_data['confidence']:.1%}",
                        'H': f"{pred_data['probabilities'].get('H', 0):.2f}",
                        'D': f"{pred_data['probabilities'].get('D', 0):.2f}",
                        'A': f"{pred_data['probabilities'].get('A', 0):.2f}"
                    })
                
                if htr_df_data:
                    htr_df = pd.DataFrame(htr_df_data)
                    st.dataframe(htr_df, use_container_width=True, hide_index=True)
                    
            # Best predictions summary
            st.subheader("🏆 Best Predictions")
            if 'FTR' in predictions:
                best_ftr = max(predictions['FTR'].items(), key=lambda x: x[1]['confidence'])
                st.success(f"**FTR:** {best_ftr[1]['prediction']} ({best_ftr[1]['confidence']:.1%} confidence) - {best_ftr[0].replace('_', ' ').title()}")
            
            if 'HTR' in predictions:
                best_htr = max(predictions['HTR'].items(), key=lambda x: x[1]['confidence'])
                st.success(f"**HTR:** {best_htr[1]['prediction']} ({best_htr[1]['confidence']:.1%} confidence) - {best_htr[0].replace('_', ' ').title()}")
        else:
            st.info("👈 Enter match details and click 'Make Prediction' to see results here!")

with tab2:
    st.header("📊 Batch Predictions from CSV")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="CSV should contain columns: B365H, B365D, B365A (and optionally FTR, HTR, League)"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**📋 Data Preview:**")
            st.dataframe(df.head(), use_container_width=True)
            
            required_cols = ['B365H', 'B365D', 'B365A']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"❌ Missing required columns: {missing_cols}")
            else:
                if st.button("🚀 Generate Predictions", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    results_list = []
                    
                    for idx, row in df.iterrows():
                        status_text.text(f"Processing match {idx + 1} of {len(df)}")
                        
                        h_odd, d_odd, a_odd = row['B365H'], row['B365D'], row['B365A']
                        league = row.get('League', 'Unknown')
                        
                        predictions, features = st.session_state.predictor.predict_single_match(
                            h_odd, d_odd, a_odd, league
                        )
                        
                        # Get best predictions
                        best_ftr = None
                        best_htr = None
                        
                        if predictions and 'FTR' in predictions:
                            best_ftr = max(predictions['FTR'].items(), key=lambda x: x[1]['confidence'])
                        if predictions and 'HTR' in predictions:
                            best_htr = max(predictions['HTR'].items(), key=lambda x: x[1]['confidence'])
                        
                        results_list.append({
                            'Match': idx + 1,
                            'Odds': f"{h_odd:.2f}/{d_odd:.2f}/{a_odd:.2f}",
                            'League': league,
                            'Best_FTR_Model': best_ftr[0] if best_ftr else 'N/A',
                            'Best_FTR_Prediction': best_ftr[1]['prediction'] if best_ftr else 'N/A',
                            'FTR_Confidence': f"{best_ftr[1]['confidence']:.3f}" if best_ftr else 'N/A',
                            'Best_HTR_Model': best_htr[0] if best_htr else 'N/A',
                            'Best_HTR_Prediction': best_htr[1]['prediction'] if best_htr else 'N/A',
                            'HTR_Confidence': f"{best_htr[1]['confidence']:.3f}" if best_htr else 'N/A',
                            'Actual_FTR': row.get('FTR', 'N/A'),
                            'Actual_HTR': row.get('HTR', 'N/A')
                        })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    status_text.text("✅ Processing complete!")
                    results_df = pd.DataFrame(results_list)
                    st.write("**🎯 Prediction Results:**")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="prediction_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

with tab3:
    st.header("ℹ️ Model Information")
    
    if st.session_state.predictor:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Loaded Models")
            if st.session_state.predictor.models:
                for model_name, model in st.session_state.predictor.models.items():
                    with st.expander(f"🤖 {model_name}"):
                        st.write(f"**Type:** {type(model).__name__}")
                        if hasattr(model, 'classes_'):
                            st.write(f"**Classes:** {list(model.classes_)}")
                        if hasattr(model, 'feature_importances_'):
                            st.write("**Has feature importances:** Yes")
                        else:
                            st.write("**Has feature importances:** No")
            else:
                st.warning("No models loaded")
        
        with col2:
            st.subheader("🛠️ Preprocessing Objects")
            
            if st.session_state.predictor.scaler:
                st.write("✅ **Scaler:** Loaded")
            else:
                st.write("❌ **Scaler:** Not found")
            
            if st.session_state.predictor.label_encoders:
                st.write("✅ **Label Encoders:** Loaded")
                for encoder_name in st.session_state.predictor.label_encoders.keys():
                    st.write(f"   - {encoder_name}")
            else:
                st.write("❌ **Label Encoders:** Not found")
            
            # Feature example
            if hasattr(st.session_state, 'current_features'):
                st.subheader("🔍 Last Generated Features")
                features_df = pd.DataFrame([st.session_state.current_features]).T
                features_df.columns = ['Value']
                st.dataframe(features_df)
    
    # File system info
    st.markdown("---")
    st.subheader("📁 Current Directory Files")
    current_files = []
    for file in os.listdir('.'):
        if file.endswith('.pkl'):
            size = os.path.getsize(file)
            current_files.append({
                'Filename': file,
                'Size (KB)': f"{size/1024:.1f}",
                'Type': 'Model' if not file.startswith(('scaler_', 'encoder_')) else 'Preprocessing'
            })
    
    if current_files:
        files_df = pd.DataFrame(current_files)
        st.dataframe(files_df, use_container_width=True)
    else:
        st.info("No .pkl files found in current directory")