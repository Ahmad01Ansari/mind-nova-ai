import joblib
import pandas as pd
import numpy as np
import os

class MindNovaBurnoutEnsemble:
    """
    MindNova Dual-Core Burnout Risk Engine.
    Routes users to the 'Integrated' model (with Slack/Meetings) or 
    'Clinical' model (Self-report only) based on feature presence.
    """
    def __init__(self, model_dir='models'):
        print("🧊 Loading MindNova Burnout Ensembles...")
        self.model_a = joblib.load(os.path.join(model_dir, 'burnout_integrated_core.pkl'))
        self.scaler_a = joblib.load(os.path.join(model_dir, 'burnout_integrated_scaler.pkl'))
        self.features_a = joblib.load(os.path.join(model_dir, 'burnout_integrated_features.pkl'))
        
        self.model_b = joblib.load(os.path.join(model_dir, 'burnout_clinical_core.pkl'))
        self.scaler_b = joblib.load(os.path.join(model_dir, 'burnout_clinical_scaler.pkl'))
        self.features_b = joblib.load(os.path.join(model_dir, 'burnout_clinical_features.pkl'))
        
    def _apply_feature_engineering(self, df):
        """
        Applies MindNova behavioral indexing to raw input data.
        Ensures the ensemble is self-contained for production.
        """
        calc_df = df.copy()
        
        # 1. Digital Overload Index
        slack = calc_df.get('SlackActivity', pd.Series(0, index=df.index)).fillna(0)
        meetings = calc_df.get('MeetingParticipation', pd.Series(0, index=df.index)).fillna(0)
        sentiment = calc_df.get('EmailSentiment', pd.Series(0.7, index=df.index)).fillna(0.7)
        calc_df['DigitalOverloadIndex'] = (slack + meetings + (1 - sentiment)) / 3
        
        # 2. Workload Intensity
        overtime = calc_df.get('OvertimeHours', pd.Series(0, index=df.index)).fillna(0)
        workload = calc_df.get('WorkloadScore', pd.Series(0.5, index=df.index)).fillna(0.5)
        calc_df['WorkloadIntensity'] = (workload + (overtime / 40)) / 2
        
        # 3. StressLoad
        stress = calc_df.get('StressLevel', pd.Series(5.0, index=df.index)).fillna(5.0)
        academic = calc_df.get('AcademicStress', pd.Series(0, index=df.index)).fillna(0)
        financial = calc_df.get('FinancialStress', pd.Series(0, index=df.index)).fillna(0)
        calc_df['StressLoad'] = (stress + academic + financial) / 30
        
        # 4. Sleep & Recovery
        sleep_h = calc_df.get('SleepHours', pd.Series(7.0, index=df.index)).fillna(7.0)
        calc_df['SleepDebt'] = np.maximum(0, (8 - sleep_h) / 8)
        
        recovery_phys = calc_df.get('PhysicalActivity', pd.Series(1.0, index=df.index)).fillna(1.0)
        recovery_sleep = calc_df.get('SleepQuality', pd.Series(5.0, index=df.index)).fillna(5.0)
        calc_df['RecoveryScore'] = ((recovery_phys / 10) + (recovery_sleep / 10)) / 2
        
        # 5. Missingness Flags
        missing_targets = ['OvertimeHours', 'SlackActivity', 'MeetingParticipation', 
                           'EmailSentiment', 'WorkloadScore', 'ExperienceYears', 'WorkHours', 'GPA']
        for col in missing_targets:
            calc_df[f'{col}_missing'] = df[col].isnull().astype(int) if col in df.columns else 1
            
        return calc_df

    def predict_risk(self, user_data):
        """
        user_data: dict or single-row DataFrame
        Returns: dict with risk results and metadata
        """
        if isinstance(user_data, dict):
            df = pd.DataFrame([user_data])
        else:
            df = user_data.copy()
            
        # 0. Apply Engineering
        df = self._apply_feature_engineering(df)
            
        # 1. Routing Logic
        # Features that define 'Integrated' status
        it_features = ['SlackActivity', 'MeetingParticipation', 'EmailSentiment']
        has_integrations = any(df.get(feat, pd.Series([np.nan])).notnull().any() for feat in it_features)
        
        if has_integrations:
            model = self.model_a
            scaler = self.scaler_a
            features = self.features_a
            selected_model = "integrated"
        else:
            model = self.model_b
            scaler = self.scaler_b
            features = self.features_b
            selected_model = "clinical"
            
        # 2. Alignment & Imputation
        # Ensure all required features exist (fill with 0/NaN if missing)
        X = pd.DataFrame(index=df.index)
        for feat in features:
            if feat in df.columns:
                X[feat] = df[feat]
            else:
                # Use 0 for missingness flags, median/0 for others
                if feat.endswith('_missing'):
                    X[feat] = 1
                else:
                    X[feat] = 0.0 # Default fallback
                    
        # 3. Scaling & Inference
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        # 4. Result Formatting
        risk_probability = float(probs[0])
        
        # Dynamic Thresholding
        threshold = 0.5 if selected_model == "integrated" else 0.4
        is_risky = risk_probability >= threshold
        
        risk_level = "safe"
        if risk_probability > 0.8: risk_level = "high"
        elif risk_probability >= threshold: risk_level = "moderate"
        
        # Simplified Confidence (Distance from decision boundary)
        confidence = 1 - abs(risk_probability - threshold) * 2
        confidence = max(0.65, min(0.98, confidence)) # Normalized range
        
        return {
            "selected_model": selected_model,
            "burnout_probability": round(risk_probability, 4),
            "risk_level": risk_level,
            "confidence_score": round(confidence, 2),
            "needs_attention": int(is_risky)
        }

    def predict_risk_batch(self, df):
        """
        df: DataFrame with multiple users
        Returns: DataFrame with risk results and metadata for all rows
        """
        results = [None] * len(df)
        
        # 0. Apply Engineering
        df_eng = self._apply_feature_engineering(df)
        
        # 1. Routing & Subsetting
        it_features = ['SlackActivity', 'MeetingParticipation', 'EmailSentiment']
        
        # Check presence of columns in the RAW dataframe (routing depends on input)
        present_it_features = [f for f in it_features if f in df.columns]
        if not present_it_features:
            has_integrations = pd.Series(False, index=df.index)
        else:
            has_integrations = df[present_it_features].notnull().any(axis=1)
        
        # 2. Process Integrated Batch
        idx_a = has_integrations[has_integrations].index
        if not idx_a.empty:
            sub_a = df_eng.loc[idx_a]
            X_a = pd.DataFrame(index=idx_a)
            for feat in self.features_a:
                X_a[feat] = sub_a[feat] if feat in sub_a.columns else (1 if feat.endswith('_missing') else 0.0)
            X_a_scaled = self.scaler_a.transform(X_a)
            probs_a = self.model_a.predict_proba(X_a_scaled)[:, 1]
            
            for i, idx in enumerate(idx_a):
                p = float(probs_a[i])
                thresh = 0.5
                results[df.index.get_loc(idx)] = {
                    "selected_model": "integrated",
                    "burnout_probability": round(p, 4),
                    "risk_level": "high" if p > 0.8 else ("moderate" if p >= thresh else "safe"),
                    "confidence_score": round(max(0.65, min(0.98, 1 - abs(p - thresh) * 2)), 2),
                    "needs_attention": int(p >= thresh)
                }
                
        # 3. Process Clinical Batch
        idx_b = has_integrations[~has_integrations].index
        if not idx_b.empty:
            sub_b = df_eng.loc[idx_b]
            X_b = pd.DataFrame(index=idx_b)
            for feat in self.features_b:
                X_b[feat] = sub_b[feat] if feat in sub_b.columns else (1 if feat.endswith('_missing') else 5.0)
            X_b_scaled = self.scaler_b.transform(X_b)
            probs_b = self.model_b.predict_proba(X_b_scaled)[:, 1]
            
            for i, idx in enumerate(idx_b):
                p = float(probs_b[i])
                thresh = 0.4
                results[df.index.get_loc(idx)] = {
                    "selected_model": "clinical",
                    "burnout_probability": round(p, 4),
                    "risk_level": "high" if p > 0.8 else ("moderate" if p >= thresh else "safe"),
                    "confidence_score": round(max(0.65, min(0.98, 1 - abs(p - thresh) * 2)), 2),
                    "needs_attention": int(p >= thresh)
                }
                
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test stub
    engine = MindNovaBurnoutEnsemble()
    test_user = {
        "StressLevel": 8.0, 
        "WorkHours": 10, 
        "SleepHours": 5,
        "SlackActivity": 0.9 # Integrated
    }
    print(f"Test (Integrated): {engine.predict_risk(test_user)}")
    
    test_clinician = {
        "StressLevel": 9.0, 
        "WorkHours": 12, 
        "JobSatisfaction": 2.0
        # No Slack
    }
    print(f"Test (Clinical): {engine.predict_risk(test_clinician)}")
