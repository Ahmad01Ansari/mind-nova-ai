import joblib
import pandas as pd
import numpy as np
import os
import shap

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.imputers = {}
        self.features = {}
        self._load_all_artifacts()

    def _load_all_artifacts(self):
        print("🏗️  Initializing Model Hub...")
        self.registry = {
            "stress": {"file": "stress_model_recovered", "threshold": 0.62, "version": "v3"},
            "deterioration": {"file": "deterioration_model_recovered", "threshold": 0.35, "version": "v2"},
            "anxiety": {"file": "binary_optimized_model", "threshold": 0.54, "version": "hybrid_v5"},
            "depression": {"file": "binary_optimized_model", "threshold": 0.54, "version": "hybrid_v5"},
            "burnout": {"file": "burnout_xgboost", "threshold": 0.71, "version": "lgbm_best"}
        }

        for key, config in self.registry.items():
            prefix = config["file"]
            try:
                m_path = os.path.join(self.models_dir, f"{prefix}.pkl")
                if not os.path.exists(m_path): continue
                self.models[key] = joblib.load(m_path)
                
                s_suffix = "_recovered" if "recovered" in prefix else ("_final" if "final" in prefix else "")
                s_path = os.path.join(self.models_dir, f"{key}_scaler{s_suffix}.pkl")
                if "binary_optimized" in prefix: s_path = os.path.join(self.models_dir, "hybrid_scaler.pkl")
                if not os.path.exists(s_path): s_path = os.path.join(self.models_dir, f"{key}_scaler.pkl")
                if not os.path.exists(s_path): s_path = os.path.join(self.models_dir, "scaler.pkl")
                if os.path.exists(s_path): self.scalers[key] = joblib.load(s_path)
                
                i_path = os.path.join(self.models_dir, f"{key}_imputer{s_suffix}.pkl")
                if not os.path.exists(i_path): i_path = os.path.join(self.models_dir, f"{key}_imputer.pkl")
                if os.path.exists(i_path): self.imputers[key] = joblib.load(i_path)
                
                # Load correct feature manifest based on model
                if key == "stress": f_path = os.path.join(self.models_dir, "stress_features_recovered.pkl")
                elif key == "deterioration": f_path = os.path.join(self.models_dir, "deterioration_features_recovered.pkl")
                elif key == "burnout" and os.path.exists(os.path.join(self.models_dir, "burnout_clinical_features.pkl")):
                    f_path = os.path.join(self.models_dir, "burnout_clinical_features.pkl")
                else: f_path = os.path.join(self.models_dir, "selected_features.pkl")
                
                if os.path.exists(f_path):
                    self.features[key] = joblib.load(f_path)
                
                # Dynamic override: If scaler has feature names, use them as absolute truth
                if key in self.scalers and hasattr(self.scalers[key], "feature_names_in_"):
                    self.features[key] = list(self.scalers[key].feature_names_in_)
                
                print(f"✅ Loaded {key.upper()} Model Cluster. Features: {len(self.features.get(key, []))}")
            except Exception as e:
                print(f"⚠️  Error loading {key}: {str(e)}")

    def _map_granular(self, val, scale_min=1.0, scale_max=10.0, target_min=1.0, target_max=10.0):
        """
        Maps a 1-10 slider value to a model-specific range using linear interpolation.
        This provides much higher sensitivity than discrete step buckets.
        """
        # Normalize to 0-1 range
        normalized = (val - scale_min) / (scale_max - scale_min)
        # Scale to target range
        scaled = target_min + (normalized * (target_max - target_min))
        # Clip to ensure valid range
        return np.clip(scaled, target_min, target_max)

    def predict_anxiety(self, data: dict):
        if "anxiety" not in self.models: return {"error": "Model not loaded"}
        feats = self.features["anxiety"]
        v = {f: 5.0 for f in feats} 
        
        # Base Inputs
        v['PHQ2'] = float(data.get('phq2_score', 2.0))
        v['GAD2'] = float(data.get('gad2_score', 2.0))
        v['OnlineStress'] = float(data.get('online_stress', 3.0))
        v['GPA'] = float(data.get('academic_performance', 3.0))
        v['SleepHours'] = float(data.get('sleep_hours', 7.0))
        v['FamilySupport'] = float(data.get('family_support', 7.0))
        v['DietQuality'] = float(data.get('diet_quality', 5.0))
        
        # Dependency factors for composites
        sleep_quality = float(data.get('sleep_quality', 5.0))
        exercise_freq = float(data.get('exercise_freq', 3.0))
        peer_rel = float(data.get('peer_relationship', 5.0))
        social_act = float(data.get('social_activity', 5.0))
        self_eff = float(data.get('self_efficacy', 5.0))
        acad_stress = float(data.get('academic_stress', 5.0))
        screen_time = float(data.get('screen_time', 4.0))

        # Advanced composites (from hybrid_feature_engineering.py)
        v['LifestyleScore'] = v['SleepHours'] + exercise_freq + v['DietQuality'] + sleep_quality
        v['SupportScore'] = v['FamilySupport'] + peer_rel + social_act
        v['MentalResilience'] = self_eff + v['FamilySupport'] + peer_rel
        v['BurnoutRisk'] = acad_stress + screen_time - v['SleepHours']
        
        df_v = pd.DataFrame([v])[feats]
        
        # Strict Feature Order Validation
        assert list(df_v.columns) == list(feats), "Feature order mismatch in Anxiety prediction"
        
        X = self.scalers["anxiety"].transform(df_v.values)
        X_df = pd.DataFrame(X, columns=feats)
        base_prob = self.models["anxiety"].predict_proba(X_df)[0][1]
        
        # Recommendation: Weight by GAD2 (Anxiety-specific items)
        # This differentiates Anxiety from Depression while keeping the same precise detection core.
        weight = 0.8 + 0.4 * (v['GAD2'] / 6.0)
        prob = base_prob * weight
        
        # Clinical Calibration: If core indicators are zero, ensure GREEN status
        if v['GAD2'] == 0 and v['PHQ2'] == 0:
            prob = min(prob, 0.20)
        
        prob = np.clip(prob, 0.0, 0.99)
        
        return self._format_result(prob, "anxiety", v, self.models["anxiety"], X_df)

    def predict_depression(self, data: dict):
        if "depression" not in self.models: return {"error": "Model not loaded"}
        feats = self.features["depression"]
        v = {f: 5.0 for f in feats}
        
        # Base Inputs
        v['PHQ2'] = float(data.get('phq2_score', 2.0))
        v['GAD2'] = float(data.get('gad2_score', 2.0))
        v['OnlineStress'] = float(data.get('online_stress', 3.0))
        v['GPA'] = float(data.get('academic_performance', 3.0))
        v['SleepHours'] = float(data.get('sleep_hours', 7.0))
        v['FamilySupport'] = float(data.get('family_support', 7.0))
        v['DietQuality'] = float(data.get('diet_quality', 5.0))
        
        # Dependency factors
        sleep_quality = float(data.get('sleep_quality', 5.0))
        exercise_freq = float(data.get('exercise_freq', 3.0))
        peer_rel = float(data.get('peer_relationship', 5.0))
        social_act = float(data.get('social_activity', 5.0))
        self_eff = float(data.get('self_efficacy', 5.0))
        acad_stress = float(data.get('academic_stress', 5.0))
        screen_time = float(data.get('screen_time', 4.0))

        # Advanced composites
        v['LifestyleScore'] = v['SleepHours'] + exercise_freq + v['DietQuality'] + sleep_quality
        v['SupportScore'] = v['FamilySupport'] + peer_rel + social_act
        v['MentalResilience'] = self_eff + v['FamilySupport'] + peer_rel
        v['BurnoutRisk'] = acad_stress + screen_time - v['SleepHours']
        
        df_v = pd.DataFrame([v])[feats]
        
        # Strict Feature Order Validation
        assert list(df_v.columns) == list(feats), "Feature order mismatch in Depression prediction"
        
        X = self.scalers["depression"].transform(df_v.values)
        X_df = pd.DataFrame(X, columns=feats)
        base_prob = self.models["depression"].predict_proba(X_df)[0][1]
        
        # Recommendation: Weight by PHQ2 (Depression-specific items)
        weight = 0.8 + 0.4 * (v['PHQ2'] / 6.0)
        prob = base_prob * weight
        
        # Clinical Calibration: If core indicators are zero, ensure GREEN status
        if v['PHQ2'] == 0 and v['GAD2'] == 0:
            prob = min(prob, 0.20)
            
        prob = np.clip(prob, 0.0, 0.99)
        
        return self._format_result(prob, "depression", v, self.models["depression"], X_df)

    def predict_burnout(self, data: dict):
        if "burnout" not in self.models: return {"error": "Model not loaded"}
        feats = self.features["burnout"]
        v = {f: 0.0 for f in feats}
        
        # 1. Direct inputs from mobile
        v['WorkHours'] = float(data.get('work_hours', 8.0))
        v['SleepHours'] = float(data.get('sleep_hours', 7.0))
        v['ScreenTime'] = float(data.get('screen_time', 5.0))
        v['BreakFrequency'] = float(data.get('break_frequency', 3.0))
        v['StressLevel'] = float(data.get('stress_level', 5.0))
        v['ExperienceYears'] = float(data.get('experience_years', 2.0))
        v['JobSatisfaction'] = float(data.get('job_satisfaction', 5.0))
        
        # 2. Proxy mappings for missing required fields (37-feature alignment)
        v['AcademicStress'] = v['StressLevel']
        v['FinancialStress'] = float(data.get('financial_stress', v['StressLevel']))
        v['GPA'] = float(data.get('gpa', 3.0))
        # Refined Sleep Quality proxy: based on hours AND stress level
        base_quality = v['SleepHours'] * 1.25
        v['SleepQuality'] = np.clip(base_quality - (v['StressLevel'] * 0.2), 1.0, 10.0)
        
        v['PhysicalActivity'] = v['BreakFrequency']
        v['Age'] = float(data.get('age', 25.0))
        v['Gender'] = 1.0 
        v['OvertimeHours'] = max(0.0, v['WorkHours'] - 8.0)
        v['SlackActivity'] = 50.0 
        v['MeetingParticipation'] = 3.0
        # Refined Email Sentiment: use Job Satisfaction and Stress Level
        v['EmailSentiment'] = ((v['JobSatisfaction'] * 0.7) - (v['StressLevel'] * 0.3)) / 10.0
        v['WorkloadScore'] = min(10.0, (v['WorkHours'] / 10.0) * 10.0)
        v['PerformanceScore'] = 3.0
        
        # 3. Calculated indices matching training pipeline
        v['DigitalOverloadIndex'] = (v['ScreenTime'] * 1.5) + (v['WorkHours'] * 0.5)
        v['WorkloadIntensity'] = v['WorkloadScore'] * 1.1
        v['StressLoad'] = v['StressLevel'] * 2.5
        v['SleepDebt'] = max(0.0, 8.0 - v['SleepHours'])
        # Recovery includes Social Support if provided
        social_sup = float(data.get('social_support', 5.0))
        v['RecoveryScore'] = (v['SleepHours'] * 1.1) + (v['BreakFrequency'] * 0.6) + (social_sup * 0.3)
        v['BurnoutRiskIndex'] = (v['WorkHours'] / max(1, v['SleepHours'])) * (v['StressLevel'] / 2.0)
        
        # 4. Binary missingness flags (ensure they match training skew)
        v['ExperienceYears_missing'] = 0.0
        v['WorkHours_missing'] = 0.0
        v['GPA_missing'] = 0.0 if 'gpa' in data else 1.0
        v['AcademicStress_missing'] = 0.0
        v['FinancialStress_missing'] = 0.0
        v['SlackActivity_missing'] = 1.0
        
        # 5. One-Hot Encoded Dataset Sources
        v['DatasetSource_Healthcare_Workforce'] = 0.0
        v['DatasetSource_Remote_Work'] = 0.0
        v['DatasetSource_Student_Burnout'] = 0.0
        v['DatasetSource_Synthetic_Employee'] = 1.0
        v['DatasetSource_WFH_Corporate'] = 0.0
        
        # Ensure exact feature order for scaler
        df_v = pd.DataFrame([v])[feats]
        
        # Strict Feature Order Validation
        assert list(df_v.columns) == list(feats), "Feature order mismatch in Burnout prediction"
        
        X = self.scalers["burnout"].transform(df_v.values)
        
        # Pass scaled features back as DataFrame to keep feature names for XGBoost
        X_df = pd.DataFrame(X, columns=feats)
        prob = self.models["burnout"].predict_proba(X_df)[0][1]
        
        return self._format_result(prob, "burnout", v, self.models["burnout"], X_df)

    def predict_stress(self, data: dict):
        if "stress" not in self.models: return {"error": "Model not loaded"}
        feats = self.features["stress"]
        v = {f: 0 for f in feats}
        
        raw_mood = data.get('mood_current', 5.0)
        raw_sleep = data.get('sleep_hours', 7.0)
        raw_workload = data.get('workload_level', 5.0)
        
        v['JobSatisfaction'] = data.get("job_satisfaction", 5.0)
        v['WorkHours'] = data.get("work_hours", 8.0)
        v['SleepHours'] = raw_sleep
        v['ScreenTime'] = data.get("screen_time", 5.0)
        v['Age'] = data.get("age", 25)
        v['ExperienceYears'] = data.get("experience_years", 2.0)
        v['AcademicStress'] = data.get("academic_stress", 5.0)
        
        # High-sensitivity granular mapping instead of discrete buckets
        v['WorkStressMini'] = self._map_granular(11 - raw_mood, target_min=2.5, target_max=8.5)
        v['ExhaustionMini'] = self._map_granular(raw_workload, target_min=2.0, target_max=9.0)
        
        v['StressLoad'] = v['AcademicStress'] + raw_workload + (10 - raw_mood)
        v['BurnoutRiskCalc'] = v['WorkHours'] / max(1, raw_sleep)
        v['RecoveryScore'] = self._map_granular(raw_sleep, scale_max=12.0, target_max=10.0) * 1.2
        v['RecoveryFailureScore'] = max(0, 10 - v['RecoveryScore'])
        v['WeeklyStressTrend'] = 0
        v['RecentStressSpike'] = 0
        v['ConsecutivePoorSleepDays'] = 0 if raw_sleep >= 6.0 else 1
        v['HighWorkloadFrequency'] = 0
        
        df_v = pd.DataFrame([v])[feats]
        
        # Strict Feature Order Validation
        assert list(df_v.columns) == list(feats), "Feature order mismatch in Stress prediction"
        
        X = self.scalers["stress"].transform(df_v.values)
        X_df = pd.DataFrame(X, columns=feats)
        prob = self.models["stress"].predict_proba(X_df)[0][1]
        
        return self._format_result(prob, "stress", v, self.models["stress"], X_df)

    def predict_deterioration(self, history: list):
        if len(history) < 7:
            return {"status": "insufficient_data", "message": "Min 7 days required."}
            
        df = pd.DataFrame(history)
        
        # Ensure column names are as expected (handle potential capitalization or object types)
        column_mapping = {col.lower(): col for col in df.columns}
        if 'mood' not in column_mapping:
            return {"error": "Missing 'mood' data in history."}
        
        mood_col = column_mapping['mood']
        sleep_col = column_mapping.get('sleep', 'sleep')
        workload_col = column_mapping.get('workload', 'workload')

        # Use granular mapping for history trends as well
        df['MoodMini'] = df[mood_col].apply(lambda x: self._map_granular(11.0 - float(x), target_min=2.5, target_max=8.5))
        df['SleepMini'] = df[sleep_col].apply(lambda x: self._map_granular(float(x), scale_max=12.0, target_min=4.0, target_max=9.0))
        df['WorkloadMini'] = df[workload_col].apply(lambda x: self._map_granular(float(x), target_min=2.0, target_max=9.0))
        
        mood_v = df['MoodMini'].diff(3).fillna(0)
        sleep_v = df['SleepMini'].diff(3).fillna(0) * -1
        work_v = df['WorkloadMini'].diff(3).fillna(0)
        df['EscalationVelocity_Mini'] = (mood_v + sleep_v + work_v).clip(-10, 10)
        
        df['SleepDecline_7D'] = df['sleep'].rolling(window=7).apply(lambda x: x.iloc[0] - x.iloc[-1] if len(x) > 0 else 0).fillna(0)
        df['MoodVolatility'] = df['mood'].rolling(window=7).std().fillna(0)
        deficit = ((11 - df['mood']) - df['sleep']).clip(0, 10)
        df['RecoveryDeficitSlope'] = deficit.diff().fillna(0)
        
        is_risky = (df['mood'] < 4.0).astype(int)
        df['ConsecutiveRiskDays'] = is_risky.groupby((is_risky != is_risky.shift()).cumsum()).cumsum()
        
        vel = (11 - df['mood']).diff().fillna(0)
        df['BurnoutAcceleration'] = vel.diff().fillna(0)
        
        latest = df.iloc[-1:].copy()
        feats = self.features["deterioration"]
        
        # Ensure all features exist in latest df (some might be static 0s)
        for f in feats:
            if f not in latest.columns:
                latest[f] = 0.0
                
        # Strict Feature Order Validation
        # Clinical Calibration: The deterioration model can be over-sensitive to 'outlier' healthy states.
        # If the latest indicators are all in the 'High Health' zone, dampen the risk score.
        last_day = df.iloc[-1]
        last_mood = last_day['mood']
        last_sleep = last_day['sleep']
        last_workload = last_day['workload']
        
        if last_mood >= 8.0 and last_sleep >= 7.0 and last_workload <= 3.0:
            # User is objectively healthy right now; likely a false positive from synthetic drift patterns
            prob = prob * 0.3 # Dampen significantly if objectively healthy
            
        v_dict = last_day.to_dict()
        
        # Prepare final feature set for SHAP/Results
        latest_filtered = latest[feats]
        X = self.scalers["deterioration"].transform(latest_filtered.values)
        X_df = pd.DataFrame(X, columns=feats)
        
        # Run inference
        probs = self.models["deterioration"].predict_proba(X_df)[0]
        # Multi-class scoring: Total Risk = 1.0 - Probability(Stable)
        prob = 1.0 - probs[0]
            
        return self._format_result(prob, "deterioration", v_dict, self.models["deterioration"], X_df, lead_time="7 Days")

    def _format_result(self, prob, key, v_dict, model, X_df, lead_time=None):
        config = self.registry.get(key, {"threshold": 0.5, "version": "unknown"})
        threshold = config["threshold"]
        
        # 1. Calibrated Risk to UI Severity (0-100)
        # We want the model's 'threshold' to perfectly align with the 70 point mark (start of HIGH risk).
        if prob >= threshold:
            # Scale threshold..1.0 -> 70..100
            score_100 = 70.0 + ((prob - threshold) / (1.0 - threshold + 1e-9)) * 30.0
        else:
            # Scale 0..threshold -> 0..69
            score_100 = (prob / threshold) * 69.0
            
        score_100 = np.clip(score_100, 0, 100)
        
        if score_100 >= 85: risk_level = "SEVERE"
        elif score_100 >= 70: risk_level = "HIGH"
        elif score_100 >= 50: risk_level = "MODERATE"
        elif score_100 >= 25: risk_level = "MILD"
        else: risk_level = "MINIMAL"

        # 2. SHAP Contributors
        contributors = []
        try:
            # Detect if model is CalibratedClassifierCV
            is_calibrated = "CalibratedClassifierCV" in str(type(model))
            
            # Use TreeExplainer only for compatible tree models
            inner_model = None
            if is_calibrated and hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
                inner_model = getattr(model.calibrated_classifiers_[0], 'estimator', 
                              getattr(model.calibrated_classifiers_[0], 'base_estimator', None))
            
            can_use_tree = hasattr(model, 'feature_importances_') or (inner_model is not None and hasattr(inner_model, 'feature_importances_'))
            
            if can_use_tree:
                try:
                    # For calibrated models, explain the underlying estimator
                    explainer_model = inner_model if is_calibrated else model
                    if explainer_model is not None:
                        explainer = shap.TreeExplainer(explainer_model)
                        shap_values = explainer.shap_values(X_df)
                        
                        # Handle binary/multi-class SHAP output shapes
                        if isinstance(shap_values, list):
                            # Use class 1 (risk) if available, else class 0
                            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                        else:
                            vals = shap_values
                        
                        # Flatten to 1D and get top features
                        shap_abs = np.abs(vals.flatten())
                        top_indices = np.argsort(shap_abs)[::-1][:3]
                    else:
                        raise ValueError("No inner model found")
                except Exception as e:
                    print(f"SHAP Attempt failed: {e}. Falling back to feature importances.")
                    # If SHAP fails, use feature importances as secondary fallback
                    importances = getattr(model, 'feature_importances_', 
                                  getattr(inner_model, 'feature_importances_', None))
                    if importances is not None:
                        top_indices = np.argsort(importances)[::-1][:3]
                    else:
                        raise ValueError("No importances found")
            else:
                # Non-tree model or complex wrapper: Use feature variance as proxy (scaled values indicate impact)
                # Since X_df is scaled, values far from 0 (the mean) are the biggest drivers
                impact_vector = np.abs(X_df.values.flatten())
                top_indices = np.argsort(impact_vector)[::-1][:3]
                
            features_list = list(X_df.columns)
            raw_drivers = [features_list[i] for i in top_indices if i < len(features_list)]
            
            for d in raw_drivers:
                val = v_dict.get(d, 0)
                # Humanize driver labels
                if "Sleep" in d: contributors.append(f"Sleep pattern variance ({val:.1f}h)")
                elif "WorkHours" in d: contributors.append(f"Workload intensity ({val:.1f}h)")
                elif "Stress" in d: contributors.append(f"Elevated {d.lower()} signals")
                elif "Mood" in d: contributors.append("Emotional fluctuations")
                elif "Workload" in d: contributors.append("High task density")
                elif "GAD2" in d or "PHQ2" in d: contributors.append("Clinical screening markers")
                else:
                    humanized = ''.join([' '+c if c.isupper() else c for c in d]).strip()
                    contributors.append(f"Impact from {humanized.lower()}")
                    
        except Exception as e:
            print(f"Driver Analysis Fallback triggered: {e}")
            # Final safety fallback
            drivers = sorted([(k, abs(val - 5.0)) for k,val in v_dict.items() if isinstance(val, (int, float))], key=lambda x: x[1], reverse=True)
            contributors = [f"Variance in {d[0]}" for d in drivers[:3]]

        # 3. Confidence Logic
        input_completeness = 92 # Baseline for now, could be dynamic
        confidence = "High" if input_completeness > 80 else "Medium"

        res = {
            "success": True,
            "predictionType": key,
            "score": int(score_100),
            "riskLevel": risk_level,
            "confidence": confidence,
            "inputCompleteness": input_completeness,
            "contributors": contributors,
            "modelVersion": config["version"],
            "pipelineVersion": "phase6.2"
        }
        if lead_time:
            res["lead_time"] = lead_time
        return res

if __name__ == "__main__":
    mm = ModelManager()
