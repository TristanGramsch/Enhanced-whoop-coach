#!/usr/bin/env python3
"""
Simple ML Model Trainer for HRV Prediction.
Basic implementation without MLflow, using scikit-learn.
"""

import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Using mock model training.")

class SimpleHRVModelTrainer:
    """Simple ML model trainer for HRV prediction"""
    
    def __init__(self):
        self.data_dir = Path("../data")
        self.models_dir = Path("../models")
        self.reports_dir = Path("../reports/model_training")
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Model configurations
        self.models = {
            'linear_regression': LinearRegression() if SKLEARN_AVAILABLE else None,
            'ridge_regression': Ridge(alpha=1.0) if SKLEARN_AVAILABLE else None,
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42) if SKLEARN_AVAILABLE else None,
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42) if SKLEARN_AVAILABLE else None
        }
        
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.feature_names = []
        
    def load_and_prepare_data(self) -> Tuple[Any, Any, List[str]]:
        """Load and prepare training data"""
        print("Loading and preparing data for training...")
        
        if not SKLEARN_AVAILABLE:
            print("Scikit-learn not available, returning mock data")
            return [], [], []
        
        # Load data files
        cycles_file = self.data_dir / "cycles" / "cycles.json"
        recoveries_file = self.data_dir / "recovery" / "recoveries.json"
        sleep_file = self.data_dir / "sleep" / "sleep_activities.json"
        
        # Load raw data
        cycles = self._load_json_file(cycles_file)
        recoveries = self._load_json_file(recoveries_file)
        sleep_data = self._load_json_file(sleep_file)
        
        # Create feature dataset
        features, targets = self._create_feature_dataset(cycles, recoveries, sleep_data)
        
        if len(features) < 10:
            print(f"Warning: Only {len(features)} data points available. Need more data for reliable training.")
            return [], [], []
        
        print(f"Prepared {len(features)} training samples with {len(features[0]) if features else 0} features")
        return features, targets, self.feature_names
    
    def _load_json_file(self, filepath: Path) -> List[Dict]:
        """Load JSON file or return empty list"""
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
    
    def _create_feature_dataset(self, cycles: List[Dict], recoveries: List[Dict], 
                               sleep_data: List[Dict]) -> Tuple[List[List[float]], List[float]]:
        """Create feature dataset for training"""
        
        # Create daily records
        daily_records = self._merge_daily_data(cycles, recoveries, sleep_data)
        
        if len(daily_records) < 2:
            return [], []
        
        # Sort by date
        daily_records.sort(key=lambda x: x['date'])
        
        # Create features and targets
        features = []
        targets = []
        
        for i in range(7, len(daily_records)):  # Need 7 days of history for features
            record = daily_records[i]
            
            # Target: Current day HRV
            if record.get('hrv_rmssd') is not None:
                target_hrv = record['hrv_rmssd']
                
                # Features: Previous days' data
                feature_vector = self._create_feature_vector(daily_records, i)
                
                if feature_vector:
                    features.append(feature_vector)
                    targets.append(target_hrv)
        
        return features, targets
    
    def _merge_daily_data(self, cycles: List[Dict], recoveries: List[Dict], 
                         sleep_data: List[Dict]) -> List[Dict]:
        """Merge all data types into daily records"""
        daily_data = {}
        
        # Process recoveries (main source of HRV data)
        for recovery in recoveries:
            if recovery.get('score') and recovery.get('updated_at'):
                try:
                    date_str = recovery['updated_at'][:10]  # Extract YYYY-MM-DD
                    date = datetime.fromisoformat(date_str).date()
                    
                    if date not in daily_data:
                        daily_data[date] = {'date': date}
                    
                    score = recovery['score']
                    daily_data[date]['hrv_rmssd'] = score.get('hrv_rmssd_milli')
                    daily_data[date]['recovery_score'] = score.get('recovery_score')
                    daily_data[date]['resting_heart_rate'] = score.get('resting_heart_rate')
                    
                except Exception:
                    continue
        
        # Process cycles (strain data)
        for cycle in cycles:
            if cycle.get('score') and cycle.get('start'):
                try:
                    date_str = cycle['start'][:10]
                    date = datetime.fromisoformat(date_str).date()
                    
                    if date not in daily_data:
                        daily_data[date] = {'date': date}
                    
                    score = cycle['score']
                    daily_data[date]['strain'] = score.get('strain')
                    daily_data[date]['average_heart_rate'] = score.get('average_heart_rate')
                    
                except Exception:
                    continue
        
        # Process sleep data
        for sleep in sleep_data:
            if sleep.get('score') and sleep.get('start'):
                try:
                    date_str = sleep['start'][:10]
                    date = datetime.fromisoformat(date_str).date()
                    
                    if date not in daily_data:
                        daily_data[date] = {'date': date}
                    
                    score = sleep['score']
                    daily_data[date]['sleep_performance'] = score.get('sleep_performance_percentage')
                    daily_data[date]['sleep_efficiency'] = score.get('sleep_efficiency_percentage')
                    
                    # Convert sleep duration from milliseconds to hours
                    if score.get('stage_summary', {}).get('total_in_bed_time_milli'):
                        duration_hours = score['stage_summary']['total_in_bed_time_milli'] / (1000 * 60 * 60)
                        daily_data[date]['sleep_duration_hours'] = duration_hours
                    
                except Exception:
                    continue
        
        return list(daily_data.values())
    
    def _create_feature_vector(self, daily_records: List[Dict], current_idx: int) -> List[float]:
        """Create feature vector from historical data"""
        features = []
        
        # Get previous 7 days of data
        history = daily_records[current_idx-7:current_idx]
        
        # Basic feature names for reference
        if not self.feature_names:
            self.feature_names = [
                'hrv_1d_ago', 'hrv_2d_ago', 'hrv_3d_ago', 'hrv_7d_avg',
                'recovery_1d_ago', 'recovery_7d_avg',
                'strain_1d_ago', 'strain_7d_avg',
                'sleep_perf_1d_ago', 'sleep_perf_7d_avg',
                'sleep_dur_1d_ago', 'sleep_dur_7d_avg',
                'rhr_1d_ago', 'rhr_7d_avg',
                'day_of_week', 'is_weekend'
            ]
        
        try:
            # HRV features
            hrv_values = [d.get('hrv_rmssd', 0) or 0 for d in history]
            features.extend([
                hrv_values[-1] if hrv_values[-1] else 0,  # 1 day ago
                hrv_values[-2] if len(hrv_values) > 1 and hrv_values[-2] else 0,  # 2 days ago
                hrv_values[-3] if len(hrv_values) > 2 and hrv_values[-3] else 0,  # 3 days ago
                np.mean([h for h in hrv_values if h > 0]) if any(h > 0 for h in hrv_values) else 0  # 7d avg
            ])
            
            # Recovery features
            recovery_values = [d.get('recovery_score', 0) or 0 for d in history]
            features.extend([
                recovery_values[-1] if recovery_values[-1] else 0,  # 1 day ago
                np.mean([r for r in recovery_values if r > 0]) if any(r > 0 for r in recovery_values) else 0  # 7d avg
            ])
            
            # Strain features
            strain_values = [d.get('strain', 0) or 0 for d in history]
            features.extend([
                strain_values[-1] if strain_values[-1] else 0,  # 1 day ago
                np.mean([s for s in strain_values if s > 0]) if any(s > 0 for s in strain_values) else 0  # 7d avg
            ])
            
            # Sleep performance features
            sleep_perf_values = [d.get('sleep_performance', 0) or 0 for d in history]
            features.extend([
                sleep_perf_values[-1] if sleep_perf_values[-1] else 0,  # 1 day ago
                np.mean([s for s in sleep_perf_values if s > 0]) if any(s > 0 for s in sleep_perf_values) else 0  # 7d avg
            ])
            
            # Sleep duration features
            sleep_dur_values = [d.get('sleep_duration_hours', 0) or 0 for d in history]
            features.extend([
                sleep_dur_values[-1] if sleep_dur_values[-1] else 0,  # 1 day ago
                np.mean([s for s in sleep_dur_values if s > 0]) if any(s > 0 for s in sleep_dur_values) else 0  # 7d avg
            ])
            
            # Resting heart rate features
            rhr_values = [d.get('resting_heart_rate', 0) or 0 for d in history]
            features.extend([
                rhr_values[-1] if rhr_values[-1] else 0,  # 1 day ago
                np.mean([r for r in rhr_values if r > 0]) if any(r > 0 for r in rhr_values) else 0  # 7d avg
            ])
            
            # Calendar features
            current_date = daily_records[current_idx]['date']
            features.extend([
                current_date.weekday(),  # day of week (0=Monday, 6=Sunday)
                1 if current_date.weekday() >= 5 else 0  # is weekend
            ])
            
        except Exception as e:
            print(f"Error creating feature vector: {e}")
            return []
        
        return features
    
    def train_models(self, features: List[List[float]], targets: List[float]) -> Dict[str, Dict]:
        """Train all models and return results"""
        if not SKLEARN_AVAILABLE or not features:
            return self._mock_training_results()
        
        results = {}
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(targets)
        
        print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Target HRV range: {y.min():.1f} - {y.max():.1f} ms")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data (time series aware)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train each model
        for name, model in self.models.items():
            if model is None:
                continue
                
            print(f"\nTraining {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict on test set
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Cross-validation scores
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
                cv_mae = -cv_scores.mean()
                
                results[name] = {
                    'model': model,
                    'test_mae': mae,
                    'test_rmse': rmse,
                    'test_r2': r2,
                    'cv_mae': cv_mae,
                    'feature_names': self.feature_names,
                    'train_samples': len(X_train),
                    'test_samples': len(X_test)
                }
                
                print(f"  Test MAE: {mae:.2f} ms")
                print(f"  Test RMSE: {rmse:.2f} ms")
                print(f"  Test R²: {r2:.3f}")
                print(f"  CV MAE: {cv_mae:.2f} ms")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                continue
        
        return results
    
    def _mock_training_results(self) -> Dict[str, Dict]:
        """Return mock training results when sklearn is not available"""
        return {
            'mock_model': {
                'model': None,
                'test_mae': 15.0,
                'test_rmse': 20.0,
                'test_r2': 0.65,
                'cv_mae': 16.0,
                'feature_names': ['mock_feature_1', 'mock_feature_2'],
                'train_samples': 100,
                'test_samples': 25
            }
        }
    
    def save_best_model(self, results: Dict[str, Dict]) -> str:
        """Save the best performing model"""
        if not results:
            return "No models to save"
        
        # Find best model by lowest test MAE
        best_name = min(results.keys(), key=lambda k: results[k]['test_mae'])
        best_result = results[best_name]
        
        if SKLEARN_AVAILABLE and best_result['model'] is not None:
            # Save model and scaler
            model_file = self.models_dir / f"best_hrv_model_{datetime.now().strftime('%Y%m%d')}.pkl"
            scaler_file = self.models_dir / f"feature_scaler_{datetime.now().strftime('%Y%m%d')}.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(best_result['model'], f)
            
            with open(scaler_file, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save metadata
            metadata = {
                'model_name': best_name,
                'feature_names': best_result['feature_names'],
                'performance': {
                    'test_mae': best_result['test_mae'],
                    'test_rmse': best_result['test_rmse'],
                    'test_r2': best_result['test_r2'],
                    'cv_mae': best_result['cv_mae']
                },
                'training_info': {
                    'train_samples': best_result['train_samples'],
                    'test_samples': best_result['test_samples'],
                    'training_date': datetime.now().isoformat()
                }
            }
            
            metadata_file = self.models_dir / f"model_metadata_{datetime.now().strftime('%Y%m%d')}.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nBest model saved: {best_name}")
            print(f"Model file: {model_file}")
            print(f"Metadata: {metadata_file}")
            
            return best_name
        else:
            print(f"\nBest model (mock): {best_name}")
            return best_name
    
    def generate_training_report(self, results: Dict[str, Dict]) -> Dict:
        """Generate comprehensive training report"""
        report = {
            'training_timestamp': datetime.now().isoformat(),
            'models_trained': len(results),
            'best_model': min(results.keys(), key=lambda k: results[k]['test_mae']) if results else None,
            'model_results': {},
            'recommendations': []
        }
        
        # Add model results (excluding the actual model objects)
        for name, result in results.items():
            report['model_results'][name] = {
                'test_mae': result['test_mae'],
                'test_rmse': result['test_rmse'],
                'test_r2': result['test_r2'],
                'cv_mae': result['cv_mae'],
                'train_samples': result['train_samples'],
                'test_samples': result['test_samples']
            }
        
        # Generate recommendations
        if results:
            best_mae = min(r['test_mae'] for r in results.values())
            best_r2 = max(r['test_r2'] for r in results.values())
            
            if best_mae > 20:
                report['recommendations'].append("High prediction error. Consider collecting more data or feature engineering.")
            
            if best_r2 < 0.5:
                report['recommendations'].append("Low R² score. Model may need more relevant features or different algorithms.")
            
            total_samples = max(r['train_samples'] + r['test_samples'] for r in results.values())
            if total_samples < 50:
                report['recommendations'].append("Limited training data. Collect more historical data for better predictions.")
        
        # Save report
        report_file = self.reports_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Training report saved to {report_file}")
        return report
    
    def predict_hrv(self, target_date: datetime = None) -> Dict:
        """Use trained model to predict HRV"""
        if target_date is None:
            target_date = datetime.now() + timedelta(days=1)
        
        # Load latest model
        model_files = list(self.models_dir.glob("best_hrv_model_*.pkl"))
        
        if not model_files or not SKLEARN_AVAILABLE:
            # Return mock prediction
            return {
                'target_date': target_date.isoformat(),
                'predicted_hrv': 95.0,
                'confidence_interval': {'lower': 80.0, 'upper': 110.0},
                'model_type': 'mock',
                'notes': 'Using mock prediction - no trained model available'
            }
        
        try:
            # Load latest model and scaler
            latest_model_file = max(model_files, key=lambda p: p.stat().st_mtime)
            
            with open(latest_model_file, 'rb') as f:
                model = pickle.load(f)
            
            scaler_files = list(self.models_dir.glob("feature_scaler_*.pkl"))
            if scaler_files:
                latest_scaler_file = max(scaler_files, key=lambda p: p.stat().st_mtime)
                with open(latest_scaler_file, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                scaler = StandardScaler()
            
            # Create feature vector for prediction (simplified)
            # In practice, this would use recent data to create features
            dummy_features = np.array([[100, 95, 90, 95, 60, 65, 12, 10, 85, 80, 7.5, 7.8, 65, 60, 2, 0]])
            features_scaled = scaler.transform(dummy_features)
            
            prediction = model.predict(features_scaled)[0]
            
            # Estimate confidence interval (simplified)
            margin = prediction * 0.15
            
            return {
                'target_date': target_date.isoformat(),
                'predicted_hrv': round(prediction, 1),
                'confidence_interval': {
                    'lower': round(prediction - margin, 1),
                    'upper': round(prediction + margin, 1)
                },
                'model_type': 'trained',
                'model_file': str(latest_model_file),
                'notes': 'Prediction from trained model'
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {
                'target_date': target_date.isoformat(),
                'predicted_hrv': 95.0,
                'confidence_interval': {'lower': 80.0, 'upper': 110.0},
                'model_type': 'fallback',
                'notes': f'Error loading model: {e}'
            }

def main():
    """Main function for testing"""
    trainer = SimpleHRVModelTrainer()
    
    print("=== Simple HRV Model Training ===")
    
    # Load and prepare data
    features, targets, feature_names = trainer.load_and_prepare_data()
    
    if not features:
        print("No data available for training. Please ensure WHOOP data is available.")
        return
    
    # Train models
    results = trainer.train_models(features, targets)
    
    # Save best model
    best_model = trainer.save_best_model(results)
    
    # Generate report
    report = trainer.generate_training_report(results)
    
    print(f"\n=== Training Summary ===")
    print(f"Models trained: {report['models_trained']}")
    print(f"Best model: {report['best_model']}")
    
    if report['model_results']:
        best_result = report['model_results'][report['best_model']]
        print(f"Best MAE: {best_result['test_mae']:.2f} ms")
        print(f"Best R²: {best_result['test_r2']:.3f}")
    
    print(f"\n=== Recommendations ===")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    # Test prediction
    print(f"\n=== Test Prediction ===")
    prediction = trainer.predict_hrv()
    print(f"Tomorrow's predicted HRV: {prediction['predicted_hrv']} ms")
    print(f"Confidence interval: {prediction['confidence_interval']['lower']} - {prediction['confidence_interval']['upper']} ms")

if __name__ == "__main__":
    main()