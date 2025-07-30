"""
Data Processor for HRV Data Intelligence.
Processes raw WHOOP data and creates structured summaries.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from shared.utils import (
    setup_logging, parse_datetime, clean_whoop_data, merge_daily_data,
    calculate_rolling_metrics, calculate_trend, categorize_hrv,
    categorize_recovery, categorize_strain, milliseconds_to_hours,
    save_json, load_json, ensure_directories
    )
    from shared.constants import DATA_DIR, REPORTS_DIR, FEATURE_WINDOWS
    from shared.schemas import HRVDataPoint, DataSummary
except ImportError:
    print("Shared modules not available, using local implementations")
    # Minimal local implementations for testing
    def setup_logging(name):
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(name)
    
    def save_json(data, filepath):
        import json
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_json(filepath):
        import json
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def ensure_directories(*dirs):
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    DATA_DIR = Path("../data")
    REPORTS_DIR = Path("../reports")

class DataProcessor:
    """Process and analyze WHOOP data for intelligence generation"""
    
    def __init__(self):
        self.logger = setup_logging("data_processor")
        self.data_dir = DATA_DIR
        self.reports_dir = REPORTS_DIR / "data_intelligence"
        ensure_directories(self.reports_dir)
        
    def load_whoop_data(self) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
        """Load all WHOOP data from files"""
        self.logger.info("Loading WHOOP data...")
        
        # Load cycles
        cycles_file = self.data_dir / "cycles" / "cycles.json"
        cycles = load_json(cycles_file) if cycles_file.exists() else []
        
        # Load recoveries
        recoveries_file = self.data_dir / "recovery" / "recoveries.json"
        recoveries = load_json(recoveries_file) if recoveries_file.exists() else []
        
        # Load sleep
        sleep_file = self.data_dir / "sleep" / "sleep_activities.json"
        sleep_data = load_json(sleep_file) if sleep_file.exists() else []
        
        # Load workouts
        workouts_file = self.data_dir / "workouts" / "workouts.json"
        workouts = load_json(workouts_file) if workouts_file.exists() else []
        
        self.logger.info(f"Loaded {len(cycles)} cycles, {len(recoveries)} recoveries, "
                        f"{len(sleep_data)} sleep records, {len(workouts)} workouts")
        
        return cycles, recoveries, sleep_data, workouts
    
    def create_daily_dataset(self) -> pd.DataFrame:
        """Create daily dataset with all WHOOP metrics"""
        cycles, recoveries, sleep_data, workouts = self.load_whoop_data()
        
        # Clean data
        cycles = [clean_whoop_data(c) for c in cycles]
        recoveries = [clean_whoop_data(r) for r in recoveries]
        sleep_data = [clean_whoop_data(s) for s in sleep_data]
        workouts = [clean_whoop_data(w) for w in workouts]
        
        # Merge into daily data
        daily_df = merge_daily_data(cycles, recoveries, sleep_data, workouts)
        
        # Sort by date
        daily_df = daily_df.sort_values('date').reset_index(drop=True)
        
        # Add rolling averages
        self._add_rolling_features(daily_df)
        
        # Add trend indicators
        self._add_trend_indicators(daily_df)
        
        self.logger.info(f"Created daily dataset with {len(daily_df)} days")
        return daily_df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> None:
        """Add rolling average features to the dataframe"""
        for window in [7, 14, 30]:
            if 'hrv_rmssd' in df.columns:
                df[f'hrv_rmssd_{window}d_avg'] = df['hrv_rmssd'].rolling(window, min_periods=1).mean()
                df[f'hrv_rmssd_{window}d_std'] = df['hrv_rmssd'].rolling(window, min_periods=1).std()
            
            if 'recovery_score' in df.columns:
                df[f'recovery_score_{window}d_avg'] = df['recovery_score'].rolling(window, min_periods=1).mean()
            
            if 'strain' in df.columns:
                df[f'strain_{window}d_avg'] = df['strain'].rolling(window, min_periods=1).mean()
            
            if 'sleep_performance' in df.columns:
                df[f'sleep_performance_{window}d_avg'] = df['sleep_performance'].rolling(window, min_periods=1).mean()
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> None:
        """Add trend indicators to the dataframe"""
        if 'hrv_rmssd' in df.columns and len(df) >= 7:
            df['hrv_trend_7d'] = df['hrv_rmssd'].rolling(7).apply(
                lambda x: self._calculate_trend_slope(x), raw=False
            )
            df['hrv_trend_direction'] = df['hrv_trend_7d'].apply(self._slope_to_direction)
        
        if 'recovery_score' in df.columns and len(df) >= 7:
            df['recovery_trend_7d'] = df['recovery_score'].rolling(7).apply(
                lambda x: self._calculate_trend_slope(x), raw=False
            )
    
    def _calculate_trend_slope(self, series: pd.Series) -> float:
        """Calculate trend slope for a series"""
        if len(series) < 3 or series.isnull().all():
            return 0.0
        
        valid_data = series.dropna()
        if len(valid_data) < 3:
            return 0.0
        
        x = np.arange(len(valid_data))
        coeffs = np.polyfit(x, valid_data, 1)
        return coeffs[0]
    
    def _slope_to_direction(self, slope: float) -> str:
        """Convert slope to direction string"""
        if pd.isna(slope):
            return "insufficient_data"
        elif slope > 0.5:
            return "increasing"
        elif slope < -0.5:
            return "decreasing"
        else:
            return "stable"
    
    def analyze_hrv_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze HRV patterns and return insights"""
        if 'hrv_rmssd' in df.columns:
            hrv_data = df['hrv_rmssd'].dropna()
            
            if len(hrv_data) == 0:
                return {"error": "No HRV data available"}
            
            analysis = {
                "overall_stats": {
                    "mean": float(hrv_data.mean()),
                    "std": float(hrv_data.std()),
                    "min": float(hrv_data.min()),
                    "max": float(hrv_data.max()),
                    "median": float(hrv_data.median()),
                    "count": len(hrv_data)
                },
                "recent_trend": self._analyze_recent_trend(hrv_data),
                "variability": self._analyze_variability(hrv_data),
                "outliers": self._detect_outliers(hrv_data),
                "categories": self._categorize_hrv_distribution(hrv_data)
            }
            
            return analysis
        
        return {"error": "No HRV column found"}
    
    def _analyze_recent_trend(self, data: pd.Series) -> Dict:
        """Analyze recent trend in HRV data"""
        if len(data) < 14:
            return {"status": "insufficient_data"}
        
        recent_data = data.tail(14)
        older_data = data.tail(28).head(14)
        
        recent_mean = recent_data.mean()
        older_mean = older_data.mean()
        
        change_percent = ((recent_mean - older_mean) / older_mean) * 100
        
        return {
            "recent_14d_mean": float(recent_mean),
            "previous_14d_mean": float(older_mean),
            "change_percent": float(change_percent),
            "direction": "improving" if change_percent > 2 else "declining" if change_percent < -2 else "stable"
        }
    
    def _analyze_variability(self, data: pd.Series) -> Dict:
        """Analyze HRV variability"""
        if len(data) < 7:
            return {"status": "insufficient_data"}
        
        recent_7d = data.tail(7)
        cv = (recent_7d.std() / recent_7d.mean()) * 100  # Coefficient of variation
        
        variability_level = "low" if cv < 15 else "moderate" if cv < 25 else "high"
        
        return {
            "coefficient_of_variation": float(cv),
            "variability_level": variability_level,
            "consistency": "high" if variability_level == "low" else "moderate" if variability_level == "moderate" else "low"
        }
    
    def _detect_outliers(self, data: pd.Series) -> Dict:
        """Detect outliers in HRV data"""
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
        return {
            "count": len(outliers),
            "percentage": float((len(outliers) / len(data)) * 100),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "recent_outliers": len(data.tail(7)[(data.tail(7) < lower_bound) | (data.tail(7) > upper_bound)])
        }
    
    def _categorize_hrv_distribution(self, data: pd.Series) -> Dict:
        """Categorize HRV values into ranges"""
        categories = data.apply(categorize_hrv).value_counts().to_dict()
        
        # Convert to percentages
        total = len(data)
        percentages = {k: (v / total) * 100 for k, v in categories.items()}
        
        return {
            "distribution": categories,
            "percentages": percentages,
            "dominant_category": max(percentages.keys(), key=percentages.get)
        }
    
    def analyze_recovery_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze recovery score patterns"""
        if 'recovery_score' in df.columns:
            recovery_data = df['recovery_score'].dropna()
            
            if len(recovery_data) == 0:
                return {"error": "No recovery data available"}
            
            analysis = {
                "overall_stats": {
                    "mean": float(recovery_data.mean()),
                    "std": float(recovery_data.std()),
                    "min": float(recovery_data.min()),
                    "max": float(recovery_data.max())
                },
                "zones": self._analyze_recovery_zones(recovery_data),
                "consistency": self._analyze_recovery_consistency(recovery_data)
            }
            
            return analysis
        
        return {"error": "No recovery score column found"}
    
    def _analyze_recovery_zones(self, data: pd.Series) -> Dict:
        """Analyze distribution across recovery zones"""
        zones = data.apply(categorize_recovery).value_counts().to_dict()
        total = len(data)
        
        return {
            "distribution": zones,
            "percentages": {k: (v / total) * 100 for k, v in zones.items()},
            "recent_7d_dominant": data.tail(7).apply(categorize_recovery).mode().iloc[0] if len(data) >= 7 else None
        }
    
    def _analyze_recovery_consistency(self, data: pd.Series) -> Dict:
        """Analyze recovery consistency"""
        if len(data) < 14:
            return {"status": "insufficient_data"}
        
        recent_std = data.tail(14).std()
        overall_std = data.std()
        
        return {
            "recent_14d_std": float(recent_std),
            "overall_std": float(overall_std),
            "consistency_trend": "improving" if recent_std < overall_std else "declining"
        }
    
    def generate_daily_summary(self, date: datetime) -> DataSummary:
        """Generate daily summary for a specific date"""
        df = self.create_daily_dataset()
        
        # Find data for the specific date
        day_data = df[df['date'] == date.date()]
        
        if day_data.empty:
            # Return summary with None values if no data
            return DataSummary(
                date=date,
                hrv_actual=None,
                hrv_predicted=None,
                hrv_trend_7day="insufficient_data",
                recovery_score=None,
                strain_score=None,
                sleep_performance=None,
                sleep_duration_hours=None,
                workout_summary="No data available",
                journal_insights=None,
                model_accuracy_recent=0.0
            )
        
        row = day_data.iloc[0]
        
        # Calculate 7-day trend
        recent_data = df[df['date'] <= date.date()].tail(7)
        hrv_trend = "insufficient_data"
        if len(recent_data) >= 3 and 'hrv_rmssd' in recent_data.columns:
            hrv_series = recent_data['hrv_rmssd'].dropna()
            if len(hrv_series) >= 3:
                trend_slope = self._calculate_trend_slope(hrv_series)
                hrv_trend = self._slope_to_direction(trend_slope)
        
        # Workout summary
        workout_summary = "No workouts"
        if not pd.isna(row.get('workout_count', 0)) and row.get('workout_count', 0) > 0:
            count = int(row.get('workout_count', 0))
            strain = row.get('workout_strain', 0)
            sport = row.get('primary_sport', 'Unknown')
            workout_summary = f"{count} workout(s), {sport}, strain: {strain:.1f}"
        
        return DataSummary(
            date=date,
            hrv_actual=row.get('hrv_rmssd'),
            hrv_predicted=None,  # Will be filled by model training module
            hrv_trend_7day=hrv_trend,
            recovery_score=row.get('recovery_score'),
            strain_score=row.get('strain'),
            sleep_performance=row.get('sleep_performance'),
            sleep_duration_hours=row.get('sleep_duration_hours'),
            workout_summary=workout_summary,
            journal_insights=None,  # Will be filled by journal analysis module
            model_accuracy_recent=0.0  # Will be filled by model training module
        )
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive data intelligence report"""
        self.logger.info("Generating comprehensive data intelligence report...")
        
        df = self.create_daily_dataset()
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "data_summary": {
                "total_days": len(df),
                "date_range": {
                    "start": df['date'].min().isoformat() if not df.empty else None,
                    "end": df['date'].max().isoformat() if not df.empty else None
                },
                "data_completeness": self._assess_data_completeness(df)
            },
            "hrv_analysis": self.analyze_hrv_patterns(df),
            "recovery_analysis": self.analyze_recovery_patterns(df),
            "correlations": self._calculate_correlations(df),
            "recommendations": self._generate_recommendations(df)
        }
        
        # Save report
        report_file = self.reports_dir / f"intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        save_json(report, report_file)
        
        self.logger.info(f"Report saved to {report_file}")
        return report
    
    def _assess_data_completeness(self, df: pd.DataFrame) -> Dict:
        """Assess completeness of data"""
        if df.empty:
            return {"status": "no_data"}
        
        key_columns = ['hrv_rmssd', 'recovery_score', 'strain', 'sleep_performance']
        completeness = {}
        
        for col in key_columns:
            if col in df.columns:
                non_null_count = df[col].notna().sum()
                completeness[col] = {
                    "available_days": int(non_null_count),
                    "percentage": float((non_null_count / len(df)) * 100)
                }
        
        return completeness
    
    def _calculate_correlations(self, df: pd.DataFrame) -> Dict:
        """Calculate correlations between key metrics"""
        key_columns = ['hrv_rmssd', 'recovery_score', 'strain', 'sleep_performance', 'sleep_duration_hours']
        available_columns = [col for col in key_columns if col in df.columns]
        
        if len(available_columns) < 2:
            return {"error": "Insufficient columns for correlation analysis"}
        
        corr_matrix = df[available_columns].corr()
        
        # Convert to dictionary format
        correlations = {}
        for i, col1 in enumerate(available_columns):
            for j, col2 in enumerate(available_columns):
                if i < j:  # Only upper triangle
                    corr_value = corr_matrix.loc[col1, col2]
                    if not pd.isna(corr_value):
                        correlations[f"{col1}_vs_{col2}"] = {
                            "correlation": float(corr_value),
                            "strength": self._interpret_correlation(corr_value)
                        }
        
        return correlations
    
    def _interpret_correlation(self, corr: float) -> str:
        """Interpret correlation strength"""
        abs_corr = abs(corr)
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate data-driven recommendations"""
        recommendations = []
        
        if 'hrv_rmssd' in df.columns:
            hrv_analysis = self.analyze_hrv_patterns(df)
            
            if hrv_analysis.get('recent_trend', {}).get('direction') == 'declining':
                recommendations.append("HRV has been declining recently. Consider focusing on recovery strategies.")
            
            variability = hrv_analysis.get('variability', {})
            if variability.get('variability_level') == 'high':
                recommendations.append("HRV shows high variability. Consider identifying and managing stressors.")
        
        if 'recovery_score' in df.columns:
            recovery_analysis = self.analyze_recovery_patterns(df)
            zones = recovery_analysis.get('zones', {}).get('percentages', {})
            
            if zones.get('red', 0) > 30:
                recommendations.append("Frequent red recovery zones detected. Prioritize sleep and stress management.")
        
        # Sleep recommendations
        if 'sleep_performance' in df.columns:
            recent_sleep = df.tail(7)['sleep_performance'].mean()
            if recent_sleep < 80:
                recommendations.append("Sleep performance below optimal. Focus on sleep hygiene and consistency.")
        
        if not recommendations:
            recommendations.append("Data patterns look healthy. Continue current lifestyle habits.")
        
        return recommendations

def main():
    """Main function for testing"""
    processor = DataProcessor()
    
    # Generate comprehensive report
    report = processor.generate_comprehensive_report()
    print("Data Intelligence Report generated successfully!")
    
    # Generate daily summary for today
    today = datetime.now()
    summary = processor.generate_daily_summary(today)
    print(f"Daily summary for {today.date()}: HRV={summary.hrv_actual}, Recovery={summary.recovery_score}")

if __name__ == "__main__":
    main()