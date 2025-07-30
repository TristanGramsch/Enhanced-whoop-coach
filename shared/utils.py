"""
Utility functions for the HRV prediction system.
Common functions for data processing, validation, and formatting.
"""

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

# Optional imports with fallbacks
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None
    np = None
from .constants import (
    DATA_QUALITY_THRESHOLDS, LOGGING_CONFIG, HRV_THRESHOLDS,
    RECOVERY_THRESHOLDS, STRAIN_THRESHOLDS, SPORT_MAPPINGS
)

def setup_logging(name: str, log_dir: Optional[Path] = None) -> logging.Logger:
    """Set up logging configuration"""
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log")
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            LOGGING_CONFIG["format"],
            datefmt=LOGGING_CONFIG["date_format"]
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    formatter = logging.Formatter(LOGGING_CONFIG["format"])
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def parse_datetime(dt_string: str) -> datetime:
    """Parse datetime string from WHOOP API"""
    if dt_string.endswith('Z'):
        dt_string = dt_string[:-1] + '+00:00'
    return datetime.fromisoformat(dt_string)

def format_datetime(dt: datetime) -> str:
    """Format datetime for WHOOP API"""
    return dt.isoformat() + 'Z'

def get_date_range(start_date: datetime, end_date: datetime) -> List[datetime]:
    """Generate list of dates between start and end"""
    dates = []
    current = start_date.date()
    end = end_date.date()
    
    while current <= end:
        dates.append(datetime.combine(current, datetime.min.time()))
        current += timedelta(days=1)
    
    return dates

def validate_hrv_value(hrv: float) -> bool:
    """Validate HRV value is within reasonable bounds"""
    return (DATA_QUALITY_THRESHOLDS["min_hrv_value"] <= hrv <= 
            DATA_QUALITY_THRESHOLDS["max_hrv_value"])

def validate_recovery_score(score: int) -> bool:
    """Validate recovery score is within bounds"""
    return (DATA_QUALITY_THRESHOLDS["min_recovery_score"] <= score <= 
            DATA_QUALITY_THRESHOLDS["max_recovery_score"])

def validate_strain_score(strain: float) -> bool:
    """Validate strain score is within bounds"""
    return (DATA_QUALITY_THRESHOLDS["min_strain"] <= strain <= 
            DATA_QUALITY_THRESHOLDS["max_strain"])

def validate_sleep_performance(performance: int) -> bool:
    """Validate sleep performance is within bounds"""
    return (DATA_QUALITY_THRESHOLDS["min_sleep_performance"] <= performance <= 
            DATA_QUALITY_THRESHOLDS["max_sleep_performance"])

def categorize_hrv(hrv: float) -> str:
    """Categorize HRV value into descriptive ranges"""
    if hrv < HRV_THRESHOLDS["very_low"]:
        return "very_low"
    elif hrv < HRV_THRESHOLDS["low"]:
        return "low"
    elif hrv < HRV_THRESHOLDS["moderate"]:
        return "moderate"
    elif hrv < HRV_THRESHOLDS["high"]:
        return "high"
    else:
        return "very_high"

def categorize_recovery(score: int) -> str:
    """Categorize recovery score into color zones"""
    if score < RECOVERY_THRESHOLDS["red"]:
        return "red"
    elif score < RECOVERY_THRESHOLDS["yellow"]:
        return "yellow"
    else:
        return "green"

def categorize_strain(strain: float) -> str:
    """Categorize strain score into intensity levels"""
    if strain < STRAIN_THRESHOLDS["low"]:
        return "low"
    elif strain < STRAIN_THRESHOLDS["moderate"]:
        return "moderate"
    elif strain < STRAIN_THRESHOLDS["high"]:
        return "high"
    else:
        return "very_high"

def get_sport_name(sport_id: int) -> str:
    """Get sport name from sport ID"""
    return SPORT_MAPPINGS.get(sport_id, "Unknown")

def calculate_rolling_metrics(data, window: int) -> Dict[str, float]:
    """Calculate rolling statistics for a time series"""
    if PANDAS_AVAILABLE and hasattr(data, 'rolling'):
        # Use pandas Series method
        if len(data) < window:
            return {"mean": float('nan'), "std": float('nan'), "min": float('nan'), "max": float('nan')}
        
        rolling = data.rolling(window=window, min_periods=1)
        
        return {
            "mean": rolling.mean().iloc[-1],
            "std": rolling.std().iloc[-1],
            "min": rolling.min().iloc[-1],
            "max": rolling.max().iloc[-1]
        }
    else:
        # Use built-in statistics for list/array
        if isinstance(data, (list, tuple)):
            data_list = list(data)
        else:
            data_list = [data]
        
        if len(data_list) < window:
            return {"mean": float('nan'), "std": float('nan'), "min": float('nan'), "max": float('nan')}
        
        # Use last 'window' elements
        recent_data = data_list[-window:]
        
        import statistics
        return {
            "mean": statistics.mean(recent_data),
            "std": statistics.stdev(recent_data) if len(recent_data) > 1 else 0.0,
            "min": min(recent_data),
            "max": max(recent_data)
        }

def calculate_trend(data, window: int = 7) -> str:
    """Calculate trend direction over a window"""
    if PANDAS_AVAILABLE and hasattr(data, 'tail') and np is not None:
        # Use pandas Series method
        if len(data) < window:
            return "insufficient_data"
        
        recent_data = data.tail(window)
        
        # Simple linear trend
        x = np.arange(len(recent_data))
        coeffs = np.polyfit(x, recent_data, 1)
        slope = coeffs[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    else:
        # Use built-in methods for list/array
        if isinstance(data, (list, tuple)):
            data_list = list(data)
        else:
            data_list = [data]
        
        if len(data_list) < window:
            return "insufficient_data"
        
        # Use last 'window' elements
        recent_data = data_list[-window:]
        
        # Simple linear trend (slope calculation)
        if len(recent_data) < 2:
            return "stable"
        
        # Calculate simple slope
        x_values = list(range(len(recent_data)))
        n = len(recent_data)
        sum_x = sum(x_values)
        sum_y = sum(recent_data)
        sum_xy = sum(x * y for x, y in zip(x_values, recent_data))
        sum_x_squared = sum(x * x for x in x_values)
        
        # Linear regression slope formula
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x * sum_x)
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"

def milliseconds_to_hours(milliseconds: int) -> float:
    """Convert milliseconds to hours"""
    return milliseconds / (1000 * 60 * 60)

def hours_to_milliseconds(hours: float) -> int:
    """Convert hours to milliseconds"""
    return int(hours * 60 * 60 * 1000)

def clean_whoop_data(data: Dict) -> Dict:
    """Clean and validate WHOOP data"""
    cleaned = data.copy()
    
    # Parse datetime fields
    datetime_fields = ['created_at', 'updated_at', 'start', 'end']
    for field in datetime_fields:
        if field in cleaned and cleaned[field]:
            try:
                cleaned[field] = parse_datetime(cleaned[field])
            except Exception:
                cleaned[field] = None
    
    # Validate numeric fields
    if 'score' in cleaned and cleaned['score']:
        score = cleaned['score']
        
        if 'hrv_rmssd_milli' in score:
            if not validate_hrv_value(score['hrv_rmssd_milli']):
                score['hrv_rmssd_milli'] = None
                
        if 'recovery_score' in score:
            if not validate_recovery_score(score['recovery_score']):
                score['recovery_score'] = None
                
        if 'strain' in score:
            if not validate_strain_score(score['strain']):
                score['strain'] = None
    
    return cleaned

def merge_daily_data(cycles: List[Dict], recoveries: List[Dict], 
                    sleep_data: List[Dict], workouts: List[Dict]):
    """Merge all WHOOP data types into daily summary"""
    
    # Create daily records
    daily_data = []
    
    # Group data by date
    cycles_by_date = {}
    for cycle in cycles:
        if cycle.get('start'):
            date = cycle['start'].date()
            cycles_by_date[date] = cycle
    
    recoveries_by_cycle = {r['cycle_id']: r for r in recoveries if r.get('cycle_id')}
    sleep_by_date = {}
    workouts_by_date = {}
    
    for sleep in sleep_data:
        if sleep.get('start'):
            date = sleep['start'].date()
            if date not in sleep_by_date:
                sleep_by_date[date] = []
            sleep_by_date[date].append(sleep)
    
    for workout in workouts:
        if workout.get('start'):
            date = workout['start'].date()
            if date not in workouts_by_date:
                workouts_by_date[date] = []
            workouts_by_date[date].append(workout)
    
    # Create daily summary for each date
    all_dates = set(cycles_by_date.keys())
    all_dates.update(sleep_by_date.keys())
    all_dates.update(workouts_by_date.keys())
    
    for date in sorted(all_dates):
        daily_record = {"date": date}
        
        # Cycle data
        if date in cycles_by_date:
            cycle = cycles_by_date[date]
            daily_record["cycle_id"] = cycle.get("id")
            if cycle.get("score"):
                daily_record["strain"] = cycle["score"].get("strain")
                daily_record["average_heart_rate"] = cycle["score"].get("average_heart_rate")
        
        # Recovery data
        cycle_id = daily_record.get("cycle_id")
        if cycle_id and cycle_id in recoveries_by_cycle:
            recovery = recoveries_by_cycle[cycle_id]
            if recovery.get("score"):
                daily_record["recovery_score"] = recovery["score"].get("recovery_score")
                daily_record["hrv_rmssd"] = recovery["score"].get("hrv_rmssd_milli")
                daily_record["resting_heart_rate"] = recovery["score"].get("resting_heart_rate")
        
        # Sleep data
        if date in sleep_by_date:
            sleep_list = sleep_by_date[date]
            main_sleep = max(sleep_list, key=lambda s: 
                           s.get("score", {}).get("stage_summary", {}).get("total_in_bed_time_milli", 0))
            
            if main_sleep.get("score"):
                score = main_sleep["score"]
                stage_summary = score.get("stage_summary", {})
                
                daily_record["sleep_performance"] = score.get("sleep_performance_percentage")
                daily_record["sleep_duration_hours"] = milliseconds_to_hours(
                    stage_summary.get("total_in_bed_time_milli", 0))
                daily_record["sleep_efficiency"] = score.get("sleep_efficiency_percentage")
        
        # Workout data
        if date in workouts_by_date:
            workout_list = workouts_by_date[date]
            total_strain = sum(w.get("score", {}).get("strain", 0) for w in workout_list)
            daily_record["workout_strain"] = total_strain
            daily_record["workout_count"] = len(workout_list)
            
            # Primary workout (highest strain)
            if workout_list:
                primary_workout = max(workout_list, key=lambda w: 
                                    w.get("score", {}).get("strain", 0))
                daily_record["primary_sport"] = primary_workout.get("sport_name")
        
        daily_data.append(daily_record)
    
    if PANDAS_AVAILABLE:
        return pd.DataFrame(daily_data)
    else:
        return daily_data

def save_json(data: Any, filepath: Path) -> None:
    """Save data as JSON file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(filepath: Path) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def ensure_directories(*dirs: Path) -> None:
    """Ensure directories exist"""
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

def get_missing_data_report(df) -> Dict[str, float]:
    """Generate missing data report for DataFrame or list of dicts"""
    if PANDAS_AVAILABLE and hasattr(df, 'columns'):
        # Use pandas DataFrame method
        total_rows = len(df)
        missing_report = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            missing_report[column] = missing_percentage
        
        return missing_report
    else:
        # Use list of dicts method
        if not isinstance(df, list) or not df:
            return {}
        
        total_rows = len(df)
        all_keys = set()
        for record in df:
            if isinstance(record, dict):
                all_keys.update(record.keys())
        
        missing_report = {}
        for key in all_keys:
            missing_count = sum(1 for record in df if record.get(key) is None)
            missing_percentage = (missing_count / total_rows) * 100
            missing_report[key] = missing_percentage
        
        return missing_report