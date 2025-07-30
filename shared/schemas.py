"""
Data schemas for the HRV prediction system.
Defines standardized data structures for WHOOP API data and internal processing.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Union
from enum import Enum

class ScoreState(Enum):
    """WHOOP score states"""
    SCORED = "SCORED"
    PENDING_SCORE = "PENDING_SCORE"
    UNSCORABLE = "UNSCORABLE"

@dataclass
class UserProfile:
    """User profile data from WHOOP API"""
    user_id: int
    email: str
    first_name: str
    last_name: str

@dataclass
class BodyMeasurements:
    """Body measurements data from WHOOP API"""
    height_meter: float
    weight_kilogram: float
    max_heart_rate: int

@dataclass
class CycleScore:
    """Cycle score data from WHOOP API"""
    strain: float
    kilojoule: float
    average_heart_rate: int
    max_heart_rate: int

@dataclass
class Cycle:
    """Physiological cycle data from WHOOP API"""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    start: datetime
    end: Optional[datetime]
    timezone_offset: str
    score_state: ScoreState
    score: Optional[CycleScore]

@dataclass
class RecoveryScore:
    """Recovery score data from WHOOP API"""
    user_calibrating: bool
    recovery_score: int
    resting_heart_rate: int
    hrv_rmssd_milli: float
    spo2_percentage: float
    skin_temp_celsius: float

@dataclass
class Recovery:
    """Recovery data from WHOOP API"""
    cycle_id: int
    sleep_id: str
    user_id: int
    created_at: datetime
    updated_at: datetime
    score_state: ScoreState
    score: Optional[RecoveryScore]

@dataclass
class StageSummary:
    """Sleep stage summary from WHOOP API"""
    total_in_bed_time_milli: int
    total_awake_time_milli: int
    total_no_data_time_milli: int
    total_light_sleep_time_milli: int
    total_slow_wave_sleep_time_milli: int
    total_rem_sleep_time_milli: int
    sleep_cycle_count: int
    disturbance_count: int

@dataclass
class SleepNeeded:
    """Sleep needed calculation from WHOOP API"""
    baseline_milli: int
    need_from_sleep_debt_milli: int
    need_from_recent_strain_milli: int
    need_from_recent_nap_milli: int

@dataclass
class SleepScore:
    """Sleep score data from WHOOP API"""
    stage_summary: StageSummary
    sleep_needed: SleepNeeded
    respiratory_rate: float
    sleep_performance_percentage: int
    sleep_consistency_percentage: int
    sleep_efficiency_percentage: float

@dataclass
class Sleep:
    """Sleep activity data from WHOOP API"""
    id: str
    v1_id: Optional[int]
    user_id: int
    created_at: datetime
    updated_at: datetime
    start: datetime
    end: datetime
    timezone_offset: str
    nap: bool
    score_state: ScoreState
    score: Optional[SleepScore]

@dataclass
class ZoneDurations:
    """Heart rate zone durations from WHOOP API"""
    zone_zero_milli: int
    zone_one_milli: int
    zone_two_milli: int
    zone_three_milli: int
    zone_four_milli: int
    zone_five_milli: int

@dataclass
class WorkoutScore:
    """Workout score data from WHOOP API"""
    strain: float
    average_heart_rate: int
    max_heart_rate: int
    kilojoule: float
    percent_recorded: int
    distance_meter: Optional[float]
    altitude_gain_meter: Optional[float]
    altitude_change_meter: Optional[float]
    zone_durations: ZoneDurations

@dataclass
class Workout:
    """Workout activity data from WHOOP API"""
    id: str
    v1_id: Optional[int]
    user_id: int
    created_at: datetime
    updated_at: datetime
    start: datetime
    end: datetime
    timezone_offset: str
    sport_name: str
    score_state: ScoreState
    score: Optional[WorkoutScore]
    sport_id: Optional[int]

# Internal schemas for analysis and prediction

@dataclass
class HRVDataPoint:
    """Single HRV data point for analysis"""
    date: datetime
    hrv_rmssd: float
    recovery_score: int
    strain: float
    sleep_performance: int
    sleep_duration_hours: float
    resting_heart_rate: int
    
@dataclass
class ModelFeatures:
    """Features for ML model training"""
    date: datetime
    hrv_target: float  # Target HRV for prediction
    hrv_7day_avg: float
    hrv_14day_avg: float
    hrv_30day_avg: float
    strain_yesterday: float
    strain_7day_avg: float
    sleep_performance_yesterday: int
    sleep_duration_yesterday: float
    sleep_debt: float
    rhr_yesterday: int
    workout_strain_yesterday: float
    day_of_week: int
    is_weekend: bool

@dataclass
class Prediction:
    """HRV prediction result"""
    date: datetime
    predicted_hrv: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    model_name: str
    features_used: Dict[str, float]
    prediction_timestamp: datetime

@dataclass
class JournalEntry:
    """Journal entry with extracted features"""
    date: datetime
    transcript: str
    sentiment_score: float
    stress_level: int  # 1-10 scale
    sleep_quality_mentioned: Optional[int]  # 1-10 scale if mentioned
    exercise_intensity_mentioned: Optional[int]  # 1-10 scale if mentioned
    alcohol_mentioned: bool
    caffeine_mentioned: bool
    illness_mentioned: bool
    travel_mentioned: bool
    work_stress_mentioned: bool
    
@dataclass
class DataSummary:
    """Daily data summary for LLM agent"""
    date: datetime
    hrv_actual: Optional[float]
    hrv_predicted: Optional[float]
    hrv_trend_7day: str  # "increasing", "decreasing", "stable"
    recovery_score: Optional[int]
    strain_score: Optional[float]
    sleep_performance: Optional[int]
    sleep_duration_hours: Optional[float]
    workout_summary: str
    journal_insights: Optional[str]
    model_accuracy_recent: float  # Recent model accuracy percentage