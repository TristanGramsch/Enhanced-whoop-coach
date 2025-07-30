"""
Constants and configuration values for the HRV prediction system.
"""

from pathlib import Path

# File paths
DATA_DIR = Path("../data")
MODELS_DIR = Path("../models")
REPORTS_DIR = Path("../reports")
LOGS_DIR = Path("../logs")

# Data subdirectories
WHOOP_DATA_DIR = DATA_DIR / "whoop"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"
JOURNAL_DATA_DIR = DATA_DIR / "journal"

# HRV thresholds and metrics
HRV_THRESHOLDS = {
    "very_low": 20.0,
    "low": 35.0,
    "moderate": 50.0,
    "high": 65.0,
    "very_high": 80.0
}

RECOVERY_THRESHOLDS = {
    "red": 33,      # Poor recovery
    "yellow": 66,   # Moderate recovery  
    "green": 100    # Good recovery
}

STRAIN_THRESHOLDS = {
    "low": 10.0,
    "moderate": 14.0,
    "high": 18.0,
    "very_high": 21.0
}

SLEEP_PERFORMANCE_THRESHOLDS = {
    "poor": 70,
    "fair": 80,
    "good": 90,
    "excellent": 100
}

# Model configuration
MODEL_CONFIG = {
    "train_test_split": 0.8,
    "validation_split": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "max_features_per_model": 20,
    "model_types": ["xgboost", "random_forest", "linear_regression", "lstm"],
    "hyperparameter_trials": 100
}

# Feature engineering
FEATURE_WINDOWS = {
    "short_term": 7,    # days
    "medium_term": 14,  # days
    "long_term": 30     # days
}

ROLLING_METRICS = [
    "mean", "std", "min", "max", "trend"
]

# Data quality thresholds
DATA_QUALITY_THRESHOLDS = {
    "min_hrv_value": 5.0,
    "max_hrv_value": 200.0,
    "min_recovery_score": 0,
    "max_recovery_score": 100,
    "min_strain": 0.0,
    "max_strain": 25.0,
    "min_sleep_performance": 0,
    "max_sleep_performance": 100,
    "missing_data_tolerance": 0.1  # 10% missing data tolerance
}

# Journal analysis
JOURNAL_CONFIG = {
    "sentiment_model": "vader",
    "stress_keywords": [
        "stressed", "anxiety", "worried", "overwhelmed", "pressure",
        "tense", "nervous", "frustrated", "exhausted", "burnout"
    ],
    "sleep_keywords": [
        "sleep", "tired", "sleepy", "insomnia", "restless", "dream",
        "nightmare", "wake", "woke", "bed", "nap"
    ],
    "exercise_keywords": [
        "workout", "exercise", "gym", "run", "running", "bike", "cycling",
        "swim", "swimming", "yoga", "lift", "weights", "cardio", "training"
    ],
    "alcohol_keywords": [
        "alcohol", "drink", "drinking", "beer", "wine", "cocktail",
        "whiskey", "vodka", "rum", "gin", "bar", "drunk"
    ],
    "caffeine_keywords": [
        "coffee", "caffeine", "espresso", "latte", "tea", "energy drink",
        "cola", "soda", "pre-workout", "stimulant"
    ]
}

# LLM agent configuration  
LLM_CONFIG = {
    "model": "gpt-4",
    "temperature": 0.3,
    "max_tokens": 1000,
    "prediction_confidence_threshold": 0.7,
    "context_window_days": 14
}

# API configuration
API_CONFIG = {
    "whoop_api_base": "https://api.prod.whoop.com/developer/v2",
    "rate_limit_delay": 1.0,  # seconds between requests
    "max_retries": 3,
    "timeout": 30.0
}

# Orchestration
DAGSTER_CONFIG = {
    "daily_run_time": "06:00",  # UTC
    "timezone": "UTC",
    "max_concurrent_ops": 3,
    "retry_policy": {
        "max_retries": 2,
        "delay": 60  # seconds
    }
}

# Logging
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}

# Visualization
VIZ_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 100,
    "style": "seaborn-v0_8",
    "color_palette": ["#2E8B57", "#FF6B35", "#4682B4", "#FFD700", "#DC143C"],
    "font_size": 12
}

# Sport ID mappings from WHOOP API
SPORT_MAPPINGS = {
    -1: "Activity",
    0: "Running", 
    1: "Cycling",
    16: "Baseball",
    17: "Basketball",
    18: "Rowing",
    19: "Fencing",
    20: "Field Hockey",
    21: "Football",
    22: "Golf",
    24: "Ice Hockey",
    25: "Lacrosse",
    27: "Rugby",
    28: "Sailing",
    29: "Skiing",
    30: "Soccer",
    31: "Softball",
    32: "Squash",
    33: "Swimming",
    34: "Tennis",
    35: "Track & Field",
    36: "Volleyball",
    37: "Water Polo",
    38: "Wrestling",
    39: "Boxing",
    42: "Dance",
    43: "Pilates",
    44: "Yoga",
    45: "Weightlifting",
    47: "Cross Country Skiing",
    48: "Functional Fitness",
    49: "Duathlon",
    51: "Gymnastics",
    52: "Hiking/Rucking",
    53: "Horseback Riding",
    55: "Kayaking",
    56: "Martial Arts",
    57: "Mountain Biking",
    59: "Powerlifting",
    60: "Rock Climbing",
    61: "Paddleboarding",
    62: "Triathlon",
    63: "Walking",
    64: "Surfing",
    65: "Elliptical",
    66: "Stairmaster",
    70: "Meditation",
    71: "Other",
    96: "HIIT"
}

# File formats
FILE_FORMATS = {
    "data_format": "parquet",  # or "csv"
    "model_format": "joblib",
    "report_format": "json"
}

# Journal analysis keywords and patterns
BEHAVIORAL_KEYWORDS = {
    "exercise": ["exercise", "workout", "run", "gym", "training", "cardio", "fitness", "sport"],
    "alcohol": ["alcohol", "drink", "beer", "wine", "cocktail", "drinking", "drunk"],
    "caffeine": ["coffee", "caffeine", "espresso", "tea", "energy drink"],
    "work_stress": ["work", "job", "meeting", "deadline", "office", "boss", "project"],
    "travel": ["travel", "trip", "flight", "vacation", "hotel", "journey"]
}

HEALTH_KEYWORDS = {
    "sleep_quality": ["sleep", "rest", "bed", "tired", "exhausted", "insomnia"],
    "energy_levels": ["energy", "energetic", "fatigue", "tired", "alert", "drowsy"],
    "mood": ["happy", "sad", "angry", "frustrated", "excited", "calm", "stressed"],
    "pain": ["pain", "ache", "sore", "hurt", "headache", "migraine"]
}

STRESS_KEYWORDS = [
    "stress", "stressed", "anxiety", "anxious", "overwhelmed", "pressure",
    "tense", "worried", "nervous", "panic", "burnout"
]

JOURNAL_KEYWORDS = {
    "stress": ["stress", "stressed", "anxiety", "anxious", "overwhelmed", "pressure"],
    "sleep": ["sleep", "tired", "fatigue", "exhausted", "rest", "insomnia"],
    "exercise": ["workout", "exercise", "gym", "run", "training", "cardio"],
    "nutrition": ["food", "eat", "diet", "nutrition", "meal", "hungry"],
    "mood": ["happy", "sad", "angry", "frustrated", "excited", "calm"]
}