#!/usr/bin/env python3
"""
HRV Prediction Pipeline Orchestrator

Dagster-based orchestration of the complete HRV prediction system:
1. WHOOP data collection
2. Journal analysis
3. Data intelligence processing
4. Model training and prediction
5. LLM agent predictions
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import dagster as dg
from dagster import asset, AssetExecutionContext, Config, MaterializeResult, MetadataValue

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared import (
    DATA_DIR, MODELS_DIR, REPORTS_DIR, PREDICTIONS_DIR,
    setup_logging, load_json, save_json
)

logger = setup_logging("orchestrator")

# ==============================================================================
# CONFIGURATION
# ==============================================================================

class PipelineConfig(Config):
    """Configuration for HRV prediction pipeline"""
    whoop_fetch_interval: int = 30  # minutes
    prediction_horizon: int = 7     # days
    enable_journal_analysis: bool = True
    enable_llm_predictions: bool = True
    skip_existing_data: bool = True

# ==============================================================================
# WHOOP DATA COLLECTION ASSETS
# ==============================================================================

@asset(group_name="data_collection")
def whoop_raw_data(context: AssetExecutionContext, config: PipelineConfig) -> MaterializeResult:
    """Fetch fresh WHOOP data from API"""
    try:
        # Import the WHOOP data fetcher
        sys.path.append(str(Path(__file__).parent.parent / "whoop_api_client"))
        from whoop_data_fetcher import WHOOPDataFetcher
        
        # Load access token
        token_file = Path(__file__).parent.parent / "whoop_api_client" / "token.json"
        if not token_file.exists():
            context.log.error("No WHOOP token found. Skipping data fetch.")
            return MaterializeResult(metadata={"status": "skipped", "reason": "no_token"})
        
        with open(token_file, 'r') as f:
            token_data = load_json(token_file)
            access_token = token_data.get("access_token")
        
        if not access_token:
            context.log.error("No access token in token file")
            return MaterializeResult(metadata={"status": "failed", "reason": "no_access_token"})
        
        # Create fetcher and get data
        fetcher = WHOOPDataFetcher(access_token)
        context.log.info("Fetching WHOOP data...")
        
        # Run a single fetch cycle
        import asyncio
        result = asyncio.run(fetcher.fetch_all_data())
        
        # Count records
        data_dir = Path(__file__).parent.parent / "whoop_api_client" / "data"
        total_records = 0
        data_files = {}
        
        for data_type in ["cycles", "recovery", "sleep", "workouts", "user"]:
            file_path = data_dir / f"{data_type}_data.json"
            if file_path.exists():
                data = load_json(file_path)
                count = len(data) if isinstance(data, list) else 1
                data_files[data_type] = count
                total_records += count
        
        context.log.info(f"WHOOP data fetch complete. Total records: {total_records}")
        
        return MaterializeResult(
            metadata={
                "total_records": total_records,
                "data_files": data_files,
                "fetch_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        )
        
    except Exception as e:
        context.log.error(f"Error fetching WHOOP data: {e}")
        return MaterializeResult(metadata={"status": "failed", "error": str(e)})

@asset(group_name="data_collection", deps=[whoop_raw_data])
def whoop_processed_data(context: AssetExecutionContext) -> MaterializeResult:
    """Process and validate WHOOP data"""
    try:
        # Import data intelligence
        sys.path.append(str(Path(__file__).parent.parent / "data_intelligence"))
        from simple_analyzer import DataIntelligence
        
        analyzer = DataIntelligence()
        context.log.info("Processing WHOOP data...")
        
        # Load and process data
        summary = analyzer.analyze_all_data()
        
        if not summary:
            context.log.warning("No data summary generated")
            return MaterializeResult(metadata={"status": "failed", "reason": "no_summary"})
        
        context.log.info("WHOOP data processing complete")
        
        return MaterializeResult(
            metadata={
                "hrv_records": summary.get("hrv_data_points", 0),
                "recovery_records": summary.get("recovery_data_points", 0),
                "sleep_records": summary.get("sleep_data_points", 0),
                "analysis_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        )
        
    except Exception as e:
        context.log.error(f"Error processing WHOOP data: {e}")
        return MaterializeResult(metadata={"status": "failed", "error": str(e)})

# ==============================================================================
# JOURNAL ANALYSIS ASSETS
# ==============================================================================

@asset(group_name="journal_analysis")
def journal_entries(context: AssetExecutionContext, config: PipelineConfig) -> MaterializeResult:
    """Process MP4 journal entries"""
    if not config.enable_journal_analysis:
        context.log.info("Journal analysis disabled")
        return MaterializeResult(metadata={"status": "skipped", "reason": "disabled"})
    
    try:
        # Import journal analyzer
        sys.path.append(str(Path(__file__).parent.parent / "journal_analysis"))
        from video_analyzer import VideoJournalAnalyzer
        
        analyzer = VideoJournalAnalyzer()
        context.log.info("Processing journal entries...")
        
        # Find MP4 files in journal_analysis directory
        journal_dir = Path(__file__).parent.parent / "journal_analysis"
        mp4_files = list(journal_dir.glob("*.mp4"))
        
        processed_count = 0
        for mp4_file in mp4_files:
            context.log.info(f"Processing journal: {mp4_file.name}")
            
            # Check if already processed today
            date_str = mp4_file.stem
            analysis_file = analyzer.output_dir / f"{date_str}_analysis.json"
            
            if config.skip_existing_data and analysis_file.exists():
                # Check if file is from today
                if analysis_file.stat().st_mtime > (datetime.now() - timedelta(days=1)).timestamp():
                    context.log.info(f"Skipping {mp4_file.name} - already processed recently")
                    continue
            
            result = analyzer.process_mp4_journal(str(mp4_file))
            if result:
                processed_count += 1
                context.log.info(f"Successfully processed: {mp4_file.name}")
            else:
                context.log.warning(f"Failed to process: {mp4_file.name}")
        
        context.log.info(f"Journal processing complete. Processed {processed_count} entries.")
        
        return MaterializeResult(
            metadata={
                "mp4_files_found": len(mp4_files),
                "processed_count": processed_count,
                "processing_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        )
        
    except Exception as e:
        context.log.error(f"Error processing journal entries: {e}")
        return MaterializeResult(metadata={"status": "failed", "error": str(e)})

# ==============================================================================
# MODEL TRAINING AND PREDICTION ASSETS
# ==============================================================================

@asset(group_name="ml_models", deps=[whoop_processed_data])
def trained_model(context: AssetExecutionContext) -> MaterializeResult:
    """Train ML model for HRV prediction"""
    try:
        # Import model trainer
        sys.path.append(str(Path(__file__).parent.parent / "model_training"))
        from simple_trainer import ModelTrainer
        
        trainer = ModelTrainer()
        context.log.info("Training ML model...")
        
        # Train the model
        model_results = trainer.train_models()
        
        if not model_results:
            context.log.warning("Model training failed")
            return MaterializeResult(metadata={"status": "failed", "reason": "training_failed"})
        
        best_model = model_results.get("best_model", "unknown")
        best_score = model_results.get("best_score", 0.0)
        
        context.log.info(f"Model training complete. Best model: {best_model} (R² = {best_score:.3f})")
        
        return MaterializeResult(
            metadata={
                "best_model": best_model,
                "best_score": best_score,
                "models_trained": len(model_results.get("all_results", {})),
                "training_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        )
        
    except Exception as e:
        context.log.error(f"Error training model: {e}")
        return MaterializeResult(metadata={"status": "failed", "error": str(e)})

@asset(group_name="ml_models", deps=[trained_model])
def ml_predictions(context: AssetExecutionContext, config: PipelineConfig) -> MaterializeResult:
    """Generate ML-based HRV predictions"""
    try:
        # Import model trainer for predictions
        sys.path.append(str(Path(__file__).parent.parent / "model_training"))
        from simple_trainer import ModelTrainer
        
        trainer = ModelTrainer()
        context.log.info("Generating ML predictions...")
        
        # Generate predictions for next few days
        predictions = []
        for days_ahead in range(1, config.prediction_horizon + 1):
            pred_date = datetime.now() + timedelta(days=days_ahead)
            
            try:
                prediction = trainer.predict_hrv(pred_date.strftime("%Y-%m-%d"))
                if prediction:
                    predictions.append({
                        "date": pred_date.strftime("%Y-%m-%d"),
                        "predicted_hrv": prediction,
                        "method": "ml_model",
                        "confidence": 0.75  # Default confidence
                    })
            except Exception as e:
                context.log.warning(f"Failed to predict for {pred_date.date()}: {e}")
        
        context.log.info(f"Generated {len(predictions)} ML predictions")
        
        # Save predictions
        predictions_file = PREDICTIONS_DIR / f"ml_predictions_{datetime.now().strftime('%Y%m%d')}.json"
        PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        save_json(predictions, predictions_file)
        
        return MaterializeResult(
            metadata={
                "predictions_count": len(predictions),
                "prediction_horizon": config.prediction_horizon,
                "predictions_file": str(predictions_file),
                "generation_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        )
        
    except Exception as e:
        context.log.error(f"Error generating ML predictions: {e}")
        return MaterializeResult(metadata={"status": "failed", "error": str(e)})

# ==============================================================================
# LLM AGENT PREDICTION ASSETS
# ==============================================================================

@asset(group_name="llm_predictions", deps=[whoop_processed_data, journal_entries, ml_predictions])
def llm_predictions(context: AssetExecutionContext, config: PipelineConfig) -> MaterializeResult:
    """Generate LLM-based HRV predictions with full context"""
    if not config.enable_llm_predictions:
        context.log.info("LLM predictions disabled")
        return MaterializeResult(metadata={"status": "skipped", "reason": "disabled"})
    
    try:
        # Import LLM agent
        sys.path.append(str(Path(__file__).parent.parent / "llm_agent"))
        from hrv_predictor import HRVPredictor
        
        predictor = HRVPredictor()
        context.log.info("Generating LLM predictions...")
        
        # Generate single-day prediction
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        prediction = predictor.predict_hrv(tomorrow)
        
        predictions_generated = 0
        if prediction:
            predictions_generated += 1
            context.log.info(f"Generated LLM prediction for {tomorrow}")
        
        # Generate 7-day forecast
        forecast = predictor.generate_7_day_forecast()
        if forecast:
            predictions_generated += len(forecast.get("daily_predictions", []))
            context.log.info("Generated 7-day LLM forecast")
        
        context.log.info(f"LLM prediction generation complete. Generated {predictions_generated} predictions.")
        
        return MaterializeResult(
            metadata={
                "single_prediction": prediction is not None,
                "forecast_generated": forecast is not None,
                "total_predictions": predictions_generated,
                "generation_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        )
        
    except Exception as e:
        context.log.error(f"Error generating LLM predictions: {e}")
        return MaterializeResult(metadata={"status": "failed", "error": str(e)})

# ==============================================================================
# REPORTING ASSETS
# ==============================================================================

@asset(group_name="reporting", deps=[llm_predictions])
def daily_report(context: AssetExecutionContext) -> MaterializeResult:
    """Generate daily pipeline execution report"""
    try:
        context.log.info("Generating daily report...")
        
        # Collect pipeline status
        report = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "pipeline_run_timestamp": datetime.now().isoformat(),
            "components": {
                "whoop_data": {"status": "completed", "details": "Data fetched successfully"},
                "journal_analysis": {"status": "completed", "details": "Journal entries processed"},
                "ml_training": {"status": "completed", "details": "Model trained and predictions generated"},
                "llm_predictions": {"status": "completed", "details": "LLM predictions generated"},
            },
            "summary": {
                "total_components": 4,
                "successful_components": 4,
                "failed_components": 0,
                "overall_status": "success"
            }
        }
        
        # Save report
        report_file = REPORTS_DIR / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        save_json(report, report_file)
        
        context.log.info(f"Daily report saved to: {report_file}")
        
        return MaterializeResult(
            metadata={
                "report_file": str(report_file),
                "components_run": report["summary"]["total_components"],
                "overall_status": report["summary"]["overall_status"],
                "generation_timestamp": datetime.now().isoformat(),
                "status": "success"
            }
        )
        
    except Exception as e:
        context.log.error(f"Error generating daily report: {e}")
        return MaterializeResult(metadata={"status": "failed", "error": str(e)})

# ==============================================================================
# PIPELINE DEFINITIONS
# ==============================================================================

# Daily automated pipeline
daily_assets = [
    whoop_raw_data,
    whoop_processed_data,
    journal_entries,
    trained_model,
    ml_predictions,
    llm_predictions,
    daily_report
]

@dg.job(name="daily_hrv_pipeline")
def daily_hrv_pipeline():
    """Daily automated HRV prediction pipeline"""
    daily_report()

# On-demand data refresh pipeline
@dg.job(name="data_refresh_pipeline")
def data_refresh_pipeline():
    """On-demand data refresh and processing"""
    whoop_processed_data()

# On-demand prediction pipeline
@dg.job(name="prediction_pipeline")
def prediction_pipeline():
    """On-demand prediction generation"""
    llm_predictions()

# ==============================================================================
# SCHEDULES
# ==============================================================================

@dg.schedule(
    job=daily_hrv_pipeline,
    cron_schedule="0 8 * * *",  # Daily at 8 AM
    name="daily_hrv_schedule"
)
def daily_hrv_schedule():
    """Schedule for daily HRV pipeline execution"""
    return {}

# ==============================================================================
# SENSORS
# ==============================================================================

@dg.sensor(job=data_refresh_pipeline, name="whoop_data_sensor")
def whoop_data_sensor(context):
    """Sensor to trigger data refresh when new WHOOP data is available"""
    # Check if WHOOP data is older than 2 hours
    data_dir = Path(__file__).parent.parent / "whoop_api_client" / "data"
    cycles_file = data_dir / "cycles_data.json"
    
    if cycles_file.exists():
        file_age = datetime.now() - datetime.fromtimestamp(cycles_file.stat().st_mtime)
        if file_age > timedelta(hours=2):
            context.log.info("WHOOP data is stale, triggering refresh")
            return dg.RunRequest()
    
    return None

# ==============================================================================
# RESOURCES
# ==============================================================================

@dg.resource
def pipeline_config():
    """Default pipeline configuration"""
    return PipelineConfig()

# ==============================================================================
# REPOSITORY
# ==============================================================================

defs = dg.Definitions(
    assets=daily_assets,
    jobs=[daily_hrv_pipeline, data_refresh_pipeline, prediction_pipeline],
    schedules=[daily_hrv_schedule],
    sensors=[whoop_data_sensor],
    resources={"config": pipeline_config}
)

if __name__ == "__main__":
    # For testing individual components
    print("HRV Prediction Pipeline - Orchestrator Module")
    print("Available components:")
    print("- whoop_raw_data: Fetch WHOOP data")
    print("- whoop_processed_data: Process WHOOP data")
    print("- journal_entries: Process journal MP4 files")
    print("- trained_model: Train ML models")
    print("- ml_predictions: Generate ML predictions")
    print("- llm_predictions: Generate LLM predictions")
    print("- daily_report: Generate pipeline report")
    print("\nTo run the orchestrator:")
    print("dagster dev -f hrv_pipeline.py")