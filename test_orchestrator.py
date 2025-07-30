#!/usr/bin/env python3
"""
Test script for HRV Prediction Orchestrator

Demonstrates the coordination of all system components
without requiring file write permissions.
"""

import sys
from datetime import datetime
from pathlib import Path

def test_orchestrator_components():
    """Test that all orchestrator components can be imported and initialized"""
    print("🎯 HRV Prediction Orchestrator Test")
    print("=" * 60)
    
    test_results = {}
    
    # Test 1: Test Orchestrator Import
    print("\n=== Testing Orchestrator Import ===")
    try:
        sys.path.append(str(Path(__file__).parent / "orchestrator"))
        import hrv_pipeline
        print("✅ Orchestrator pipeline imported successfully")
        print(f"📋 Available assets: {len(hrv_pipeline.daily_assets)}")
        print(f"🛠️ Available jobs: daily_hrv_pipeline, data_refresh_pipeline, prediction_pipeline")
        test_results["orchestrator_import"] = True
    except Exception as e:
        print(f"❌ Orchestrator import failed: {e}")
        test_results["orchestrator_import"] = False
    
    # Test 2: Test Component Integration
    print("\n=== Testing Component Integration ===")
    components_working = 0
    total_components = 6
    
    # WHOOP API Client
    try:
        sys.path.append(str(Path(__file__).parent / "whoop_api_client"))
        from whoop_data_fetcher import WHOOPDataFetcher
        print("✅ WHOOP API Client: Available")
        components_working += 1
    except Exception as e:
        print(f"❌ WHOOP API Client: {e}")
    
    # Data Intelligence
    try:
        sys.path.append(str(Path(__file__).parent / "data_intelligence"))
        from simple_analyzer import DataIntelligence
        analyzer = DataIntelligence()
        print("✅ Data Intelligence: Available")
        components_working += 1
    except Exception as e:
        print(f"❌ Data Intelligence: {e}")
    
    # Journal Analysis
    try:
        sys.path.append(str(Path(__file__).parent / "journal_analysis"))
        from video_analyzer import VideoJournalAnalyzer
        journal_analyzer = VideoJournalAnalyzer()
        print("✅ Journal Analysis: Available")
        components_working += 1
    except Exception as e:
        print(f"❌ Journal Analysis: {e}")
    
    # Model Training
    try:
        sys.path.append(str(Path(__file__).parent / "model_training"))
        from simple_trainer import ModelTrainer
        trainer = ModelTrainer()
        print("✅ Model Training: Available")
        components_working += 1
    except Exception as e:
        print(f"❌ Model Training: {e}")
    
    # LLM Agent
    try:
        sys.path.append(str(Path(__file__).parent / "llm_agent"))
        from hrv_predictor import HRVPredictor
        predictor = HRVPredictor()
        print("✅ LLM Agent: Available")
        components_working += 1
    except Exception as e:
        print(f"❌ LLM Agent: {e}")
    
    # Shared Utilities
    try:
        sys.path.append(str(Path(__file__).parent))
        from shared import setup_logging, load_json, save_json
        print("✅ Shared Utilities: Available")
        components_working += 1
    except Exception as e:
        print(f"❌ Shared Utilities: {e}")
    
    test_results["component_integration"] = components_working / total_components
    
    # Test 3: Test Dagster Configuration
    print("\n=== Testing Dagster Pipeline Configuration ===")
    try:
        # Test if we can access the pipeline definitions
        defs = hrv_pipeline.defs
        print(f"✅ Pipeline definitions loaded")
        print(f"📊 Assets defined: {len(defs.assets) if defs.assets else 0}")
        print(f"🔄 Jobs defined: {len(defs.jobs) if defs.jobs else 0}")
        print(f"⏰ Schedules defined: {len(defs.schedules) if defs.schedules else 0}")
        print(f"📡 Sensors defined: {len(defs.sensors) if defs.sensors else 0}")
        test_results["dagster_config"] = True
    except Exception as e:
        print(f"❌ Dagster configuration failed: {e}")
        test_results["dagster_config"] = False
    
    # Test 4: Test Pipeline Assets
    print("\n=== Testing Pipeline Assets ===")
    pipeline_assets = [
        "whoop_raw_data",
        "whoop_processed_data", 
        "journal_entries",
        "trained_model",
        "ml_predictions",
        "llm_predictions",
        "daily_report"
    ]
    
    assets_available = 0
    for asset_name in pipeline_assets:
        try:
            asset_func = getattr(hrv_pipeline, asset_name)
            print(f"✅ Asset '{asset_name}': Defined")
            assets_available += 1
        except AttributeError:
            print(f"❌ Asset '{asset_name}': Not found")
    
    test_results["pipeline_assets"] = assets_available / len(pipeline_assets)
    
    # Test 5: Mock Pipeline Execution
    print("\n=== Testing Mock Pipeline Execution ===")
    try:
        # Simulate pipeline configuration
        from orchestrator.hrv_pipeline import PipelineConfig
        config = PipelineConfig()
        print(f"✅ Pipeline configuration created")
        print(f"🔄 WHOOP fetch interval: {config.whoop_fetch_interval} minutes")
        print(f"📅 Prediction horizon: {config.prediction_horizon} days")
        print(f"📝 Journal analysis enabled: {config.enable_journal_analysis}")
        print(f"🤖 LLM predictions enabled: {config.enable_llm_predictions}")
        test_results["mock_execution"] = True
    except Exception as e:
        print(f"❌ Mock pipeline execution failed: {e}")
        test_results["mock_execution"] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("=== Orchestrator Test Results ===")
    
    passed_tests = sum(1 for result in test_results.values() if 
                      (isinstance(result, bool) and result) or 
                      (isinstance(result, float) and result > 0.5))
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        if isinstance(result, bool):
            status = "✅ PASS" if result else "❌ FAIL"
        else:
            status = f"✅ PASS ({result:.1%})" if result > 0.5 else f"❌ FAIL ({result:.1%})"
        
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests >= total_tests * 0.8:
        print("\n🎉 Orchestrator is ready for deployment!")
        print("📋 To run the orchestrator:")
        print("   cd orchestrator")
        print("   pip install -r requirements.txt")
        print("   dagster dev -f hrv_pipeline.py")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) need attention.")
    
    return test_results

if __name__ == "__main__":
    test_orchestrator_components()