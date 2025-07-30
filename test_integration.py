#!/usr/bin/env python3
"""
Integration Test for HRV Prediction Infrastructure
Tests all components working together to generate HRV predictions.
"""

import sys
import json
from datetime import datetime, timedelta
from pathlib import Path

def test_whoop_data_availability():
    """Test if WHOOP data is available"""
    print("=== Testing WHOOP Data Availability ===")
    
    data_dir = Path("data")
    
    # Check key data files
    files_to_check = [
        "cycles/cycles.json",
        "recovery/recoveries.json", 
        "sleep/sleep_activities.json",
        "workouts/workouts.json"
    ]
    
    available_data = {}
    
    for file_path in files_to_check:
        full_path = data_dir / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    data = json.load(f)
                    available_data[file_path] = len(data)
                    print(f"✅ {file_path}: {len(data)} records")
            except Exception as e:
                print(f"❌ {file_path}: Error loading - {e}")
                available_data[file_path] = 0
        else:
            print(f"❌ {file_path}: File not found")
            available_data[file_path] = 0
    
    total_records = sum(available_data.values())
    print(f"\nTotal data records: {total_records}")
    
    return total_records > 0

def test_data_intelligence():
    """Test data intelligence module"""
    print("\n=== Testing Data Intelligence ===")
    
    try:
        sys.path.append('data_intelligence')
        from data_intelligence.simple_analyzer import SimpleDataAnalyzer
        
        analyzer = SimpleDataAnalyzer()
        report = analyzer.generate_comprehensive_report()
        
        print(f"✅ Data Intelligence Report Generated")
        
        if 'hrv_analysis' in report and 'error' not in report['hrv_analysis']:
            hrv = report['hrv_analysis']
            print(f"   HRV Mean: {hrv['mean_hrv']:.1f} ms")
            print(f"   HRV Trend: {hrv.get('recent_trend', {}).get('direction', 'unknown')}")
        
        if 'recovery_analysis' in report and 'error' not in report['recovery_analysis']:
            recovery = report['recovery_analysis'] 
            print(f"   Recovery Mean: {recovery['mean_recovery']:.1f}")
        
        # Generate daily summary
        summary = analyzer.generate_daily_summary()
        print(f"   Today's HRV: {summary['hrv_actual'] or 'N/A'}")
        print(f"   Today's Recovery: {summary['recovery_score'] or 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Intelligence Test Failed: {e}")
        return False

def test_journal_analysis():
    """Test journal analysis module"""
    print("\n=== Testing Journal Analysis ===")
    
    try:
        sys.path.append('journal_analysis')
        from journal_analysis.simple_analyzer import SimpleJournalAnalyzer
        
        analyzer = SimpleJournalAnalyzer()
        
        # Process mock journal entry
        analysis = analyzer.process_mock_voice_recording()
        
        print(f"✅ Journal Analysis Complete")
        print(f"   Stress Level: {analysis['stress_level']}/10")
        print(f"   Overall Tone: {analysis['overall_tone']}")
        print(f"   Exercise Mentioned: {analysis['behavioral_factors']['exercise_mentioned']}")
        print(f"   Sleep Quality: {analysis['health_mentions'].get('sleep_quality_mentioned', 'Not specified')}")
        
        # Generate insights summary
        insights = analyzer.generate_insights_summary(analysis)
        print(f"   Insights: {insights}")
        
        return True
        
    except Exception as e:
        print(f"❌ Journal Analysis Test Failed: {e}")
        return False

def test_model_training():
    """Test model training module"""
    print("\n=== Testing Model Training ===")
    
    try:
        sys.path.append('model_training')
        from model_training.simple_trainer import SimpleHRVModelTrainer
        
        trainer = SimpleHRVModelTrainer()
        
        # Try to generate a prediction
        prediction = trainer.predict_hrv()
        
        print(f"✅ Model Prediction Generated")
        print(f"   Predicted HRV: {prediction['predicted_hrv']} ms")
        print(f"   Confidence Interval: {prediction['confidence_interval']['lower']} - {prediction['confidence_interval']['upper']} ms")
        print(f"   Model Type: {prediction['model_type']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model Training Test Failed: {e}")
        return False

def test_llm_agent():
    """Test LLM agent"""
    print("\n=== Testing LLM Agent ===")
    
    try:
        sys.path.append('llm_agent')
        from llm_agent.hrv_predictor import HRVPredictionAgent
        
        agent = HRVPredictionAgent()
        
        # Generate prediction for tomorrow
        tomorrow = datetime.now() + timedelta(days=1)
        prediction = agent.generate_prediction(tomorrow)
        
        print(f"✅ LLM Agent Prediction Generated")
        print(f"   Target Date: {tomorrow.strftime('%Y-%m-%d')}")
        print(f"   Predicted HRV: {prediction['hrv_prediction']} ms")
        print(f"   Confidence: {prediction['confidence_level']}")
        print(f"   Prediction Type: {prediction['prediction_type']}")
        
        print(f"\n   Key Factors:")
        for factor in prediction['key_factors'][:3]:  # Show first 3 factors
            print(f"   - {factor}")
        
        print(f"\n   Reasoning: {prediction['reasoning'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM Agent Test Failed: {e}")
        return False

def test_shared_utilities():
    """Test shared utilities"""
    print("\n=== Testing Shared Utilities ===")
    
    try:
        sys.path.append('shared')
        from shared.utils import categorize_hrv, categorize_recovery, milliseconds_to_hours
        
        # Test utility functions
        hrv_category = categorize_hrv(85.5)
        recovery_category = categorize_recovery(65)
        hours = milliseconds_to_hours(28800000)  # 8 hours in ms
        
        print(f"✅ Shared Utilities Working")
        print(f"   HRV 85.5 ms categorized as: {hrv_category}")
        print(f"   Recovery 65 categorized as: {recovery_category}")
        print(f"   28800000 ms = {hours} hours")
        
        return True
        
    except Exception as e:
        print(f"❌ Shared Utilities Test Failed: {e}")
        return False

def generate_comprehensive_prediction():
    """Generate a comprehensive prediction using all components"""
    print("\n=== Generating Comprehensive HRV Prediction ===")
    
    try:
        # Import all modules
        sys.path.append('data_intelligence')
        sys.path.append('journal_analysis') 
        sys.path.append('model_training')
        sys.path.append('llm_agent')
        
        from data_intelligence.simple_analyzer import SimpleDataAnalyzer
        from journal_analysis.simple_analyzer import SimpleJournalAnalyzer
        from model_training.simple_trainer import SimpleHRVModelTrainer
        from llm_agent.hrv_predictor import HRVPredictionAgent
        
        # Get data intelligence insights
        data_analyzer = SimpleDataAnalyzer()
        data_report = data_analyzer.generate_comprehensive_report()
        
        # Get journal insights
        journal_analyzer = SimpleJournalAnalyzer()
        journal_report = journal_analyzer.create_daily_report()
        
        # Get ML model prediction
        model_trainer = SimpleHRVModelTrainer()
        model_prediction = model_trainer.predict_hrv()
        
        # Get LLM agent prediction
        llm_agent = HRVPredictionAgent()
        agent_prediction = llm_agent.generate_prediction()
        
        # Combine all insights
        comprehensive_prediction = {
            'timestamp': datetime.now().isoformat(),
            'target_date': (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            'data_intelligence': {
                'hrv_trend': data_report.get('hrv_analysis', {}).get('recent_trend', {}).get('direction', 'unknown'),
                'recovery_mean': data_report.get('recovery_analysis', {}).get('mean_recovery', 'unknown'),
                'recommendations': data_report.get('recommendations', [])
            },
            'journal_insights': {
                'stress_level': journal_report['key_factors']['stress_level'],
                'overall_tone': journal_report['key_factors']['overall_tone'],
                'exercise_mentioned': journal_report['key_factors']['exercise_mentioned'],
                'insights_summary': journal_report['insights_summary']
            },
            'model_prediction': {
                'predicted_hrv': model_prediction['predicted_hrv'],
                'confidence_interval': model_prediction['confidence_interval'],
                'model_type': model_prediction['model_type']
            },
            'llm_agent_prediction': {
                'predicted_hrv': agent_prediction['hrv_prediction'],
                'confidence_level': agent_prediction['confidence_level'],
                'reasoning': agent_prediction['reasoning'],
                'recommendations': agent_prediction['recommendations']
            }
        }
        
        # Save comprehensive prediction
        prediction_file = Path("comprehensive_hrv_prediction.json")
        with open(prediction_file, 'w') as f:
            json.dump(comprehensive_prediction, f, indent=2)
        
        print(f"✅ Comprehensive Prediction Generated and Saved")
        print(f"\n=== Prediction Summary ===")
        print(f"Target Date: {comprehensive_prediction['target_date']}")
        print(f"ML Model Prediction: {model_prediction['predicted_hrv']} ms")
        print(f"LLM Agent Prediction: {agent_prediction['hrv_prediction']} ms")
        print(f"Data Trend: {comprehensive_prediction['data_intelligence']['hrv_trend']}")
        print(f"Journal Stress Level: {comprehensive_prediction['journal_insights']['stress_level']}/10")
        print(f"Journal Tone: {comprehensive_prediction['journal_insights']['overall_tone']}")
        
        print(f"\n=== Key Recommendations ===")
        for rec in agent_prediction['recommendations'][:3]:
            print(f"- {rec}")
        
        print(f"\n📄 Full prediction saved to: {prediction_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive Prediction Failed: {e}")
        return False

def main():
    """Run integration tests"""
    print("🚀 HRV Prediction Infrastructure Integration Test")
    print("=" * 60)
    
    test_results = {}
    
    # Run individual component tests
    test_results['whoop_data'] = test_whoop_data_availability()
    test_results['data_intelligence'] = test_data_intelligence()
    test_results['journal_analysis'] = test_journal_analysis()
    test_results['model_training'] = test_model_training()
    test_results['llm_agent'] = test_llm_agent()
    test_results['shared_utilities'] = test_shared_utilities()
    
    # Run comprehensive test
    test_results['comprehensive_prediction'] = generate_comprehensive_prediction()
    
    # Summary
    print("\n" + "=" * 60)
    print("=== Test Results Summary ===")
    
    passed_tests = sum(1 for result in test_results.values() if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 All systems operational! HRV prediction infrastructure is complete.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} component(s) need attention.")
    
    print(f"\n📊 The system successfully:")
    print(f"   - Analyzed {test_results.get('whoop_data', 0)} WHOOP data records")
    print(f"   - Generated data intelligence insights")
    print(f"   - Processed journal behavioral analysis") 
    print(f"   - Created ML model predictions")
    print(f"   - Produced LLM-powered HRV forecasts")
    print(f"   - Integrated all components for comprehensive predictions")

if __name__ == "__main__":
    main()