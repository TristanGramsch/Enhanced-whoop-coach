"""
HRV Prediction LLM Agent.
Uses OpenAI GPT to analyze data and generate HRV predictions with explanations.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install openai")

class HRVPredictionAgent:
    """LLM-based agent for HRV prediction and analysis"""
    
    def __init__(self, api_key: str = None):
        self.data_dir = Path("../data")
        self.reports_dir = Path("../reports")
        self.predictions_dir = Path("../predictions")
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        if OPENAI_AVAILABLE and api_key:
            openai.api_key = api_key
            self.openai_available = True
        else:
            self.openai_available = False
            print("Warning: OpenAI API not configured. Using mock predictions.")
        
        # Load historical data for context
        self.historical_data = self._load_historical_context()
    
    def _load_historical_context(self) -> Dict:
        """Load recent historical data for context"""
        context = {
            "recent_summaries": [],
            "data_intelligence": {},
            "model_predictions": []
        }
        
        # Load recent data intelligence reports
        intelligence_dir = self.reports_dir / "data_intelligence"
        if intelligence_dir.exists():
            reports = sorted(intelligence_dir.glob("*.json"))
            if reports:
                latest_report = reports[-1]
                try:
                    with open(latest_report, 'r') as f:
                        context["data_intelligence"] = json.load(f)
                except Exception as e:
                    print(f"Error loading intelligence report: {e}")
        
        return context
    
    def generate_prediction(self, target_date: datetime = None) -> Dict:
        """Generate HRV prediction for target date"""
        if target_date is None:
            target_date = datetime.now() + timedelta(days=1)
        
        # Gather data for prediction
        context_data = self._gather_prediction_context(target_date)
        
        # Generate prediction
        if self.openai_available:
            prediction = self._generate_ai_prediction(context_data, target_date)
        else:
            prediction = self._generate_mock_prediction(context_data, target_date)
        
        # Save prediction
        self._save_prediction(prediction)
        
        return prediction
    
    def _gather_prediction_context(self, target_date: datetime) -> Dict:
        """Gather all relevant context for prediction"""
        context = {
            "target_date": target_date.isoformat(),
            "current_trends": self._analyze_current_trends(),
            "recent_metrics": self._get_recent_metrics(),
            "sleep_patterns": self._analyze_sleep_patterns(),
            "strain_patterns": self._analyze_strain_patterns(),
            "recovery_patterns": self._analyze_recovery_patterns(),
            "data_quality": self._assess_data_quality(),
            "seasonal_factors": self._consider_seasonal_factors(target_date)
        }
        
        return context
    
    def _analyze_current_trends(self) -> Dict:
        """Analyze current HRV and recovery trends"""
        intelligence = self.historical_data.get("data_intelligence", {})
        
        trends = {
            "hrv_trend": "stable",
            "recovery_trend": "stable",
            "confidence": "medium"
        }
        
        if "hrv_analysis" in intelligence:
            hrv_analysis = intelligence["hrv_analysis"]
            
            if "recent_trend" in hrv_analysis:
                trends["hrv_trend"] = hrv_analysis["recent_trend"].get("direction", "stable")
                trends["hrv_change_percent"] = hrv_analysis["recent_trend"].get("change_percent", 0)
            
            if "variability" in hrv_analysis:
                variability = hrv_analysis["variability"]
                if variability.get("variability_level") == "high":
                    trends["confidence"] = "low"
                elif variability.get("variability_level") == "low":
                    trends["confidence"] = "high"
        
        return trends
    
    def _get_recent_metrics(self) -> Dict:
        """Get recent HRV and recovery metrics"""
        # Load recent recovery data
        recoveries_file = self.data_dir / "recovery" / "recoveries.json"
        recent_metrics = {
            "recent_hrv": [],
            "recent_recovery": [],
            "recent_strain": []
        }
        
        if recoveries_file.exists():
            try:
                with open(recoveries_file, 'r') as f:
                    recoveries = json.load(f)
                
                # Get last 7 days of data
                cutoff_date = datetime.now() - timedelta(days=7)
                
                for recovery in recoveries[-50:]:  # Check recent records
                    if recovery.get('score'):
                        score = recovery['score']
                        
                        if score.get('hrv_rmssd_milli'):
                            recent_metrics["recent_hrv"].append(score['hrv_rmssd_milli'])
                        
                        if score.get('recovery_score'):
                            recent_metrics["recent_recovery"].append(score['recovery_score'])
            
            except Exception as e:
                print(f"Error loading recent metrics: {e}")
        
        return recent_metrics
    
    def _analyze_sleep_patterns(self) -> Dict:
        """Analyze recent sleep patterns"""
        sleep_file = self.data_dir / "sleep" / "sleep_activities.json"
        patterns = {
            "average_performance": None,
            "average_duration": None,
            "consistency": "unknown"
        }
        
        if sleep_file.exists():
            try:
                with open(sleep_file, 'r') as f:
                    sleep_data = json.load(f)
                
                recent_performances = []
                recent_durations = []
                
                for sleep in sleep_data[-14:]:  # Last 14 days
                    if sleep.get('score'):
                        score = sleep['score']
                        
                        if score.get('sleep_performance_percentage'):
                            recent_performances.append(score['sleep_performance_percentage'])
                        
                        if score.get('stage_summary', {}).get('total_in_bed_time_milli'):
                            duration_hours = score['stage_summary']['total_in_bed_time_milli'] / (1000 * 60 * 60)
                            recent_durations.append(duration_hours)
                
                if recent_performances:
                    patterns["average_performance"] = sum(recent_performances) / len(recent_performances)
                
                if recent_durations:
                    patterns["average_duration"] = sum(recent_durations) / len(recent_durations)
                    
                    # Assess consistency
                    if len(recent_durations) > 3:
                        std_dev = (sum((x - patterns["average_duration"]) ** 2 for x in recent_durations) / len(recent_durations)) ** 0.5
                        if std_dev < 0.5:
                            patterns["consistency"] = "high"
                        elif std_dev < 1.0:
                            patterns["consistency"] = "moderate"
                        else:
                            patterns["consistency"] = "low"
            
            except Exception as e:
                print(f"Error analyzing sleep patterns: {e}")
        
        return patterns
    
    def _analyze_strain_patterns(self) -> Dict:
        """Analyze recent strain patterns"""
        cycles_file = self.data_dir / "cycles" / "cycles.json"
        patterns = {
            "average_strain": None,
            "recent_trend": "stable",
            "high_strain_days": 0
        }
        
        if cycles_file.exists():
            try:
                with open(cycles_file, 'r') as f:
                    cycles = json.load(f)
                
                recent_strains = []
                
                for cycle in cycles[-14:]:  # Last 14 days
                    if cycle.get('score', {}).get('strain'):
                        strain = cycle['score']['strain']
                        recent_strains.append(strain)
                        
                        if strain > 15:  # High strain threshold
                            patterns["high_strain_days"] += 1
                
                if recent_strains:
                    patterns["average_strain"] = sum(recent_strains) / len(recent_strains)
                    
                    # Analyze trend
                    if len(recent_strains) >= 7:
                        recent_avg = sum(recent_strains[-3:]) / 3
                        previous_avg = sum(recent_strains[-7:-3]) / 4
                        
                        if recent_avg > previous_avg * 1.1:
                            patterns["recent_trend"] = "increasing"
                        elif recent_avg < previous_avg * 0.9:
                            patterns["recent_trend"] = "decreasing"
            
            except Exception as e:
                print(f"Error analyzing strain patterns: {e}")
        
        return patterns
    
    def _analyze_recovery_patterns(self) -> Dict:
        """Analyze recovery score patterns"""
        intelligence = self.historical_data.get("data_intelligence", {})
        
        patterns = {
            "average_recovery": None,
            "zone_distribution": {},
            "consistency": "unknown"
        }
        
        if "recovery_analysis" in intelligence:
            recovery_analysis = intelligence["recovery_analysis"]
            
            if "mean_recovery" in recovery_analysis:
                patterns["average_recovery"] = recovery_analysis["mean_recovery"]
            
            if "recovery_zones" in recovery_analysis:
                patterns["zone_distribution"] = recovery_analysis["recovery_zones"].get("percentages", {})
        
        return patterns
    
    def _assess_data_quality(self) -> Dict:
        """Assess the quality and completeness of available data"""
        intelligence = self.historical_data.get("data_intelligence", {})
        
        quality = {
            "data_completeness": "unknown",
            "reliability": "medium",
            "recommendation": ""
        }
        
        if "data_summary" in intelligence:
            data_summary = intelligence["data_summary"]
            
            if "data_completeness" in data_summary:
                completeness = data_summary["data_completeness"]
                
                # Check HRV data completeness
                hrv_percentage = completeness.get("hrv_rmssd", {}).get("percentage", 0)
                
                if hrv_percentage > 80:
                    quality["data_completeness"] = "high"
                    quality["reliability"] = "high"
                elif hrv_percentage > 60:
                    quality["data_completeness"] = "medium"
                    quality["reliability"] = "medium"
                else:
                    quality["data_completeness"] = "low"
                    quality["reliability"] = "low"
                    quality["recommendation"] = "Prediction confidence limited by insufficient data."
        
        return quality
    
    def _consider_seasonal_factors(self, target_date: datetime) -> Dict:
        """Consider seasonal and calendar factors"""
        factors = {
            "day_of_week": target_date.strftime("%A"),
            "is_weekend": target_date.weekday() >= 5,
            "month": target_date.strftime("%B"),
            "seasonal_note": ""
        }
        
        # Weekend prediction adjustments
        if factors["is_weekend"]:
            factors["seasonal_note"] = "Weekend - potential for different sleep/recovery patterns"
        
        # Seasonal considerations
        month = target_date.month
        if month in [12, 1, 2]:  # Winter
            factors["seasonal_note"] += " Winter season may affect HRV patterns."
        elif month in [6, 7, 8]:  # Summer
            factors["seasonal_note"] += " Summer season - consider heat/activity impacts."
        
        return factors
    
    def _generate_ai_prediction(self, context: Dict, target_date: datetime) -> Dict:
        """Generate prediction using OpenAI API"""
        prompt = self._build_prediction_prompt(context, target_date)
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert HRV analyst providing precise, data-driven predictions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            prediction_text = response.choices[0].message.content
            return self._parse_ai_response(prediction_text, context, target_date)
            
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._generate_mock_prediction(context, target_date)
    
    def _build_prediction_prompt(self, context: Dict, target_date: datetime) -> str:
        """Build the prediction prompt for the LLM"""
        prompt = f"""
You are an expert Heart Rate Variability (HRV) analyst. Based on the comprehensive data below, provide a precise HRV prediction for {target_date.strftime('%Y-%m-%d')}.

## Current Data Context:

### Recent Trends:
- HRV Trend: {context['current_trends']['hrv_trend']}
- Recovery Trend: {context['current_trends']['recovery_trend']}
- Confidence Level: {context['current_trends']['confidence']}

### Recent Metrics (last 7 days):
- Recent HRV values: {context['recent_metrics']['recent_hrv'][-7:]}
- Recent Recovery scores: {context['recent_metrics']['recent_recovery'][-7:]}

### Sleep Patterns:
- Average Sleep Performance: {context['sleep_patterns']['average_performance']}%
- Average Sleep Duration: {context['sleep_patterns']['average_duration']} hours
- Sleep Consistency: {context['sleep_patterns']['consistency']}

### Strain Patterns:
- Average Strain: {context['strain_patterns']['average_strain']}
- Recent Strain Trend: {context['strain_patterns']['recent_trend']}
- High Strain Days (last 14d): {context['strain_patterns']['high_strain_days']}

### Recovery Patterns:
- Average Recovery Score: {context['recovery_patterns']['average_recovery']}
- Recovery Zone Distribution: {context['recovery_patterns']['zone_distribution']}

### Data Quality:
- Data Completeness: {context['data_quality']['data_completeness']}
- Reliability: {context['data_quality']['reliability']}

### Contextual Factors:
- Target Date: {context['seasonal_factors']['day_of_week']} ({target_date.strftime('%Y-%m-%d')})
- Weekend: {context['seasonal_factors']['is_weekend']}
- Seasonal Note: {context['seasonal_factors']['seasonal_note']}

## Required Output Format:

Please provide your analysis in this exact JSON format:

{{
  "hrv_prediction": <predicted HRV value in milliseconds>,
  "confidence_level": "<high/medium/low>",
  "confidence_interval": {{
    "lower": <lower bound>,
    "upper": <upper bound>
  }},
  "key_factors": [
    "<factor 1>",
    "<factor 2>",
    "<factor 3>"
  ],
  "reasoning": "<detailed explanation of your prediction logic>",
  "recommendations": [
    "<recommendation 1>",
    "<recommendation 2>"
  ],
  "risk_factors": [
    "<risk factor 1>",
    "<risk factor 2>"
  ]
}}

Focus on:
1. Statistical patterns in the recent data
2. Impact of sleep quality and strain on HRV
3. Recovery trends and their predictive value
4. Any notable patterns or anomalies
5. Practical recommendations for optimizing HRV
"""
        return prompt
    
    def _parse_ai_response(self, response_text: str, context: Dict, target_date: datetime) -> Dict:
        """Parse and validate AI response"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                ai_prediction = json.loads(json_str)
                
                # Validate and structure prediction
                prediction = {
                    "timestamp": datetime.now().isoformat(),
                    "target_date": target_date.isoformat(),
                    "prediction_type": "ai_generated",
                    "hrv_prediction": ai_prediction.get("hrv_prediction"),
                    "confidence_level": ai_prediction.get("confidence_level", "medium"),
                    "confidence_interval": ai_prediction.get("confidence_interval", {}),
                    "key_factors": ai_prediction.get("key_factors", []),
                    "reasoning": ai_prediction.get("reasoning", ""),
                    "recommendations": ai_prediction.get("recommendations", []),
                    "risk_factors": ai_prediction.get("risk_factors", []),
                    "context_data": context
                }
                
                return prediction
            
        except Exception as e:
            print(f"Error parsing AI response: {e}")
        
        # Fallback to mock prediction if parsing fails
        return self._generate_mock_prediction(context, target_date)
    
    def _generate_mock_prediction(self, context: Dict, target_date: datetime) -> Dict:
        """Generate mock prediction when AI is not available"""
        # Simple rule-based prediction
        recent_hrv = context['recent_metrics']['recent_hrv']
        
        if recent_hrv:
            # Use recent average as baseline
            baseline_hrv = sum(recent_hrv[-7:]) / len(recent_hrv[-7:])
            
            # Adjust based on trends
            trend = context['current_trends']['hrv_trend']
            if trend == "improving":
                predicted_hrv = baseline_hrv * 1.05
            elif trend == "declining":
                predicted_hrv = baseline_hrv * 0.95
            else:
                predicted_hrv = baseline_hrv
        else:
            predicted_hrv = 50.0  # Default baseline
        
        # Generate mock prediction
        prediction = {
            "timestamp": datetime.now().isoformat(),
            "target_date": target_date.isoformat(),
            "prediction_type": "rule_based_mock",
            "hrv_prediction": round(predicted_hrv, 1),
            "confidence_level": "medium",
            "confidence_interval": {
                "lower": round(predicted_hrv * 0.85, 1),
                "upper": round(predicted_hrv * 1.15, 1)
            },
            "key_factors": [
                f"Recent HRV trend: {context['current_trends']['hrv_trend']}",
                f"Sleep performance: {context['sleep_patterns']['average_performance']}%",
                f"Recovery pattern: {context['recovery_patterns']['average_recovery']}"
            ],
            "reasoning": f"Mock prediction based on recent HRV average of {baseline_hrv:.1f}ms and {trend} trend.",
            "recommendations": [
                "Maintain consistent sleep schedule",
                "Monitor recovery scores",
                "Consider stress management if HRV declining"
            ],
            "risk_factors": [
                "Limited by mock prediction algorithm",
                "Actual AI analysis not available"
            ],
            "context_data": context
        }
        
        return prediction
    
    def _save_prediction(self, prediction: Dict) -> None:
        """Save prediction to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"hrv_prediction_{timestamp}.json"
        filepath = self.predictions_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(prediction, f, indent=2, default=str)
        
        print(f"Prediction saved to {filepath}")
    
    def generate_weekly_forecast(self) -> List[Dict]:
        """Generate 7-day HRV forecast"""
        forecasts = []
        
        for i in range(1, 8):
            target_date = datetime.now() + timedelta(days=i)
            prediction = self.generate_prediction(target_date)
            forecasts.append(prediction)
        
        # Save weekly forecast
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        forecast_file = self.predictions_dir / f"weekly_forecast_{timestamp}.json"
        
        with open(forecast_file, 'w') as f:
            json.dump(forecasts, f, indent=2, default=str)
        
        print(f"Weekly forecast saved to {forecast_file}")
        return forecasts

def main():
    """Main function for testing"""
    # Initialize agent (without API key for testing)
    agent = HRVPredictionAgent()
    
    print("=== HRV Prediction Agent ===")
    
    # Generate tomorrow's prediction
    tomorrow = datetime.now() + timedelta(days=1)
    prediction = agent.generate_prediction(tomorrow)
    
    print(f"\n=== Prediction for {tomorrow.strftime('%Y-%m-%d')} ===")
    print(f"Predicted HRV: {prediction['hrv_prediction']} ms")
    print(f"Confidence: {prediction['confidence_level']}")
    print(f"Range: {prediction['confidence_interval']['lower']} - {prediction['confidence_interval']['upper']} ms")
    
    print(f"\n=== Key Factors ===")
    for factor in prediction['key_factors']:
        print(f"- {factor}")
    
    print(f"\n=== Reasoning ===")
    print(prediction['reasoning'])
    
    print(f"\n=== Recommendations ===")
    for rec in prediction['recommendations']:
        print(f"- {rec}")
    
    # Generate weekly forecast
    print(f"\n=== Weekly Forecast ===")
    forecasts = agent.generate_weekly_forecast()
    
    for i, forecast in enumerate(forecasts, 1):
        target_date = datetime.fromisoformat(forecast['target_date'])
        print(f"Day {i} ({target_date.strftime('%m/%d')}): {forecast['hrv_prediction']} ms")

if __name__ == "__main__":
    main()