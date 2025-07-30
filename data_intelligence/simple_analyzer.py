#!/usr/bin/env python3
"""
Simple Data Analyzer for HRV Data Intelligence.
Basic analysis without external dependencies.
"""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

class SimpleDataAnalyzer:
    """Simple data analyzer using only built-in Python libraries"""
    
    def __init__(self):
        self.data_dir = Path("../data")
        self.reports_dir = Path("../reports/data_intelligence")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_whoop_data(self) -> Dict[str, List[Dict]]:
        """Load all WHOOP data from files"""
        print("Loading WHOOP data...")
        
        data = {}
        
        # Load cycles
        cycles_file = self.data_dir / "cycles" / "cycles.json"
        data['cycles'] = self._load_json_file(cycles_file)
        
        # Load recoveries
        recoveries_file = self.data_dir / "recovery" / "recoveries.json"
        data['recoveries'] = self._load_json_file(recoveries_file)
        
        # Load sleep
        sleep_file = self.data_dir / "sleep" / "sleep_activities.json"
        data['sleep'] = self._load_json_file(sleep_file)
        
        # Load workouts
        workouts_file = self.data_dir / "workouts" / "workouts.json"
        data['workouts'] = self._load_json_file(workouts_file)
        
        print(f"Loaded {len(data['cycles'])} cycles, {len(data['recoveries'])} recoveries, "
              f"{len(data['sleep'])} sleep records, {len(data['workouts'])} workouts")
        
        return data
    
    def _load_json_file(self, filepath: Path) -> List[Dict]:
        """Load JSON file or return empty list"""
        if filepath.exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
    
    def extract_hrv_data(self, recoveries: List[Dict]) -> List[Dict]:
        """Extract HRV data points from recovery data"""
        hrv_data = []
        
        for recovery in recoveries:
            if recovery.get('score') and recovery['score'].get('hrv_rmssd_milli'):
                data_point = {
                    'date': recovery.get('updated_at', recovery.get('created_at')),
                    'hrv_rmssd': recovery['score']['hrv_rmssd_milli'],
                    'recovery_score': recovery['score'].get('recovery_score'),
                    'resting_heart_rate': recovery['score'].get('resting_heart_rate'),
                    'cycle_id': recovery.get('cycle_id')
                }
                hrv_data.append(data_point)
        
        # Sort by date
        hrv_data.sort(key=lambda x: x['date'])
        return hrv_data
    
    def analyze_hrv_trends(self, hrv_data: List[Dict]) -> Dict:
        """Analyze HRV trends and patterns"""
        if not hrv_data:
            return {"error": "No HRV data available"}
        
        hrv_values = [d['hrv_rmssd'] for d in hrv_data if d['hrv_rmssd']]
        
        if not hrv_values:
            return {"error": "No valid HRV values"}
        
        # Basic statistics
        analysis = {
            "total_readings": len(hrv_values),
            "mean_hrv": statistics.mean(hrv_values),
            "median_hrv": statistics.median(hrv_values),
            "min_hrv": min(hrv_values),
            "max_hrv": max(hrv_values),
            "std_dev": statistics.stdev(hrv_values) if len(hrv_values) > 1 else 0
        }
        
        # Recent trend (last 7 days vs previous 7 days)
        if len(hrv_values) >= 14:
            recent_7 = hrv_values[-7:]
            previous_7 = hrv_values[-14:-7]
            
            recent_avg = statistics.mean(recent_7)
            previous_avg = statistics.mean(previous_7)
            
            change_percent = ((recent_avg - previous_avg) / previous_avg) * 100
            
            analysis["recent_trend"] = {
                "recent_7d_avg": recent_avg,
                "previous_7d_avg": previous_avg,
                "change_percent": change_percent,
                "direction": "improving" if change_percent > 2 else "declining" if change_percent < -2 else "stable"
            }
        
        # Categorize HRV levels
        analysis["hrv_categories"] = self._categorize_hrv_values(hrv_values)
        
        return analysis
    
    def _categorize_hrv_values(self, hrv_values: List[float]) -> Dict:
        """Categorize HRV values into ranges"""
        categories = {"very_low": 0, "low": 0, "moderate": 0, "high": 0, "very_high": 0}
        
        for hrv in hrv_values:
            if hrv < 20:
                categories["very_low"] += 1
            elif hrv < 35:
                categories["low"] += 1
            elif hrv < 50:
                categories["moderate"] += 1
            elif hrv < 65:
                categories["high"] += 1
            else:
                categories["very_high"] += 1
        
        total = len(hrv_values)
        percentages = {k: (v / total) * 100 for k, v in categories.items()}
        
        return {
            "counts": categories,
            "percentages": percentages,
            "dominant_category": max(percentages.keys(), key=percentages.get)
        }
    
    def analyze_recovery_patterns(self, recoveries: List[Dict]) -> Dict:
        """Analyze recovery score patterns"""
        recovery_scores = []
        
        for recovery in recoveries:
            if recovery.get('score') and recovery['score'].get('recovery_score'):
                recovery_scores.append(recovery['score']['recovery_score'])
        
        if not recovery_scores:
            return {"error": "No recovery data available"}
        
        analysis = {
            "total_readings": len(recovery_scores),
            "mean_recovery": statistics.mean(recovery_scores),
            "median_recovery": statistics.median(recovery_scores),
            "min_recovery": min(recovery_scores),
            "max_recovery": max(recovery_scores),
            "std_dev": statistics.stdev(recovery_scores) if len(recovery_scores) > 1 else 0
        }
        
        # Categorize into recovery zones
        zones = {"red": 0, "yellow": 0, "green": 0}
        for score in recovery_scores:
            if score < 33:
                zones["red"] += 1
            elif score < 66:
                zones["yellow"] += 1
            else:
                zones["green"] += 1
        
        total = len(recovery_scores)
        analysis["recovery_zones"] = {
            "counts": zones,
            "percentages": {k: (v / total) * 100 for k, v in zones.items()}
        }
        
        return analysis
    
    def analyze_sleep_patterns(self, sleep_data: List[Dict]) -> Dict:
        """Analyze sleep patterns"""
        sleep_performances = []
        sleep_durations = []
        
        for sleep in sleep_data:
            if sleep.get('score'):
                score = sleep['score']
                if score.get('sleep_performance_percentage'):
                    sleep_performances.append(score['sleep_performance_percentage'])
                
                if score.get('stage_summary', {}).get('total_in_bed_time_milli'):
                    # Convert milliseconds to hours
                    duration_hours = score['stage_summary']['total_in_bed_time_milli'] / (1000 * 60 * 60)
                    sleep_durations.append(duration_hours)
        
        analysis = {}
        
        if sleep_performances:
            analysis["sleep_performance"] = {
                "mean": statistics.mean(sleep_performances),
                "median": statistics.median(sleep_performances),
                "min": min(sleep_performances),
                "max": max(sleep_performances)
            }
        
        if sleep_durations:
            analysis["sleep_duration"] = {
                "mean_hours": statistics.mean(sleep_durations),
                "median_hours": statistics.median(sleep_durations),
                "min_hours": min(sleep_durations),
                "max_hours": max(sleep_durations)
            }
        
        return analysis
    
    def generate_daily_summary(self, target_date: str = None) -> Dict:
        """Generate summary for a specific date (defaults to today)"""
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        data = self.load_whoop_data()
        
        # Find data for the target date
        summary = {
            "date": target_date,
            "hrv_actual": None,
            "recovery_score": None,
            "strain_score": None,
            "sleep_performance": None,
            "workout_count": 0,
            "workout_summary": "No workouts"
        }
        
        # Check recovery data
        for recovery in data['recoveries']:
            if self._is_same_date(recovery.get('updated_at', ''), target_date):
                if recovery.get('score'):
                    summary["hrv_actual"] = recovery['score'].get('hrv_rmssd_milli')
                    summary["recovery_score"] = recovery['score'].get('recovery_score')
                break
        
        # Check cycles for strain
        for cycle in data['cycles']:
            if self._is_same_date(cycle.get('start', ''), target_date):
                if cycle.get('score'):
                    summary["strain_score"] = cycle['score'].get('strain')
                break
        
        # Check sleep data
        for sleep in data['sleep']:
            if self._is_same_date(sleep.get('start', ''), target_date):
                if sleep.get('score'):
                    summary["sleep_performance"] = sleep['score'].get('sleep_performance_percentage')
                break
        
        # Check workouts
        workout_count = 0
        workout_strains = []
        for workout in data['workouts']:
            if self._is_same_date(workout.get('start', ''), target_date):
                workout_count += 1
                if workout.get('score', {}).get('strain'):
                    workout_strains.append(workout['score']['strain'])
        
        summary["workout_count"] = workout_count
        if workout_count > 0:
            total_strain = sum(workout_strains)
            summary["workout_summary"] = f"{workout_count} workout(s), total strain: {total_strain:.1f}"
        
        return summary
    
    def _is_same_date(self, datetime_str: str, target_date: str) -> bool:
        """Check if datetime string matches target date"""
        if not datetime_str:
            return False
        
        try:
            # Parse datetime and extract date
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d') == target_date
        except:
            return False
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        print("Generating comprehensive data analysis report...")
        
        data = self.load_whoop_data()
        
        # Extract HRV data for analysis
        hrv_data = self.extract_hrv_data(data['recoveries'])
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "data_summary": {
                "total_cycles": len(data['cycles']),
                "total_recoveries": len(data['recoveries']),
                "total_sleep_records": len(data['sleep']),
                "total_workouts": len(data['workouts']),
                "hrv_data_points": len(hrv_data)
            },
            "hrv_analysis": self.analyze_hrv_trends(hrv_data),
            "recovery_analysis": self.analyze_recovery_patterns(data['recoveries']),
            "sleep_analysis": self.analyze_sleep_patterns(data['sleep']),
            "recommendations": self._generate_recommendations(hrv_data, data['recoveries'])
        }
        
        # Save report
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f"simple_analysis_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Report saved to {report_file}")
        return report
    
    def _generate_recommendations(self, hrv_data: List[Dict], recoveries: List[Dict]) -> List[str]:
        """Generate simple recommendations based on data"""
        recommendations = []
        
        if hrv_data:
            hrv_values = [d['hrv_rmssd'] for d in hrv_data if d['hrv_rmssd']]
            
            if hrv_values:
                recent_hrv = statistics.mean(hrv_values[-7:]) if len(hrv_values) >= 7 else statistics.mean(hrv_values)
                overall_avg = statistics.mean(hrv_values)
                
                if recent_hrv < overall_avg * 0.9:
                    recommendations.append("Recent HRV is below average. Consider focusing on recovery.")
                
                if recent_hrv < 30:
                    recommendations.append("HRV is relatively low. Prioritize sleep and stress management.")
        
        # Recovery zone analysis
        recovery_scores = [r['score']['recovery_score'] for r in recoveries 
                          if r.get('score', {}).get('recovery_score')]
        
        if recovery_scores:
            recent_recovery = statistics.mean(recovery_scores[-7:]) if len(recovery_scores) >= 7 else statistics.mean(recovery_scores)
            
            if recent_recovery < 50:
                recommendations.append("Recent recovery scores are in the yellow/red zone. Focus on rest.")
        
        if not recommendations:
            recommendations.append("Data patterns look healthy. Continue current habits.")
        
        return recommendations

def main():
    """Main function for testing"""
    analyzer = SimpleDataAnalyzer()
    
    print("=== HRV Data Intelligence Report ===")
    
    # Generate comprehensive report
    report = analyzer.generate_comprehensive_report()
    
    print("\n=== Key Insights ===")
    if 'hrv_analysis' in report and 'error' not in report['hrv_analysis']:
        hrv = report['hrv_analysis']
        print(f"HRV Mean: {hrv['mean_hrv']:.1f} ms")
        print(f"HRV Range: {hrv['min_hrv']:.1f} - {hrv['max_hrv']:.1f} ms")
        
        if 'recent_trend' in hrv:
            trend = hrv['recent_trend']
            print(f"Recent Trend: {trend['direction']} ({trend['change_percent']:.1f}%)")
    
    if 'recovery_analysis' in report and 'error' not in report['recovery_analysis']:
        recovery = report['recovery_analysis']
        print(f"Recovery Mean: {recovery['mean_recovery']:.1f}")
        
        zones = recovery['recovery_zones']['percentages']
        print(f"Recovery Zones - Green: {zones['green']:.1f}%, Yellow: {zones['yellow']:.1f}%, Red: {zones['red']:.1f}%")
    
    print("\n=== Recommendations ===")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # Generate daily summary
    print(f"\n=== Today's Summary ===")
    daily = analyzer.generate_daily_summary()
    print(f"Date: {daily['date']}")
    print(f"HRV: {daily['hrv_actual'] or 'N/A'}")
    print(f"Recovery: {daily['recovery_score'] or 'N/A'}")
    print(f"Strain: {daily['strain_score'] or 'N/A'}")
    print(f"Sleep Performance: {daily['sleep_performance'] or 'N/A'}%")
    print(f"Workouts: {daily['workout_summary']}")

if __name__ == "__main__":
    main()