#!/usr/bin/env python3
"""
Simple Journal Analysis for HRV Context.
Basic text analysis without heavy dependencies.
"""

import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

class SimpleJournalAnalyzer:
    """Simple journal analyzer using basic text processing"""
    
    def __init__(self):
        self.data_dir = Path("../data/journal")
        self.reports_dir = Path("../reports/journal_analysis")
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Keyword patterns for analysis
        self.patterns = {
            'stress_keywords': [
                'stressed', 'anxiety', 'worried', 'overwhelmed', 'pressure',
                'tense', 'nervous', 'frustrated', 'exhausted', 'burnout',
                'difficult', 'challenging', 'busy', 'hectic'
            ],
            'positive_keywords': [
                'good', 'great', 'excellent', 'amazing', 'fantastic',
                'happy', 'relaxed', 'calm', 'peaceful', 'energetic',
                'motivated', 'accomplished', 'successful'
            ],
            'sleep_keywords': [
                'sleep', 'tired', 'sleepy', 'insomnia', 'restless', 
                'dream', 'nightmare', 'wake', 'woke', 'bed', 'nap',
                'slept', 'sleeping', 'drowsy', 'fatigue'
            ],
            'exercise_keywords': [
                'workout', 'exercise', 'gym', 'run', 'running', 'bike', 
                'cycling', 'swim', 'swimming', 'yoga', 'lift', 'weights',
                'cardio', 'training', 'fitness', 'sport', 'active'
            ],
            'alcohol_keywords': [
                'alcohol', 'drink', 'drinking', 'beer', 'wine', 'cocktail',
                'whiskey', 'vodka', 'rum', 'gin', 'bar', 'drunk', 'tipsy'
            ],
            'caffeine_keywords': [
                'coffee', 'caffeine', 'espresso', 'latte', 'tea', 
                'energy drink', 'cola', 'soda', 'stimulant'
            ],
            'work_keywords': [
                'work', 'job', 'office', 'meeting', 'deadline', 'project',
                'boss', 'colleague', 'client', 'presentation', 'busy'
            ],
            'travel_keywords': [
                'travel', 'trip', 'flight', 'airport', 'vacation', 
                'hotel', 'journey', 'visiting', 'away'
            ]
        }
    
    def analyze_text(self, text: str, date: datetime = None) -> Dict:
        """Analyze text for behavioral insights"""
        if date is None:
            date = datetime.now()
        
        text_lower = text.lower()
        
        analysis = {
            'date': date.isoformat(),
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentiment_indicators': self._analyze_sentiment(text_lower),
            'behavioral_factors': self._extract_behavioral_factors(text_lower),
            'health_mentions': self._extract_health_mentions(text_lower),
            'lifestyle_factors': self._extract_lifestyle_factors(text_lower),
            'stress_level': self._assess_stress_level(text_lower),
            'overall_tone': self._assess_overall_tone(text_lower)
        }
        
        return analysis
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Basic sentiment analysis using keyword matching"""
        positive_count = sum(1 for word in self.patterns['positive_keywords'] if word in text)
        stress_count = sum(1 for word in self.patterns['stress_keywords'] if word in text)
        
        total_words = len(text.split())
        
        return {
            'positive_mentions': positive_count,
            'stress_mentions': stress_count,
            'positive_ratio': positive_count / max(total_words, 1),
            'stress_ratio': stress_count / max(total_words, 1),
            'sentiment_balance': positive_count - stress_count
        }
    
    def _extract_behavioral_factors(self, text: str) -> Dict:
        """Extract behavioral factors that might affect HRV"""
        factors = {}
        
        # Exercise mentions
        exercise_mentions = sum(1 for word in self.patterns['exercise_keywords'] if word in text)
        factors['exercise_mentioned'] = exercise_mentions > 0
        factors['exercise_frequency'] = exercise_mentions
        
        # Alcohol mentions
        alcohol_mentions = sum(1 for word in self.patterns['alcohol_keywords'] if word in text)
        factors['alcohol_mentioned'] = alcohol_mentions > 0
        factors['alcohol_frequency'] = alcohol_mentions
        
        # Caffeine mentions
        caffeine_mentions = sum(1 for word in self.patterns['caffeine_keywords'] if word in text)
        factors['caffeine_mentioned'] = caffeine_mentions > 0
        factors['caffeine_frequency'] = caffeine_mentions
        
        # Work stress
        work_mentions = sum(1 for word in self.patterns['work_keywords'] if word in text)
        factors['work_stress_mentioned'] = work_mentions > 0
        factors['work_stress_frequency'] = work_mentions
        
        # Travel
        travel_mentions = sum(1 for word in self.patterns['travel_keywords'] if word in text)
        factors['travel_mentioned'] = travel_mentions > 0
        
        return factors
    
    def _extract_health_mentions(self, text: str) -> Dict:
        """Extract health-related mentions"""
        health = {}
        
        # Sleep mentions
        sleep_mentions = sum(1 for word in self.patterns['sleep_keywords'] if word in text)
        health['sleep_mentioned'] = sleep_mentions > 0
        health['sleep_frequency'] = sleep_mentions
        
        # Specific sleep quality indicators
        if any(word in text for word in ['good sleep', 'slept well', 'rested']):
            health['sleep_quality_mentioned'] = 'good'
        elif any(word in text for word in ['bad sleep', 'poor sleep', 'restless', 'insomnia']):
            health['sleep_quality_mentioned'] = 'poor'
        else:
            health['sleep_quality_mentioned'] = None
        
        # Energy levels
        if any(word in text for word in ['energetic', 'energized', 'energy']):
            health['energy_mentioned'] = 'high'
        elif any(word in text for word in ['tired', 'exhausted', 'fatigue', 'drained']):
            health['energy_mentioned'] = 'low'
        else:
            health['energy_mentioned'] = None
        
        return health
    
    def _extract_lifestyle_factors(self, text: str) -> Dict:
        """Extract lifestyle factors"""
        lifestyle = {}
        
        # Meal timing and quality
        if any(word in text for word in ['breakfast', 'lunch', 'dinner', 'meal', 'eat', 'food']):
            lifestyle['nutrition_mentioned'] = True
        else:
            lifestyle['nutrition_mentioned'] = False
        
        # Social activities
        if any(word in text for word in ['friends', 'family', 'social', 'party', 'gathering']):
            lifestyle['social_activity'] = True
        else:
            lifestyle['social_activity'] = False
        
        # Screen time / technology
        if any(word in text for word in ['phone', 'computer', 'screen', 'internet', 'social media']):
            lifestyle['screen_time_mentioned'] = True
        else:
            lifestyle['screen_time_mentioned'] = False
        
        return lifestyle
    
    def _assess_stress_level(self, text: str) -> int:
        """Assess stress level on 1-10 scale"""
        stress_indicators = sum(1 for word in self.patterns['stress_keywords'] if word in text)
        positive_indicators = sum(1 for word in self.patterns['positive_keywords'] if word in text)
        
        total_words = len(text.split())
        
        if total_words == 0:
            return 5  # Neutral
        
        stress_ratio = stress_indicators / total_words
        positive_ratio = positive_indicators / total_words
        
        # Scale 1-10 where 1 is very low stress, 10 is very high stress
        base_stress = 5  # neutral
        
        if stress_ratio > 0.05:  # High stress words
            base_stress += 3
        elif stress_ratio > 0.02:
            base_stress += 1
        
        if positive_ratio > 0.05:  # Many positive words
            base_stress -= 2
        elif positive_ratio > 0.02:
            base_stress -= 1
        
        # Specific high stress phrases
        if any(phrase in text for phrase in ['very stressed', 'extremely tired', 'overwhelmed', 'burnout']):
            base_stress += 2
        
        return max(1, min(10, base_stress))
    
    def _assess_overall_tone(self, text: str) -> str:
        """Assess overall emotional tone"""
        sentiment = self._analyze_sentiment(text)
        
        if sentiment['sentiment_balance'] > 2:
            return 'very_positive'
        elif sentiment['sentiment_balance'] > 0:
            return 'positive'
        elif sentiment['sentiment_balance'] == 0:
            return 'neutral'
        elif sentiment['sentiment_balance'] > -2:
            return 'negative'
        else:
            return 'very_negative'
    
    def process_mock_voice_recording(self, filename: str = None) -> Dict:
        """Process a voice recording (mock transcription)"""
        if filename is None:
            filename = "2025-07-30.mp4"
        
        # Mock transcription since we don't have Whisper
        mock_transcriptions = {
            "2025-07-30.mp4": """
            Today was a pretty good day overall. I woke up feeling rested after getting about 8 hours of sleep. 
            Had my usual coffee this morning and felt energized for my workout. Did a 30-minute run and felt great afterwards.
            Work was busy with a few meetings, but nothing too stressful. 
            Had lunch with a colleague which was nice and relaxing.
            In the evening, I did some yoga to wind down and prepare for bed.
            Feeling grateful and looking forward to tomorrow.
            """
        }
        
        transcription = mock_transcriptions.get(filename, "No transcription available for this file.")
        
        analysis = self.analyze_text(transcription)
        analysis['audio_file'] = filename
        analysis['transcription'] = transcription.strip()
        analysis['transcription_method'] = 'mock'
        
        return analysis
    
    def save_journal_entry(self, analysis: Dict) -> str:
        """Save journal analysis to file"""
        date_str = analysis['date'][:10]  # Extract YYYY-MM-DD
        filename = f"journal_analysis_{date_str}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Journal analysis saved to {filepath}")
        return str(filepath)
    
    def generate_insights_summary(self, analysis: Dict) -> str:
        """Generate a summary of insights for the LLM agent"""
        insights = []
        
        # Stress assessment
        stress_level = analysis['stress_level']
        if stress_level >= 7:
            insights.append(f"High stress indicators detected (level {stress_level}/10)")
        elif stress_level >= 5:
            insights.append(f"Moderate stress level ({stress_level}/10)")
        else:
            insights.append(f"Low stress level ({stress_level}/10)")
        
        # Behavioral factors
        behavioral = analysis['behavioral_factors']
        if behavioral['exercise_mentioned']:
            insights.append("Exercise activity mentioned")
        
        if behavioral['alcohol_mentioned']:
            insights.append("Alcohol consumption mentioned")
        
        if behavioral['caffeine_mentioned']:
            insights.append("Caffeine intake mentioned")
        
        # Health mentions
        health = analysis['health_mentions']
        if health.get('sleep_quality_mentioned'):
            insights.append(f"Sleep quality described as {health['sleep_quality_mentioned']}")
        
        if health.get('energy_mentioned'):
            insights.append(f"Energy levels described as {health['energy_mentioned']}")
        
        # Overall tone
        tone = analysis['overall_tone']
        insights.append(f"Overall emotional tone: {tone}")
        
        return "; ".join(insights)
    
    def create_daily_report(self, date: datetime = None) -> Dict:
        """Create daily journal analysis report"""
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime('%Y-%m-%d')
        
        # Look for existing journal entry
        journal_file = self.data_dir / f"journal_analysis_{date_str}.json"
        
        if journal_file.exists():
            with open(journal_file, 'r') as f:
                analysis = json.load(f)
        else:
            # Process mock recording if available
            analysis = self.process_mock_voice_recording()
            self.save_journal_entry(analysis)
        
        # Generate report
        report = {
            'date': date_str,
            'analysis_available': True,
            'insights_summary': self.generate_insights_summary(analysis),
            'key_factors': {
                'stress_level': analysis['stress_level'],
                'overall_tone': analysis['overall_tone'],
                'exercise_mentioned': analysis['behavioral_factors']['exercise_mentioned'],
                'sleep_quality': analysis['health_mentions'].get('sleep_quality_mentioned'),
                'energy_level': analysis['health_mentions'].get('energy_mentioned')
            },
            'full_analysis': analysis
        }
        
        # Save report
        report_file = self.reports_dir / f"daily_journal_report_{date_str}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    """Main function for testing"""
    analyzer = SimpleJournalAnalyzer()
    
    print("=== Journal Analysis ===")
    
    # Process mock voice recording
    analysis = analyzer.process_mock_voice_recording()
    
    print(f"\n=== Transcription ===")
    print(analysis['transcription'])
    
    print(f"\n=== Analysis Results ===")
    print(f"Stress Level: {analysis['stress_level']}/10")
    print(f"Overall Tone: {analysis['overall_tone']}")
    print(f"Word Count: {analysis['word_count']}")
    
    print(f"\n=== Behavioral Factors ===")
    behavioral = analysis['behavioral_factors']
    print(f"Exercise mentioned: {behavioral['exercise_mentioned']}")
    print(f"Alcohol mentioned: {behavioral['alcohol_mentioned']}")
    print(f"Caffeine mentioned: {behavioral['caffeine_mentioned']}")
    print(f"Work stress: {behavioral['work_stress_mentioned']}")
    
    print(f"\n=== Health Mentions ===")
    health = analysis['health_mentions']
    print(f"Sleep mentioned: {health['sleep_mentioned']}")
    print(f"Sleep quality: {health.get('sleep_quality_mentioned', 'Not specified')}")
    print(f"Energy level: {health.get('energy_mentioned', 'Not specified')}")
    
    # Save analysis
    analyzer.save_journal_entry(analysis)
    
    # Generate daily report
    report = analyzer.create_daily_report()
    
    print(f"\n=== Daily Report Summary ===")
    print(report['insights_summary'])

if __name__ == "__main__":
    main()