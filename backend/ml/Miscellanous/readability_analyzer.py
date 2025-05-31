import re
import logging
from typing import Dict, List, Any
import string

logger = logging.getLogger(__name__)

class ReadabilityAnalyzer:
    """Analyzer for text readability metrics."""
    
    def __init__(self):
        """Initialize the readability analyzer."""
        self.metrics = {
            'sentence_length': {'weight': 0.25, 'optimal': 15, 'tolerance': 5},
            'word_length': {'weight': 0.15, 'optimal': 5, 'tolerance': 2},
            'paragraph_length': {'weight': 0.20, 'optimal': 3, 'tolerance': 2},
            'bullet_point_ratio': {'weight': 0.20, 'optimal': 0.3, 'tolerance': 0.1},
            'buzzword_density': {'weight': 0.20, 'optimal': 0.05, 'tolerance': 0.05}
        }
        
        # Common business/resume buzzwords to avoid overusing
        self.buzzwords = [
            'synergy', 'leverage', 'strategic', 'streamline', 'optimize', 'innovate',
            'paradigm', 'proactive', 'dynamic', 'empower', 'disruptive', 'cutting-edge',
            'best-of-breed', 'mission-critical', 'robust', 'scalable', 'enterprise',
            'solution', 'ecosystem', 'innovative', 'best-practice', 'next-generation',
            'world-class', 'results-driven', 'customer-centric', 'value-added',
            'thought-leadership', 'game-changer'
        ]
    
    def analyze_text(self, text: str) -> float:
        """Analyze the readability of the text and return a score between 0 and 1."""
        if not text:
            return 0.5  # Default middle score
            
        try:
            scores = {}
            
            # Split text into sentences, words, and paragraphs
            sentences = self._split_into_sentences(text)
            words = self._split_into_words(text)
            paragraphs = self._split_into_paragraphs(text)
            
            # Calculate metrics
            if sentences:
                avg_sentence_length = len(words) / len(sentences)
                scores['sentence_length'] = self._score_metric(
                    avg_sentence_length, 
                    self.metrics['sentence_length']['optimal'],
                    self.metrics['sentence_length']['tolerance']
                )
            else:
                scores['sentence_length'] = 0.5
            
            if words:
                avg_word_length = sum(len(w) for w in words) / len(words)
                scores['word_length'] = self._score_metric(
                    avg_word_length,
                    self.metrics['word_length']['optimal'],
                    self.metrics['word_length']['tolerance']
                )
            else:
                scores['word_length'] = 0.5
            
            if paragraphs:
                # Average sentences per paragraph
                sentences_per_paragraph = [len(self._split_into_sentences(p)) for p in paragraphs]
                if sentences_per_paragraph:
                    avg_paragraph_length = sum(sentences_per_paragraph) / len(sentences_per_paragraph)
                    scores['paragraph_length'] = self._score_metric(
                        avg_paragraph_length,
                        self.metrics['paragraph_length']['optimal'],
                        self.metrics['paragraph_length']['tolerance']
                    )
                else:
                    scores['paragraph_length'] = 0.5
            else:
                scores['paragraph_length'] = 0.5
            
            # Calculate bullet point ratio
            bullet_point_lines = sum(1 for line in text.split('\n') if line.strip().startswith(('•', '-', '*')))
            total_lines = len(text.split('\n'))
            if total_lines > 0:
                bullet_ratio = bullet_point_lines / total_lines
                scores['bullet_point_ratio'] = self._score_metric(
                    bullet_ratio,
                    self.metrics['bullet_point_ratio']['optimal'],
                    self.metrics['bullet_point_ratio']['tolerance']
                )
            else:
                scores['bullet_point_ratio'] = 0.5
            
            # Calculate buzzword density
            if words:
                buzzword_count = sum(1 for word in words if word.lower() in self.buzzwords)
                buzzword_density = buzzword_count / len(words)
                scores['buzzword_density'] = 1.0 - buzzword_density * 10  # Lower density is better
                scores['buzzword_density'] = max(0.0, min(1.0, scores['buzzword_density']))
            else:
                scores['buzzword_density'] = 0.5
            
            # Calculate weighted average
            weighted_score = 0.0
            total_weight = 0.0
            
            for metric, score in scores.items():
                weighted_score += score * self.metrics[metric]['weight']
                total_weight += self.metrics[metric]['weight']
            
            if total_weight > 0:
                final_score = weighted_score / total_weight
            else:
                final_score = 0.5
                
            # Ensure score is between 0 and 1
            final_score = max(0.0, min(1.0, final_score))
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error analyzing readability: {str(e)}")
            return 0.5
    
    def _score_metric(self, value: float, optimal: float, tolerance: float) -> float:
        """Score a metric based on how close it is to the optimal value."""
        # Calculate the distance from the optimal value
        distance = abs(value - optimal)
        
        # If within tolerance, give high score
        if distance <= tolerance:
            return 1.0 - (distance / tolerance) * 0.5  # Scale from 1.0 down to 0.5
        else:
            # Calculate how far beyond tolerance
            beyond_tolerance = distance - tolerance
            # Scale down from 0.5 based on how far beyond
            return max(0.0, 0.5 - (beyond_tolerance / optimal) * 0.5)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple regex for splitting on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_into_words(self, text: str) -> List[str]:
        """Split text into words."""
        # Remove punctuation and split on whitespace
        translator = str.maketrans('', '', string.punctuation)
        text_no_punct = text.translate(translator)
        return [w for w in text_no_punct.split() if w.strip()]
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
