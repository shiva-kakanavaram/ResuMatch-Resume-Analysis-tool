import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
from typing import Dict, List, Any, Tuple
import re

logger = logging.getLogger(__name__)

class EnhancedScoringSystem:
    def __init__(self):
        """Initialize the enhanced scoring system with TF-IDF capabilities"""
        self.nlp = spacy.load("en_core_web_sm")
        self.tfidf_vectorizer = TfidfVectorizer(
            min_df=1, 
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)  # Include both unigrams and bigrams
        )
        
        # Define section weights for scoring
        self.section_weights = {
            'skills': {
                'weight': 0.3,
                'subcategories': {
                    'programming_languages': 0.25,
                    'web_frameworks': 0.20,
                    'databases': 0.15,
                    'cloud_platforms': 0.15,
                    'tools': 0.15,
                    'soft_skills': 0.10
                }
            },
            'experience': {
                'weight': 0.35,
                'subcategories': {
                    'relevance': 0.4,
                    'duration': 0.3,
                    'responsibilities': 0.2,
                    'achievements': 0.1
                }
            },
            'education': {
                'weight': 0.15,
                'subcategories': {
                    'degree_level': 0.6,
                    'field_relevance': 0.3,
                    'institution_quality': 0.1
                }
            },
            'readability': {
                'weight': 0.1,
                'subcategories': {
                    'formatting': 0.4,
                    'clarity': 0.3,
                    'structure': 0.3
                }
            },
            'achievements': {
                'weight': 0.1,
                'subcategories': {
                    'quantified_results': 0.6,
                    'recognition': 0.4
                }
            }
        }
        
        # Industry-specific skill importance
        self.industry_skill_weights = {
            'software_engineering': {
                'programming_languages': 1.5,
                'web_frameworks': 1.3,
                'databases': 1.2,
                'cloud_platforms': 1.2,
                'tools': 1.0,
                'soft_skills': 0.8
            },
            'data_science': {
                'programming_languages': 1.3,
                'databases': 1.4,
                'cloud_platforms': 1.1,
                'tools': 1.2,
                'web_frameworks': 0.8,
                'soft_skills': 0.9
            },
            'product_management': {
                'soft_skills': 1.5,
                'tools': 1.2,
                'programming_languages': 0.7,
                'web_frameworks': 0.7,
                'databases': 0.6,
                'cloud_platforms': 0.6
            }
        }
        
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for TF-IDF analysis"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Process with spaCy for lemmatization
        doc = self.nlp(text)
        lemmatized = " ".join([token.lemma_ for token in doc 
                              if not token.is_stop and not token.is_punct])
        
        return lemmatized
        
    def calculate_section_tfidf_score(self, section_text: str, reference_corpus: List[str]) -> float:
        """Calculate TF-IDF based score for a section against a reference corpus"""
        if not section_text or not reference_corpus:
            return 0.0
            
        # Preprocess the section text
        processed_text = self.preprocess_text(section_text)
        if not processed_text:
            return 0.0
            
        # Add the processed text to the corpus for vectorization
        all_texts = [processed_text] + [self.preprocess_text(text) for text in reference_corpus]
        
        # Fit and transform the corpus
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Calculate average similarity with reference corpus
            section_vector = tfidf_matrix[0:1]  # The first document is our section
            corpus_vectors = tfidf_matrix[1:]  # The rest are reference corpus
            
            if corpus_vectors.shape[0] == 0:
                return 0.0
                
            # Calculate cosine similarities
            similarities = cosine_similarity(section_vector, corpus_vectors).flatten()
            
            # Return average similarity
            return float(np.mean(similarities))
            
        except Exception as e:
            logger.error(f"Error calculating TF-IDF score: {str(e)}")
            return 0.0
            
    def score_skills(self, skills: Dict, industry: str = None) -> Dict[str, float]:
        """Score skills using TF-IDF and industry-specific weights"""
        if not skills:
            return {'score': 0.0, 'subscores': {}}
            
        subscores = {}
        total_score = 0.0
        total_weight = 0.0
        
        # Get industry weights or use default
        industry_weights = self.industry_skill_weights.get(
            industry, 
            {cat: 1.0 for cat in self.section_weights['skills']['subcategories']}
        )
        
        for category, weight in self.section_weights['skills']['subcategories'].items():
            category_skills = skills.get(category, [])
            
            # Apply industry-specific weight
            adjusted_weight = weight * industry_weights.get(category, 1.0)
            total_weight += adjusted_weight
            
            # Calculate category score based on number and quality of skills
            if isinstance(category_skills, list):
                # More skills in important categories = higher score
                category_score = min(1.0, len(category_skills) / 5)  # Cap at 5 skills per category
            else:
                category_score = 0.0
                
            subscores[category] = category_score
            total_score += category_score * adjusted_weight
            
        # Normalize final score
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Bonus for having skills in multiple categories
        categories_with_skills = sum(1 for category in skills if skills.get(category))
        diversity_bonus = min(0.2, categories_with_skills * 0.05)  # Up to 0.2 bonus
        
        final_score = min(1.0, final_score + diversity_bonus)
        
        return {
            'score': final_score,
            'subscores': subscores
        }
        
    def score_experience(self, experience: List[Dict], job_description: str = None) -> Dict[str, float]:
        """Score experience using TF-IDF and weighted criteria"""
        if not experience:
            return {'score': 0.0, 'subscores': {}}
            
        subscores = {}
        
        # Calculate duration score
        total_months = 0
        for exp in experience:
            if not isinstance(exp, dict):
                continue
                
            duration = exp.get('duration', 0)
            # Handle string durations
            if isinstance(duration, str):
                try:
                    duration = int(duration.strip())
                except (ValueError, TypeError):
                    duration = 0
            total_months += duration
            
        if total_months >= 60:  # 5+ years
            duration_score = 1.0
        elif total_months >= 36:  # 3+ years
            duration_score = 0.8
        elif total_months >= 24:  # 2+ years
            duration_score = 0.6
        elif total_months >= 12:  # 1+ year
            duration_score = 0.4
        else:
            duration_score = 0.2
            
        subscores['duration'] = duration_score
        
        # Calculate relevance score using TF-IDF if job description is provided
        if job_description:
            # Combine all experience descriptions
            all_descriptions = []
            for exp in experience:
                if isinstance(exp, dict):
                    desc = exp.get('description', '')
                    if desc:
                        all_descriptions.append(desc)
                        
            combined_experience = " ".join(all_descriptions)
            relevance_score = self.calculate_section_tfidf_score(combined_experience, [job_description])
        else:
            # Without job description, base on position diversity
            companies = set()
            titles = set()
            for exp in experience:
                if isinstance(exp, dict):
                    if 'company' in exp:
                        companies.add(exp['company'])
                    if 'title' in exp:
                        titles.add(exp['title'])
                        
            diversity_score = min(1.0, (len(companies) * 0.15 + len(titles) * 0.15))
            relevance_score = 0.5 + (diversity_score * 0.5)  # Base 0.5 + up to 0.5 for diversity
            
        subscores['relevance'] = relevance_score
        
        # Calculate responsibilities score
        resp_score = 0.0
        for exp in experience:
            if not isinstance(exp, dict):
                continue
                
            responsibilities = exp.get('responsibilities', [])
            if responsibilities and isinstance(responsibilities, list):
                resp_score += min(0.25, len(responsibilities) * 0.05)  # Up to 0.25 per position
                
        subscores['responsibilities'] = min(1.0, resp_score)
        
        # Calculate achievements score
        achievement_keywords = ['achieved', 'increased', 'reduced', 'improved', 'led', 'developed']
        metric_patterns = [
            r'\d+(?:\.\d+)?%',  # Percentage
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Money
            r'\d+(?:\.\d+)?x',   # Multiplier
        ]
        
        achievement_score = 0.0
        for exp in experience:
            if not isinstance(exp, dict):
                continue
                
            # Check responsibilities for achievement keywords and metrics
            responsibilities = exp.get('responsibilities', [])
            if not responsibilities and 'description' in exp:
                # If no responsibilities list but has description, use that
                responsibilities = [exp['description']]
                
            for resp in responsibilities:
                if not isinstance(resp, str):
                    continue
                    
                resp_text = resp.lower()
                
                # Check for achievement keywords
                if any(keyword in resp_text for keyword in achievement_keywords):
                    achievement_score += 0.1
                    
                # Extra points for quantified achievements
                if any(re.search(pattern, resp_text) for pattern in metric_patterns):
                    achievement_score += 0.2
                    
        subscores['achievements'] = min(1.0, achievement_score)
        
        # Calculate final weighted score
        weights = self.section_weights['experience']['subcategories']
        final_score = (
            duration_score * weights['duration'] +
            relevance_score * weights['relevance'] +
            subscores['responsibilities'] * weights['responsibilities'] +
            subscores['achievements'] * weights['achievements']
        )
        
        return {
            'score': final_score,
            'subscores': subscores
        }
        
    def score_education(self, education: List[Dict]) -> Dict[str, float]:
        """Score education using weighted criteria"""
        if not education:
            return {'score': 0.0, 'subscores': {}}
            
        subscores = {}
        
        # Score for degree level
        degree_scores = {
            'phd': 1.0,
            'doctorate': 1.0,
            'master': 0.8,
            'bachelor': 0.6,
            'associate': 0.4,
            'diploma': 0.3,
            'certificate': 0.2
        }
        
        max_degree_score = 0.0
        for edu in education:
            if not isinstance(edu, dict):
                continue
                
            degree = str(edu.get('degree', '')).lower()
            
            # Find the highest degree level
            for level, score in degree_scores.items():
                if level in degree:
                    max_degree_score = max(max_degree_score, score)
                    
        subscores['degree_level'] = max_degree_score
        
        # Score for field relevance (simplified version)
        tech_fields = ['computer', 'software', 'information', 'data', 'engineering', 'science', 'mathematics']
        field_score = 0.0
        
        for edu in education:
            if not isinstance(edu, dict):
                continue
                
            field = str(edu.get('field', '')).lower()
            if any(tech_term in field for tech_term in tech_fields):
                field_score = 1.0
                break
                
        subscores['field_relevance'] = field_score
        
        # Institution quality is difficult to assess automatically
        # This would ideally use a database of institution rankings
        # For now, we'll use a placeholder score
        subscores['institution_quality'] = 0.5
        
        # Calculate final weighted score
        weights = self.section_weights['education']['subcategories']
        final_score = (
            subscores['degree_level'] * weights['degree_level'] +
            subscores['field_relevance'] * weights['field_relevance'] +
            subscores['institution_quality'] * weights['institution_quality']
        )
        
        return {
            'score': final_score,
            'subscores': subscores
        }
        
    def calculate_overall_score(self, section_scores: Dict[str, Dict[str, float]]) -> float:
        """Calculate overall resume score based on weighted section scores"""
        overall_score = 0.0
        
        for section, details in self.section_weights.items():
            if section in section_scores:
                section_score = section_scores[section]['score']
                section_weight = details['weight']
                overall_score += section_score * section_weight
                
        return overall_score
