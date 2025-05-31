import spacy
import re
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class JobAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize skill categories and common requirements
        self.skill_categories = {
            'technical': {
                'patterns': [
                    r'(?i)(python|java|javascript|typescript|c\+\+|ruby|php|scala|go|rust)',
                    r'(?i)(sql|mysql|postgresql|mongodb|oracle|redis)',
                    r'(?i)(aws|azure|gcp|cloud)',
                    r'(?i)(docker|kubernetes|jenkins|ci/cd)',
                    r'(?i)(react|angular|vue|node\.js|django|flask|spring)'
                ],
                'weight': 0.4
            },
            'soft': {
                'patterns': [
                    r'(?i)(communication|teamwork|leadership|problem.solving)',
                    r'(?i)(analytical|critical.thinking|decision.making)',
                    r'(?i)(time.management|organization|planning)',
                    r'(?i)(creativity|innovation|adaptability)'
                ],
                'weight': 0.3
            },
            'education': {
                'patterns': [
                    r'(?i)(bachelor|master|phd|degree)',
                    r'(?i)(computer science|engineering|information technology)',
                    r'(?i)(mathematics|physics|statistics)'
                ],
                'weight': 0.2
            },
            'experience': {
                'patterns': [
                    r'(?i)(\d+[\+]?\s*(?:years?|yrs?))',
                    r'(?i)(senior|junior|lead|principal)',
                    r'(?i)(experience in|background in|knowledge of)'
                ],
                'weight': 0.1
            }
        }
        
        # Experience level mapping
        self.experience_levels = {
            'entry': r'(?i)(entry|junior|0-2|1-2|graduate)',
            'mid': r'(?i)(mid|intermediate|2-5|3-5)',
            'senior': r'(?i)(senior|lead|principal|5\+|7\+|10\+)',
            'manager': r'(?i)(manager|director|head)'
        }
        
        # Required vs preferred pattern
        self.requirement_patterns = {
            'required': r'(?i)(required|must have|essential|necessary)',
            'preferred': r'(?i)(preferred|nice to have|desirable|plus)'
        }
    
    def _extract_requirements(self, text: str) -> Dict[str, Dict[str, List[str]]]:
        """Extract and categorize job requirements"""
        requirements = {
            'required': defaultdict(list),
            'preferred': defaultdict(list)
        }
        
        # Process text by sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        for sentence in sentences:
            # Determine if requirement is required or preferred
            requirement_type = 'preferred'
            for req_type, pattern in self.requirement_patterns.items():
                if re.search(pattern, sentence):
                    requirement_type = req_type
                    break
            
            # Extract skills from each category
            for category, config in self.skill_categories.items():
                for pattern in config['patterns']:
                    matches = re.finditer(pattern, sentence)
                    for match in matches:
                        skill = match.group(0).lower()
                        if skill not in requirements[requirement_type][category]:
                            requirements[requirement_type][category].append(skill)
        
        return requirements
    
    def _extract_experience_level(self, text: str) -> str:
        """Determine the experience level required"""
        for level, pattern in self.experience_levels.items():
            if re.search(pattern, text):
                return level
        return 'not_specified'
    
    def _calculate_match_score(self, resume_skills: Dict[str, List[Dict]], 
                             job_requirements: Dict[str, Dict[str, List[str]]]) -> Tuple[float, Dict]:
        """Calculate match score between resume skills and job requirements"""
        scores = {}
        matches = defaultdict(list)
        missing = defaultdict(list)
        
        # Calculate score for each category
        total_score = 0
        total_weight = 0
        
        for category, config in self.skill_categories.items():
            weight = config['weight']
            total_weight += weight
            
            # Get resume skills for this category
            resume_category_skills = {
                skill['name'].lower() 
                for skills in resume_skills.values() 
                for skill in skills
            }
            
            # Get required and preferred skills for this category
            required_skills = set(job_requirements['required'].get(category, []))
            preferred_skills = set(job_requirements['preferred'].get(category, []))
            
            # Calculate matches
            required_matches = resume_category_skills & required_skills
            preferred_matches = resume_category_skills & preferred_skills
            
            # Calculate category score
            if required_skills:
                required_score = len(required_matches) / len(required_skills) * 0.7
            else:
                required_score = 0.7  # Full score if no required skills
                
            if preferred_skills:
                preferred_score = len(preferred_matches) / len(preferred_skills) * 0.3
            else:
                preferred_score = 0.3  # Full score if no preferred skills
            
            category_score = (required_score + preferred_score) * 100
            
            # Store matches and missing skills
            matches[category].extend(list(required_matches | preferred_matches))
            missing[category].extend(list(
                (required_skills | preferred_skills) - resume_category_skills
            ))
            
            # Add weighted score to total
            total_score += category_score * weight
            scores[category] = category_score
        
        # Normalize total score
        final_score = total_score / total_weight if total_weight > 0 else 0
        
        return final_score, {
            'category_scores': scores,
            'matched_skills': dict(matches),
            'missing_skills': dict(missing)
        }
    
    def analyze_job_match(self, resume_analysis: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """Analyze how well a resume matches a job description"""
        try:
            # Extract job requirements
            requirements = self._extract_requirements(job_description)
            
            # Determine experience level
            experience_level = self._extract_experience_level(job_description)
            
            # Calculate match score
            match_score, match_details = self._calculate_match_score(
                resume_analysis['sections']['skills'],
                requirements
            )
            
            # Generate recommendations
            recommendations = []
            
            # Add skill-based recommendations
            for category, missing_skills in match_details['missing_skills'].items():
                if missing_skills:
                    recommendations.append({
                        'type': 'skill_gap',
                        'category': category,
                        'skills': missing_skills[:3],  # Top 3 missing skills
                        'suggestion': f"Consider adding experience with: {', '.join(missing_skills[:3])}"
                    })
            
            # Add experience-based recommendations
            if experience_level != 'not_specified':
                resume_experience = sum(
                    exp.get('duration_months', 0) or 0 
                    for exp in resume_analysis['sections']['experience']
                )
                if experience_level == 'senior' and resume_experience < 60:  # 5 years
                    recommendations.append({
                        'type': 'experience_gap',
                        'current_months': resume_experience,
                        'suggestion': "This position typically requires 5+ years of experience. Highlight relevant projects and achievements."
                    })
            
            return {
                'match_score': match_score,
                'experience_level': experience_level,
                'requirements': requirements,
                'match_details': match_details,
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing job match: {str(e)}")
            raise
