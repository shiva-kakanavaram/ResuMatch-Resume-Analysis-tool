import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

class CareerPathAnalyzer:
    def __init__(self):
        """Initialize the career path analyzer"""
        self.job_categories = {
            'software_engineering': ['software', 'developer', 'engineer', 'programming'],
            'data_science': ['data scientist', 'machine learning', 'ai', 'analytics'],
            'product_management': ['product manager', 'product owner', 'program manager'],
            'design': ['designer', 'ux', 'ui', 'user experience'],
            'marketing': ['marketing', 'seo', 'content', 'social media'],
            'sales': ['sales', 'business development', 'account manager'],
            'operations': ['operations', 'project manager', 'scrum master']
        }
        
    def analyze_career_path(self, resume_sections: dict) -> dict:
        """Analyze career path based on resume sections"""
        if not resume_sections:
            return {
                'predicted_category': 'Not Available',
                'experience_level': 'Entry Level',
                'career_trajectory': 'Not enough data',
                'growth_potential': 0.5
            }
            
        # Extract experience details
        experience = resume_sections.get('experience', [])
        skills = resume_sections.get('skills', {})
        
        # Predict job category
        predicted_category = self._predict_job_category(experience, skills)
        
        # Determine experience level
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
            
        if total_months < 24:
            level = 'Entry Level'
        elif total_months < 60:
            level = 'Mid Level'
        else:
            level = 'Senior Level'
            
        # Analyze career trajectory
        trajectory = self._analyze_trajectory(experience)
        
        # Calculate growth potential
        growth_potential = self._calculate_growth_potential(experience, skills, predicted_category)
        
        return {
            'predicted_category': predicted_category,
            'experience_level': level,
            'career_trajectory': trajectory,
            'growth_potential': growth_potential
        }
        
    def _predict_job_category(self, experience: list, skills: dict) -> str:
        """Predict job category based on experience and skills"""
        # Count keyword occurrences for each category
        category_scores = {category: 0 for category in self.job_categories}
        
        # Check experience titles and descriptions
        for exp in experience:
            title = str(exp.get('title', '')).lower()
            description = ' '.join(str(r) for r in exp.get('responsibilities', [])).lower()
            
            for category, keywords in self.job_categories.items():
                for keyword in keywords:
                    if keyword in title or keyword in description:
                        category_scores[category] += 2 if keyword in title else 1
                        
        # Check skills
        all_skills = []
        for skill_list in skills.values():
            if isinstance(skill_list, list):
                all_skills.extend(str(skill) for skill in skill_list)
            
        skill_text = ' '.join(all_skills).lower()
        for category, keywords in self.job_categories.items():
            for keyword in keywords:
                if keyword in skill_text:
                    category_scores[category] += 1
                    
        # Get category with highest score
        if not category_scores:
            return 'Not Available'
            
        predicted_category = max(category_scores.items(), key=lambda x: x[1])[0]
        return predicted_category if category_scores[predicted_category] > 0 else 'Not Available'
        
    def _analyze_trajectory(self, experience: list) -> str:
        """Analyze career trajectory based on experience"""
        if not experience:
            return 'Not enough data'
            
        # Sort experience by start date
        sorted_exp = sorted(experience, key=lambda x: x.get('start_date', ''), reverse=True)
        
        # Look for promotions and role changes
        role_changes = []
        for i in range(len(sorted_exp) - 1):
            curr_title = sorted_exp[i].get('title', '').lower()
            prev_title = sorted_exp[i + 1].get('title', '').lower()
            
            if 'senior' in curr_title and 'senior' not in prev_title:
                role_changes.append('promotion')
            elif curr_title != prev_title:
                role_changes.append('role_change')
                
        if len(role_changes) >= 2:
            return 'Rapid Growth'
        elif len(role_changes) == 1:
            return 'Steady Progress'
        else:
            return 'Stable'
            
    def _calculate_growth_potential(self, experience: list, skills: dict, predicted_category: str) -> float:
        """Calculate growth potential based on experience and skills"""
        score = 0.5  # Base score
        
        # Points for diverse experience
        companies = set()
        for exp in experience:
            if isinstance(exp, dict) and 'company' in exp:
                companies.add(exp.get('company', ''))
                
        score += min(0.2, len(companies) * 0.05)  # Up to 0.2 for diverse companies
        
        # Points for relevant skills
        if isinstance(skills, dict):
            relevant_skills = []
            
            # Check if we have the new structure with categorized skills
            if any(key in skills for key in ['programming_languages', 'web_frameworks', 'databases']):
                # Flatten all skills
                for category in skills.values():
                    if isinstance(category, list):
                        relevant_skills.extend(category)
            else:
                # Old structure
                for category in skills.values():
                    if isinstance(category, list):
                        relevant_skills.extend(category)
                        
            score += min(0.3, len(relevant_skills) * 0.02)  # Up to 0.3 for skills
        
        # Adjust based on predicted category
        if predicted_category in ['software_engineering', 'data_science']:
            score += 0.1  # Bonus for high-growth fields
            
        return min(1.0, score)
