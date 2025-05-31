import json
import re
from pathlib import Path
import logging
from typing import Dict, List, Set, Optional
import requests
from collections import defaultdict

logger = logging.getLogger(__name__)

class SkillTaxonomyManager:
    def __init__(self):
        """Initialize the skill taxonomy manager"""
        self.data_dir = Path(__file__).parent / "data"
        self.taxonomy_file = self.data_dir / "onet_taxonomy.json"
        self.synonyms_file = self.data_dir / "skill_synonyms.json"
        
        # Load or create taxonomy data
        self.taxonomy = self._load_taxonomy()
        self.skill_synonyms = self._load_synonyms()
        self.related_skills = self._build_related_skills()
        
        # Basic skill categories
        self.skill_categories = {
            'programming_languages': {
                'patterns': [
                    r'python|java|javascript|c\+\+|ruby|php|swift|kotlin|go|rust|typescript',
                    r'scala|perl|r|matlab|shell|bash|powershell|sql|html|css'
                ],
                'keywords': [
                    'programming', 'coding', 'development', 'software', 'engineering',
                    'backend', 'frontend', 'full-stack', 'developer'
                ]
            },
            'web_frameworks': {
                'patterns': [
                    r'django|flask|fastapi|spring|express|react|angular|vue|next\.?js',
                    r'node\.?js|rails|laravel|asp\.net|symfony|gatsby|nuxt\.?js'
                ],
                'keywords': [
                    'web', 'framework', 'frontend', 'backend', 'full-stack',
                    'development', 'application', 'api'
                ]
            },
            'databases': {
                'patterns': [
                    r'mysql|postgresql|mongodb|redis|elasticsearch|cassandra|dynamodb',
                    r'oracle|sql\s*server|sqlite|neo4j|couchdb|mariadb'
                ],
                'keywords': [
                    'database', 'db', 'data', 'storage', 'nosql', 'sql', 'rdbms',
                    'query', 'acid'
                ]
            },
            'cloud_platforms': {
                'patterns': [
                    r'aws|amazon|ec2|s3|lambda|azure|gcp|google\s*cloud',
                    r'heroku|digitalocean|kubernetes|docker|terraform|cloudformation'
                ],
                'keywords': [
                    'cloud', 'deployment', 'infrastructure', 'devops', 'ci/cd',
                    'container', 'orchestration'
                ]
            },
            'soft_skills': {
                'patterns': [
                    r'leadership|communication|teamwork|problem[\s-]solving|analytical',
                    r'project\s*management|agile|scrum|collaboration|time\s*management'
                ],
                'keywords': [
                    'skill', 'ability', 'competency', 'proficiency', 'expertise',
                    'experience', 'knowledge'
                ]
            }
        }
        
        # Load proficiency indicators
        self.proficiency_indicators = {
            'expert': ['expert', 'advanced', 'senior', 'lead', 'architect'],
            'intermediate': ['intermediate', 'proficient', 'experienced'],
            'beginner': ['beginner', 'basic', 'familiar', 'learning']
        }
    
    def extract_skills(self, text: str) -> Dict[str, List[Dict]]:
        """Extract and categorize skills from text"""
        skills = {category: [] for category in self.skill_categories}
        
        # Convert text to lowercase for better matching
        text_lower = text.lower()
        
        for category, patterns in self.skill_categories.items():
            # Find all skills in the category
            for pattern in patterns['patterns']:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    skill = match.group()
                    
                    # Check if skill is in a relevant context
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(text_lower), match.end() + 50)
                    context = text_lower[context_start:context_end]
                    
                    if self._is_valid_skill_context(context, patterns['keywords']):
                        # Get proficiency level
                        proficiency = self._detect_proficiency(context)
                        
                        # Add skill if not already present
                        skill_entry = {
                            'name': skill,
                            'proficiency': proficiency,
                            'context': context.strip()
                        }
                        
                        if not any(s['name'] == skill for s in skills[category]):
                            skills[category].append(skill_entry)
        
        return skills
    
    def _is_valid_skill_context(self, context: str, keywords: List[str]) -> bool:
        """Check if the skill appears in a valid context"""
        # Count keyword matches in context
        keyword_matches = sum(1 for keyword in keywords if keyword in context)
        
        # Check if context contains common false positive indicators
        false_positives = [
            'not familiar',
            'no experience',
            'want to learn',
            'planning to learn'
        ]
        
        if any(fp in context for fp in false_positives):
            return False
        
        return keyword_matches >= 1
    
    def _detect_proficiency(self, context: str) -> Dict:
        """Detect proficiency level from context"""
        for level, indicators in self.proficiency_indicators.items():
            if any(indicator in context for indicator in indicators):
                return {
                    'level': level,
                    'score': {
                        'expert': 1.0,
                        'intermediate': 0.7,
                        'beginner': 0.4
                    }.get(level, 0.0)
                }
        
        # Default to intermediate if no clear indicators
        return {
            'level': 'intermediate',
            'score': 0.7
        }
    
    def get_related_skills(self, skill: str) -> List[str]:
        """Get related skills for a given skill"""
        related = []
        skill_lower = skill.lower()
        
        # Find which category the skill belongs to
        for category, patterns in self.skill_categories.items():
            for pattern in patterns['patterns']:
                if re.search(pattern, skill_lower):
                    # Add other skills from the same category
                    for other_pattern in patterns['patterns']:
                        matches = re.finditer(other_pattern, pattern)
                        for match in matches:
                            related_skill = match.group()
                            if related_skill != skill_lower:
                                related.append(related_skill)
        
        return list(set(related))
    
    def _load_taxonomy(self) -> Dict:
        """Load or create O*NET skill taxonomy"""
        if self.taxonomy_file.exists():
            with open(self.taxonomy_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_synonyms(self) -> Dict:
        """Load or create skill synonyms"""
        if self.synonyms_file.exists():
            with open(self.synonyms_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _build_related_skills(self) -> Dict:
        """Build related skills mapping"""
        related = defaultdict(set)
        for skill, info in self.taxonomy.items():
            if 'related' in info:
                for rel in info['related']:
                    related[skill].add(rel)
                    related[rel].add(skill)
        return dict(related)
    
    def normalize_skill(self, skill: str) -> str:
        """Normalize skill name using synonyms"""
        skill_lower = skill.lower()
        for canonical, synonyms in self.skill_synonyms.items():
            if skill_lower in synonyms or skill_lower == canonical.lower():
                return canonical
        return skill
    
    def get_skill_info(self, skill: str) -> Optional[Dict]:
        """Get detailed information about a skill"""
        skill_lower = skill.lower()
        
        # Check direct match
        for category, skills in self.skill_categories.items():
            if skill_lower in skills:
                return {
                    'name': skill_lower,
                    'category': category,
                    'description': f'{category.replace("_", " ").title()} skills',
                    'synonyms': self.skill_synonyms.get(skill_lower, []),
                    'related_skills': list(self.related_skills.get(skill_lower, set()))
                }
        
        # Check synonyms
        for main_skill, synonyms in self.skill_synonyms.items():
            if skill_lower in synonyms:
                return self.get_skill_info(main_skill)
        
        return None
    
    def detect_skill_proficiency(self, skill: str, context: str) -> Dict:
        """Detect skill proficiency from context"""
        skill_lower = skill.lower()
        normalized_skill = self.normalize_skill(skill_lower)
        
        # Default proficiency
        proficiency = {
            'level': 'not_specified',
            'confidence': 0.5,
            'context': context
        }
        
        # Proficiency levels
        proficiency_levels = {
            'expert': {
                'keywords': ['expert', 'advanced', 'senior', 'lead'],
                'years': 5,
                'score': 0.9
            },
            'intermediate': {
                'keywords': ['intermediate', 'experienced', 'proficient'],
                'years': 2,
                'score': 0.7
            },
            'beginner': {
                'keywords': ['beginner', 'basic', 'familiar', 'learning'],
                'years': 0,
                'score': 0.4
            }
        }
        
        # Check for proficiency indicators in context
        for level, info in proficiency_levels.items():
            if any(kw in context.lower() for kw in info['keywords']):
                proficiency['level'] = level
                proficiency['confidence'] = info['score']
                break
        
        return proficiency
    
    def categorize_skill(self, skill: str) -> Optional[Dict[str, str]]:
        """Categorize a skill into its type and category"""
        skill_info = self.get_skill_info(skill)
        if skill_info:
            return {
                'type': 'technical',
                'category': skill_info['category']
            }
        return None
