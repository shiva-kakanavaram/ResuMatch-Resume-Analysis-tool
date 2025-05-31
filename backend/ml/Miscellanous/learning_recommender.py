import json
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LearningRecommender:
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.resources_file = self.data_dir / "learning_resources.json"
        
        # Load learning resources
        self.resources = self._load_resources()
        
        # Define learning paths by career level
        self.learning_paths = {
            'entry': {
                'focus_areas': ['fundamentals', 'tools', 'best_practices'],
                'certification_priority': 'low',
                'project_complexity': 'basic'
            },
            'mid': {
                'focus_areas': ['advanced_concepts', 'specialization', 'team_skills'],
                'certification_priority': 'medium',
                'project_complexity': 'intermediate'
            },
            'senior': {
                'focus_areas': ['architecture', 'leadership', 'mentoring'],
                'certification_priority': 'high',
                'project_complexity': 'advanced'
            },
            'lead': {
                'focus_areas': ['system_design', 'team_management', 'strategy'],
                'certification_priority': 'high',
                'project_complexity': 'complex'
            }
        }
    
    def _load_resources(self) -> Dict:
        """Load or create learning resources data"""
        try:
            if self.resources_file.exists():
                with open(self.resources_file, 'r') as f:
                    return json.load(f)
            else:
                # Create basic learning resources
                resources = {
                    'platforms': {
                        'coursera': {
                            'url': 'https://www.coursera.org',
                            'specialties': ['academic', 'professional', 'certificates'],
                            'cost': 'paid',
                            'features': ['certificates', 'university_partnerships']
                        },
                        'udemy': {
                            'url': 'https://www.udemy.com',
                            'specialties': ['professional', 'practical'],
                            'cost': 'paid',
                            'features': ['lifetime_access', 'practice_exercises']
                        },
                        'pluralsight': {
                            'url': 'https://www.pluralsight.com',
                            'specialties': ['technology', 'software_development'],
                            'cost': 'subscription',
                            'features': ['skill_assessments', 'learning_paths']
                        }
                    },
                    'certifications': {
                        'aws': {
                            'tracks': ['cloud', 'devops', 'security'],
                            'levels': ['foundational', 'associate', 'professional'],
                            'provider': 'Amazon'
                        },
                        'microsoft': {
                            'tracks': ['azure', 'development', 'data'],
                            'levels': ['fundamentals', 'associate', 'expert'],
                            'provider': 'Microsoft'
                        },
                        'google': {
                            'tracks': ['cloud', 'data', 'machine_learning'],
                            'levels': ['associate', 'professional'],
                            'provider': 'Google'
                        }
                    },
                    'communities': {
                        'stack_overflow': {
                            'url': 'https://stackoverflow.com',
                            'type': 'q&a',
                            'focus': 'programming'
                        },
                        'github': {
                            'url': 'https://github.com',
                            'type': 'project_collaboration',
                            'focus': 'open_source'
                        },
                        'dev.to': {
                            'url': 'https://dev.to',
                            'type': 'blog_community',
                            'focus': 'technology'
                        }
                    },
                    'books': {
                        'software_engineering': [
                            'Clean Code by Robert Martin',
                            'Design Patterns by Gang of Four',
                            'The Pragmatic Programmer'
                        ],
                        'career_development': [
                            'The Manager\'s Path',
                            'Soft Skills: The Software Developer\'s Life Manual',
                            'The Clean Coder'
                        ]
                    }
                }
                
                # Save for future use
                self.data_dir.mkdir(exist_ok=True)
                with open(self.resources_file, 'w') as f:
                    json.dump(resources, f, indent=2)
                
                return resources
        except Exception as e:
            logger.error(f"Error loading learning resources: {str(e)}")
            return {}
    
    def _recommend_courses(self, skill_gaps: List[str], career_level: str) -> List[Dict]:
        """Recommend courses based on skill gaps and career level"""
        recommendations = []
        learning_path = self.learning_paths.get(career_level, self.learning_paths['entry'])
        
        for skill in skill_gaps:
            # Find relevant platforms
            relevant_platforms = []
            for platform, details in self.resources['platforms'].items():
                if any(specialty in skill.lower() for specialty in details['specialties']):
                    relevant_platforms.append({
                        'platform': platform,
                        'url': details['url'],
                        'features': details['features']
                    })
            
            if relevant_platforms:
                recommendations.append({
                    'skill': skill,
                    'platforms': relevant_platforms,
                    'focus_areas': learning_path['focus_areas'],
                    'complexity': learning_path['project_complexity']
                })
        
        return recommendations
    
    def _recommend_certifications(self, career_path: Dict, career_level: str) -> List[Dict]:
        """Recommend certifications based on career path and level"""
        recommendations = []
        learning_path = self.learning_paths.get(career_level, self.learning_paths['entry'])
        
        if learning_path['certification_priority'] == 'low':
            return []  # Skip certifications for entry level
        
        for cert_provider, cert_details in self.resources['certifications'].items():
            # Check if certification tracks align with career path
            matching_tracks = []
            for track in cert_details['tracks']:
                if track in career_path['track'].lower() or track in career_path.get('domain', '').lower():
                    matching_tracks.append(track)
            
            if matching_tracks:
                # Determine appropriate certification level
                if career_level in ['entry', 'mid']:
                    cert_level = cert_details['levels'][0]  # Foundational/Associate
                else:
                    cert_level = cert_details['levels'][-1]  # Professional/Expert
                
                recommendations.append({
                    'provider': cert_provider,
                    'tracks': matching_tracks,
                    'level': cert_level,
                    'priority': learning_path['certification_priority']
                })
        
        return recommendations
    
    def _recommend_projects(self, skill_gaps: List[str], career_level: str) -> List[Dict]:
        """Recommend practice projects based on skill gaps and level"""
        complexity = self.learning_paths[career_level]['project_complexity']
        
        project_templates = {
            'basic': [
                {
                    'type': 'personal_portfolio',
                    'description': 'Build a personal portfolio website',
                    'skills': ['html', 'css', 'javascript']
                },
                {
                    'type': 'crud_app',
                    'description': 'Create a simple CRUD application',
                    'skills': ['backend', 'database', 'api']
                }
            ],
            'intermediate': [
                {
                    'type': 'full_stack_app',
                    'description': 'Build a full-stack web application with authentication',
                    'skills': ['frontend', 'backend', 'database', 'auth']
                },
                {
                    'type': 'api_service',
                    'description': 'Create a RESTful API service with documentation',
                    'skills': ['api', 'backend', 'documentation']
                }
            ],
            'advanced': [
                {
                    'type': 'distributed_system',
                    'description': 'Build a distributed system with microservices',
                    'skills': ['microservices', 'cloud', 'system_design']
                },
                {
                    'type': 'data_pipeline',
                    'description': 'Create a data processing pipeline',
                    'skills': ['data_engineering', 'etl', 'big_data']
                }
            ]
        }
        
        # Filter projects that help with skill gaps
        recommended_projects = []
        for project in project_templates.get(complexity, []):
            if any(skill.lower() in skill_gaps for skill in project['skills']):
                recommended_projects.append(project)
        
        return recommended_projects
    
    def _recommend_resources(self, career_path: Dict) -> Dict:
        """Recommend learning resources based on career path"""
        domain = career_path['domain']
        track = career_path['track']
        
        recommended_resources = {
            'communities': [],
            'books': []
        }
        
        # Recommend communities
        for community, details in self.resources['communities'].items():
            if details['focus'] in [domain, track, 'technology']:
                recommended_resources['communities'].append({
                    'name': community,
                    'url': details['url'],
                    'type': details['type']
                })
        
        # Recommend books
        for category, books in self.resources['books'].items():
            if category in [domain, 'career_development']:
                recommended_resources['books'].extend(books)
        
        return recommended_resources
    
    def get_recommendations(self, skill_gaps: List[str], career_path: Dict = None, career_level: str = None) -> Dict:
        """Get personalized learning recommendations"""
        try:
            recommendations = {
                'courses': [],
                'certifications': [],
                'resources': [],
                'practice_projects': []
            }
            
            # Get skill-based recommendations
            for skill in skill_gaps:
                recommendations['courses'].extend(self._recommend_courses([skill], career_level))
                recommendations['certifications'].extend(self._recommend_certifications({'track': skill}, career_level))
                recommendations['resources'].extend(self._recommend_resources({'domain': skill})['communities'])
                recommendations['practice_projects'].extend(self._recommend_projects([skill], career_level))
            
            # Get career path specific recommendations if available
            if career_path and isinstance(career_path, dict):
                domain = career_path.get('domain', 'software_development')
                level = career_path.get('level', 'entry')
                
                # Add domain-specific recommendations
                domain_recs = self._recommend_resources(career_path)
                for key in recommendations:
                    if key in domain_recs:
                        recommendations[key].extend(domain_recs[key])
            
            # Deduplicate recommendations
            for key in recommendations:
                recommendations[key] = list(set(recommendations[key]))
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating learning recommendations: {str(e)}")
            return {
                'courses': [],
                'certifications': [],
                'resources': [],
                'practice_projects': []
            }
