import logging
from typing import Dict, List, Tuple, Optional, Union
import spacy
from pathlib import Path
import json
import re
from dateutil import parser
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import traceback
import os
import time

from .skill_taxonomy import SkillTaxonomyManager
from .readability_analyzer import ReadabilityAnalyzer
from .career_path_analyzer import CareerPathAnalyzer
from .learning_recommender import LearningRecommender
from .resume_parser import ResumeParser  # Import ResumeParser

logger = logging.getLogger(__name__)

class JobMatcher:
    """Class to match resumes with job descriptions."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        # Initialize any required models or dictionaries
        self._init_models()
    
    def _init_models(self):
        """Initialize any required models for job matching."""
        # This would load ML models, but for now we'll use keyword matching
        pass
    
    def extract_keywords(self, text):
        """Extract keywords from text using basic NLP techniques."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        # Remove stopwords (a more comprehensive list would be used in production)
        stopwords = {'and', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'is', 'are'}
        keywords = [word for word in words if word not in stopwords and len(word) > 2]
        # Return unique keywords
        return list(set(keywords))
    
    def extract_requirements(self, job_description):
        """
        Extract requirements from a job description.
        
        Args:
            job_description: Job description text
            
        Returns:
            Dictionary of requirements by category
        """
        requirements = {
            'required_skills': [],
            'preferred_skills': [],
            'experience': [],
            'education': []
        }
        
        # Convert to lowercase for easier matching
        text = job_description.lower()
        
        # Extract skills
        skill_sections = re.findall(r'required skills[^\n]*:(.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if skill_sections:
            skills_text = skill_sections[0]
            # Split by bullets or commas
            skills = re.findall(r'[-•*]\s*([^,\n-•*]+)|([^,\n-•*]+)', skills_text)
            requirements['required_skills'] = [s[0].strip() or s[1].strip() for s in skills if any(s)]
        
        # Look for preferred/nice to have skills
        preferred_sections = re.findall(r'(?:preferred|nice to have)[^\n]*:(.*?)(?:\n\n|\Z)', text, re.DOTALL)
        if preferred_sections:
            preferred_text = preferred_sections[0]
            # Split by bullets or commas
            skills = re.findall(r'[-•*]\s*([^,\n-•*]+)|([^,\n-•*]+)', preferred_text)
            requirements['preferred_skills'] = [s[0].strip() or s[1].strip() for s in skills if any(s)]
        
        # Extract experience
        experience_match = re.search(r'(\d+)(?:\+)?\s*(?:years?|yrs?)(?:\s*of)?\s*experience', text)
        if experience_match:
            requirements['experience'] = [f"{experience_match.group(1)}+ years"]
        
        # Extract education
        education_terms = ['bachelor', 'master', 'phd', 'degree', 'bs', 'ms', 'ba', 'ma']
        for term in education_terms:
            if term in text:
                requirements['education'].append(f"Requires {term}")
                break
        
        self.logger.info(f"Extracted {len(requirements['required_skills'])} required skills and {len(requirements['preferred_skills'])} preferred skills")
        return requirements
    
    def calculate_match(self, resume_data, job_description):
        """
        Calculate how well a resume matches a job description.
        
        Args:
            resume_data: Dictionary containing parsed resume data
            job_description: Job description text
            
        Returns:
            Float between 0 and 1 representing match percentage
        """
        try:
            # Extract requirements from job description
            requirements = self.extract_requirements(job_description)
            
            # Extract keywords from job description
            job_keywords = self.extract_keywords(job_description)
            
            # Get skills from resume
            resume_skills = []
            skills_dict = resume_data.get('skills', {})
            for skill_type, skills in skills_dict.items():
                if isinstance(skills, list):
                    resume_skills.extend([s.lower() for s in skills])
            
            # Get keywords from experience and summary
            experience_text = " ".join([
                f"{exp.get('title', '')} {exp.get('company', '')} "
                f"{' '.join(exp.get('responsibilities', []))}"
                for exp in resume_data.get('experience', [])
            ])
            
            summary_text = resume_data.get('summary', '')
            
            # Extract keywords from resume text
            resume_experience_keywords = self.extract_keywords(experience_text)
            resume_summary_keywords = self.extract_keywords(summary_text)
            
            # Combine all resume keywords
            all_resume_keywords = set(resume_skills + resume_experience_keywords + resume_summary_keywords)
            
            # Calculate match based on keyword overlap
            matching_keywords = [kw for kw in job_keywords if kw in all_resume_keywords]
            
            # Calculate match percentage
            if not job_keywords:
                return 0.5  # Default if no keywords found
            
            match_percentage = len(matching_keywords) / len(job_keywords)
            
            # Apply weights to balance the score
            # Typically skill matches are more important than generic keyword matches
            weighted_match = min(1.0, match_percentage * 1.2)  # Slight boost but cap at 1.0
            
            self.logger.info(f"Job match calculation: {len(matching_keywords)} matching keywords out of {len(job_keywords)}")
            return weighted_match
            
        except Exception as e:
            self.logger.error(f"Error in job matching: {str(e)}")
            traceback.print_exc()
            return 0.5  # Default score on error

class ResumeAnalyzer:
    """Analyze resumes and provide feedback and scoring."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.job_matcher = JobMatcher(self.logger)
            
        # Initialize managers
        self._init_skills_manager()
        self._init_readability_manager()
        self._init_career_path_manager()
        self._init_learning_recommender()
        self._init_industry_keywords()
        
        # Define section patterns
        self.section_patterns = {
            'summary': re.compile(r'\b(summary|profile|objective|about)\b', re.IGNORECASE),
            'experience': re.compile(r'\b(experience|work|employment|job|professional background)\b', re.IGNORECASE),
            'education': re.compile(r'\b(education|academic|degree|university|college|school)\b', re.IGNORECASE),
            'skills': re.compile(r'\b(skills|expertise|proficiencies|competencies|qualifications)\b', re.IGNORECASE),
            'projects': re.compile(r'\b(projects|portfolio|works)\b', re.IGNORECASE),
            'certifications': re.compile(r'\b(certifications|certificates|credentials|qualifications)\b', re.IGNORECASE)
        }
    
    def analyze(self, resume_data, job_description=None, industry=None, timeout=30):
        """
        Analyze a parsed resume and provide feedback.
        
        Args:
            resume_data: Dictionary with parsed resume data
            job_description: Optional job description for matching
            industry: Optional industry context for analysis
            timeout: Maximum seconds to allow for analysis (default 30)
            
        Returns:
            Dictionary with analysis results
        """
        try:
            self.logger.info("Starting resume analysis")
            start_time = time.time()
            
            # Create a function to check for timeouts
            def check_timeout(section=None):
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self.logger.warning(f"Analysis timed out after {elapsed:.2f}s during {section or 'analysis'}")
                    raise TimeoutError(f"Resume analysis timed out after {elapsed:.2f}s")
                return elapsed
            
            # Validate input
            if not resume_data:
                self.logger.warning("Empty resume data provided")
                return {
                    "overall_score": 0,
                    "section_scores": {},
                    "recommendations": ["No resume data provided or resume parsing failed."]
                }
            
            # Create sections dict to store the scores for each section
            section_scores = {}
            recommendations = []
            
            # Analyze contact information
            try:
                check_timeout("contact")
                contact_score, contact_recs = self._analyze_contact(resume_data.get('contact_info', {}))
                section_scores['contact'] = contact_score
                recommendations.extend(contact_recs)
            except TimeoutError:
                self.logger.warning("Skipping remaining analysis due to timeout")
                section_scores['contact'] = 0.5
                recommendations.append("Analysis timed out. Try with a simpler resume.")
                return self._create_timeout_result(section_scores, recommendations, check_timeout())
            except Exception as e:
                self.logger.error(f"Error analyzing contact info: {str(e)}")
                section_scores['contact'] = 0.5
            
            # Analyze summary
            try:
                check_timeout("summary")
                summary_score, summary_recs = self._analyze_summary(resume_data.get('summary', ''), job_description)
                section_scores['summary'] = summary_score
                recommendations.extend(summary_recs)
            except TimeoutError:
                self.logger.warning("Skipping remaining analysis due to timeout")
                section_scores['summary'] = 0.5
                return self._create_timeout_result(section_scores, recommendations, check_timeout())
            except Exception as e:
                self.logger.error(f"Error analyzing summary: {str(e)}")
                section_scores['summary'] = 0.5
            
            # Analyze experience
            try:
                check_timeout("experience")
                experience_score, experience_recs = self._analyze_experience(resume_data.get('experience', []))
                section_scores['experience'] = experience_score
                recommendations.extend(experience_recs)
            except TimeoutError:
                self.logger.warning("Skipping remaining analysis due to timeout")
                section_scores['experience'] = 0.5
                return self._create_timeout_result(section_scores, recommendations, check_timeout())
            except Exception as e:
                self.logger.error(f"Error analyzing experience: {str(e)}")
                section_scores['experience'] = 0.5
            
            # Analyze skills
            try:
                check_timeout("skills")
                skills_score, skills_recs = self._analyze_skills(resume_data.get('skills', {}), job_description)
                section_scores['skills'] = skills_score
                recommendations.extend(skills_recs)
            except TimeoutError:
                self.logger.warning("Skipping remaining analysis due to timeout")
                section_scores['skills'] = 0.5
                return self._create_timeout_result(section_scores, recommendations, check_timeout())
            except Exception as e:
                self.logger.error(f"Error analyzing skills: {str(e)}")
                section_scores['skills'] = 0.5
            
            # Analyze education
            try:
                check_timeout("education")
                education_score, education_recs = self._analyze_education(resume_data.get('education', []))
                section_scores['education'] = education_score
                recommendations.extend(education_recs)
            except TimeoutError:
                self.logger.warning("Skipping remaining analysis due to timeout")
                section_scores['education'] = 0.5
                return self._create_timeout_result(section_scores, recommendations, check_timeout())
            except Exception as e:
                self.logger.error(f"Error analyzing education: {str(e)}")
                section_scores['education'] = 0.5
            
            # Analyze projects
            try:
                check_timeout("projects")
                projects_score, projects_recs = self._analyze_projects(resume_data.get('projects', []))
                section_scores['projects'] = projects_score
                recommendations.extend(projects_recs)
            except TimeoutError:
                self.logger.warning("Skipping remaining analysis due to timeout")
                section_scores['projects'] = 0.5
                return self._create_timeout_result(section_scores, recommendations, check_timeout())
            except Exception as e:
                self.logger.error(f"Error analyzing projects: {str(e)}")
                section_scores['projects'] = 0.5
            
            # Analyze resume format
            try:
                check_timeout("format")
                format_score, format_recs = self._analyze_format(resume_data)
                section_scores['format'] = format_score
                recommendations.extend(format_recs)
            except TimeoutError:
                self.logger.warning("Skipping remaining analysis due to timeout")
                section_scores['format'] = 0.5
                return self._create_timeout_result(section_scores, recommendations, check_timeout())
            except Exception as e:
                self.logger.error(f"Error analyzing format: {str(e)}")
                section_scores['format'] = 0.5
            
            # Calculate overall score (weighted average of section scores)
            scores_list = list(section_scores.values())
            if scores_list:
                # Convert from decimal to percentage (0-100 scale)
                overall_score = int(sum(scores_list) / len(scores_list) * 100)
            else:
                overall_score = 0
            
            # Job match analysis if job description is provided
            job_match = {}
            if job_description:
                try:
                    check_timeout("job_match")
                    # Calculate match score using job matcher
                    match_score = self.job_matcher.calculate_match(resume_data, job_description)
                    
                    # Extract job requirements
                    requirements = self.job_matcher.extract_requirements(job_description)
                    required_skills = requirements.get('required_skills', [])
                    preferred_skills = requirements.get('preferred_skills', [])
                    
                    # Get all skills from resume
                    all_resume_skills = []
                    skills_dict = resume_data.get('skills', {})
                    for skill_category, skills in skills_dict.items():
                        if isinstance(skills, list):
                            all_resume_skills.extend(skills)
                    
                    # Convert to lowercase for matching
                    resume_skills_lower = [s.lower() for s in all_resume_skills]
                    
                    # Find matching and missing skills
                    matching_skills = []
                    missing_skills = []
                    
                    for skill in required_skills:
                        skill_lower = skill.lower()
                        if any(skill_lower in rs for rs in resume_skills_lower):
                            matching_skills.append(skill)
                        else:
                            missing_skills.append(skill)
                    
                    # Extract keywords for matching
                    keywords = self.job_matcher.extract_keywords(job_description)
                    matching_keywords = []
                    
                    # Calculate job match percentage
                    job_match = {
                        "match_score": match_score,
                        "matching_skills": matching_skills,
                        "missing_skills": missing_skills[:5],  # Limit to top 5 missing skills
                        "keywords": keywords[:10]  # Limit to top 10 keywords
                    }
                except TimeoutError:
                    self.logger.warning("Job matching timed out")
                    job_match = {
                        "match_score": 50,  # Default score
                        "matching_skills": [],
                        "missing_skills": [],
                        "keywords": [],
                        "error": "Job matching timed out"
                    }
                except Exception as e:
                    self.logger.error(f"Error in job matching: {str(e)}")
                    job_match = {
                        "error": f"Job matching failed: {str(e)}"
                    }
            
            total_time = time.time() - start_time
            self.logger.info(f"Analysis complete. Overall score: {overall_score} (took {total_time:.2f}s)")
            
            # Create result dictionary
            result = {
                "overall_score": overall_score,
                "section_scores": section_scores,
                "recommendations": recommendations,
                "processing_time": f"{total_time:.2f}s"
            }
            
            # Add job match if it exists
            if job_description:
                result["job_match"] = job_match
            
            return result
            
        except TimeoutError as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Timeout in analysis: {str(e)}")
            return {
                "overall_score": 50,  # Default score
                "error": f"Analysis timed out after {elapsed:.2f}s",
                "section_scores": section_scores if 'section_scores' in locals() else {},
                "recommendations": ["Analysis timed out due to complexity. Try with a shorter resume."],
                "processing_time": f"{elapsed:.2f}s"
            }
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Error analyzing resume: {str(e)}")
            traceback.print_exc()
            return {
                "overall_score": 0,
                "error": str(e),
                "section_scores": {},
                "recommendations": ["An error occurred during analysis."],
                "processing_time": f"{elapsed:.2f}s"
            }
    
    def _create_timeout_result(self, section_scores, recommendations, elapsed_time):
        """Create a valid result object when a timeout occurs."""
        # Fill in missing scores with defaults so the frontend doesn't break
        required_sections = ['contact', 'summary', 'experience', 'education', 'skills', 'format']
        for section in required_sections:
            if section not in section_scores:
                section_scores[section] = 0.5  # Default mid-range score
                
        # Calculate a reduced overall score based on what we've processed
        score_values = list(section_scores.values())
        if score_values:
            overall_score = sum(score_values) / len(score_values)
        else:
            overall_score = 0.5  # Default
        
        # Ensure we return at least one recommendation
        if not recommendations:
            recommendations.append({
                "section": "general",
                "text": "Analysis timed out. Try simplifying your resume or using a cleaner format.",
                "priority": "high"
            })
        
        return {
            "overall_score": overall_score,
            "section_scores": section_scores,
            "recommendations": recommendations,
            "timeout": True,
            "processed_time": elapsed_time,
            "message": f"Analysis timed out after {elapsed_time:.2f} seconds. Results may be incomplete."
        }
    
    def _load_scoring_criteria(self):
        """Load scoring criteria from config file or use defaults."""
        try:
            import os
            import json
            
            criteria_path = os.path.join(os.path.dirname(__file__), '../config/scoring_criteria.json')
            if os.path.exists(criteria_path):
                with open(criteria_path, 'r') as f:
                    self.scoring_criteria = json.load(f)
                    logger.info("Loaded scoring criteria from file")
                    return
        except Exception as e:
            logger.warning(f"Error loading scoring criteria: {str(e)}")
            
        # Default scoring criteria if file not found or error occurs
        self.scoring_criteria = {
            "skills": {
                "min_count": 5,
                "optimal_count": 15,
                "tech_weight": 0.6,
                "soft_weight": 0.4
            },
            "experience": {
                "min_years": 1,
                "optimal_years": 5,
                "description_min_length": 50,
                "description_optimal_length": 200
            },
            "education": {
                "degree_weights": {
                    "bachelor": 0.7,
                    "master": 0.9,
                    "phd": 1.0,
                    "associate": 0.5,
                    "certificate": 0.3
                }
            },
            "summary": {
                "min_length": 50,
                "optimal_length": 200,
                "keyword_density": 0.05
            }
        }
        # Use default criteria silently
        
    def _load_industry_profiles(self):
        """Load industry profiles from config file or use defaults."""
        try:
            import os
            import json
            
            profiles_path = os.path.join(os.path.dirname(__file__), '../config/industry_profiles.json')
            if os.path.exists(profiles_path):
                with open(profiles_path, 'r') as f:
                    self.industry_profiles = json.load(f)
                    logger.info("Loaded industry profiles from file")
                    return
        except Exception as e:
            logger.warning(f"Error loading industry profiles: {str(e)}")
            
        # Default industry profiles if file not found or error occurs
        self.industry_profiles = {
            "technology": {
                "key_skills": ["programming", "development", "software", "cloud", "data", "analysis"],
                "education_emphasis": 0.7,
                "experience_emphasis": 0.8,
                "certification_emphasis": 0.6
            },
            "finance": {
                "key_skills": ["financial", "analysis", "accounting", "risk", "compliance", "audit"],
                "education_emphasis": 0.9,
                "experience_emphasis": 0.8,
                "certification_emphasis": 0.7
            },
            "healthcare": {
                "key_skills": ["medical", "patient", "care", "health", "clinical", "treatment"],
                "education_emphasis": 0.9,
                "experience_emphasis": 0.7,
                "certification_emphasis": 0.8
            },
            "marketing": {
                "key_skills": ["marketing", "brand", "social media", "content", "campaign", "digital"],
                "education_emphasis": 0.6,
                "experience_emphasis": 0.9,
                "certification_emphasis": 0.5
            },
            "default": {
                "key_skills": ["communication", "management", "leadership", "problem solving"],
                "education_emphasis": 0.7,
                "experience_emphasis": 0.8,
                "certification_emphasis": 0.6
            }
        }
        # Use default profiles silently
    
    def _find_section_boundaries(self, text: str) -> Dict[str, Tuple[int, int]]:
        """Find the start and end lines of each section in the resume"""
        print("Finding section boundaries...")
        if not text:
            print("Empty text")
            return {}
            
        lines = text.split('\n')
        print(f"Found {len(lines)} lines")
        sections = {}
        
        # Common section headers and their variations
        section_patterns = {
            'summary': r'(?i)^(?:summary|profile|objective|about|professional\s+summary)',
            'education': r'(?i)^(?:education|academic|qualification|degree)',
            'experience': r'(?i)^(?:experience|employment|work\s+history|professional\s+experience)',
            'skills': r'(?i)^(?:skills|technical\s+skills|expertise|competencies)',
            'projects': r'(?i)^(?:projects|personal\s+projects|academic\s+projects)',
            'certifications': r'(?i)^(?:certifications|certificates|accreditations)',
            'awards': r'(?i)^(?:awards|honors|achievements)',
            'languages': r'(?i)^(?:languages|language\s+proficiency)',
            'interests': r'(?i)^(?:interests|hobbies|activities)',
            'references': r'(?i)^(?:references|recommendations)'
        }
        
        # First pass: find all section headers
        section_starts = []
        
        # Handle contact section (usually at the top)
        contact_end = 0
        print("Looking for contact section...")
        contact_patterns = [
            r'@',
            r'phone',
            r'\d{3}[-.]?\d{3}[-.]?\d{4}',
            r'linkedin\.com',
            r'^[A-Z][a-z]+\s+[A-Z][a-z]+$'  # Name pattern
        ]
        
        for i in range(min(5, len(lines))):
            line = lines[i]
            print(f"Checking line {i}: {line}")
            if line:
                for pattern in contact_patterns:
                    if re.search(pattern, line, re.I):
                        sections['contact'] = (0, i + 1)
                        contact_end = i + 1
                        print(f"Found contact section: {sections['contact']}")
                        break
                if 'contact' in sections:
                    break
        
        # Find other sections
        print("Looking for other sections...")
        for i in range(contact_end, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
                
            # Check if line is a section header
            for section, pattern in section_patterns.items():
                if re.match(pattern, line):
                    section_starts.append((i, section))
                    print(f"Found section {section} at line {i}")
                    break
        
        # Second pass: determine section boundaries
        print(f"Found {len(section_starts)} sections")
        if section_starts:
            for i in range(len(section_starts)):
                start_idx, section = section_starts[i]
                
                # End is either the start of next section or end of resume
                if i < len(section_starts) - 1:
                    end_idx = section_starts[i + 1][0]
                else:
                    end_idx = len(lines)
                
                # Trim trailing empty lines
                while end_idx > start_idx and not lines[end_idx - 1].strip():
                    end_idx -= 1
                
                sections[section] = (start_idx, end_idx)
                print(f"Section {section}: {sections[section]}")
        
        return sections
    
    def _extract_section_text(self, text: str, section: str) -> str:
        """Extract text for a specific section"""
        if not text:
            return ""
            
        lines = text.split('\n')
        boundaries = self._find_section_boundaries(text)
        
        if section in boundaries:
            start, end = boundaries[section]
            section_lines = lines[start:end]
            return '\n'.join(line for line in section_lines if line.strip())
        
        # If section not found, try to find it using pattern matching
        section_patterns = {
            'summary': r'(?i)^(?:summary|profile|objective|about|professional\s+summary)',
            'education': r'(?i)^(?:education|academic|qualification|degree)',
            'experience': r'(?i)^(?:experience|employment|work\s+history|professional\s+experience)',
            'skills': r'(?i)^(?:skills|technical\s+skills|expertise|competencies)',
        }
        
        if section in section_patterns:
            pattern = section_patterns[section]
            section_lines = []
            in_section = False
            
            for line in lines:
                if re.match(pattern, line.strip(), re.I):
                    in_section = True
                    continue
                elif in_section and line.strip() and any(re.match(p, line.strip(), re.I) for p in section_patterns.values()):
                    break
                elif in_section and line.strip():
                    section_lines.append(line)
            
            return '\n'.join(section_lines)
        
        return ""
    
    def _extract_contact_info(self, text: str) -> Dict:
        """Extract contact information from resume text"""
        # Get contact section text
        contact_text = self._extract_section_text(text, 'contact')
        if not contact_text:
            contact_text = '\n'.join(text.split('\n')[:5])  # Use first 5 lines if no contact section found
        
        doc = self.nlp(contact_text)
        
        contact_info = {
            'name': None,
            'email': None,
            'phone': None,
            'location': None,
            'links': []
        }
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and not contact_info['name']:
                contact_info['name'] = ent.text
            elif ent.label_ == 'GPE' and not contact_info['location']:
                contact_info['location'] = ent.text
        
        # Extract email using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, contact_text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Extract phone using regex
        phone_patterns = [
            r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',  # (123) 456-7890
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890
            r'\b\+\d{1,2}\s\d{3}\s\d{3}\s\d{4}\b'  # +1 123 456 7890
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, contact_text)
            if phones:
                contact_info['phone'] = phones[0]
                break
        
        # Extract links (LinkedIn, GitHub, etc.)
        link_patterns = {
            'linkedin': r'linkedin\.com/in/[\w-]+',
            'github': r'github\.com/[\w-]+',
            'portfolio': r'(?:portfolio\.com|\.io|\.dev)/[\w-]+'
        }
        
        for platform, pattern in link_patterns.items():
            links = re.findall(pattern, contact_text, re.I)
            if links:
                contact_info['links'].append({
                    'platform': platform,
                    'url': links[0]
                })
        
        return contact_info
    
    def _extract_education(self, text: str) -> List[Dict]:
        """Extract education information from resume text"""
        education_section = []
        
        # Common education keywords and degree patterns
        edu_keywords = ['university', 'college', 'institute', 'school']
        degree_patterns = [
            r'bachelor(?:\'s)?\s+(?:of|in)\s+[\w\s]+',
            r'master(?:\'s)?\s+(?:of|in)\s+[\w\s]+',
            r'phd|ph\.d\.',
            r'b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?'
        ]
        
        # Find education section
        lines = text.lower().split('\n')
        in_education = False
        current_edu = {}
        
        for i, line in enumerate(lines):
            line = line.strip()
            original_line = text.split('\n')[i].strip()
            
            # Check if we're entering education section
            if 'education' in line and not in_education:
                in_education = True
                continue
            
            # Check if we're leaving education section
            if in_education and any(section in line for section in ['experience', 'skills', 'projects']):
                in_education = False
                if current_edu:
                    education_section.append(current_edu)
                    current_edu = {}
                continue
            
            if in_education and line:
                # Check for university/college
                if any(keyword in line for keyword in edu_keywords):
                    if current_edu:
                        education_section.append(current_edu)
                    current_edu = {'institution': original_line, 'degree': None, 'year': None}
                    
                    # Extract year if present
                    year_match = re.search(r'\b20\d{2}\b', original_line)
                    if year_match:
                        current_edu['year'] = year_match.group()
                
                # Check for degree
                elif current_edu and any(re.search(pattern, line) for pattern in degree_patterns):
                    current_edu['degree'] = original_line
                    
                    # Extract year if present
                    year_match = re.search(r'\b20\d{2}\b', original_line)
                    if year_match:
                        current_edu['year'] = year_match.group()
                
                # Check for GPA
                elif current_edu and ('gpa' in line or 'grade' in line):
                    gpa_match = re.search(r'\d+\.\d+', line)
                    if gpa_match:
                        current_edu['gpa'] = float(gpa_match.group())
        
        # Add last education entry if exists
        if current_edu:
            education_section.append(current_edu)
        
        return education_section
    
    def _extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience information from resume text"""
        experience_text = self._extract_section_text(text, 'experience')
        if not experience_text:
            return []
        
        experiences = []
        current_experience = None
        
        # Common date patterns
        date_patterns = [
            r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
            r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
            r'\s*(?:\d{4})',  # Month Year
            r'\d{1,2}/\d{4}',  # MM/YYYY
            r'\d{4}',  # YYYY
            r'Present|Current|Now'  # Current job indicators
        ]
        
        date_pattern = '|'.join(f'({p})' for p in date_patterns)
        
        # Split into entries (usually separated by newlines)
        entries = experience_text.split('\n')
        
        for entry in entries:
            entry = entry.strip()
            if not entry:
                continue
            
            # Look for date patterns
            dates = re.findall(date_pattern, entry, re.I)
            dates = [d for d in dates if any(d)]  # Remove empty matches
            
            if dates:
                # This line likely starts a new experience entry
                if current_experience:
                    experiences.append(current_experience)
                
                # Parse title and company
                title_company = re.sub(date_pattern, '', entry, flags=re.I).strip()
                title_company = re.split(r'\s*[|@]\s*', title_company)
                
                current_experience = {
                    'title': title_company[0].strip() if title_company else '',
                    'company': title_company[1].strip() if len(title_company) > 1 else '',
                    'dates': self._parse_dates(dates),
                    'responsibilities': []
                }
            elif current_experience and entry.strip().startswith(('•', '-', '*')):
                # This is likely a bullet point
                responsibility = entry.strip().lstrip('•-* ')
                if responsibility:
                    current_experience['responsibilities'].append(responsibility)
        
        # Add the last experience
        if current_experience:
            experiences.append(current_experience)
        
        return experiences
    
    def _parse_dates(self, dates: List[str]) -> Dict:
        """Parse dates into start and end dates"""
        from datetime import datetime
        
        def parse_date(date_str: str) -> Optional[datetime]:
            if isinstance(date_str, tuple):
                # Find first non-empty string in tuple
                date_str = next((s for s in date_str if s), '')
            
            date_str = str(date_str).strip().lower()
            
            if date_str in ['present', 'current', 'now']:
                return datetime.now()
            
            try:
                # Try different date formats
                for fmt in ['%B %Y', '%b %Y', '%m/%Y', '%Y']:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
            except Exception:
                pass
            
            return None
        
        result = {
            'start': None,
            'end': None,
            'duration': None
        }
        
        if len(dates) >= 2:
            start_date = parse_date(dates[0])
            end_date = parse_date(dates[1])
            
            if start_date and end_date:
                result['start'] = start_date.strftime('%Y-%m')
                result['end'] = end_date.strftime('%Y-%m')
                
                # Calculate duration in months
                months = (end_date.year - start_date.year) * 12
                months += end_date.month - start_date.month
                result['duration'] = months
        
        return result
    
    def _extract_skills(self, text: str) -> Dict:
        """Extract and categorize skills using the skill taxonomy manager"""
        # Initialize skill categories
        skills = {
            'programming_languages': [],
            'web_frameworks': [],
            'databases': [],
            'cloud_platforms': [],
            'soft_skills': []
        }
        
        # Extract skill mentions
        doc = self.nlp(text)
        potential_skills = set()
        
        # Add noun phrases as potential skills
        for chunk in doc.noun_chunks:
            potential_skills.add(chunk.text.lower())
        
        # Add individual tokens that might be skills
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                potential_skills.add(token.text.lower())
        
        # Get skills from taxonomy
        extracted_skills = self.skill_manager.extract_skills(text)
        
        # Merge extracted skills into our categories
        for category, skill_list in extracted_skills.items():
            if category in skills:
                skills[category].extend(skill_list)
        
        return skills
    
    def _calculate_skills_score(self, skills: Dict) -> float:
        """Calculate score for skills section"""
        score = 0
        
        # Base score for having skills section
        if skills:
            score += 20
        
        # Score for each skill category
        for category, skill_list in skills.items():
            if skill_list:
                score += min(40, len(skill_list) * 5)  # Up to 40 points per category
        
        return min(100, score)
    
    def _calculate_experience_score(self, experience: List[Dict]) -> float:
        """Calculate score for experience section"""
        if not experience:
            return 0
            
        # Calculate total experience duration in months
        total_months = 0
        for exp in experience:
            duration = exp.get('duration', 0)
            # Handle string durations
            if isinstance(duration, str):
                try:
                    duration = int(duration.strip())
                except (ValueError, TypeError):
                    duration = 0
            total_months += duration
            
        # Calculate score based on total experience
        if total_months >= 60:  # 5+ years
            duration_score = 0.5
        elif total_months >= 36:  # 3+ years
            duration_score = 0.4
        elif total_months >= 24:  # 2+ years
            duration_score = 0.3
        elif total_months >= 12:  # 1+ year
            duration_score = 0.2
        else:
            duration_score = 0.1
            
        # Calculate score based on number of positions
        num_positions = len(experience)
        if num_positions >= 3:
            positions_score = 0.3
        elif num_positions == 2:
            positions_score = 0.2
        else:
            positions_score = 0.1
            
        # Calculate score based on job descriptions
        description_score = 0.0
        for exp in experience:
            description = exp.get('description', '')
            if isinstance(description, str) and len(description) > 100:
                description_score += 0.05
                
        description_score = min(0.2, description_score)  # Cap at 0.2
        
        # Combine scores
        total_score = duration_score + positions_score + description_score
        
        return min(1.0, total_score)
    
    def _calculate_education_score(self, education_data: List, industry: str = None) -> float:
        """Analyze education section and return a score between 0 and 1"""
        if not education_data or not isinstance(education_data, list):
            return 0.3  # Default score
            
        # Initialize score components
        degree_score = 0.0
        details_score = 0.0
        
        # Count valid education entries
        valid_entries = 0
        
        for entry in education_data:
            # Check if entry is dictionary with expected fields
            if isinstance(entry, dict):
                valid_entries += 1
                
                # Score degree type if present
                if 'degree' in entry and entry['degree']:
                    degree_name = entry['degree'].lower()
                    if any(term in degree_name for term in ['phd', 'doctorate', 'doctor']):
                        degree_score += 1.0
                    elif any(term in degree_name for term in ['master', 'mba', 'ms', 'ma']):
                        degree_score += 0.8
                    elif any(term in degree_name for term in ['bachelor', 'bs', 'ba', 'bsc']):
                        degree_score += 0.6
                    elif any(term in degree_name for term in ['associate', 'certification']):
                        degree_score += 0.4
                    else:
                        degree_score += 0.3
                
                # Score completeness of entry
                details_count = sum(1 for field in ['institution', 'graduation_date', 'gpa', 'field'] 
                                   if field in entry and entry[field])
                details_score += min(1.0, details_count / 4)
            
            # If it's just a string, give partial credit
            elif isinstance(entry, str) and entry.strip():
                valid_entries += 0.5
                details_score += 0.3
        
        # Normalize scores based on entries
        if valid_entries > 0:
            degree_score /= max(1, valid_entries)
            details_score /= max(1, valid_entries)
        
        # Combine scores with weights
        final_score = (degree_score * 0.6) + (details_score * 0.4)
        
        # Industry-specific adjustments
        if industry:
            industry_profiles = getattr(self, 'industry_profiles', {})
            if industry in industry_profiles:
                education_emphasis = industry_profiles[industry].get('education_emphasis', 0.7)
                # Boost score for industries that value education highly
                if education_emphasis > 0.8:
                    final_score = min(1.0, final_score * 1.2)
        
        return final_score

    def _calculate_achievements_score(self, experience: List[Dict]) -> float:
        """Calculate score for achievements"""
        if not experience or not isinstance(experience, list):
            return 0.0
            
        score = 0.0
        achievement_keywords = ['achieved', 'increased', 'reduced', 'improved', 'led', 'developed', 'created', 'implemented']
        metric_patterns = [
            r'\d+(?:\.\d+)?%',  # Percentage
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # Money
            r'\d+(?:\.\d+)?x',   # Multiplier
            r'\d+(?:\.\d+)?\s*(?:million|k|thousand)',  # Large numbers
            r'\d+(?:\.\d+)?\s*(?:users?|customers?|clients?)'  # User metrics
        ]
        
        # Check experience section for achievements
        total_achievements = 0
        
        for exp in experience:
            if not isinstance(exp, dict):
                continue
                
            # Get responsibilities from experience
            responsibilities = []
            if 'responsibilities' in exp and isinstance(exp['responsibilities'], list):
                responsibilities = exp['responsibilities']
            elif 'description' in exp and isinstance(exp['description'], str):
                # If no responsibilities list but has description, use that
                responsibilities = [exp['description']]
                
            # Check each responsibility for achievements
            for resp in responsibilities:
                if not isinstance(resp, str):
                    continue
                    
                resp_text = resp.lower()
                
                # Check for achievement keywords
                if any(keyword in resp_text for keyword in achievement_keywords):
                    total_achievements += 1
                    
                # Check for metrics
                if any(re.search(pattern, resp_text, re.IGNORECASE) for pattern in metric_patterns):
                    total_achievements += 1
                    
        # Base score for having achievements
        if total_achievements > 0:
            score = 0.3  # Base score for having any achievements
            
        # Add points based on number of achievements
        if total_achievements >= 5:
            score += 0.5  # Full points for 5+ achievements
        else:
            score += (total_achievements / 5) * 0.5  # Partial points
            
        # Add bonus for metrics
        metric_count = 0
        for exp in experience:
            if not isinstance(exp, dict):
                continue
                
            responsibilities = []
            if 'responsibilities' in exp and isinstance(exp['responsibilities'], list):
                responsibilities = exp['responsibilities']
            elif 'description' in exp and isinstance(exp['description'], str):
                responsibilities = [exp['description']]
                
            for resp in responsibilities:
                if not isinstance(resp, str):
                    continue
                    
                if any(re.search(pattern, resp, re.IGNORECASE) for pattern in metric_patterns):
                    metric_count += 1
                    
        score += min(0.2, metric_count * 0.05)  # Up to 20% bonus for metrics
        
        return min(1.0, score)

    def _analyze_skills(self, skills, job_description=None):
        """
        Analyze the skills section of a resume.
        
        Args:
            skills: Dictionary of skills by category
            job_description: Optional job description for matching
            
        Returns:
            Tuple of (score, recommendations)
        """
        score = 0
        recommendations = []
        
        # Return lower score if skills is empty
        if not skills:
            recommendations.append("Add a dedicated skills section with relevant technical and soft skills.")
            return 0.2, recommendations
        
        # Check for empty or nearly empty skills
        total_skills = sum(len(skill_list) for skill_list in skills.values())
        
        if total_skills == 0:
            recommendations.append("Add a dedicated skills section with relevant technical and soft skills.")
            return 0.2, recommendations
        elif total_skills < 5:
            recommendations.append("Expand your skills section to include more relevant skills (aim for at least 10-15 skills).")
            score = 0.4
        
        # Calculate skill coverage score
        programming_languages = skills.get("programming_languages", [])
        web_frameworks = skills.get("web_frameworks", [])
        databases = skills.get("databases", [])
        cloud_platforms = skills.get("cloud_platforms", [])
        tools = skills.get("tools", [])
        soft_skills = skills.get("soft_skills", [])
        
        # Ensure we have a good mix of technical and soft skills
        has_programming = len(programming_languages) > 0
        has_frameworks = len(web_frameworks) > 0
        has_databases = len(databases) > 0
        has_cloud = len(cloud_platforms) > 0
        has_tools = len(tools) > 0
        has_soft_skills = len(soft_skills) > 0
        
        # Count categories with skills
        categories_with_skills = sum([has_programming, has_frameworks, has_databases, has_cloud, has_tools, has_soft_skills])
        
        # Base skill balance score on category coverage
        if categories_with_skills >= 5:
            # Great coverage across different skill types
            balance_score = 1.0
        elif categories_with_skills >= 3:
            # Good coverage
            balance_score = 0.8
            recommendations.append("Add skills from more categories (try to cover programming languages, frameworks, databases, cloud platforms, and soft skills).")
        elif categories_with_skills >= 2:
            # Acceptable coverage
            balance_score = 0.6
            recommendations.append("Your skills section is too narrow. Include a wider range of technical and soft skills.")
        else:
            # Poor coverage
            balance_score = 0.4
            recommendations.append("Your skills section lacks diversity. Include skills from multiple categories including programming, frameworks, tools, and soft skills.")
        
        # Check skill specificity - avoid generic skills
        specificity_score = 0.0
        # Programming languages should be specific
        if has_programming:
            specific_languages = any(lang for lang in programming_languages if lang not in ["programming", "coding", "development"])
            if specific_languages:
                specificity_score += 0.2
        
        # Frameworks should be named specifically
        if has_frameworks:
            specific_frameworks = any(fw for fw in web_frameworks if fw not in ["frameworks", "libraries", "web development"])
            if specific_frameworks:
                specificity_score += 0.2
        
        # Databases should be named specifically
        if has_databases:
            specific_databases = any(db for db in databases if db not in ["database", "data management"])
            if specific_databases:
                specificity_score += 0.2
        
        # Cloud platforms should be named specifically
        if has_cloud:
            specific_cloud = any(cp for cp in cloud_platforms if cp not in ["cloud", "devops"])
            if specific_cloud:
                specificity_score += 0.2
        
        # Tools should be specific
        if has_tools and len(tools) >= 3:
            specificity_score += 0.2
            
        # Soft skills should be diverse and specific
        if has_soft_skills and len(soft_skills) >= 3:
            specificity_score += 0.2
        
        # Cap specificity score at 1.0
        specificity_score = min(1.0, specificity_score)
        
        # Check quantity - ideal resume has 15-20 relevant skills
        quantity_score = 0.0
        if total_skills >= 15:
            quantity_score = 1.0
        elif total_skills >= 10:
            quantity_score = 0.8
        elif total_skills >= 5:
            quantity_score = 0.6
            recommendations.append("Add more skills to your resume - aim for 15-20 relevant skills.")
        else:
            quantity_score = 0.4
            recommendations.append("Your skills section is too sparse. Add at least 10-15 relevant skills.")
        
        # If job description is provided, check relevance
        relevance_score = 0.7  # Default is moderate relevance
        if job_description:
            # Extract job requirements
            requirements = self.job_matcher.extract_requirements(job_description)
            
            # Flatten skills for matching
            all_skills = []
            for skill_category in skills.values():
                all_skills.extend(skill_category)
            
            # Calculate match percentage
            matched_skills = 0
            required_skills = requirements.get("required_skills", [])
            
            if required_skills:
                # First check exact matches
                for req_skill in required_skills:
                    if any(req_skill.lower() in skill.lower() for skill in all_skills):
                        matched_skills += 1
                
                # If few exact matches, try semantic matching
                if matched_skills / max(1, len(required_skills)) < 0.3:
                    semantic_matches = self.job_matcher.match_skills_semantically(all_skills, required_skills)
                    matched_skills = max(matched_skills, semantic_matches)
                
                # Calculate relevance score
                match_percentage = matched_skills / max(1, len(required_skills))
                
                if match_percentage >= 0.7:
                    relevance_score = 1.0
                elif match_percentage >= 0.5:
                    relevance_score = 0.8
                    recommendations.append("Your skills don't fully align with job requirements. Add more of the specific skills mentioned in the job description.")
                elif match_percentage >= 0.3:
                    relevance_score = 0.6
                    recommendations.append("Your skills have low alignment with job requirements. Tailor your skills section to match the job description more closely.")
                else:
                    relevance_score = 0.4
                    recommendations.append("Your skills section doesn't match the job requirements. Customize your resume with skills mentioned in the job posting.")
        
        # Calculate overall skills score
        # Weight the components: balance (30%), specificity (20%), quantity (20%), relevance (30%)
        weighted_score = (balance_score * 0.3) + (specificity_score * 0.2) + (quantity_score * 0.2) + (relevance_score * 0.3)
        
        # Apply the base score
        score = max(score, weighted_score)
        
        # Generate recommendations for improvement
        if total_skills < 10:
            recommendations.append("Add more skills to your resume, especially those relevant to your target role.")
        
        missing_categories = []
        if not has_programming and not has_frameworks:
            missing_categories.append("technical skills (programming languages or frameworks)")
        if not has_databases and not has_cloud:
            missing_categories.append("infrastructure skills (databases or cloud platforms)")
        if not has_soft_skills:
            missing_categories.append("soft skills")
        
        if missing_categories:
            categories_str = ", ".join(missing_categories)
            recommendations.append(f"Add {categories_str} to your skills section for better balance.")
        
        return score, recommendations

    def _analyze_experience(self, experience):
        """
        Analyze work experience.
        
        Args:
            experience: List of experience dictionaries
            
        Returns:
            Tuple of (score, recommendations)
        """
        score = 0
        recommendations = []
        
        # Check if experience exists
        if not experience:
            recommendations.append("Add your work experience to your resume, including job titles, companies, dates, and accomplishments.")
            return 0.1, recommendations
        
        # Score based on number of entries
        num_entries = len(experience)
        if num_entries >= 3:
            score = 0.6  # Good number of entries
        elif num_entries >= 1:
            score = 0.4  # Acceptable but could use more
        else:
            score = 0.2  # Too few entries
        
        # Check for completeness of entries
        complete_entries = 0
        entries_with_dates = 0
        entries_with_responsibilities = 0
        total_bullets = 0
        entries_with_companies = 0
        entries_with_titles = 0
        
        for entry in experience:
            # Check if entry has all key components
            has_title = bool(entry.get('title'))
            has_company = bool(entry.get('company'))
            has_dates = bool(entry.get('start_date')) or bool(entry.get('end_date'))
            responsibilities = entry.get('responsibilities', [])
            has_responsibilities = len(responsibilities) > 0
            
            if has_title:
                entries_with_titles += 1
            
            if has_company:
                entries_with_companies += 1
            
            if has_dates:
                entries_with_dates += 1
            
            if has_responsibilities:
                entries_with_responsibilities += 1
                total_bullets += len(responsibilities)
            
            if has_title and has_company and has_dates and has_responsibilities:
                complete_entries += 1
        
        # Calculate completion percentage
        if num_entries > 0:
            completion_score = complete_entries / num_entries
            
            # Adjust score based on completion
            if completion_score >= 0.8:
                score += 0.3  # Most entries are complete
            elif completion_score >= 0.5:
                score += 0.2  # More than half are complete
            
            # Check for specific components
            if entries_with_titles < num_entries:
                recommendations.append("Ensure all experience entries include your job title.")
            
            if entries_with_companies < num_entries:
                recommendations.append("Ensure all experience entries include the company name.")
            
            if entries_with_dates < num_entries:
                recommendations.append("Include dates (month/year) for all work experiences.")
            
            if entries_with_responsibilities < num_entries:
                recommendations.append("Add bullet points describing your responsibilities and achievements for each role.")
        
        # Check for bullet point quality
        if total_bullets > 0:
            bullets_per_job = total_bullets / num_entries
            
            if bullets_per_job < 2:
                recommendations.append("Add more detail to your job descriptions (at least 3-5 bullet points per job).")
            elif bullets_per_job > 8:
                recommendations.append("Consider focusing on the most important achievements (3-5 bullet points per job is ideal).")
        
        # Check for accomplishment statements
        accomplishment_keywords = ['achieved', 'improved', 'increased', 'reduced', 'created', 'developed', 'implemented', 'led']
        has_accomplishments = False
        
        for entry in experience:
            for resp in entry.get('responsibilities', []):
                if any(keyword in resp.lower() for keyword in accomplishment_keywords):
                    has_accomplishments = True
                    break
            if has_accomplishments:
                break
        
        if not has_accomplishments:
            recommendations.append("Focus on accomplishments, not just duties. Use action verbs and quantify results where possible.")
        else:
            score += 0.1  # Bonus for having accomplishments
        
        return min(1.0, score), recommendations

    def _analyze_contact(self, contact):
        """
        Analyze contact information.
        
        Args:
            contact: Dictionary of contact information
            
        Returns:
            Tuple of (score, recommendations)
        """
        score = 0
        recommendations = []
        
        if not contact:
            recommendations.append("Add contact information including name, phone number, email, and location.")
            return 0.1, recommendations
        
        # Check essential fields
        has_name = bool(contact.get('name'))
        has_email = bool(contact.get('email'))
        has_phone = bool(contact.get('phone'))
        has_location = bool(contact.get('location'))
        has_linkedin = bool(contact.get('linkedin')) or any(
            link.get('platform') == 'linkedin' for link in contact.get('links', [])
        )
        
        # Score based on completeness
        completeness = 0
        if has_name:
            completeness += 0.2
        else:
            recommendations.append("Add your full name to the contact section.")
            
        if has_email:
            completeness += 0.2
        else:
            recommendations.append("Add your email address to the contact section.")
            
        if has_phone:
            completeness += 0.2
        else:
            recommendations.append("Add your phone number to the contact section.")
            
        if has_location:
            completeness += 0.1
        else:
            recommendations.append("Add your location (city, state) to the contact section.")
            
        if has_linkedin:
            completeness += 0.1
        else:
            recommendations.append("Add your LinkedIn profile to the contact section.")
            
        # Extra for additional professional links
        if len(contact.get('links', [])) > 1:
            completeness += 0.1
            
        score = min(1.0, completeness)
        
        return score, recommendations
        
    def _analyze_summary(self, summary, job_description=None):
        """
        Analyze the summary/objective section of a resume.
        
        Args:
            summary: Summary text
            job_description: Optional job description for matching
            
        Returns:
            Tuple of (score, recommendations)
        """
        score = 0
        recommendations = []
        
        if not summary:
            recommendations.append("Add a professional summary that highlights your key qualifications and career goals.")
            return 0.1, recommendations
        
        # Check summary length
        words = summary.split()
        word_count = len(words)
        
        if word_count < 30:
            score = 0.3
            recommendations.append("Your summary is too brief. Aim for 50-100 words that highlight key qualifications and career goals.")
        elif word_count < 50:
            score = 0.6
            recommendations.append("Consider expanding your summary to better highlight your qualifications (aim for 50-100 words).")
        elif word_count > 200:
            score = 0.7
            recommendations.append("Your summary is quite long. Consider trimming it to 100 words or less for better readability.")
        else:
            score = 0.9
        
        # Check job relevance if job description is provided
        if job_description:
            # Extract keywords from job description
            if hasattr(self, 'job_matcher') and self.job_matcher:
                job_keywords = self.job_matcher.extract_keywords(job_description)
                matching_keywords = [kw for kw in job_keywords if kw.lower() in summary.lower()]
                
                keyword_match_ratio = len(matching_keywords) / max(1, len(job_keywords))
                if keyword_match_ratio < 0.2:
                    recommendations.append("Your summary doesn't match the job description well. Include more relevant keywords from the job posting.")
                    score -= 0.2
        
        return max(0.1, score), recommendations
        
    def _analyze_education(self, education):
        """
        Analyze education section.
        
        Args:
            education: List of education dictionaries
            
        Returns:
            Tuple of (score, recommendations)
        """
        score = 0
        recommendations = []
        
        if not education:
            recommendations.append("Add your educational background including degrees, institutions, and graduation dates.")
            return 0.3, recommendations  # Still give some points as education may not be critical for everyone
        
        # Check completeness of education entries
        complete_entries = 0
        entries_with_degree = 0
        entries_with_institution = 0
        entries_with_dates = 0
        entries_with_gpa = 0
        
        for entry in education:
            # Check for key components
            has_degree = bool(entry.get('degree'))
            has_institution = bool(entry.get('institution'))
            has_dates = bool(entry.get('start_date')) or bool(entry.get('end_date')) or bool(entry.get('year'))
            has_gpa = bool(entry.get('gpa'))
            
            if has_degree:
                entries_with_degree += 1
            
            if has_institution:
                entries_with_institution += 1
            
            if has_dates:
                entries_with_dates += 1
            
            if has_gpa:
                entries_with_gpa += 1
            
            if has_degree and has_institution and has_dates:
                complete_entries += 1
        
        # Score based on completeness
        if len(education) > 0:
            completion_ratio = complete_entries / len(education)
            
            if completion_ratio >= 0.8:
                score = 0.8  # Most entries are complete
            elif completion_ratio >= 0.5:
                score = 0.6  # More than half complete
            else:
                score = 0.4  # Less than half complete
                
            # Check for specific missing components
            if entries_with_degree < len(education):
                recommendations.append("Include your degree type for all education entries.")
            
            if entries_with_institution < len(education):
                recommendations.append("Include the institution name for all education entries.")
            
            if entries_with_dates < len(education):
                recommendations.append("Include graduation dates or attendance years for all education entries.")
        
        # Bonus for having GPA if it's good
        if entries_with_gpa > 0:
            score += 0.1
            
        return min(1.0, score), recommendations
        
    def _analyze_projects(self, projects):
        """
        Analyze projects section.
        
        Args:
            projects: List of project dictionaries
            
        Returns:
            Tuple of (score, recommendations)
        """
        score = 0
        recommendations = []
        
        # Projects are optional but valuable
        if not projects:
            recommendations.append("Consider adding a projects section to showcase relevant work examples.")
            return 0.5, recommendations  # Still give decent score as projects are optional
        
        # Score based on number of projects
        num_projects = len(projects)
        if num_projects >= 3:
            score = 0.7  # Good number of projects
        elif num_projects >= 1:
            score = 0.6  # At least one project
            
        # Check completeness of project entries
        complete_projects = 0
        projects_with_description = 0
        projects_with_technologies = 0
        
        for project in projects:
            has_title = bool(project.get('title'))
            has_description = bool(project.get('description')) and len(str(project.get('description', ''))) > 20
            has_technologies = bool(project.get('technologies')) and len(project.get('technologies', [])) > 0
            
            if has_description:
                projects_with_description += 1
            
            if has_technologies:
                projects_with_technologies += 1
            
            if has_title and has_description and has_technologies:
                complete_projects += 1
        
        # Score based on completeness
        if num_projects > 0:
            completion_ratio = complete_projects / num_projects
            
            if completion_ratio >= 0.8:
                score += 0.2  # Most projects are complete
            elif completion_ratio >= 0.5:
                score += 0.1  # More than half complete
                
            # Check for specific missing components
            if projects_with_description < num_projects:
                recommendations.append("Add detailed descriptions for all projects explaining what problem you solved and how.")
            
            if projects_with_technologies < num_projects:
                recommendations.append("List the technologies used for each project.")
        
        return min(1.0, score), recommendations
        
    def _analyze_format(self, resume_data):
        """
        Analyze the overall format and readability of the resume.
        
        Args:
            resume_data: The complete resume data
            
        Returns:
            Tuple of (score, recommendations)
        """
        score = 0.6  # Default reasonable score
        recommendations = []
        
        # Check for basic sections
        has_contact = bool(resume_data.get('contact_info'))
        has_summary = bool(resume_data.get('summary'))
        has_experience = len(resume_data.get('experience', [])) > 0
        has_education = len(resume_data.get('education', [])) > 0
        has_skills = bool(resume_data.get('skills'))
        
        # Count essential sections
        essential_sections = sum([has_contact, has_experience, has_education, has_skills])
        
        if essential_sections <= 2:
            score = 0.3
            recommendations.append("Your resume is missing essential sections. Include contact info, experience, education, and skills.")
        elif essential_sections == 3:
            score = 0.5
            
        # Specific section checks
        if not has_contact:
            recommendations.append("Add a contact section with your name, phone, email, and location.")
            
        if not has_summary and has_experience:
            recommendations.append("Consider adding a professional summary to highlight your key qualifications.")
            
        if not has_skills:
            recommendations.append("Add a skills section that categorizes your technical and soft skills.")
            
        # Format recommendations
        if has_experience and not has_education:
            recommendations.append("In most cases, both education and experience sections are expected on a resume.")
            
        return score, recommendations
        
    def _prioritize_recommendations(self, recommendations, section_scores, weights):
        """
        Prioritize recommendations based on section importance and scores.
        
        Args:
            recommendations: List of recommendations
            section_scores: Dictionary of section scores
            weights: Dictionary of section weights
            
        Returns:
            Prioritized list of recommendations
        """
        if not recommendations:
            return []
            
        # Create priority categories
        high_priority = []
        medium_priority = []
        low_priority = []
        
        # Assign priority based on section importance and score
        for rec in recommendations:
            # Determine which section this recommendation belongs to based on keywords
            rec_lower = rec.lower()
            section = None
            
            if any(kw in rec_lower for kw in ['contact', 'name', 'email', 'phone']):
                section = 'contact'
            elif any(kw in rec_lower for kw in ['summary', 'objective', 'profile']):
                section = 'summary'
            elif any(kw in rec_lower for kw in ['experience', 'job', 'role', 'work']):
                section = 'experience'
            elif any(kw in rec_lower for kw in ['skill', 'technical', 'soft']):
                section = 'skills'
            elif any(kw in rec_lower for kw in ['education', 'degree', 'university']):
                section = 'education'
            elif any(kw in rec_lower for kw in ['project', 'portfolio']):
                section = 'projects'
            else:
                section = 'format'
                
            # Get section score and weight
            score = section_scores.get(section, 0.5)
            weight = weights.get(section, 10)
            
            # Assign priority based on score and weight
            # Low scores in high-weight sections are highest priority
            if score < 0.4 and weight > 15:
                high_priority.append(rec)
            elif score < 0.6 or weight > 20:
                medium_priority.append(rec)
            else:
                low_priority.append(rec)
                
        # Combine priorities to create final list
        return high_priority + medium_priority + low_priority
    def _init_skills_manager(self):
        """Initialize the skills taxonomy manager."""
        try:
            self.logger.info("Initialized skill taxonomy manager")
        except Exception as e:
            self.logger.warning(f"Failed to initialize skills manager: {str(e)}")
            
    def _init_readability_manager(self):
        """Initialize the readability analyzer."""
        try:
            self.logger.info("Initialized readability analyzer")
        except Exception as e:
            self.logger.warning(f"Failed to initialize readability analyzer: {str(e)}")
            
    def _init_career_path_manager(self):
        """Initialize the career path analyzer."""
        try:
            self.logger.info("Initialized career path analyzer")
        except Exception as e:
            self.logger.warning(f"Failed to initialize career path analyzer: {str(e)}")
            
    def _init_learning_recommender(self):
        """Initialize the learning recommendations manager."""
        try:
            self.logger.info("Initialized learning recommender")
        except Exception as e:
            self.logger.warning(f"Failed to initialize learning recommender: {str(e)}")
            
    def _init_industry_keywords(self):
        """Initialize industry-specific keywords from profiles."""
        self.industry_keywords = {}
        try:
            # In a real implementation, this would load industry-specific keywords
            # from a database or file
            self.industry_keywords = {
                'technology': ['software', 'development', 'programming', 'cloud', 'api', 'agile'],
                'finance': ['analysis', 'investment', 'banking', 'financial', 'trading', 'risk'],
                'healthcare': ['patient', 'clinical', 'medical', 'health', 'care', 'treatment'],
                'marketing': ['brand', 'campaign', 'content', 'digital', 'analytics', 'social'],
                'education': ['teaching', 'curriculum', 'student', 'learning', 'education', 'classroom']
            }
            self.logger.info(f"Initialized keywords for {len(self.industry_keywords)} industries")
        except Exception as e:
            self.logger.warning(f"Failed to initialize industry keywords: {str(e)}")

    # Add method to ensure API compatibility
    def analyze_resume(self, resume_data, job_description=None, industry=None):
        """
        API-compatible method that calls analyze.
        
        Args:
            resume_data: Dictionary with parsed resume data
            job_description: Optional job description for matching
            industry: Optional industry context for analysis
            
        Returns:
            Dictionary with analysis results
        """
        return self.analyze(resume_data, job_description, industry)
