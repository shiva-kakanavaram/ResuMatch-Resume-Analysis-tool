import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import spacy
from spacy.tokens import Doc
import numpy as np
from collections import defaultdict
import dateutil.parser
import json
import os
import warnings
import time
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class EnhancedResumeParser:
    """Enhanced resume parser that extracts structured information from resumes."""
    
    def __init__(self, model_name="en_core_web_sm"):
        """
        Initialize the parser with specified NLP model.
        
        Args:
            model_name: Name of the spaCy model to use
        """
        try:
            self.nlp = spacy.load(model_name)
            self.logger = logger
            self.logger.info("Parser initialized successfully.")
        except Exception as e:
            self.logger = logger
            self.logger.error(f"Error loading spaCy model: {str(e)}")
            raise
        
        # Enhanced section patterns with variations
        self.section_patterns = {
            'summary': [
                r'(?i)^(?:summary|profile|professional\s+summary|career\s+summary|objective)$',
                r'(?i)^(?:about\s+me|about)$'
            ],
            'experience': [
                r'(?i)^(?:experience|work\s+experience|professional\s+experience|employment\s+history|work\s+history)$',
                r'(?i)^(?:career\s+history|employment)$'
            ],
            'education': [
                r'(?i)^(?:education|academic\s+background|qualifications|academic\s+qualifications)$',
                r'(?i)^(?:degrees|academic\s+history)$'
            ],
            'skills': [
                r'(?i)^(?:skills|technical\s+skills|core\s+competencies|expertise|proficiencies)$',
                r'(?i)^(?:technical\s+expertise|competencies)$'
            ],
            'certifications': [
                r'(?i)^(?:certifications|certificates|professional\s+certifications|accreditations)$',
                r'(?i)^(?:qualifications|professional\s+qualifications)$'
            ],
            'projects': [
                r'(?i)^(?:projects|personal\s+projects|portfolio|project\s+experience)$',
                r'(?i)^(?:project\s+work|project\s+history)$'
            ]
        }
        
        # Enhanced date patterns
        self.date_patterns = [
            re.compile(r'(?i)(?P<month>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)\s+(?P<year>\d{4})'),
            re.compile(r'(?i)(?P<year>\d{4})\s*-\s*(?P<month>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)'),
            re.compile(r'(?i)(?P<month>\d{1,2})/(?P<year>\d{4})'),
            re.compile(r'(?i)(?:present|current)'),
            re.compile(r'(?i)(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+(?P<year>\d{4})'),
            # Add additional date patterns
            re.compile(r'(?i)(?P<month>\d{1,2})/(?P<year>\d{2})'),  # MM/YY format
            re.compile(r'(?i)(?P<year>\d{4})\s*to\s*(?:present|current|now)'),  # YYYY to Present
            re.compile(r'(?i)(?P<month>\d{1,2})/(?P<year>\d{4})\s*-\s*(?:present|current|now)'),  # MM/YYYY - Present
            re.compile(r'(?i)(?P<month>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)\s+(?P<year>\d{4})\s*[-–—]\s*(?:present|current|now)'),  # Month YYYY - Present
            re.compile(r'(?i)(?P<year>\d{4})\s*[-–—]\s*(?P<year2>\d{4})'),  # YYYY-YYYY
            re.compile(r'(?i)(?P<month>(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})'),  # Month Day, YYYY
            re.compile(r'(?i)(?P<month>\d{1,2})-(?P<month2>\d{1,2})/(?P<year>\d{4})'),  # MM-MM/YYYY (date range in same year)
            re.compile(r'(?i)(?P<year>\d{4})[-–—/](?:present|current|now|ongoing)')  # YYYY-Present with various separators
        ]
        
        # Enhanced degree patterns
        self.degree_patterns = {
            'bachelor': [
                r'(?i)(?:b\.?s\.?|b\.?a\.?|bachelor(?:\'s)?\s+(?:of\s+)?(?:science|arts|engineering|business|technology|computer\s+science))',
                r'(?i)(?:b\.?e\.?|bachelor\s+of\s+engineering)',
                r'(?i)(?:b\.?tech\.?|bachelor\s+of\s+technology)'
            ],
            'master': [
                r'(?i)(?:m\.?s\.?|m\.?a\.?|m\.?e\.?|m\.?b\.?a\.?|master(?:\'s)?\s+(?:of\s+)?(?:science|arts|engineering|business|technology|computer\s+science))',
                r'(?i)(?:m\.?tech\.?|master\s+of\s+technology)',
                r'(?i)(?:m\.?sc\.?|master\s+of\s+science)'
            ],
            'phd': [
                r'(?i)(?:ph\.?d\.?|doctorate|doctor\s+of\s+philosophy)',
                r'(?i)(?:d\.?phil\.?|doctor\s+of\s+philosophy)'
            ],
            'associate': [
                r'(?i)(?:a\.?s\.?|a\.?a\.?|associate(?:\'s)?\s+(?:of\s+)?(?:science|arts))',
                r'(?i)(?:a\.?a\.?s\.?|associate\s+of\s+applied\s+science)'
            ]
        }
        
        # Enhanced contact patterns
        self.email_pattern = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+", re.IGNORECASE)
        self.phone_pattern = re.compile(r"(?:\+\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}")
        self.linkedin_pattern = re.compile(r"(?i)linkedin\.com/in/[\w-]+")
        
        # Skill categorization patterns
        self.technical_skill_patterns = [
            r'(?i)(?:programming|development|engineering|database|cloud|devops|testing|security)',
            r'(?i)(?:framework|library|tool|platform|system|software|hardware)'
        ]
        self.soft_skill_patterns = [
            r'(?i)(?:leadership|communication|management|problem\s+solving|teamwork|collaboration)',
            r'(?i)(?:analytical|creative|strategic|innovative|adaptable|flexible)'
        ]
        
        # Initialize skill taxonomies
        self.technical_skills = set()
        self.soft_skills = set()
        self.programming_languages = set()
    
    def parse_resume(self, text):
        """Parse the resume text into structured data."""
        if not text:
            return {}
            
        # Extract sections from the text
        sections = self._extract_sections(text)
        if not sections:
            self.logger.warning("No sections found in the text.")
            
        parsed_data = {}
        
        # Extract contact information
        contact_info = self._extract_contact_info(text)
        parsed_data["contact_info"] = contact_info
        
        # Extract summary
        summary = ""
        if "summary" in sections:
            summary = sections["summary"].get("content", "")
        parsed_data["summary"] = summary
        
        # Extract experience
        experience = []
        if "experience" in sections:
            experience = self._extract_experience(sections["experience"].get("content", ""))
        if not experience:
            # Try to extract experience from the full text if section-based extraction failed
            self.logger.info("No experience found in sections, attempting to extract from full text")
            experience = self._extract_experience_from_text(text)
        self.logger.info(f"Extracted experience entries: {len(experience)}")
        parsed_data["experience"] = experience
        
        # Extract education
        education = []
        if "education" in sections:
            education = self._extract_education(sections["education"].get("content", ""))
        self.logger.info(f"Extracted education entries: {len(education)}")
        parsed_data["education"] = education
        
        # Extract skills
        skills = {}
        if "skills" in sections:
            skills = self._extract_skills(sections["skills"].get("content", ""))
        if not any(skills.values()):
            # If no skills found in the skills section, try to extract from full text
            self.logger.info("No skills found in sections, attempting to extract from full text")
            skills = self._extract_skills_from_full_text(text)
        self.logger.info(f"Extracted skills categories: {len(skills)}")
        parsed_data["skills"] = skills
        
        # Extract certifications
        certifications = []
        if "certifications" in sections:
            certifications = self._extract_certifications(sections["certifications"].get("content", ""))
        self.logger.info(f"Extracted certifications: {len(certifications)}")
        parsed_data["certifications"] = certifications
        
        # Extract projects
        projects = []
        if "projects" in sections:
            projects = self._extract_projects(sections["projects"].get("content", ""))
        
        # If no projects found in the projects section, check if experience entries might be projects
        if not projects:
            project_entries = []
            
            # First, check if there's a "Projects" section in experience
            projects_in_exp = False
            projects_start_idx = -1
            
            # Loop through experience to find any that look like projects
            for i, entry in enumerate(experience):
                title = entry.get("title", "")
                if title == "Projects":
                    projects_in_exp = True
                    projects_start_idx = i
                    break
            
            # If we found a "Projects" marker, extract subsequent entries as projects
            if projects_in_exp and projects_start_idx >= 0:
                for i in range(projects_start_idx + 1, len(experience)):
                    if any(s in experience[i].get("title", "").lower() for s in ["education", "skills", "interests", "languages"]):
                        break  # Stop at next section
                        
                    project = {
                        "title": experience[i].get("title", ""),
                        "date": "",
                        "description": "",
                        "technologies": []
                    }
                    
                    # Extract date from title if possible
                    date_match = re.search(r'\b(20\d{2})\b', experience[i].get("title", ""))
                    if date_match:
                        project["date"] = date_match.group(1)
                    elif "company" in experience[i] and re.search(r'\b(20\d{2})\b', experience[i].get("company", "")):
                        date_match = re.search(r'\b(20\d{2})\b', experience[i].get("company", ""))
                        if date_match:
                            project["date"] = date_match.group(1)
                    
                    # Extract description from responsibilities
                    if "responsibilities" in experience[i]:
                        description_parts = []
                        tech_list = []
                        
                        for resp in experience[i].get("responsibilities", []):
                            if resp.startswith("Technologies:"):
                                tech_text = resp.replace("Technologies:", "").strip()
                                techs = re.split(r'[,;/|]', tech_text)
                                tech_list = [t.strip() for t in techs if t.strip()]
                            else:
                                description_parts.append(resp)
                        
                        project["description"] = "\n".join(description_parts)
                        project["technologies"] = tech_list
                    
                    projects.append(project)
            
            # If still no projects, look for individual project entries that weren't under a "Projects" section
            if not projects:
                for entry in experience:
                    # Check if this looks like a project (contains "project", "website", "app", etc.)
                    title = entry.get("title", "").lower()
                    if any(kw in title for kw in ["project", "website", "app", "system", "application", "portfolio"]):
                        project = {
                            "title": entry.get("title", ""),
                            "date": "",
                            "description": "",
                            "technologies": []
                        }
                        
                        # Extract date from title or company
                        date_match = re.search(r'\b(20\d{2})\b', entry.get("title", ""))
                        if date_match:
                            project["date"] = date_match.group(1)
                        elif "company" in entry and re.search(r'\b(20\d{2})\b', entry.get("company", "")):
                            date_match = re.search(r'\b(20\d{2})\b', entry.get("company", ""))
                            if date_match:
                                project["date"] = date_match.group(1)
                        
                        # Extract description and technologies from responsibilities
                        if "responsibilities" in entry:
                            description_parts = []
                            tech_list = []
                            
                            for resp in entry.get("responsibilities", []):
                                if resp.startswith("Technologies:"):
                                    tech_text = resp.replace("Technologies:", "").strip()
                                    techs = re.split(r'[,;/|]', tech_text)
                                    tech_list = [t.strip() for t in techs if t.strip()]
                                else:
                                    description_parts.append(resp)
                            
                            project["description"] = "\n".join(description_parts)
                            project["technologies"] = tech_list
                        
                        projects.append(project)
        
        self.logger.info(f"Extracted projects: {len(projects)}")
        parsed_data["projects"] = projects
        
        # Extract entities
        entities = self._extract_entities(text)
        self.logger.info(f"Extracted entities: {len(entities)}")
        parsed_data["entities"] = entities
        
        return parsed_data
    
    def _extract_sections(self, text):
        """Extract sections from the text based on the section patterns."""
        sections = {}
        
        # Define section patterns if not already defined
        if not hasattr(self, 'section_patterns'):
            self.section_patterns = {
                'summary': [r'(?:^|\n)(?:Summary|SUMMARY|Profile|PROFILE|About Me|ABOUT ME).*?(?=\n\s*(?:[A-Z][a-zA-Z\s]+:|\n\s*[A-Z][A-Z\s]+\n)|$)'],
                'experience': [r'(?:^|\n)(?:Experience|EXPERIENCE|Work\s+Experience|WORK\s+EXPERIENCE|Professional\s+Experience|Employment).*?(?=\n\s*(?:[A-Z][a-zA-Z\s]+:|\n\s*[A-Z][A-Z\s]+\n)|$)'],
                'education': [r'(?:^|\n)(?:Education|EDUCATION|Academic|ACADEMIC|Degrees|DEGREES).*?(?=\n\s*(?:[A-Z][a-zA-Z\s]+:|\n\s*[A-Z][A-Z\s]+\n)|$)'],
                'skills': [r'(?:^|\n)(?:Skills|SKILLS|Technical\s+Skills|TECHNICAL\s+SKILLS|Competencies|COMPETENCIES).*?(?=\n\s*(?:[A-Z][a-zA-Z\s]+:|\n\s*[A-Z][A-Z\s]+\n)|$)'],
                'projects': [r'(?:^|\n)(?:Projects|PROJECTS|Project\s+Experience|PROJECT\s+EXPERIENCE).*?(?=\n\s*(?:[A-Z][a-zA-Z\s]+:|\n\s*[A-Z][A-Z\s]+\n)|$)'],
                'certifications': [r'(?:^|\n)(?:Certifications|CERTIFICATIONS|Certificates|CERTIFICATES).*?(?=\n\s*(?:[A-Z][a-zA-Z\s]+:|\n\s*[A-Z][A-Z\s]+\n)|$)']
            }
        
        # First identify all section headers
        section_headers = []
        # More flexible section header pattern
        section_header_pattern = r'(?:^|\n)\s*((?:Summary|Experience|Work(?:\s+Experience)?|Professional(?:\s+Experience)?|Employment|Education|Academic|Degrees|Skills|Technical(?:\s+Skills)?|Competencies|Projects|Project(?:\s+Experience)?|Certifications|Certificates)(?:\s*:|\s*$|\n))'
        
        for match in re.finditer(section_header_pattern, text, re.IGNORECASE):
            header_text = match.group(1).strip()
            section_name = header_text.lower().rstrip(':')
            
            # Normalize section names
            if any(term in section_name for term in ['summary', 'profile', 'about']):
                normalized_name = 'summary'
            elif any(term in section_name for term in ['experience', 'work', 'professional', 'employment']):
                normalized_name = 'experience'
            elif any(term in section_name for term in ['education', 'academic', 'degrees']):
                normalized_name = 'education'
            elif any(term in section_name for term in ['skill', 'technical', 'competencies']):
                normalized_name = 'skills'
            elif any(term in section_name for term in ['project', 'portfolio']):
                normalized_name = 'projects'
            elif any(term in section_name for term in ['certification', 'certificate']):
                normalized_name = 'certifications'
            else:
                continue  # Skip unrecognized headers
            
            section_headers.append({
                'name': normalized_name,
                'position': match.start(1),
                'text': match.group(1)
            })
        
        # If no headers were found using regex, try a simpler approach with common section names
        if not section_headers:
            self.logger.warning("No section headers found using regex, trying simpler approach")
            common_sections = ['summary', 'experience', 'education', 'skills', 'projects', 'certifications']
            for section_name in common_sections:
                # Look for lines that contain just the section name (case-insensitive)
                pattern = r'(?:^|\n)\s*' + re.escape(section_name) + r'\s*(?:\:|\n|$)'
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    section_headers.append({
                        'name': section_name,
                        'position': match.start(),
                        'text': match.group(0).strip()
                    })
        
        # Sort by position in the document
        section_headers.sort(key=lambda x: x['position'])
        
        self.logger.debug(f"Found {len(section_headers)} section headers: {[h['name'] for h in section_headers]}")
        
        # Extract content between headers
        for i, header in enumerate(section_headers):
            start_pos = header['position'] + len(header['text'])
            end_pos = len(text)
            
            # If not the last header, end at the next header
            if i < len(section_headers) - 1:
                end_pos = section_headers[i + 1]['position']
            
            # Extract content
            content = text[start_pos:end_pos].strip()
            
            # Add to sections dictionary
            sections[header['name']] = {'content': content, 'start_idx': start_pos, 'end_idx': end_pos}
        
        # If we still have no sections, make a last-ditch effort by looking for keywords
        if not sections:
            self.logger.warning("No sections found after pattern matching, attempting keyword-based extraction")
            lines = text.split('\n')
            current_section = None
            section_content = []
            
            for line in lines:
                line_lower = line.lower()
                
                # Check if this line might be a section header
                if any(keyword in line_lower for keyword in ['summary', 'profile', 'about me']):
                    if current_section:
                        sections[current_section] = {'content': '\n'.join(section_content).strip()}
                    current_section = 'summary'
                    section_content = []
                elif any(keyword in line_lower for keyword in ['experience', 'work', 'professional', 'employment']):
                    if current_section:
                        sections[current_section] = {'content': '\n'.join(section_content).strip()}
                    current_section = 'experience'
                    section_content = []
                elif any(keyword in line_lower for keyword in ['education', 'academic', 'degrees']):
                    if current_section:
                        sections[current_section] = {'content': '\n'.join(section_content).strip()}
                    current_section = 'education'
                    section_content = []
                elif any(keyword in line_lower for keyword in ['skill', 'technical', 'competencies']):
                    if current_section:
                        sections[current_section] = {'content': '\n'.join(section_content).strip()}
                    current_section = 'skills'
                    section_content = []
                elif any(keyword in line_lower for keyword in ['project', 'portfolio']):
                    if current_section:
                        sections[current_section] = {'content': '\n'.join(section_content).strip()}
                    current_section = 'projects'
                    section_content = []
                elif any(keyword in line_lower for keyword in ['certification', 'certificate']):
                    if current_section:
                        sections[current_section] = {'content': '\n'.join(section_content).strip()}
                    current_section = 'certifications'
                    section_content = []
                elif current_section:
                    # Add line to current section
                    section_content.append(line)
            
            # Add the last section if any
            if current_section and section_content:
                sections[current_section] = {'content': '\n'.join(section_content).strip()}
        
        if not sections:
            self.logger.warning("No sections found in the text")
        else:
            self.logger.info(f"Extracted {len(sections)} sections: {list(sections.keys())}")
        
        return sections
    
    def _extract_contact_info(self, text):
        """Extract contact information from text using various methods."""
        if not text:
            return {}
        
        contact_info = {
            "name": None,
            "email": None,
            "phone": None,
            "linkedin": None,
            "github": None,
            "website": None,
            "location": None,
        }
        
        doc = self.nlp(text)
        
        # Extract name - try different approaches
        # First look for a standalone name on the first line
        lines = text.split('\n')
        if lines and len(lines[0].strip().split()) <= 5:
            potential_name = lines[0].strip()
            # Check if it looks like a name (no signs of email, phone, etc.)
            if not re.search(r'[@:\/\\\d]', potential_name) and len(potential_name) > 0:
                contact_info["name"] = potential_name
        
        # If name not found, try NLP
        if not contact_info["name"]:
            person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if person_entities:
                # Most likely the first PERSON entity is the name
                contact_info["name"] = person_entities[0]
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            contact_info["email"] = email_matches[0]
        
        # Extract phone numbers - handle multiple formats
        # Standard formats: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123-456-7890
        phone_patterns = [
            r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US/Canada: (123) 456-7890 or 123-456-7890
            r'\b\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # International: +1 (123) 456-7890
            r'\b\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # International: +1 123-456-7890
            r'\b\+\d{1,3}[-.\s]?\d{9,11}\b',  # International compact: +12134567890
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # Simple: 123-456-7890
        ]
        
        for pattern in phone_patterns:
            phone_matches = re.findall(pattern, text)
            if phone_matches:
                # Format the phone number consistently
                raw_phone = phone_matches[0]
                # Extract digits only
                digits = re.sub(r'\D', '', raw_phone)
                
                # Format based on length
                if len(digits) == 10:  # Standard US number
                    contact_info["phone"] = f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
                elif len(digits) > 10:  # International
                    if digits.startswith('1') and len(digits) == 11:  # US with country code
                        contact_info["phone"] = f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:11]}"
                    else:
                        # Keep original format for other international numbers
                        contact_info["phone"] = raw_phone
                else:
                    # Keep original if format is unclear
                    contact_info["phone"] = raw_phone
                break
        
        # Extract LinkedIn profile
        linkedin_patterns = [
            r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+(?:/)?',
            r'linkedin\.com/in/[A-Za-z0-9_-]+',
            r'linkedin:[A-Za-z0-9_-]+'
        ]
        
        for pattern in linkedin_patterns:
            linkedin_matches = re.findall(pattern, text.lower())
            if linkedin_matches:
                linkedin_url = linkedin_matches[0]
                # Ensure it has https:// prefix
                if not linkedin_url.startswith('http'):
                    if linkedin_url.startswith('linkedin:'):
                        username = linkedin_url.split(':')[1]
                        linkedin_url = f"https://www.linkedin.com/in/{username}"
                    elif not linkedin_url.startswith('www.'):
                        linkedin_url = f"https://{linkedin_url}"
                    else:
                        linkedin_url = f"https://{linkedin_url}"
                
                contact_info["linkedin"] = linkedin_url
                break
        
        # Extract GitHub profile
        github_patterns = [
            r'(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+(?:/)?',
            r'github\.com/[A-Za-z0-9_-]+',
            r'github:[A-Za-z0-9_-]+'
        ]
        
        for pattern in github_patterns:
            github_matches = re.findall(pattern, text.lower())
            if github_matches:
                github_url = github_matches[0]
                # Ensure it has https:// prefix
                if not github_url.startswith('http'):
                    if github_url.startswith('github:'):
                        username = github_url.split(':')[1]
                        github_url = f"https://github.com/{username}"
                    elif not github_url.startswith('www.'):
                        github_url = f"https://{github_url}"
                    else:
                        github_url = f"https://{github_url}"
                
                contact_info["github"] = github_url
                break
        
        # Extract website
        website_patterns = [
            r'(?:https?://)?(?:www\.)?[A-Za-z0-9][A-Za-z0-9-]{1,61}[A-Za-z0-9](?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?',
            r'[A-Za-z0-9][A-Za-z0-9-]{1,61}[A-Za-z0-9](?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?'
        ]
        
        for pattern in website_patterns:
            website_matches = re.findall(pattern, text.lower())
            if website_matches:
                for match in website_matches:
                    # Skip if it's LinkedIn, GitHub, or an email domain
                    if ('linkedin.com' in match or 'github.com' in match or 
                        any(domain in match for domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])):
                        continue
                    
                    website_url = match
                    # Ensure it has https:// prefix
                    if not website_url.startswith('http'):
                        website_url = f"https://{website_url}"
                    
                    contact_info["website"] = website_url
                    break
                
                if contact_info["website"]:
                    break
        
        # Extract location
        # Look for potential locations using NLP
        gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        if gpe_entities:
            # Check for city, state pattern
            for i in range(len(gpe_entities) - 1):
                city_state = f"{gpe_entities[i]}, {gpe_entities[i+1]}"
                if re.search(r'[A-Za-z\s-]+,\s+[A-Z]{2}', city_state):
                    contact_info["location"] = city_state
                    break
            
            # If no city, state pattern found, use the first location
            if not contact_info["location"]:
                contact_info["location"] = gpe_entities[0]
        
        # If no location found with NLP, try regex patterns
        if not contact_info["location"]:
            # Look for City, State pattern (e.g., New York, NY)
            location_pattern = r'([A-Z][a-zA-Z\s-]+),\s+([A-Z]{2})'
            location_matches = re.findall(location_pattern, text)
            if location_matches:
                city, state = location_matches[0]
                contact_info["location"] = f"{city}, {state}"
        
        # Clean up any HTML or excess whitespace
        for key, value in contact_info.items():
            if value and isinstance(value, str):
                # Remove HTML tags
                clean_value = re.sub(r'<[^>]+>', '', value)
                # Remove excess whitespace
                clean_value = re.sub(r'\s+', ' ', clean_value).strip()
                contact_info[key] = clean_value
        
        return {k: v for k, v in contact_info.items() if v}
    
    def _extract_experience_from_text(self, text):
        """
        Extract experience entries from resume text.
        
        Args:
            text (str): The resume text
            
        Returns:
            list: List of experience entries with company, title, date, and description
        """
        # Start timer for timeout protection
        start_time = time.time()
        max_processing_time = 5  # maximum 5 seconds for this function
        
        self.logger.info("Extracting experience entries from text")
        experience_entries = []
        
        # Find experience section
        experience_section = None
        sections = self._extract_sections(text)
        if sections and "experience" in sections:
            experience_section = sections["experience"]
            
        # Try to extract from dedicated experience section if found
        if experience_section:
            try:
                entries = self._extract_experience(experience_section)
                if entries:
                    experience_entries = entries
                    self.logger.info(f"Extracted {len(entries)} experience entries from section")
            except Exception as e:
                self.logger.error(f"Error extracting experience from section: {e}")
                # Fall back to full text extraction
                
        # If no entries were found or extraction failed, try whole text approach
        if not experience_entries:
            try:
                # Check for timeout
                if time.time() - start_time > max_processing_time:
                    self.logger.warning("Experience extraction timeout (when finding section)")
                    return []
                    
                self.logger.info("No experience found in sections, attempting to extract from full text")
                parsed_experience = []
                
                # Look for experience section headers
                exp_headers = ["experience", "work experience", "employment", "work history", "professional experience"]
                for header in exp_headers:
                    pattern = re.compile(rf'\b{re.escape(header)}\b', re.IGNORECASE)
                    match = pattern.search(text)
                    if match:
                        # Extract from the header to the next section header or end of text
                        start_idx = match.end()
                        next_section_match = re.search(r'\b(education|skills|projects|certifications|awards|references)\b', text[start_idx:], re.IGNORECASE)
                        if next_section_match:
                            end_idx = start_idx + next_section_match.start()
                            section_text = text[start_idx:end_idx]
                        else:
                            section_text = text[start_idx:]
                            
                        # Check for timeout
                        if time.time() - start_time > max_processing_time:
                            self.logger.warning("Experience extraction timeout (when processing section)")
                            return []
                            
                        # Try to extract entries from this section
                        try:
                            parsed_experience = self._extract_experience({"content": section_text})
                            if parsed_experience:
                                break  # Found valid entries, no need to try other headers
                        except Exception as e:
                            self.logger.error(f"Error extracting from section text: {e}")
                            
                if parsed_experience:
                    experience_entries = parsed_experience
                    self.logger.info(f"Extracted {len(parsed_experience)} experience entries from full text")
                else:
                    self.logger.warning("Could not find experience section in text")
            except Exception as e:
                self.logger.error(f"Error in full-text experience extraction: {e}")
                traceback.print_exc()
                
        self.logger.info(f"Extracted experience entries: {len(experience_entries)}")
        return experience_entries
    
    def _extract_education_from_text(self, text):
        """Extract education information from the entire text when a specific section is not found."""
        if not text:
            self.logger.info("No text provided for education extraction")
            return []
            
        # Try to find education section
        education_section = None
        
        # Common education section headers
        edu_headers = [
            r'(?:^|\n)(?:EDUCATION|Education|ACADEMIC BACKGROUND|Academic Background)(?:\s*:|\s*\n)',
            r'(?:^|\n)(?:EDUCATION AND TRAINING|Education and Training)(?:\s*:|\s*\n)',
            r'(?:^|\n)(?:EDUCATIONAL QUALIFICATIONS|Educational Qualifications)(?:\s*:|\s*\n)'
        ]
        
        # Try to find education section using patterns
        for pattern in edu_headers:
            matches = re.search(pattern, text)
            if matches:
                start_idx = matches.end()
                
                # Find the end of the section (next section header or end of text)
                next_section_match = re.search(r'(?:\n)(?:[A-Z][A-Z\s]+)(?:\s*:|\s*\n)', text[start_idx:])
                if next_section_match:
                    end_idx = start_idx + next_section_match.start()
                    education_section = text[start_idx:end_idx].strip()
                else:
                    # If no next section found, take the rest of the text
                    education_section = text[start_idx:].strip()
                    break
                            
        if not education_section:
            # Try another approach: look for degree-related keywords
            degree_keywords = [
                r'\b(?:Bachelor[\'s]*|Master[\'s]*|MBA|Ph\.?D\.?|Doctorate|Associate[\'s]*)\b',
                r'\b(?:B\.S\.|M\.S\.|B\.A\.|M\.A\.|B\.E\.|M\.E\.)\b',
                r'\b(?:University|College|Institute|School)\s+of\b'
            ]
            
            lines = text.split('\n')
            education_lines = []
            
            for i, line in enumerate(lines):
                if any(re.search(pattern, line, re.IGNORECASE) for pattern in degree_keywords):
                    # Found a line with education-related content
                    # Include this line and a few following lines
                    education_lines.append(line)
                    
                    # Add following lines until we encounter an empty line or another section
                    j = i + 1
                    while j < len(lines) and j < i + 10:  # Look at most 10 lines ahead
                        if not lines[j].strip():
                            break
                        if re.match(r'^[A-Z][A-Z\s]+:?$', lines[j]):  # Potential new section
                            break
                        education_lines.append(lines[j])
                        j += 1
            
            if education_lines:
                education_section = '\n'.join(education_lines)
        
        if not education_section:
            self.logger.warning("Could not find education section in text")
            return []
            
        # Now process the education section
        entries_text = self._split_education_entries(education_section)
        
        entries = []
        for entry_text in entries_text:
                    entry = self._parse_education_entry(entry_text)
                    if entry:
                        entries.append(entry)
                
        self.logger.info(f"Extracted {len(entries)} education entries from text")
        return entries
    
    def _extract_skills_from_text(self, text):
        """Extract skills from the entire text when a specific skills section is not found."""
        if not text:
            self.logger.info("No text provided for skills extraction")
            return {
                "programming_languages": [],
                "web_frameworks": [],
                "databases": [],
                "cloud_platforms": [],
                "tools": [],
                "soft_skills": []
            }
            
        # Try to find skills section
        skills_section = None
        
        # Common skills section headers
        skills_headers = [
            r'(?:^|\n)(?:SKILLS|Skills|TECHNICAL SKILLS|Technical Skills|CORE COMPETENCIES|Core Competencies)(?:\s*:|\s*\n)',
            r'(?:^|\n)(?:TECHNOLOGIES|Technologies|TOOLS|Tools|PROGRAMMING LANGUAGES|Programming Languages)(?:\s*:|\s*\n)'
        ]
        
        # Try to find skills section using patterns
        for pattern in skills_headers:
            matches = re.search(pattern, text)
            if matches:
                start_idx = matches.end()
                
                # Find the end of the section (next section header or end of text)
                next_section_match = re.search(r'(?:\n)(?:[A-Z][A-Z\s]+)(?:\s*:|\s*\n)', text[start_idx:])
                if next_section_match:
                    end_idx = start_idx + next_section_match.start()
                    skills_section = text[start_idx:end_idx].strip()
            else:
                    # If no next section found, take the rest of the text
                    skills_section = text[start_idx:].strip()
                    break
                
        if not skills_section:
            self.logger.warning("Could not find skills section in text")
            return {
                "programming_languages": [],
                "web_frameworks": [],
                "databases": [],
                "cloud_platforms": [],
                "tools": [],
                "soft_skills": []
            }
            
        # Extract skills by category
        return self._extract_skills_from_section(skills_section)
    
    def _extract_skills_from_full_text(self, text):
        """Extract skills from the entire text using a more accurate approach with pattern recognition."""
        if not text:
            return {
                "programming_languages": [],
                "web_frameworks": [],
                "databases": [],
                "cloud_platforms": [],
                "tools": [],
                "soft_skills": []
            }
            
        # Define common skills in each category
        categories = {
            "programming_languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php", 
                "swift", "kotlin", "scala", "rust", "perl", "html", "css", "sql", "r", "matlab"
            ],
            "web_frameworks": [
                "react", "angular", "vue", "django", "flask", "spring", "express", "node.js",
                "asp.net", "laravel", "rails", "fastapi", "next.js", "gatsby", "nuxt", "svelte"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "sql server", "oracle", "sqlite", "redis",
                "cassandra", "dynamodb", "mariadb", "firebase", "couchdb", "neo4j"
            ],
            "cloud_platforms": [
                "aws", "azure", "gcp", "google cloud", "alibaba cloud", "ibm cloud",
                "oracle cloud", "digitalocean", "heroku", "netlify", "vercel"
            ],
            "tools": [
                "git", "github", "docker", "kubernetes", "jenkins", "terraform", "ansible", "puppet",
                "chef", "prometheus", "grafana", "elasticsearch", "logstash", "kibana", "jira",
                "confluence", "slack", "gitlab", "bitbucket", "travis ci", "circleci"
            ],
            "soft_skills": [
                "communication", "teamwork", "leadership", "problem solving", "critical thinking",
                "time management", "adaptability", "creativity", "collaboration", "organization"
            ]
        }
        
        # Common words to exclude (false positives)
        excluded_words = [
            # Common resume words
            "summary", "experience", "education", "skills", "projects", "certifications",
            "work", "job", "career", "history", "professional", "profile", "objective", 
            "about", "contact", "references", "interests", "activities", "achievements",
            # Personal information words
            "name", "email", "phone", "address", "city", "state", "country", "zip", "postal",
            # Common verbs
            "worked", "developed", "implemented", "created", "managed", "designed", "built",
            "led", "coordinated", "analyzed", "improved", "increased", "decreased", "reduced",
            # Other common words
            "university", "college", "school", "degree", "bachelor", "master", "phd", "gpa",
            "years", "months", "dates", "present", "current", "junior", "senior", "lead",
            "inc", "llc", "corp", "company", "organization", "team", "department", "division"
        ]
        
        # Initialize skills dictionary
        skills = {
            "programming_languages": [],
            "web_frameworks": [],
            "databases": [],
            "cloud_platforms": [],
            "tools": [],
            "soft_skills": []
        }
        
        # Try to find skills section using patterns
        skills_section = None
        skills_patterns = [
            r'(?:skills|technical skills|core competencies)[:\n]+((?:.+\n)+)',
            r'(?:^|\n)skills[:\n]+((?:.+\n)+)'
        ]
        
        for pattern in skills_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                skills_section = match.group(1)
                break
        
        if skills_section:
            # Look for specific skill category patterns
            prog_lang_pattern = r'(?:programming\s+languages|languages|coding\s+languages)[:\s]+([^:\n]+)'
            frameworks_pattern = r'(?:web\s+frameworks|frameworks|libraries)[:\s]+([^:\n]+)'
            databases_pattern = r'(?:databases|db|data\s+storage)[:\s]+([^:\n]+)'
            cloud_pattern = r'(?:cloud|cloud\s+platforms)[:\s]+([^:\n]+)'
            tools_pattern = r'(?:tools|devops\s+tools)[:\s]+([^:\n]+)'
            soft_skills_pattern = r'(?:soft\s+skills|interpersonal\s+skills)[:\s]+([^:\n]+)'
            
            # Extract programming languages
            prog_lang_match = re.search(prog_lang_pattern, skills_section, re.IGNORECASE)
            if prog_lang_match:
                langs = re.split(r'[,;/|]', prog_lang_match.group(1))
                for lang in langs:
                    lang = lang.strip()
                    if lang and lang.lower() not in excluded_words and len(lang) > 1:
                        if lang not in skills["programming_languages"]:
                            skills["programming_languages"].append(lang)
            
            # Extract frameworks
            frameworks_match = re.search(frameworks_pattern, skills_section, re.IGNORECASE)
            if frameworks_match:
                frameworks = re.split(r'[,;/|]', frameworks_match.group(1))
                for framework in frameworks:
                    framework = framework.strip()
                    if framework and framework.lower() not in excluded_words and len(framework) > 1:
                        if framework not in skills["web_frameworks"]:
                            skills["web_frameworks"].append(framework)
            
            # Extract databases
            databases_match = re.search(databases_pattern, skills_section, re.IGNORECASE)
            if databases_match:
                dbs = re.split(r'[,;/|]', databases_match.group(1))
                for db in dbs:
                    db = db.strip()
                    if db and db.lower() not in excluded_words and len(db) > 1:
                        if db not in skills["databases"]:
                            skills["databases"].append(db)
            
            # Extract cloud platforms
            cloud_match = re.search(cloud_pattern, skills_section, re.IGNORECASE)
            if cloud_match:
                clouds = re.split(r'[,;/|]', cloud_match.group(1))
                for cloud in clouds:
                    cloud = cloud.strip()
                    if cloud and cloud.lower() not in excluded_words and len(cloud) > 1:
                        if cloud not in skills["cloud_platforms"]:
                            skills["cloud_platforms"].append(cloud)
            
            # Extract tools
            tools_match = re.search(tools_pattern, skills_section, re.IGNORECASE)
            if tools_match:
                tools = re.split(r'[,;/|]', tools_match.group(1))
                for tool in tools:
                    tool = tool.strip()
                    if tool and tool.lower() not in excluded_words and len(tool) > 1:
                        if tool not in skills["tools"]:
                            skills["tools"].append(tool)
            
            # Extract soft skills
            soft_skills_match = re.search(soft_skills_pattern, skills_section, re.IGNORECASE)
            if soft_skills_match:
                soft_skills = re.split(r'[,;/|]', soft_skills_match.group(1))
                for soft_skill in soft_skills:
                    soft_skill = soft_skill.strip()
                    if soft_skill and soft_skill.lower() not in excluded_words and len(soft_skill) > 1:
                        if soft_skill not in skills["soft_skills"]:
                            skills["soft_skills"].append(soft_skill)
        
        # If any category is empty, try to find skills throughout the text
        if not any(skills.values()):
            self.logger.info("No skills section found, extracting skills from full text")
            
            # Check each category
            for category, terms in categories.items():
                for skill in terms:
                    # Search for the skill with word boundaries to avoid partial matches
                    pattern = r'\b' + re.escape(skill) + r'\b'
                    if re.search(pattern, text, re.IGNORECASE):
                        if skill.capitalize() not in skills[category]:
                            skills[category].append(skill.capitalize())
        
        # Clean up and remove duplicates
        for category in skills:
            # Remove any items with newlines or other text that clearly isn't a skill
            skills[category] = [skill for skill in skills[category] if '\n' not in skill and len(skill) < 30]
            
            # Make sure we have unique items
            skills[category] = list(dict.fromkeys(skills[category]))
            
        return skills
    
    def _split_experience_entries(self, text):
        """Split experience text into individual entries."""
        if not text:
            return []
            
        # Common patterns for experience entries - job titles at beginning of line
        job_title_patterns = [
            r'(?:^|\n)(?:Senior|Lead|Principal|Junior|Associate)?\s*(?:Software|Web|Frontend|Backend|Full-Stack|DevOps|Data|ML|AI|Cloud|Mobile|Systems|QA|Test)?\s*(?:Engineer|Developer|Architect|Consultant|Manager|Analyst|Scientist|Designer|Administrator|Specialist|Lead|Director|Officer|Intern).*?\n',
            r'(?:^|\n)(?:[A-Z][a-z]+\s+)+(?:Engineer|Developer|Architect|Consultant|Manager|Analyst|Scientist|Designer|Administrator|Specialist|Lead|Director|Officer|Intern).*?\n',
            r'(?:^|\n)(?:[A-Z][a-z]+\s+)+(?:at|with|for)\s+(?:[A-Z][a-z]+\s*)+.*?\n'
        ]
        
        # Exclude education-related terms from job titles 
        education_terms = ["university", "college", "institute", "school", "academy", "bachelor", "master", "doctor", "phd", "mba", "ms", "ma", "bs", "ba", "degree", "education"]
        
        # Try each pattern
        for pattern in job_title_patterns:
            entries = []
            matches = re.findall(pattern, text + '\n', re.MULTILINE)
            
            if len(matches) > 1:
                # Found multiple job titles, use them to split the text
                positions = []
                for match in matches:
                    # Skip if it appears to be an education entry
                    if any(term in match.lower() for term in education_terms):
                        continue
                    
                    pos = text.find(match)
                    if pos != -1:
                        positions.append((pos, match))
                
                # Sort by position
                positions.sort(key=lambda x: x[0])
                
                if len(positions) > 0:
                    # Split text using positions
                    for i in range(len(positions)):
                        start_pos = positions[i][0]
                        end_pos = positions[i+1][0] if i < len(positions) - 1 else len(text)
                        entry = text[start_pos:end_pos].strip()
                        if entry:
                            entries.append(entry)
                    
                    return entries
        
        # If no good split found, try by paragraphs
        return self._split_experience_by_paragraphs(text)
        
    def _split_experience_by_paragraphs(self, text):
        """Split experience section by paragraphs."""
        if not text:
            return []
            
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Filter out empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _clean_section_text(self, text):
        """Clean and normalize section text."""
        if not text:
            return None
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove section headers that might be included
        text = re.sub(r'^(?:summary|profile|objective|about)(?:\s*:)?\s*', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def _extract_experience_from_section(self, section):
        """
        Extract experience entries from a single section.
        
        Args:
            section: Dictionary containing the section text and name
        
        Returns:
            List of experience entries
        """
        if not section or not section.get("text"):
            return []
        
        entries = []
        section_text = section.get("text", "")
        
        # Split into entries
        entry_texts = self._split_experience_entries(section_text)
        
        # Parse each entry
        for entry_text in entry_texts:
            parsed_entry = self._parse_experience_entry(entry_text)
            if parsed_entry:
                entries.append(parsed_entry)
        
        return entries
    
    def _extract_skills(self, section):
        """Extract skills from a section."""
        # Handle the case where section is a string
        if isinstance(section, str):
            section_text = section
        elif isinstance(section, dict) and "content" in section:
            section_text = section.get("content", "")
        else:
            self.logger.info("No skills section content found")
            return {
                "programming_languages": [],
                "web_frameworks": [],
                "databases": [],
                "cloud_platforms": [],
                "tools": [],
                "soft_skills": []
            }
        
        # Extract skills from the section text
        return self._extract_skills_from_section(section_text)
    
    def _extract_skills_from_section(self, section_text):
        """Extract skills from a section of text."""
        if not section_text:
            return {
                "programming_languages": [],
                "web_frameworks": [],
                "databases": [],
                "cloud_platforms": [],
                "tools": [],
                "soft_skills": []
            }
            
        # Initialize result structure
        skills = {
            "programming_languages": [],
            "web_frameworks": [],
            "databases": [],
            "cloud_platforms": [],
            "tools": [],
            "soft_skills": []
        }
        
        # Define common skills in each category
        categories = {
            "programming_languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php", 
                "swift", "kotlin", "scala", "rust", "perl", "html", "css", "sql", "r", "matlab"
            ],
            "web_frameworks": [
                "react", "angular", "vue", "django", "flask", "spring", "express", "node.js",
                "asp.net", "laravel", "rails", "fastapi", "next.js", "gatsby", "nuxt", "svelte"
            ],
            "databases": [
                "mysql", "postgresql", "mongodb", "sql server", "oracle", "sqlite", "redis",
                "cassandra", "dynamodb", "mariadb", "firebase", "couchdb", "neo4j"
            ],
            "cloud_platforms": [
                "aws", "azure", "gcp", "google cloud", "alibaba cloud", "ibm cloud",
                "oracle cloud", "digitalocean", "heroku", "netlify", "vercel"
            ],
            "tools": [
                "git", "docker", "kubernetes", "jenkins", "terraform", "ansible", "puppet",
                "chef", "prometheus", "grafana", "elasticsearch", "logstash", "kibana", "jira",
                "confluence", "slack", "gitlab", "bitbucket", "travis ci", "circleci"
            ]
        }
        
        # Extract all potential skills from the section text
        # First, split by common delimiters
        potential_skills = []
        
        # Handle different formats
        # Format 1: Category: Skill1, Skill2
        category_pattern = r'(?i)(.*?)(?::|,|-|•|\*)\s*(.*)'
        
        # Try to find categories within the text
        lines = section_text.split('\n')
        current_category = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a category line
            category_match = re.match(category_pattern, line)
            if category_match:
                potential_category = category_match.group(1).strip().lower()
                skill_text = category_match.group(2).strip()
                
                # Check if it matches one of our categories
                if potential_category in ["languages", "programming languages", "programming", "coding languages"]:
                    current_category = "programming_languages"
                    if skill_text:
                        # Split skills by common delimiters
                        skill_items = re.split(r'[,;/|•]', skill_text)
                        for skill in skill_items:
                            skill = skill.strip()
                            if skill:
                                potential_skills.append((skill.lower(), current_category))
                elif potential_category in ["frameworks", "web frameworks", "libraries", "web"]:
                    current_category = "web_frameworks"
                    if skill_text:
                        # Split skills by common delimiters
                        skill_items = re.split(r'[,;/|•]', skill_text)
                        for skill in skill_items:
                            skill = skill.strip()
                            if skill:
                                potential_skills.append((skill.lower(), current_category))
                elif potential_category in ["databases", "db", "data storage"]:
                    current_category = "databases"
                    if skill_text:
                        # Split skills by common delimiters
                        skill_items = re.split(r'[,;/|•]', skill_text)
                        for skill in skill_items:
                            skill = skill.strip()
                            if skill:
                                potential_skills.append((skill.lower(), current_category))
                elif potential_category in ["cloud", "cloud platforms", "cloud services"]:
                    current_category = "cloud_platforms"
                    if skill_text:
                        # Split skills by common delimiters
                        skill_items = re.split(r'[,;/|•]', skill_text)
                        for skill in skill_items:
                            skill = skill.strip()
                            if skill:
                                potential_skills.append((skill.lower(), current_category))
                elif potential_category in ["tools", "devops tools", "software tools"]:
                    current_category = "tools"
                    if skill_text:
                        # Split skills by common delimiters
                        skill_items = re.split(r'[,;/|•]', skill_text)
                        for skill in skill_items:
                            skill = skill.strip()
                            if skill:
                                potential_skills.append((skill.lower(), current_category))
                elif potential_category in ["soft skills", "personal skills", "interpersonal"]:
                    current_category = "soft_skills"
                    if skill_text:
                        # Split skills by common delimiters
                        skill_items = re.split(r'[,;/|•]', skill_text)
                        for skill in skill_items:
                            skill = skill.strip()
                            if skill:
                                potential_skills.append((skill.lower(), current_category))
                else:
                    # If no specific category, try to determine from the content
                    # Split skills by common delimiters
                    skill_items = re.split(r'[,;/|•]', line)
                    for skill in skill_items:
                        skill = skill.strip()
                        if skill:
                            potential_skills.append((skill.lower(), None))
            else:
                # If it's not a category line but we have a current category
                if current_category:
                    # Split skills by common delimiters
                    skill_items = re.split(r'[,;/|•]', line)
                    for skill in skill_items:
                        skill = skill.strip()
                        if skill:
                            potential_skills.append((skill.lower(), current_category))
                else:
                    # If no category context, split by delimiters
                    skill_items = re.split(r'[,;/|•]', line)
                    for skill in skill_items:
                        skill = skill.strip()
                        if skill:
                            potential_skills.append((skill.lower(), None))
        
        # If we have skills without categories, categorize them based on our predefined lists
        for skill, category in potential_skills:
            if category:
                # If category is already determined, add to that category
                if category in skills:
                    # Add if not already there
                    if skill not in skills[category]:
                        skills[category].append(skill)
            else:
                # Try to determine category
                categorized = False
                for cat_name, cat_skills in categories.items():
                    # Check if the skill is in this category's list
                    if any(skill == s or skill in s or s in skill for s in cat_skills):
                        skills[cat_name].append(skill)
                        categorized = True
                        break
                
                # If not categorized, add to tools (could be a specific tool not in our list)
                if not categorized:
                    skills["soft_skills"].append(skill)
        
        # Remove duplicates and sort
        for category in skills:
            skills[category] = sorted(list(set(skills[category])))
            
        # Make first letter uppercase for better readability
        for category in skills:
            skills[category] = [skill.capitalize() for skill in skills[category]]
        
        return skills
    
    def _calculate_parsing_confidence(self, resume_data):
        """Calculate a confidence score for the parsed resume"""
        # Start with base confidence
        confidence = 0.5
        
        # Add points for each successfully parsed section
        if resume_data.get("contact_info", {}).get("email"):
            confidence += 0.1
            
        if resume_data.get("summary"):
            confidence += 0.1
            
        if resume_data.get("experience") and len(resume_data["experience"]) > 0:
            confidence += 0.1
            # Add bonus for detailed experience entries
            detailed_entries = sum(1 for exp in resume_data["experience"] 
                                 if exp.get("company") and exp.get("title") and exp.get("dates"))
            confidence += min(0.1, detailed_entries * 0.02)
            
        if resume_data.get("education") and len(resume_data["education"]) > 0:
            confidence += 0.1
            
        if resume_data.get("skills") and len(resume_data["skills"]) > 0:
            confidence += 0.1
            
        return min(1.0, confidence)
        
    def _create_empty_resume(self):
        """Create an empty resume structure with all fields initialized."""
        return {
            "entities": [],
            "contact_info": {
                "name": None,
                "email": None,
                "phone": None,
                "location": None,
                "website": None
            },
            "summary": None,
            "experience": [],
            "education": [],
            "skills": [],
            "certifications": [],
            "projects": [],
            "metadata": {
                "parser_version": "enhanced-v2.0",
                "parse_date": datetime.now().isoformat(),
                "parsing_confidence": 0.0,
                "section_coverage": {
                    "summary": 0,
                    "experience": 0,
                    "education": 0,
                    "skills": 0,
                    "contact": 0
                }
            }
        }
    
    def _extract_contact_info(self, text):
        """Extract contact information from text using various methods."""
        if not text:
            return {}
        
            contact_info = {
                "name": None,
                "email": None,
                "phone": None,
                "linkedin": None,
            "github": None,
            "website": None,
            "location": None,
        }
        
        doc = self.nlp(text)
        
        # Extract name - try different approaches
        # First look for a standalone name on the first line
        lines = text.split('\n')
        if lines and len(lines[0].strip().split()) <= 5:
            potential_name = lines[0].strip()
            # Check if it looks like a name (no signs of email, phone, etc.)
            if not re.search(r'[@:\/\\\d]', potential_name) and len(potential_name) > 0:
                contact_info["name"] = potential_name
        
        # If name not found, try NLP
        if not contact_info["name"]:
            person_entities = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
            if person_entities:
                # Most likely the first PERSON entity is the name
                contact_info["name"] = person_entities[0]
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            contact_info["email"] = email_matches[0]
        
        # Extract phone numbers - handle multiple formats
        # Standard formats: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123-456-7890
        phone_patterns = [
            r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # US/Canada: (123) 456-7890 or 123-456-7890
            r'\b\+\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',  # International: +1 (123) 456-7890
            r'\b\+\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # International: +1 123-456-7890
            r'\b\+\d{1,3}[-.\s]?\d{9,11}\b',  # International compact: +12134567890
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'  # Simple: 123-456-7890
        ]
        
        for pattern in phone_patterns:
            phone_matches = re.findall(pattern, text)
            if phone_matches:
                # Format the phone number consistently
                raw_phone = phone_matches[0]
                # Extract digits only
                digits = re.sub(r'\D', '', raw_phone)
                
                # Format based on length
                if len(digits) == 10:  # Standard US number
                    contact_info["phone"] = f"({digits[0:3]}) {digits[3:6]}-{digits[6:10]}"
                elif len(digits) > 10:  # International
                    if digits.startswith('1') and len(digits) == 11:  # US with country code
                        contact_info["phone"] = f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:11]}"
                    else:
                        # Keep original format for other international numbers
                        contact_info["phone"] = raw_phone
                else:
                    # Keep original if format is unclear
                    contact_info["phone"] = raw_phone
                    break
            
        # Extract LinkedIn profile
        linkedin_patterns = [
            r'(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9_-]+(?:/)?',
            r'linkedin\.com/in/[A-Za-z0-9_-]+',
            r'linkedin:[A-Za-z0-9_-]+'
        ]
        
        for pattern in linkedin_patterns:
            linkedin_matches = re.findall(pattern, text.lower())
            if linkedin_matches:
                linkedin_url = linkedin_matches[0]
                # Ensure it has https:// prefix
                if not linkedin_url.startswith('http'):
                    if linkedin_url.startswith('linkedin:'):
                        username = linkedin_url.split(':')[1]
                        linkedin_url = f"https://www.linkedin.com/in/{username}"
                    elif not linkedin_url.startswith('www.'):
                        linkedin_url = f"https://{linkedin_url}"
                    else:
                        linkedin_url = f"https://{linkedin_url}"
                
                contact_info["linkedin"] = linkedin_url
                break
                    
        # Extract GitHub profile
        github_patterns = [
            r'(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9_-]+(?:/)?',
            r'github\.com/[A-Za-z0-9_-]+',
            r'github:[A-Za-z0-9_-]+'
        ]
        
        for pattern in github_patterns:
            github_matches = re.findall(pattern, text.lower())
            if github_matches:
                github_url = github_matches[0]
                # Ensure it has https:// prefix
                if not github_url.startswith('http'):
                    if github_url.startswith('github:'):
                        username = github_url.split(':')[1]
                        github_url = f"https://github.com/{username}"
                    elif not github_url.startswith('www.'):
                        github_url = f"https://{github_url}"
                    else:
                        github_url = f"https://{github_url}"
                
                contact_info["github"] = github_url
                break
        
        # Extract website
        website_patterns = [
            r'(?:https?://)?(?:www\.)?[A-Za-z0-9][A-Za-z0-9-]{1,61}[A-Za-z0-9](?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?',
            r'[A-Za-z0-9][A-Za-z0-9-]{1,61}[A-Za-z0-9](?:\.[a-zA-Z]{2,})+(?:/[^\s]*)?'
        ]
        
        for pattern in website_patterns:
            website_matches = re.findall(pattern, text.lower())
            if website_matches:
                for match in website_matches:
                    # Skip if it's LinkedIn, GitHub, or an email domain
                    if ('linkedin.com' in match or 'github.com' in match or 
                        any(domain in match for domain in ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])):
                        continue
                    
                    website_url = match
                    # Ensure it has https:// prefix
                    if not website_url.startswith('http'):
                        website_url = f"https://{website_url}"
                    
                    contact_info["website"] = website_url
                    break
                
                if contact_info["website"]:
                    break
        
        # Extract location
        # Look for potential locations using NLP
        gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        if gpe_entities:
            # Check for city, state pattern
            for i in range(len(gpe_entities) - 1):
                city_state = f"{gpe_entities[i]}, {gpe_entities[i+1]}"
                if re.search(r'[A-Za-z\s-]+,\s+[A-Z]{2}', city_state):
                    contact_info["location"] = city_state
                    break
            
            # If no city, state pattern found, use the first location
            if not contact_info["location"]:
                contact_info["location"] = gpe_entities[0]
        
        # If no location found with NLP, try regex patterns
        if not contact_info["location"]:
            # Look for City, State pattern (e.g., New York, NY)
            location_pattern = r'([A-Z][a-zA-Z\s-]+),\s+([A-Z]{2})'
            location_matches = re.findall(location_pattern, text)
            if location_matches:
                city, state = location_matches[0]
                contact_info["location"] = f"{city}, {state}"
        
        # Clean up any HTML or excess whitespace
        for key, value in contact_info.items():
            if value and isinstance(value, str):
                # Remove HTML tags
                clean_value = re.sub(r'<[^>]+>', '', value)
                # Remove excess whitespace
                clean_value = re.sub(r'\s+', ' ', clean_value).strip()
                contact_info[key] = clean_value
        
        return {k: v for k, v in contact_info.items() if v}

    def _extract_education(self, section):
        """Extract education entries from a section."""
        # Handle the case where section is a string
        if isinstance(section, str):
            section_text = section
        elif isinstance(section, dict) and "content" in section:
            section_text = section.get("content", "")
        else:
            self.logger.info("No education section content found")
            return []
            
        entries = []
        
        if not section_text:
            self.logger.warning("Empty education section content")
            return []
        
        # Split education into entries
        entries_text = self._split_education_entries(section_text)
        
        # Parse each entry
        for entry_text in entries_text:
            entry = self._parse_education_entry(entry_text)
            if entry:
                entries.append(entry)
                
        self.logger.info(f"Extracted {len(entries)} education entries")
        return entries
        
    def _split_education_entries(self, text):
        """Split education section text into separate entries."""
        if not text:
            return []
        
        # Common patterns for education entries
        patterns = [
            # Degree name at beginning of line
            r'(?:^|\n)(?:Bachelor|Master|Doctor|Ph\.?D|MBA|B\.?S|M\.?S|B\.?A|M\.?A|Associate|Diploma|Certificate).*?\n',
            # University or college name at beginning of line
            r'(?:^|\n)(?:University|College|Institute|School|Academy) of.*?\n',
            # Years at beginning of line (for education)
            r'(?:^|\n)(?:19|20)\d{2}(?:\s*[-–]\s*(?:19|20)\d{2}|\s*[-–]\s*Present|\s*[-–]\s*current|\s*[-–]\s*ongoing).*?\n'
        ]
        
        # Look for university/college names specifically
        university_pattern = r'(?:^|\n)(?:[A-Z][a-z]+\s*)+(?:University|College|Institute|School|Academy).*?\n'
        university_matches = re.findall(university_pattern, text + '\n', re.MULTILINE | re.IGNORECASE)
        if university_matches:
            patterns.insert(0, university_pattern)  # Add to the beginning of patterns
        
        # Try to split by different patterns
        entries = []
        for pattern in patterns:
            splits = re.split(pattern, text, flags=re.IGNORECASE)
            
            # If we got more than one entry
            if len(splits) > 1:
                # First split is before first match, may be empty
                if splits[0].strip():
                    entries.append(splits[0].strip())
                
                # Recombine the matches with their content
                matches = re.findall(pattern, text, flags=re.IGNORECASE)
                for i, match in enumerate(matches):
                    if i < len(splits) - 1:
                        entry = match.strip() + '\n' + splits[i+1].strip()
                        entries.append(entry)
                
                # Break on first successful split pattern
                break
                
        # If no patterns worked, try paragraphs
        if not entries:
            entries = self._split_experience_by_paragraphs(text)
            
        # Remove very short entries that might be headers
        entries = [entry for entry in entries if len(entry.strip()) > 15]
            
        return entries
        
    def _parse_education_entry(self, entry_text):
        """Parse an education entry to extract degree, institution, dates, and other details."""
        if not entry_text or len(entry_text.strip()) < 10:
            return None
            
        entry_text = entry_text.strip()
        lines = entry_text.split('\n')
        
        education = {
            "degree": "",
            "institution": "",
            "start_date": "",
            "end_date": "",
            "gpa": "",
            "major": "",
            "minor": "",
            "courses": []
        }
        
        # First parse any structured format like "Degree at Institution | Date"
        first_line = lines[0].strip() if lines else ""
        
        # Parse first line with typical formats: 
        # - Degree at Institution
        # - Degree, Institution
        # - Degree | Institution
        # - Institution - Degree
        
        # First, extract year/date information
        year_pattern = r'(?<!\d)(?:19|20)\d{2}(?!\d)'
        date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+(\d{4})'
        year_range_pattern = r'(?<!\d)(\d{4})\s*(?:-|to|–|until)\s*((?:\d{4})|Present|Current|Now)(?!\d)'
        
        # Extract dates
        years = re.findall(year_pattern, first_line)
        if years:
            if len(years) == 1:
                education["end_date"] = years[0]
            elif len(years) == 2:
                education["start_date"] = years[0]
                education["end_date"] = years[1]
            else:
            # Check for month-year format
                date_match = re.search(date_pattern, first_line)
            if date_match:
                education["end_date"] = f"{date_match.group(1)} {date_match.group(2)}"
            
            # Check for year range
            year_range_match = re.search(year_range_pattern, first_line)
            if year_range_match:
                education["start_date"] = year_range_match.group(1)
                education["end_date"] = year_range_match.group(2)
        
        # Remove dates from first line for cleaner parsing
        date_free_line = re.sub(year_pattern, '', first_line)
        date_free_line = re.sub(date_pattern, '', date_free_line)
        date_free_line = re.sub(year_range_pattern, '', date_free_line)
        date_free_line = re.sub(r'\s*\|\s*', ' | ', date_free_line)  # Normalize separators
        date_free_line = re.sub(r'\s+-\s+', ' - ', date_free_line)
        date_free_line = re.sub(r'\s+,\s+', ', ', date_free_line)
        date_free_line = re.sub(r'\s+', ' ', date_free_line).strip()  # Normalize spaces
        
        # Try to match different patterns
        degree_institution_match = None
        
        # Pattern 1: Degree at Institution
        match = re.search(r'(.+?)\s+at\s+(.+)', date_free_line)
        if match:
            degree_institution_match = match
            education["degree"] = match.group(1).strip()
            education["institution"] = match.group(2).strip()
        
        # Pattern 2: Degree | Institution or Degree, Institution
        if not degree_institution_match:
            match = re.search(r'(.+?)(?:\s+\|\s+|,\s+)(.+)', date_free_line)
            if match:
                degree_institution_match = match
                part1 = match.group(1).strip()
                part2 = match.group(2).strip()
                
                # Determine which part is the degree and which is the institution
                degree_keywords = ["bachelor", "master", "phd", "doctor", "associate", "degree", 
                                 "bs", "ba", "ms", "ma", "mba", "b.s.", "m.s.", "b.a.", "m.a."]
                institution_keywords = ["university", "college", "institute", "school", "academy"]
                
                part1_lower = part1.lower()
                part2_lower = part2.lower()
                
                is_part1_degree = any(kw in part1_lower for kw in degree_keywords)
                is_part2_degree = any(kw in part2_lower for kw in degree_keywords)
                is_part1_institution = any(kw in part1_lower for kw in institution_keywords)
                is_part2_institution = any(kw in part2_lower for kw in institution_keywords)
                
                if is_part1_degree and is_part2_institution:
                    education["degree"] = part1
                    education["institution"] = part2
                elif is_part2_degree and is_part1_institution:
                    education["degree"] = part2
                    education["institution"] = part1
                elif is_part1_degree:
                    education["degree"] = part1
                    education["institution"] = part2
                elif is_part2_degree:
                    education["degree"] = part2
                    education["institution"] = part1
                elif is_part1_institution:
                    education["institution"] = part1
                    education["degree"] = part2
                elif is_part2_institution:
                    education["institution"] = part2
                    education["degree"] = part1
                else:
                    # Default assumption: first part is degree, second is institution
                    education["degree"] = part1
                    education["institution"] = part2
        
        # Pattern 3: Just a degree or institution (no separator)
        if not degree_institution_match and date_free_line:
            degree_keywords = ["bachelor", "master", "phd", "doctor", "associate", "degree", 
                             "bs", "ba", "ms", "ma", "mba", "b.s.", "m.s.", "b.a.", "m.a."]
            institution_keywords = ["university", "college", "institute", "school", "academy"]
            
            if any(kw in date_free_line.lower() for kw in degree_keywords):
                education["degree"] = date_free_line
            elif any(kw in date_free_line.lower() for kw in institution_keywords):
                education["institution"] = date_free_line
            else:
                # Default to institution if we can't determine
                education["institution"] = date_free_line
        
        # Extract GPA if present
        gpa_pattern = r'(?:gpa|grade point average)[:\s]+([0-4]\.\d+|[0-4])'
        for line in lines:
            gpa_match = re.search(gpa_pattern, line.lower())
            if gpa_match:
                education["gpa"] = gpa_match.group(1)
                break
        
        # Extract major/minor if present
        major_pattern = r'(?:major|concentration)[:\s]+([\w\s]+)'
        minor_pattern = r'(?:minor)[:\s]+([\w\s]+)'
        
        for line in lines:
            major_match = re.search(major_pattern, line.lower())
            if major_match:
                education["major"] = major_match.group(1).strip()
            
            minor_match = re.search(minor_pattern, line.lower())
            if minor_match:
                education["minor"] = minor_match.group(1).strip()
        
        # If degree is empty but institution contains degree info, try to extract it
        if not education["degree"] and education["institution"]:
            degree_keywords = ["Bachelor", "Master", "PhD", "Doctor", "Associate", "BS", "BA", "MS", "MA", "MBA", "B.S.", "M.S.", "B.A.", "M.A."]
            for keyword in degree_keywords:
                if keyword in education["institution"]:
                    # Try to extract the degree from the institution
                    parts = re.split(r'(?:\s+in\s+|\s+of\s+)', education["institution"], maxsplit=1)
                    if len(parts) > 1:
                        degree_part = parts[0]
                        if any(keyword in degree_part for keyword in degree_keywords):
                            education["degree"] = degree_part.strip()
                            education["institution"] = parts[1].strip()
                            break
        
        # Clean up any separators from degree and institution
        if education["degree"]:
            education["degree"] = re.sub(r'\s*\|\s*|\s*-\s*', ' ', education["degree"]).strip()
        if education["institution"]:
            education["institution"] = re.sub(r'\s*\|\s*|\s*-\s*', ' ', education["institution"]).strip()
        
        # Validate the entry
        if not education["institution"] and not education["degree"]:
            self.logger.warning("Could not extract institution or degree from education entry")
            return None
        
        return education

    def _extract_experience(self, section):
        """Extract experience entries from a section."""
        import time
        import logging
        logger = logging.getLogger(__name__)
        
        start_time = time.time()
        timeout_seconds = 3  # Set a 3 second timeout
        parsed_entries = []
        
        def check_timeout():
            """Check if processing has exceeded the timeout"""
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"Experience extraction approaching timeout after {timeout_seconds} seconds")
                raise TimeoutError(f"Experience extraction timed out after {timeout_seconds} seconds")
        
        try:
            # Check if input is a dictionary (from document processor) or string
            section_content = ""
            if isinstance(section, dict):
                section_content = section.get('content', '')
            elif isinstance(section, str):
                section_content = section
            else:
                logger.warning(f"Unexpected section type in _extract_experience: {type(section)}")
                return []
                
            # Log section size to help with debugging
            logger.info(f"Experience section length: {len(section_content)} characters")
            
            # Limit processing for extremely large sections
            if len(section_content) > 10000:
                section_content = section_content[:10000]
                logger.warning("Experience section too large, truncating to first 10000 characters")
            
            # Split into entries - try to find entry boundaries intelligently
            entries = []
            
            check_timeout()  # Check timeout before splitting
            
            # Different approaches to split the experience section
            # 1. Split by double newlines (common in most resumes)
            if '\n\n' in section_content:
                entries = [entry.strip() for entry in section_content.split('\n\n') if entry.strip()]
            # 2. Try to split by patterns like "Company Name (Date - Date)"
            elif not entries:
                try:
                    entries = re.split(r'\n(?=[A-Z][a-z]+\s+[A-Z][a-z]+|\w+\s+\w+\s+\(\d{4}\s*[-–—]\s*\d{4}|\w+\s+\w+\s+\|\s*\d{4})', section_content)
                    entries = [entry.strip() for entry in entries if entry.strip()]
                except Exception as e:
                    logger.error(f"Error during regex split: {str(e)}")
                    # Fall back to simpler splitting
                    entries = [entry.strip() for entry in section_content.split('\n') if entry.strip()]
            
            check_timeout()  # Check timeout after splitting
            
            # If unreasonable number of entries, re-split using a different method
            if len(entries) > 15 or len(entries) == 1:
                logger.info(f"Re-splitting experience section; found {len(entries)} entries initially")
                # Fallback: try to split by job titles or companies (capitalized lines)
                try:
                    entries = re.split(r'\n(?=[A-Z][A-Za-z\s,]+(?:\n|:))', section_content)
                    entries = [entry.strip() for entry in entries if entry.strip()]
                except Exception as e:
                    logger.error(f"Error during regex re-split: {str(e)}")
                    # Last resort: split by lines and group 3-4 lines together
                    lines = [line.strip() for line in section_content.split('\n') if line.strip()]
                    entries = []
                    for i in range(0, len(lines), 3):
                        entry = '\n'.join(lines[i:i+3])
                        if entry.strip():
                            entries.append(entry)
            
            # Cap number of entries to prevent excessive processing
            max_entries = 10
            if len(entries) > max_entries:
                logger.warning(f"Too many experience entries detected ({len(entries)}), limiting to {max_entries}")
                entries = entries[:max_entries]
            
            # Log the number of entries detected
            logger.info(f"Detected {len(entries)} experience entries")
            
            # Process each entry
            for entry in entries:
                check_timeout()  # Check for timeout before processing each entry
                
                # Skip if entry is too short to be meaningful
                if len(entry) < 10:
                    continue
                    
                # Parse the entry
                try:
                    parsed_entry = self._parse_experience_entry(entry)
                    if parsed_entry and parsed_entry.get('title') and (parsed_entry.get('company') or parsed_entry.get('date')):
                        parsed_entries.append(parsed_entry)
                except Exception as e:
                    logger.error(f"Error parsing experience entry: {str(e)}")
                    continue
            
            # Return the parsed entries
            return parsed_entries
            
        except TimeoutError as e:
            logger.error(f"Experience extraction timeout: {str(e)}")
            # Return whatever we've managed to parse so far
            return parsed_entries if parsed_entries else []
        except Exception as e:
            logger.error(f"Error in experience extraction: {str(e)}", exc_info=True)
            return []

    def _parse_experience_entry(self, entry_text):
        """Parse a single experience entry to extract job title, company, dates, and responsibilities."""
        if not entry_text or len(entry_text.strip()) < 10:
            return None
            
        entry_text = entry_text.strip()
        lines = entry_text.split('\n')
        
        experience = {
            "title": "",
            "company": "",
            "start_date": "",
            "end_date": "",
            "responsibilities": []
        }
        
        # Extract job title and company from the first line
        if lines:
            first_line = lines[0].strip()
            
            # First, look for and extract date patterns
            date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+(\d{4})\s*(?:-|to|–|until)\s*((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)[\s,]+\d{4}|Present|Current|Now)'
            year_pattern = r'(?<!\d)(\d{4})\s*(?:-|to|–|until)\s*((?:\d{4})|Present|Current|Now)(?!\d)'
            single_year_pattern = r'(?<!\d)(20\d{2}|19\d{2})(?!\d)'
            
            # Check for date patterns
            date_info = {}
            date_match = re.search(date_pattern, first_line, re.IGNORECASE)
            if date_match:
                date_info["start"] = f"{date_match.group(1)} {date_match.group(2)}"
                date_info["end"] = date_match.group(3)
                date_info["full_match"] = date_match.group(0)
            else:
                # Check for year range
                year_match = re.search(year_pattern, first_line)
                if year_match:
                    date_info["start"] = year_match.group(1)
                    date_info["end"] = year_match.group(2)
                    date_info["full_match"] = year_match.group(0)
                else:
                    # Check for single year
                    single_year_match = re.search(single_year_pattern, first_line)
                    if single_year_match:
                        date_info["end"] = single_year_match.group(1)
                        date_info["full_match"] = single_year_match.group(0)
            
            # Remove the date from the first line if found
            if date_info.get("full_match"):
                first_line = first_line.replace(date_info["full_match"], "").strip()
                # Clean up any resulting artifacts
                first_line = re.sub(r'\s*\|\s*$|\s*-\s*$|\s*,\s*$', '', first_line).strip()
                
                # Set the dates in the experience object
                if "start" in date_info:
                    experience["start_date"] = date_info["start"]
                if "end" in date_info:
                    experience["end_date"] = date_info["end"]
            
            # Now extract title and company from the cleaned first line
            # Look for common separators: "|", "-", "at", ","
            title_company_match = re.search(r'(.*?)(?:\s+\|\s+|\s+-\s+|\s+at\s+|\s+,\s+|\s+with\s+|\s+for\s+)(.*)', first_line)
            
            if title_company_match:
                part1 = title_company_match.group(1).strip()
                part2 = title_company_match.group(2).strip()
                
                # Try to determine which part is title and which is company
                # Titles often contain words like "developer", "engineer", "manager"
                title_keywords = ["developer", "engineer", "manager", "analyst", "lead", "director",
                                 "administrator", "specialist", "consultant", "architect", "designer",
                                 "coordinator", "officer", "head", "chief", "senior", "junior", "associate"]
                
                company_keywords = ["inc", "llc", "corp", "corporation", "company", "co", "group",
                                   "technologies", "solutions", "systems", "services", "limited"]
                
                part1_lower = part1.lower()
                part2_lower = part2.lower()
                
                is_part1_title = any(kw in part1_lower for kw in title_keywords)
                is_part2_company = any(kw in part2_lower for kw in company_keywords)
                
                if is_part1_title and is_part2_company:
                    experience["title"] = part1
                    experience["company"] = part2
                elif is_part1_title or not is_part2_company:
                    # If part1 has title keywords or part2 doesn't have company keywords
                    experience["title"] = part1
                    experience["company"] = part2
                else:
                    # Default: assume part1 is company, part2 is title
                    experience["company"] = part1
                    experience["title"] = part2
            else:
                # No clear separator, make a best guess
                company_keywords = ["inc", "llc", "corp", "corporation", "company", "co", "group",
                                   "technologies", "solutions", "systems", "services", "limited"]
                
                if any(kw in first_line.lower() for kw in company_keywords):
                    # This is likely just a company
                    experience["company"] = first_line
                else:
                    # Assume it's a job title
                    experience["title"] = first_line
        
        # If we didn't find dates in the first line, look in the second line
        if not experience["start_date"] and not experience["end_date"] and len(lines) > 1:
            second_line = lines[1].strip()
            
            # Check date patterns again
            date_match = re.search(date_pattern, second_line, re.IGNORECASE)
            if date_match:
                experience["start_date"] = f"{date_match.group(1)} {date_match.group(2)}"
                experience["end_date"] = date_match.group(3)
            else:
                # Check for year range
                year_match = re.search(year_pattern, second_line)
                if year_match:
                    experience["start_date"] = year_match.group(1)
                    experience["end_date"] = year_match.group(2)
                else:
                    # Check for single year
                    single_year_match = re.search(single_year_pattern, second_line)
                    if single_year_match:
                        experience["end_date"] = single_year_match.group(1)
            
            # If the second line is just a date or contains a company, extract it
            second_line_without_date = second_line
            if date_match:
                second_line_without_date = second_line.replace(date_match.group(0), "").strip()
            elif year_match:
                second_line_without_date = second_line.replace(year_match.group(0), "").strip()
            elif single_year_match:
                second_line_without_date = second_line.replace(single_year_match.group(0), "").strip()
            
            # If there's text left in the second line after removing the date, and we don't have a company yet,
            # set it as the company
            if second_line_without_date and not experience["company"]:
                experience["company"] = second_line_without_date
        
        # Extract responsibilities (usually bullet points)
        responsibility_lines = []
        in_responsibilities = False
        start_idx = 1  # Start from second line by default
        
        # If second line has date/company info, start from third line
        if len(lines) > 1 and (re.search(date_pattern, lines[1], re.IGNORECASE) or 
                              re.search(year_pattern, lines[1]) or 
                              experience["company"] == lines[1].strip()):
            start_idx = 2
        
        for line in lines[start_idx:]:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Check if line starts with bullet point or number
            if line.startswith(('•', '-', '*')) or re.match(r'^\d+\.', line):
                in_responsibilities = True
                # Remove the bullet point or number
                cleaned_line = re.sub(r'^[•\-*]\s*|^\d+\.\s*', '', line).strip()
                if cleaned_line:
                    responsibility_lines.append(cleaned_line)
            elif in_responsibilities:
                # Continuation of a previous bullet point
                responsibility_lines.append(line)
            elif not experience["company"] and not re.search(date_pattern, line, re.IGNORECASE) and not re.search(year_pattern, line):
                # If company not found yet and this isn't a date line, it might be the company
                experience["company"] = line
        
        # If no bullet points found, treat remaining lines as responsibilities
        if not responsibility_lines and len(lines) > start_idx + 1:
            for line in lines[start_idx:]:
                if line.strip() and not re.search(date_pattern, line, re.IGNORECASE) and not re.search(year_pattern, line):
                    responsibility_lines.append(line.strip())
        
        experience["responsibilities"] = responsibility_lines
        
        # Clean up the title and company fields
        # Remove date information if it got included by mistake
        if experience["title"]:
            # Remove date patterns
            experience["title"] = re.sub(date_pattern, '', experience["title"], flags=re.IGNORECASE).strip()
            experience["title"] = re.sub(year_pattern, '', experience["title"]).strip()
            experience["title"] = re.sub(single_year_pattern, '', experience["title"]).strip()
            
            # Remove trailing separators
            experience["title"] = re.sub(r'\s*\|\s*$|\s*-\s*$|\s*,\s*$', '', experience["title"]).strip()
            
            # Truncate if too long
            if len(experience["title"]) > 100:
                experience["title"] = experience["title"][:100].strip()
        
        if experience["company"]:
            # Remove date patterns
            experience["company"] = re.sub(date_pattern, '', experience["company"], flags=re.IGNORECASE).strip()
            experience["company"] = re.sub(year_pattern, '', experience["company"]).strip()
            experience["company"] = re.sub(single_year_pattern, '', experience["company"]).strip()
            
            # Remove trailing separators
            experience["company"] = re.sub(r'\s*\|\s*$|\s*-\s*$|\s*,\s*$', '', experience["company"]).strip()
            
            # Truncate if too long
            if len(experience["company"]) > 100:
                experience["company"] = experience["company"][:100].strip()
        
        # Validate the entry
        if not experience["title"] and not experience["company"]:
            self.logger.warning("Could not extract job title or company from experience entry")
            return None
        
        return experience

    def _extract_projects(self, section_content):
        """Extract projects from the projects section."""
        # Handle the case where section is a string or dictionary
        if isinstance(section_content, dict) and "content" in section_content:
            section_text = section_content.get("content", "")
        elif isinstance(section_content, str):
            section_text = section_content
        else:
            self.logger.info("No project section content to extract")
            return []
        
        self.logger.info(f"Projects section length: {len(section_text)}")
        
        # Split the content into project entries
        project_entries = self._split_project_entries(section_text)
        
        if not project_entries:
            self.logger.warning("No project entries found in the projects section")
            return []
        
        # Filter out entries that are too short (likely not valid projects)
        project_entries = [entry for entry in project_entries if len(entry) > 15]
        
        self.logger.info(f"Found {len(project_entries)} potential project entries")
        
        # If we have an unreasonable number of entries, something went wrong
        if len(project_entries) > 15:
            self.logger.warning(f"Found {len(project_entries)} project entries, which seems excessive. Using the first 15.")
            project_entries = project_entries[:15]
        
        # Parse each project entry
        projects = []
        for i, entry in enumerate(project_entries):
            try:
                project = self._parse_project_entry(entry)
                if project:
                    projects.append(project)
            except Exception as e:
                self.logger.error(f"Error parsing project entry {i}: {str(e)}")
        
        self.logger.info(f"Successfully extracted {len(projects)} projects")
        return projects
    
    def _split_project_entries(self, text):
        """Split projects section into individual project entries."""
        # Try several patterns to identify project entries
        entries = []
        
        # Pattern 1: Projects with a title and possibly a date
        pattern1 = r'(?:^|\n)(?:\d+\.\s*)?([A-Z][^:\n]+)(?:\s+\(([^)]+)\))?[:]\s*((?:.+\n)+)'
        matches = re.finditer(pattern1, text, re.MULTILINE)
        for match in matches:
            entries.append(match.group(0))
        
        # Pattern 2: Projects starting with "Project:"
        pattern2 = r'(?:^|\n)(?:Project|PROJECT):?\s+([^\n]+)(?:\s+\(([^)]+)\))?\s*((?:.+\n)*)'
        if not entries:
            matches = re.finditer(pattern2, text, re.MULTILINE)
            for match in matches:
                entries.append(match.group(0))
        
        # Pattern 3: Paragraphs separated by blank lines (fallback)
        if not entries:
            paragraphs = re.split(r'\n\s*\n', text)
            entries = [p.strip() for p in paragraphs if p.strip()]
        
        return entries
    
    def _parse_project_entry(self, entry):
        """Parse a project entry into structured data."""
        project = {
            "title": "",
            "date": "",
            "description": "",
            "technologies": []
        }
        
        # Try to extract the title
        title_pattern = r'(?:^|\n)(?:\d+\.\s*)?([A-Z][^:\n()]+(?: [^:\n()]+)*)'
        title_match = re.search(title_pattern, entry)
        if title_match:
            project["title"] = title_match.group(1).strip()
        else:
            # Try alternate pattern
            title_pattern2 = r'Project:?\s+([^\n(]+)'
            title_match2 = re.search(title_pattern2, entry, re.IGNORECASE)
            if title_match2:
                project["title"] = title_match2.group(1).strip()
            else:
                # Use the first line as the title
                first_line = entry.split('\n')[0].strip()
                if len(first_line) < 80:  # Reasonable length for a title
                    project["title"] = first_line
        
        # Try to extract date
        date_pattern = r'\(([^)]*\d{4}[^)]*)\)'
        date_match = re.search(date_pattern, entry)
        if date_match:
            project["date"] = date_match.group(1).strip()
        
        # Extract description (everything after the title and date)
        desc_pattern = r'(?:^|\n)(?:\d+\.\s*)?[A-Z][^:\n]+(?: [^:\n]+)*(?:\s+\([^)]+\))?:?\s*\n?((?:.+\n)+)'
        desc_match = re.search(desc_pattern, entry)
        if desc_match:
            project["description"] = desc_match.group(1).strip()
        else:
            # Fallback: use everything after the first line
            lines = entry.split('\n')
            if len(lines) > 1:
                project["description"] = '\n'.join(lines[1:]).strip()
            else:
                # If there's only one line, use the entire entry
                project["description"] = entry.strip()
        
        # Try to extract technologies
        tech_pattern = r'(?:Technologies|Tech Stack|Tools)(?:\s+used)?:?\s+([^\n.]+)'
        tech_match = re.search(tech_pattern, entry, re.IGNORECASE)
        if tech_match:
            technologies = re.split(r'[,;/|]', tech_match.group(1))
            project["technologies"] = [tech.strip() for tech in technologies if tech.strip()]
        else:
            # Try to identify technologies in the description
            common_tech_patterns = [
                r'using\s+([A-Za-z0-9+#]+(?:\s+and\s+[A-Za-z0-9+#]+)?)',
                r'with\s+([A-Za-z0-9+#]+(?:\s+and\s+[A-Za-z0-9+#]+)?)',
                r'built\s+(?:using|with)?\s+([A-Za-z0-9+#]+(?:\s+and\s+[A-Za-z0-9+#]+)?)',
                r'developed\s+(?:using|with)?\s+([A-Za-z0-9+#]+(?:\s+and\s+[A-Za-z0-9+#]+)?)'
            ]
            
            for pattern in common_tech_patterns:
                matches = re.finditer(pattern, project["description"], re.IGNORECASE)
                for match in matches:
                    tech = match.group(1).strip()
                    if tech and tech not in project["technologies"]:
                        project["technologies"].append(tech)
        
        # Validate we have at least a title or description
        if not project["title"] and not project["description"]:
            return None
            
        return project
    
    def _extract_technologies_from_text(self, text):
        """Extract technologies mentioned in text."""
        technologies = []
        # Common tech keywords
        tech_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|Ruby|PHP|Go|Rust|Swift|Kotlin)\b',
            r'\b(?:React|Angular|Vue|Node\.js|Express|Django|Flask|Spring|Laravel|ASP\.NET)\b',
            r'\b(?:MySQL|PostgreSQL|MongoDB|SQLite|Oracle|SQL Server|Redis|Cassandra)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Terraform|Jenkins|Git|GitHub|Jira)\b'
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            technologies.extend([match for match in matches if match.lower() not in [t.lower() for t in technologies]])
        
        return technologies
        
    def _extract_certifications(self, section_content):
        """Extract certifications from certifications section."""
        # Handle the case where section is a string or dictionary
        if isinstance(section_content, dict) and "content" in section_content:
            section_text = section_content.get("content", "")
        elif isinstance(section_content, str):
            section_text = section_content
        else:
            self.logger.info("No certification section content to extract")
            return []
        
        if not section_text.strip():
            return []
            
        # Split by lines and look for certification patterns
        lines = section_text.split('\n')
        certifications = []
        
        # Common certification patterns
        cert_patterns = [
            r'(?i)(?:certified|certificate in|certification in|certified in)\s+([\w\s]+)',
            r'(?i)([\w\s]+)\s+(?:certification|certificate)',
            r'(?i)([\w\s]+)\s+(?:certified|cert)'
        ]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to match certification patterns
            cert_name = None
            for pattern in cert_patterns:
                match = re.search(pattern, line)
                if match:
                    cert_name = match.group(1).strip()
                    break
            
            # If no pattern matched, use the entire line
            if not cert_name and len(line) < 100:  # Avoid using very long lines
                cert_name = line
            
            if cert_name:
                # Extract date if present
                date_match = re.search(r'(?:issued|completed|obtained|earned|in)\s+(\w+\s+\d{4}|\d{4})', line, re.IGNORECASE)
                date = date_match.group(1) if date_match else ""
                
                # Extract issuer if present
                issuer_match = re.search(r'(?:by|from|through)\s+([\w\s]+)', line, re.IGNORECASE)
                issuer = issuer_match.group(1).strip() if issuer_match else ""
                
                certifications.append({
                    "name": cert_name,
                    "date": date,
                    "issuer": issuer
                })
        
        return certifications

    def _extract_entities(self, text):
        """Extract named entities from the resume text using spaCy."""
        if not text:
            return []
            
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities 
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_
            })
            
        return entities