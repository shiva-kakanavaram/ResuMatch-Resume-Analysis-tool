import spacy
import re
import json
from typing import Dict, List, Tuple, Any, Optional
import logging
import datetime
from dateutil.parser import parse as parse_date
import PyPDF2
import pdfplumber
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import traceback

logger = logging.getLogger(__name__)

class ResumeParser:
    def __init__(self):
        # Load larger spaCy model for better context understanding
        self.nlp = spacy.load("en_core_web_md")
        
        # Enhanced section patterns with more variations
        self.section_patterns = {
            'summary': r'(?i)(?:summary|profile|objective|about\s+me|career\s+objective)',
            'experience': r'(?i)(?:experience|employment|work\s+history|professional\s+experience|work\s+experience|career\s+history)',
            'education': r'(?i)(?:education|academic\s+background|qualifications|academic\s+qualifications)',
            'skills': r'(?i)(?:skills|expertise|technologies|technical\s+skills|core\s+competencies)',
            'certifications': r'(?i)(?:certifications?|licenses?|professional\s+certifications?)',
            'projects': r'(?i)(?:projects?|portfolio|personal\s+projects?)',
            'awards': r'(?i)(?:awards?|honors?|achievements?|recognition)',
            'languages': r'(?i)(?:languages?|language\s+proficiency)',
            'publications': r'(?i)(?:publications?|papers?|research\s+publications?)',
            'references': r'(?i)(?:references?|professional\s+references?)'
        }
        
        # Initialize TF-IDF vectorizer for section detection
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            stop_words='english',
            max_features=1000
        )
        
        # Section header examples for training
        self.section_examples = {
            'summary': [
                "Professional Summary", "Career Objective", "About Me",
                "Executive Summary", "Profile Summary", "Career Profile"
            ],
            'experience': [
                "Work Experience", "Professional Experience", "Employment History",
                "Career History", "Work History", "Professional Background"
            ],
            'education': [
                "Education", "Academic Background", "Qualifications",
                "Educational Background", "Academic Qualifications"
            ],
            'skills': [
                "Skills", "Technical Skills", "Core Competencies",
                "Expertise", "Technical Expertise", "Professional Skills"
            ]
        }
        
        # Train the vectorizer on section examples
        self._train_section_detector()
        
        # Initialize skill taxonomy (keeping existing taxonomy for now)
        self.skill_taxonomy = {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'ruby', 'php', 'swift',
                'kotlin', 'go', 'rust', 'typescript', 'scala', 'perl'
            ],
            'web_frameworks': [
                'django', 'flask', 'fastapi', 'spring', 'react', 'angular', 'vue',
                'express', 'rails', 'laravel', 'asp.net'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
                'cassandra', 'oracle', 'sql server', 'sqlite'
            ],
            'cloud_platforms': [
                'aws', 'azure', 'gcp', 'heroku', 'digitalocean', 'openstack',
                'kubernetes', 'docker'
            ],
            'soft_skills': [
                'communication', 'leadership', 'teamwork', 'problem solving',
                'project management', 'time management', 'analytical', 'creativity'
            ]
        }
        
        # Enhanced proficiency patterns
        self.proficiency_patterns = {
            'expert': r'(?i)(expert|advanced|proficient|mastery|senior|lead|principal|architect)',
            'intermediate': r'(?i)(intermediate|familiar|working knowledge|mid-level|experienced)',
            'beginner': r'(?i)(beginner|basic|elementary|learning|junior|entry-level)'
        }

    def _train_section_detector(self):
        """Train the TF-IDF vectorizer on section examples"""
        all_examples = []
        for section, examples in self.section_examples.items():
            all_examples.extend(examples)
        self.vectorizer.fit(all_examples)
        self.section_vectors = self.vectorizer.transform(all_examples)
        self.section_labels = [section for section, examples in self.section_examples.items() 
                             for _ in examples]

    def _find_section_boundaries(self, text: str) -> Dict[str, Tuple[int, int]]:
        """Enhanced section boundary detection using ML and regex"""
        lines = text.split('\n')
        section_boundaries = {}
        
        # First pass: Use regex patterns
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            for section, pattern in self.section_patterns.items():
                if re.search(pattern, line):
                    section_boundaries[section] = (i, len(lines))
                    break
        
        # Second pass: Use ML-based detection for missing sections
        if not section_boundaries:
            # Transform the text into TF-IDF vectors
            text_vectors = self.vectorizer.transform([line.strip() for line in lines])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(text_vectors, self.section_vectors)
            
            # Find potential section headers
            for i, scores in enumerate(similarity_scores):
                max_score = np.max(scores)
                if max_score > 0.6:  # Threshold for section detection
                    section_idx = np.argmax(scores)
                    section = self.section_labels[section_idx]
                    if section not in section_boundaries:
                        section_boundaries[section] = (i, len(lines))
        
        # Sort sections by their start position
        section_boundaries = dict(sorted(section_boundaries.items(), 
                                       key=lambda x: x[1][0]))
        
        # Adjust section boundaries to not overlap
        sections = list(section_boundaries.items())
        for i in range(len(sections) - 1):
            current_section, (start, _) = sections[i]
            next_section, (next_start, _) = sections[i + 1]
            section_boundaries[current_section] = (start, next_start)
        
        # Set the end of the last section
        if sections:
            last_section, (start, _) = sections[-1]
            section_boundaries[last_section] = (start, len(lines))
        
        return section_boundaries

    def _extract_experience(self, text: str, section_boundaries: Dict[str, Tuple[int, int]]) -> List[Dict]:
        """Enhanced experience extraction with better parsing and validation"""
        experience = []
        
        if 'experience' not in section_boundaries:
            return experience
            
        start, end = section_boundaries['experience']
        experience_text = '\n'.join(text.split('\n')[start:end])
        
        # Split into individual positions using multiple strategies
        positions = self._split_positions(experience_text)
        
        for position in positions:
            if not position.strip():
                continue
                
            # Extract position details using NLP
            position_data = self._extract_position_details(position)
            if position_data:
                experience.append(position_data)
        
        return experience

    def _split_positions(self, text: str) -> List[str]:
        """Split text into individual positions using multiple strategies"""
        positions = []
        
        # Strategy 1: Split by date patterns
        date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4}'
        date_splits = re.split(f'\n(?={date_pattern})', text)
        
        # Strategy 2: Split by bullet points
        bullet_pattern = r'^[\s•·\-*]'
        bullet_splits = []
        for split in date_splits:
            if re.search(bullet_pattern, split, re.MULTILINE):
                bullet_splits.extend(re.split(f'\n(?={bullet_pattern})', split))
            else:
                bullet_splits.append(split)
        
        # Strategy 3: Split by company names (using NLP)
        doc = self.nlp(text)
        company_splits = []
        current_split = []
        
        for sent in doc.sents:
            if any(ent.label_ == 'ORG' for ent in sent.ents):
                if current_split:
                    company_splits.append('\n'.join(current_split))
                current_split = [sent.text]
            else:
                current_split.append(sent.text)
        
        if current_split:
            company_splits.append('\n'.join(current_split))
        
        # Combine and deduplicate splits
        all_splits = set()
        for split in bullet_splits + company_splits:
            split = split.strip()
            if split and len(split.split('\n')) >= 2:  # Minimum 2 lines for a valid position
                all_splits.add(split)
        
        return list(all_splits)

    def _extract_position_details(self, position: str) -> Optional[Dict]:
        """Extract detailed position information using NLP"""
        doc = self.nlp(position)
        lines = position.split('\n')
        
        # Extract company and title using NLP
        company = None
        title = None
        
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                company = ent.text
                break
        
        # If no company found, try to extract from first line
        if not company and lines:
            first_line = lines[0].strip()
            # Try to find company after common separators
            company_match = re.search(r'(?:at|@|,)\s*([^,]+)(?:,|$)', first_line)
            if company_match:
                company = company_match.group(1).strip()
        
        # Extract title (usually before company)
        if company and first_line:
            title = first_line.split(company)[0].strip()
        elif first_line:
            title = first_line.strip()
        
        # Extract dates
        dates = []
        for line in lines:
            # Look for date patterns
            date_matches = re.findall(r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4}', line)
            dates.extend(date_matches)
        
        start_date = dates[0] if dates else ''
        end_date = dates[-1] if len(dates) > 1 else 'Present'
        
        # Extract responsibilities
        responsibilities = []
        for line in lines[1:]:  # Skip first line (title/company)
            line = line.strip()
            if not line:
                continue
            
            # Clean bullet points
            line = re.sub(r'^[•·\-*]\s*', '', line)
            if line:
                responsibilities.append(line)
        
        # Validate extracted data
        if not title or not company:
            return None
        
        return {
            'title': title,
            'company': company,
            'start_date': start_date,
            'end_date': end_date,
            'duration': self._calculate_duration(start_date, end_date),
            'responsibilities': responsibilities
        }
    
    def _extract_education(self, text: str, section_boundaries: Dict[str, Tuple[int, int]]) -> List[Dict]:
        """Extract education information from the resume text"""
        education = []
        
        if 'education' not in section_boundaries:
            return education
            
        start, end = section_boundaries['education']
        education_text = '\n'.join(text.split('\n')[start:end])
        
        # Split into individual degrees
        degrees = re.split(r'\n(?=\w)', education_text)
        
        for degree in degrees:
            if not degree.strip():
                continue
                
            lines = degree.strip().split('\n')
            if not lines:
                continue
                
            # First line usually contains degree and institution
            degree_line = lines[0]
            
            # Try to extract degree and institution
            degree_match = re.search(r'^(.*?)(?:from|at|,)?\s*([^,]+)(?:,|$)', degree_line)
            if degree_match:
                degree_name = degree_match.group(1).strip()
                institution = degree_match.group(2).strip()
            else:
                degree_name = degree_line
                institution = ''
                
            # Extract dates
            date_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4}'
            dates = re.findall(date_pattern, degree)
            
            graduation_date = dates[-1] if dates else ''
            
            # Extract GPA if present
            gpa_match = re.search(r'GPA:?\s*(\d+\.\d+)', degree)
            gpa = gpa_match.group(1) if gpa_match else None
            
            education.append({
                'degree': degree_name,
                'institution': institution,
                'graduation_date': graduation_date,
                'gpa': gpa
            })
            
        return education
        
    def _extract_skills(self, text: str, sections: Dict[str, Tuple[int, int]]) -> Dict[str, List[str]]:
        """Extract skills from the resume with improved categorization and detection"""
        skills_by_category = {
            'technical': [],
            'soft': [],
            'languages': [],
            'tools': [],
            'frameworks': [],
            'databases': [],
            'cloud': [],
            'methodologies': [],
            'other': []
        }
        
        # Define skill categories and their keywords
        skill_categories = {
            'technical': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
                'golang', 'rust', 'scala', 'perl', 'r', 'matlab', 'html', 'css', 'sql', 'nosql',
                'machine learning', 'deep learning', 'data science', 'artificial intelligence', 'ai', 'ml',
                'nlp', 'computer vision', 'cv', 'data mining', 'statistics', 'analytics', 'big data',
                'data visualization', 'etl', 'data modeling', 'data warehousing', 'data engineering',
                'full stack', 'frontend', 'backend', 'web development', 'mobile development',
                'devops', 'ci/cd', 'testing', 'qa', 'security', 'blockchain', 'iot', 'embedded systems',
                'networking', 'distributed systems', 'parallel computing', 'high performance computing',
                'algorithms', 'data structures', 'system design', 'architecture', 'microservices',
                'serverless', 'api', 'rest', 'graphql', 'soap', 'websockets', 'oauth', 'jwt',
                'authentication', 'authorization', 'encryption', 'hashing', 'cryptography'
            ],
            'soft': [
                'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking',
                'creativity', 'time management', 'organization', 'adaptability', 'flexibility',
                'collaboration', 'interpersonal', 'presentation', 'public speaking', 'negotiation',
                'conflict resolution', 'decision making', 'emotional intelligence', 'empathy',
                'customer service', 'client relations', 'mentoring', 'coaching', 'training',
                'project management', 'strategic planning', 'analytical thinking', 'attention to detail',
                'multitasking', 'prioritization', 'stress management', 'work ethic', 'self-motivation',
                'initiative', 'proactive', 'resourcefulness', 'resilience', 'persistence'
            ],
            'languages': [
                'english', 'spanish', 'french', 'german', 'italian', 'portuguese', 'russian',
                'mandarin', 'cantonese', 'japanese', 'korean', 'arabic', 'hindi', 'bengali',
                'punjabi', 'urdu', 'dutch', 'swedish', 'norwegian', 'danish', 'finnish',
                'greek', 'turkish', 'hebrew', 'polish', 'czech', 'slovak', 'hungarian',
                'romanian', 'bulgarian', 'serbo-croatian', 'ukrainian', 'thai', 'vietnamese',
                'indonesian', 'malay', 'tagalog', 'swahili', 'zulu', 'xhosa', 'afrikaans'
            ],
            'tools': [
                'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'trello', 'asana',
                'slack', 'microsoft teams', 'zoom', 'skype', 'notion', 'figma', 'sketch', 'adobe xd',
                'photoshop', 'illustrator', 'indesign', 'premiere pro', 'after effects', 'final cut pro',
                'visual studio', 'visual studio code', 'intellij', 'pycharm', 'webstorm', 'eclipse',
                'android studio', 'xcode', 'jupyter', 'rstudio', 'tableau', 'power bi', 'looker',
                'metabase', 'grafana', 'kibana', 'prometheus', 'datadog', 'new relic', 'splunk',
                'postman', 'swagger', 'soapui', 'jenkins', 'travis ci', 'circle ci', 'github actions',
                'ansible', 'puppet', 'chef', 'terraform', 'kubernetes', 'docker', 'vagrant'
            ],
            'frameworks': [
                'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt.js', 'gatsby', 'ember',
                'django', 'flask', 'fastapi', 'spring', 'spring boot', 'hibernate', 'rails',
                'laravel', 'symfony', 'express', 'nestjs', 'koa', 'meteor', 'phoenix',
                'asp.net', '.net core', 'xamarin', 'flutter', 'react native', 'ionic',
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy', 'scipy',
                'matplotlib', 'seaborn', 'plotly', 'dash', 'streamlit', 'hugging face',
                'bootstrap', 'tailwind', 'material-ui', 'semantic ui', 'bulma', 'foundation',
                'jquery', 'redux', 'mobx', 'vuex', 'rxjs', 'lodash', 'underscore', 'ramda'
            ],
            'databases': [
                'mysql', 'postgresql', 'mariadb', 'oracle', 'sql server', 'sqlite', 'mongodb',
                'couchdb', 'cassandra', 'redis', 'neo4j', 'dynamodb', 'cosmosdb', 'firebase',
                'elasticsearch', 'solr', 'hbase', 'bigtable', 'snowflake', 'redshift', 'bigquery',
                'vertica', 'teradata', 'db2', 'influxdb', 'timescaledb', 'clickhouse'
            ],
            'cloud': [
                'aws', 'amazon web services', 'ec2', 's3', 'lambda', 'dynamodb', 'rds', 'aurora',
                'azure', 'microsoft azure', 'azure functions', 'cosmos db', 'blob storage',
                'gcp', 'google cloud platform', 'google cloud functions', 'bigquery', 'cloud storage',
                'heroku', 'netlify', 'vercel', 'firebase', 'digital ocean', 'linode', 'vultr',
                'openstack', 'cloudflare', 'akamai', 'fastly', 'cdn'
            ],
            'methodologies': [
                'agile', 'scrum', 'kanban', 'lean', 'waterfall', 'extreme programming', 'xp',
                'test driven development', 'tdd', 'behavior driven development', 'bdd',
                'continuous integration', 'continuous delivery', 'continuous deployment', 'ci/cd',
                'devops', 'sre', 'site reliability engineering', 'gitflow', 'trunk based development',
                'pair programming', 'code review', 'mob programming', 'domain driven design', 'ddd',
                'microservices', 'service oriented architecture', 'soa', 'event driven architecture',
                'rest', 'graphql', 'grpc', 'soap', 'mvc', 'mvvm', 'mvp', 'clean architecture'
            ]
        }
        
        # Extract skills from the skills section if available
        if 'skills' in sections:
            start, end = sections['skills']
            skills_text = '\n'.join(text.split('\n')[start:end])
            
            # Extract skills from the skills section
            skills_list = self._extract_skills_from_text(skills_text)
            
            # Categorize skills
            for skill in skills_list:
                categorized = False
                for category, keywords in skill_categories.items():
                    if any(self._is_skill_match(skill, keyword) for keyword in keywords):
                        if skill not in skills_by_category[category]:
                            skills_by_category[category].append(skill)
                            categorized = True
                            break
                
                if not categorized:
                    if skill not in skills_by_category['other']:
                        skills_by_category['other'].append(skill)
        
        # Extract skills from the entire resume as a fallback
        if not any(skills_by_category.values()):
            logger.info("No skills found in skills section, extracting from entire resume")
            skills_list = self._extract_skills_from_text(text)
            
            # Categorize skills
            for skill in skills_list:
                categorized = False
                for category, keywords in skill_categories.items():
                    if any(self._is_skill_match(skill, keyword) for keyword in keywords):
                        if skill not in skills_by_category[category]:
                            skills_by_category[category].append(skill)
                            categorized = True
                            break
                
                if not categorized:
                    if skill not in skills_by_category['other']:
                        skills_by_category['other'].append(skill)
        
        # Try to extract skills from specific sections if we still don't have many
        if sum(len(skills) for skills in skills_by_category.values()) < 5:
            logger.info("Few skills found, trying to extract from specific sections")
            
            # Extract from summary section
            if 'summary' in sections:
                start, end = sections['summary']
                summary_text = '\n'.join(text.split('\n')[start:end])
                summary_skills = self._extract_skills_from_text(summary_text)
                
                for skill in summary_skills:
                    categorized = False
                    for category, keywords in skill_categories.items():
                        if any(self._is_skill_match(skill, keyword) for keyword in keywords):
                            if skill not in skills_by_category[category]:
                                skills_by_category[category].append(skill)
                                categorized = True
                                break
                    
                    if not categorized:
                        if skill not in skills_by_category['other']:
                            skills_by_category['other'].append(skill)
            
            # Extract from experience section
            if 'experience' in sections:
                start, end = sections['experience']
                experience_text = '\n'.join(text.split('\n')[start:end])
                experience_skills = self._extract_skills_from_text(experience_text)
                
                for skill in experience_skills:
                    categorized = False
                    for category, keywords in skill_categories.items():
                        if any(self._is_skill_match(skill, keyword) for keyword in keywords):
                            if skill not in skills_by_category[category]:
                                skills_by_category[category].append(skill)
                                categorized = True
                                break
                    
                    if not categorized:
                        if skill not in skills_by_category['other']:
                            skills_by_category['other'].append(skill)
        
        # Clean up the skills (remove duplicates, normalize)
        for category in skills_by_category:
            skills_by_category[category] = self._normalize_skills(skills_by_category[category])
        
        logger.info(f"Extracted {sum(len(skills) for skills in skills_by_category.values())} skills across {len(skills_by_category)} categories")
        return skills_by_category
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using multiple approaches"""
        skills = []
        
        # Extract skills from bullet points
        bullet_pattern = r'[•·\-\*\+◦‣⁃⦿⦾⁌⁍⧫⬝⬦⬧⬨⭘⭙]\s*([^•·\-\*\+◦‣⁃⦿⦾⁌⁍⧫\n]+)'
        bullet_matches = re.findall(bullet_pattern, text)
        for match in bullet_matches:
            # If the bullet point is short, it might be a skill
            if len(match.split()) <= 5:
                skills.append(match.strip())
        
        # Extract skills from comma-separated lists
        comma_pattern = r'(?:skills|technologies|tools|proficiencies|expertise)(?:[:\s]+)([^\.]+)'
        comma_matches = re.findall(comma_pattern, text, re.IGNORECASE)
        for match in comma_matches:
            # Split by commas and add each item
            items = [item.strip() for item in match.split(',')]
            skills.extend(items)
        
        # Extract skills from lines that look like skill lists
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # If line is short and contains technical terms, it might be a skill
            if 3 <= len(line.split()) <= 5 and not line.endswith('.'):
                skills.append(line)
        
        # Extract known programming languages and technologies
        tech_patterns = [
            r'\b(Python|Java|JavaScript|TypeScript|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go|Rust|Scala|Perl|R|MATLAB)\b',
            r'\b(HTML|CSS|SQL|NoSQL|React|Angular|Vue|Django|Flask|Spring|Rails|Laravel|Express|Node\.js)\b',
            r'\b(AWS|Azure|GCP|Docker|Kubernetes|Terraform|Jenkins|Git|GitHub|GitLab|Bitbucket|Jira|Confluence)\b',
            r'\b(MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch|Cassandra|Oracle|SQL Server|DynamoDB|Firestore)\b',
            r'\b(TensorFlow|PyTorch|Keras|scikit-learn|pandas|numpy|scipy|matplotlib|seaborn|Tableau|Power BI)\b'
        ]
        
        for pattern in tech_patterns:
            tech_matches = re.findall(pattern, text)
            skills.extend(tech_matches)
        
        # Clean up and deduplicate
        clean_skills = []
        for skill in skills:
            skill = skill.strip()
            # Remove punctuation at the end
            skill = re.sub(r'[,;\.]+$', '', skill)
            # Skip if too short or too long
            if len(skill) < 2 or len(skill) > 50:
                continue
            # Skip if it's a common word or phrase
            if skill.lower() in ['and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of']:
                continue
            # Add if not already in the list
            if skill and skill not in clean_skills:
                clean_skills.append(skill)
        
        return clean_skills
    
    def _is_skill_match(self, skill: str, keyword: str) -> bool:
        """Check if a skill matches a keyword, handling variations"""
        skill_lower = skill.lower()
        keyword_lower = keyword.lower()
        
        # Direct match
        if skill_lower == keyword_lower:
            return True
        
        # Skill contains keyword
        if keyword_lower in skill_lower:
            return True
        
        # Keyword contains skill
        if skill_lower in keyword_lower:
            return True
        
        # Handle plurals and similar variations
        if skill_lower.rstrip('s') == keyword_lower.rstrip('s'):
            return True
        
        # Handle abbreviations
        if (skill_lower.replace('.', '') == keyword_lower.replace('.', '') or
            skill_lower.replace(' ', '') == keyword_lower.replace(' ', '')):
            return True
        
        return False
    
    def _normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize and deduplicate skills"""
        normalized = []
        seen = set()
        
        for skill in skills:
            skill_lower = skill.lower()
            
            # Skip if we've seen this skill or a variation
            if skill_lower in seen:
                continue
            
            # Check if we've seen a variation
            skip = False
            for existing in normalized:
                if self._is_skill_match(skill, existing):
                    skip = True
                    break
            
            if not skip:
                normalized.append(skill)
                seen.add(skill_lower)
        
        return normalized
    
    def _calculate_section_scores(self, sections: Dict) -> Dict[str, float]:
        """Calculate scores for each section based on completeness and quality"""
        scores = {}
        
        # Contact Info Score
        contact_info = sections.get('contact_info', {})
        contact_score = 0
        if contact_info:
            # +20 for each key contact field
            if contact_info.get('name'):
                contact_score += 20
            if contact_info.get('email'):
                contact_score += 20
            if contact_info.get('phone'):
                contact_score += 20
            if contact_info.get('location'):
                contact_score += 20
            # +5 for each additional field
            for field in ['linkedin', 'github', 'website', 'portfolio']:
                if contact_info.get(field):
                    contact_score += 5
        scores['contact_info'] = min(100, contact_score)
        
        # Summary Score
        summary = sections.get('summary', '')
        summary_score = 0
        if summary:
            # Length-based score (up to 50 points)
            words = len(summary.split())
            if words >= 50:
                summary_score += 50
            else:
                summary_score += words
            
            # Content quality score (up to 50 points)
            quality_indicators = [
                'experience', 'skill', 'professional', 'background', 'expertise',
                'year', 'specialize', 'focus', 'passionate', 'committed',
                'develop', 'create', 'design', 'implement', 'manage',
                'lead', 'collaborate', 'team', 'project', 'success'
            ]
            
            quality_score = 0
            for indicator in quality_indicators:
                if indicator in summary.lower():
                    quality_score += 5
            
            summary_score += min(50, quality_score)
        scores['summary'] = min(100, summary_score)
        
        # Experience Score
        experience = sections.get('experience', [])
        experience_score = 0
        if experience:
            # Base score for having experience entries
            experience_score += min(30, len(experience) * 10)
            
            # Evaluate completeness of each entry
            completeness_score = 0
            for job in experience:
                entry_score = 0
                if job.get('title'):
                    entry_score += 5
                if job.get('company'):
                    entry_score += 5
                if job.get('start_date'):
                    entry_score += 5
                if job.get('end_date'):
                    entry_score += 5
                
                # Evaluate responsibilities
                responsibilities = job.get('responsibilities', [])
                if responsibilities:
                    # Score based on number of responsibilities (up to 10 points)
                    resp_count_score = min(10, len(responsibilities) * 2)
                    entry_score += resp_count_score
                    
                    # Score based on quality of responsibilities
                    resp_quality_score = 0
                    action_verbs = [
                        'developed', 'implemented', 'created', 'designed', 'managed',
                        'led', 'coordinated', 'achieved', 'improved', 'increased',
                        'reduced', 'negotiated', 'established', 'supervised', 'trained',
                        'analyzed', 'evaluated', 'researched', 'presented', 'authored'
                    ]
                    
                    for resp in responsibilities:
                        if any(verb in resp.lower() for verb in action_verbs):
                            resp_quality_score += 1
                    
                    entry_score += min(10, resp_quality_score)
                
                completeness_score += min(40, entry_score)
            
            experience_score += min(70, completeness_score)
        scores['experience'] = min(100, experience_score)
        
        # Education Score
        education = sections.get('education', [])
        education_score = 0
        if education:
            # Base score for having education entries
            education_score += min(40, len(education) * 20)
            
            # Evaluate completeness of each entry
            completeness_score = 0
            for edu in education:
                entry_score = 0
                if edu.get('degree'):
                    entry_score += 15
                if edu.get('field_of_study'):
                    entry_score += 10
                if edu.get('institution'):
                    entry_score += 15
                if edu.get('start_date') or edu.get('end_date'):
                    entry_score += 10
                if edu.get('gpa'):
                    entry_score += 10
                
                completeness_score += min(60, entry_score)
            
            education_score += min(60, completeness_score)
        scores['education'] = min(100, education_score)
        
        # Skills Score
        skills = sections.get('skills', {})
        skills_score = 0
        if skills:
            # Count total skills
            total_skills = sum(len(skills.get(category, [])) for category in skills)
            
            # Base score based on number of skills
            if total_skills >= 20:
                skills_score += 40
            else:
                skills_score += total_skills * 2
            
            # Score based on diversity of skill categories
            non_empty_categories = sum(1 for category in skills if skills.get(category, []))
            skills_score += min(30, non_empty_categories * 10)
            
            # Score based on relevant technical skills
            tech_skills = skills.get('technical', []) + skills.get('tools', []) + skills.get('frameworks', []) + skills.get('databases', []) + skills.get('cloud', [])
            if len(tech_skills) >= 10:
                skills_score += 30
            else:
                skills_score += len(tech_skills) * 3
        scores['skills'] = min(100, skills_score)
        
        return scores
    
    def parse_resume(self, input_data: str) -> Dict:
        """
        Parse resume from either a file path or direct text input
        
        Args:
            input_data: Either a file path (ending in .pdf or .txt) or the resume text directly
            
        Returns:
            Dict containing parsed resume sections
        """
        try:
            logger.info("Starting resume parsing")
            
            # Clean and normalize text
            text = self._clean_text(input_data)
            logger.info(f"Cleaned text length: {len(text)}")
            
            # Find section boundaries
            sections = self._find_section_boundaries_new(text)
            logger.info(f"Found sections: {list(sections.keys())}")
            
            # If no sections found, try to create artificial sections
            if not sections:
                logger.info("No sections found, creating artificial sections")
                sections = self._create_artificial_sections(text)
            
            # Extract sections
            experience = self._extract_experience_new(text, sections)
            education = self._extract_education_new(text, sections)
            skills = self._extract_skills_new(text, sections)
            
            # Extract contact information
            contact_info = self._extract_contact_info(text)
            
            # Extract summary if available or use empty string
            summary = self._extract_summary(text, sections) if 'summary' in sections else ""
            
            # Create structured resume data
            resume_data = {
                'contact_info': contact_info,
                'summary': summary,  # Always include summary field
                'experience': experience,
                'education': education,
                'skills': skills,
                'certifications': [],  # Add empty list for optional sections
                'projects': [],
                'awards': []
            }
            
            # Extract optional sections if available
            if 'certifications' in sections:
                resume_data['certifications'] = self._extract_certifications_new(text, sections)
                
            if 'projects' in sections:
                resume_data['projects'] = []  # Placeholder for project extraction
                
            if 'awards' in sections:
                resume_data['awards'] = []  # Placeholder for awards extraction
            
            logger.info(f"Parsed resume with {len(experience)} experience entries, {len(education)} education entries")
            
            return resume_data
            
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return empty structure in case of error
            return {
                'contact_info': {},
                'summary': "",  # Include empty summary field
                'experience': [],
                'education': [],
                'skills': {
                    'programming_languages': [],
                    'web_frameworks': [],
                    'databases': [],
                    'cloud_platforms': [],
                    'tools': [],
                    'soft_skills': []
                },
                'certifications': [],
                'projects': [],
                'awards': []
            }
            
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors
        text = text.replace('•', '-')  # Replace bullet points
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between camelCase
        
        # Normalize newlines
        text = text.replace('\r\n', '\n')
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Replace multiple newlines with double newline
        
        return text.strip()
    
    def _find_section_boundaries(self, text: str) -> Dict[str, Tuple[int, int]]:
        """Find the boundaries of each section in the resume text"""
        section_boundaries = {}
        
        for section, pattern in self.section_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = match.start()
                end = match.end()
                section_boundaries[section] = (start, end)
        
        return section_boundaries

    def _calculate_duration(self, start_date: str, end_date: str) -> str:
        """Calculate the duration between two dates"""
        try:
            # Handle 'Present' or similar values
            if end_date.lower() == 'present' or end_date.lower() == 'current' or end_date.lower() == 'now':
                end_date = datetime.datetime.now().strftime('%Y-%m')
            
            # Skip invalid dates or strings that aren't dates
            if not start_date or len(start_date) < 4 or not any(c.isdigit() for c in start_date):
                return ""
                
            if not end_date or len(end_date) < 4 or not any(c.isdigit() for c in end_date):
                end_date = datetime.datetime.now().strftime('%Y-%m')
            
            # Try to parse the dates
            try:
                start_date_obj = parse_date(start_date)
            except:
                # If we can't parse the start date, try extracting just the year
                year_match = re.search(r'\d{4}', start_date)
                if year_match:
                    start_date_obj = datetime.datetime(int(year_match.group(0)), 1, 1)
                else:
                    return ""
                    
            try:
                end_date_obj = parse_date(end_date)
            except:
                # If we can't parse the end date, try extracting just the year or use current date
                year_match = re.search(r'\d{4}', end_date)
                if year_match:
                    end_date_obj = datetime.datetime(int(year_match.group(0)), 12, 31)
                else:
                    end_date_obj = datetime.datetime.now()
            
            # Calculate duration
            duration = (end_date_obj.year - start_date_obj.year) * 12
            if hasattr(end_date_obj, 'month') and hasattr(start_date_obj, 'month'):
                duration += (end_date_obj.month - start_date_obj.month)
            
            return f"{duration} months"
        except Exception as e:
            logger.warning(f"Error calculating duration between {start_date} and {end_date}: {str(e)}")
            return ""
    
    def _extract_summary(self, text: str, sections: Dict[str, Tuple[int, int]]) -> str:
        """Extract summary section"""
        if 'summary' not in sections:
            return ""
            
        start, end = sections['summary']
        lines = text.split('\n')[start + 1:end]
        return ' '.join(line.strip() for line in lines if line.strip())

    def _find_section_boundaries_new(self, text: str) -> Dict[str, Tuple[int, int]]:
        """Find the start and end line numbers of each section in the resume using improved pattern matching"""
        lines = text.split('\n')
        logger.info(f"Analyzing {len(lines)} lines for section boundaries")
        
        # Initialize section boundaries
        section_boundaries = {}
        
        # Define section header patterns with more variations
        section_patterns = {
            'summary': r'(?i)(?:summary|profile|objective|about\s*me|personal\s*profile|professional\s*summary)',
            'experience': r'(?i)(?:experience|employment|work\s+history|professional\s+experience|career\s+history|work\s+experience|position|responsibilities)',
            'education': r'(?i)(?:education|academic|qualification|degree|educational\s*background|academic\s*background|college|university|school)',
            'skills': r'(?i)(?:skills|expertise|technologies|technical\s*skills|core\s*competencies|proficiencies|capabilities|tools|tech|languages|proficiencies)',
            'certifications': r'(?i)(?:certifications?|licenses?|credentials|qualifications)',
            'projects': r'(?i)(?:projects?|personal\s*projects?|academic\s*projects?|professional\s*projects?)',
            'awards': r'(?i)(?:awards?|honors?|achievements?|recognitions?)'
        }
        
        # First pass: look for common section header formats
        section_starts = []
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Skip empty lines
            if not line_stripped:
                continue
                
            # Check for various header formats
            is_header = False
            matched_section = None
            
            # Check if line is uppercase (like "EXPERIENCE")
            if line_stripped.isupper() and len(line_stripped) > 3 and len(line_stripped.split()) <= 3:
                is_header = True
            
            # Check if line has specific formatting (bold, larger font) - this would be detected in structured data
            # Not implemented here but would check doc_data['structure']['blocks'] in a full implementation
            
            # Check for section header patterns
            if is_header or len(line_stripped) < 30:  # Likely a header if short
                for section, pattern in section_patterns.items():
                    # Check if the line matches a section header pattern
                    if (re.match(f"^{pattern}[:\\s]*$", line_stripped, re.IGNORECASE) or
                        re.match(f"^{pattern}[:\\s]", line_stripped, re.IGNORECASE) or
                        line_stripped.lower() in [section.lower(), section.lower() + ':']):
                        
                        logger.info(f"Found section {section} at line {i}")
                        section_boundaries[section] = (i, None)
                        section_starts.append((i, section))
                        matched_section = section
                        break
            
            # If no exact match but line looks like a header (capitalized, short, etc.)
            if not matched_section and is_header:
                # Try fuzzy matching
                for section, pattern in section_patterns.items():
                    keywords = re.sub(r'[()\\?:|]', '', pattern).split('|')
                    for keyword in keywords:
                        keyword = re.sub(r'(?:\\s|\s)\*', ' ', keyword).strip()
                        if keyword and keyword.lower() in line_stripped.lower():
                            logger.info(f"Found section {section} at line {i} (fuzzy match)")
                            section_boundaries[section] = (i, None)
                            section_starts.append((i, section))
                            break
                    if section in section_boundaries:
                        break
        
        # Sort section starts by line number
        section_starts.sort(key=lambda x: x[0])
        
        # Set section end boundaries
        for i in range(len(section_starts)):
            start_line, section = section_starts[i]
            
            # If this is the last section, set end to end of document
            if i == len(section_starts) - 1:
                section_boundaries[section] = (start_line, len(lines))
            else:
                # Otherwise, set end to start of next section
                next_start_line = section_starts[i + 1][0]
                section_boundaries[section] = (start_line, next_start_line)
        
        # If no sections found, try more aggressive methods
        if not section_boundaries:
            logger.info("No standard sections found, trying more aggressive detection")
            section_boundaries = self._create_artificial_sections(text)
        
        # Ensure every section has both start and end defined
        for section, (start, end) in list(section_boundaries.items()):
            if end is None:
                section_boundaries[section] = (start, len(lines))
        
        return section_boundaries
    
    def _extract_experience_new(self, text: str, sections: Dict[str, Tuple[int, int]]) -> List[Dict]:
        """Extract experience section with improved handling of non-standard formats"""
        experience = []
        
        if 'experience' not in sections:
            logger.info("No experience section found")
            return experience
            
        start, end = sections['experience']
        lines = text.split('\n')[start:end]
        
        # Skip the header line
        if start + 1 < end:
            lines = lines[1:]
        
        # Join lines to process the text
        experience_text = '\n'.join(lines)
        logger.info(f"Extracting experience from {len(lines)} lines")
        
        # Check for pipe-separated format (common in modern resumes)
        # Example: "Senior Software Engineer | TechCorp Inc. | 2020-Present"
        pipe_format = False
        for line in lines[:5]:  # Check first few lines
            if '|' in line and len(line.split('|')) >= 2:
                pipe_format = True
                break
        
        if pipe_format:
            logger.info("Detected pipe-separated format for experience entries")
            current_job = None
            current_responsibilities = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check if this is a job header line (contains pipes)
                if '|' in line and len(line.split('|')) >= 2:
                    # If we were processing a job, save it before starting a new one
                    if current_job is not None:
                        current_job["responsibilities"] = current_responsibilities
                        experience.append(current_job)
                        current_responsibilities = []
                    
                    # Parse the pipe-separated line
                    parts = [part.strip() for part in line.split('|')]
                    
                    title = parts[0] if len(parts) > 0 else ""
                    company = parts[1] if len(parts) > 1 else ""
                    date_info = parts[2] if len(parts) > 2 else ""
                    
                    # Parse date information
                    start_date = ""
                    end_date = "Present"
                    if '-' in date_info:
                        date_parts = date_info.split('-')
                        start_date = date_parts[0].strip()
                        end_date = date_parts[1].strip() if len(date_parts) > 1 else "Present"
                    else:
                        start_date = date_info.strip()
                    
                    # Create new job entry
                    current_job = {
                        "title": title,
                        "company": company,
                        "start_date": start_date,
                        "end_date": end_date,
                        "duration": self._calculate_duration(start_date, end_date) if start_date else "",
                        "responsibilities": []
                    }
                else:
                    # This is a responsibility line
                    # Remove bullet points and other markers
                    clean_line = re.sub(r'^[•·\-\*\+◦‣⁃⦿⦾⁌⁍⧫⬝⬦⬧⬨⭘⭙]|\d+\.\s*', '', line).strip()
                    if clean_line and len(clean_line) > 5:
                        current_responsibilities.append(clean_line)
            
            # Add the last job if we have one
            if current_job is not None:
                current_job["responsibilities"] = current_responsibilities
                experience.append(current_job)
        else:
            # Now improve the pipe and non-pipe format detection in the experience extraction
            title_format_patterns = [
                r'(?i)(strong|bold)\s*>(.*?)</\1',  # HTML style
                r'(?i)\*\*(.*?)\*\*',               # Markdown style
                r'(?i)__(.*?)__'                    # Markdown style
            ]
            
            for i, line in enumerate(lines[:5]):  # Check first few lines
                for pattern in title_format_patterns:
                    if re.search(pattern, line):
                        logger.info(f"Detected formatted title in line: {line}")
                        pipe_format = False
                        break
            
            # Check for a date range format (common in traditional resumes)
            # Example: "Jan 2020 - Present" or "2018 - 2020"
            date_range_format = False
            for line in lines[:5]:  # Check first few lines
                if re.search(r'\d{4}\s*[-–—]\s*(?:\d{4}|[Pp]resent|[Cc]urrent)', line):
                    date_range_format = True
                    logger.info(f"Detected date range format in line: {line}")
                    break
                if re.search(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}', line, re.IGNORECASE):
                    date_range_format = True
                    logger.info(f"Detected month-year date range format in line: {line}")
                    break
            
            # Define patterns for experience extraction
            job_title_patterns = [
                r'(?i)(senior|junior|lead|principal|staff)?\s*(software|systems|data|frontend|backend|full\s*stack|web|mobile|cloud|devops|ml|ai)?\s*(engineer|developer|architect|scientist|analyst|consultant|manager|director|designer)',
                r'(?i)(project|product|program|technical|it|hr|marketing|sales|operations|finance)\s*(manager|lead|director|coordinator|specialist|analyst)',
                r'(?i)(intern|internship|co-op|trainee|apprentice)',
                r'(?i)(ceo|cto|cio|cfo|coo|vp|vice president|head of|chief)'
            ]
            
            company_indicators = [
                r'(?i)(?:at|@|with|for)\s+([A-Z][A-Za-z0-9\s\.\,\&\-]+)',
                r'(?i)([A-Z][A-Za-z0-9\s\.\,\&\-]+)(?:\s+Inc\.?|\s+LLC\.?|\s+Ltd\.?|\s+Corp\.?|\s+Corporation|\s+Company)'
            ]
            
            date_patterns = [
                r'(?i)(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s\.]+\d{4}',
                r'(?i)\d{2}/\d{2}/\d{4}',
                r'(?i)\d{2}-\d{2}-\d{4}',
                r'(?i)\d{4}',
                r'(?i)present'
            ]
            
            # First, try to identify job entries by looking for job titles
            potential_entries = []
            current_entry = {"lines": []}
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Check if this line looks like a job title
                is_job_title = False
                for pattern in job_title_patterns:
                    if re.search(pattern, line):
                        # If we already have an entry, save it
                        if current_entry["lines"]:
                            potential_entries.append(current_entry)
                        
                        # Start a new entry
                        current_entry = {"lines": [line], "title_line_index": 0}
                        is_job_title = True
                        break
                
                # If not a job title, check if it contains a date (possible job header)
                if not is_job_title:
                    has_date = False
                    for pattern in date_patterns:
                        if re.search(pattern, line):
                            has_date = True
                            break
                    
                    # If line has a date and looks like a header (short line)
                    if has_date and len(line.split()) < 10:
                        # If we already have an entry, save it
                        if current_entry["lines"]:
                            potential_entries.append(current_entry)
                        
                        # Start a new entry
                        current_entry = {"lines": [line], "title_line_index": 0}
                    else:
                        # Add to current entry
                        current_entry["lines"].append(line)
        
            # Add the last entry if not empty
            if current_entry["lines"]:
                potential_entries.append(current_entry)
            
            # If we couldn't identify separate entries, treat the entire section as one entry
            if not potential_entries:
                logger.info("Could not identify separate job entries, treating as one entry")
                potential_entries = [{"lines": lines, "title_line_index": 0}]
            
            # Process each potential entry
            for entry in potential_entries:
                entry_lines = entry["lines"]
                if not entry_lines:
                    continue
                
                job_entry = {
                    "title": "",
                    "company": "",
                    "start_date": "",
                    "end_date": "",
                    "duration": "",
                    "responsibilities": []
                }
                
                # Extract title and company from the first few lines
                title_candidates = []
                company_candidates = []
                dates = []
                
                # Look at the first 3 lines (or fewer if entry is shorter)
                header_lines = entry_lines[:min(3, len(entry_lines))]
                header_text = " ".join(header_lines)
                
                # Extract dates
                for pattern in date_patterns:
                    found_dates = re.findall(pattern, header_text)
                    if found_dates:
                        dates.extend(found_dates)
                
                # Extract job title
                for pattern in job_title_patterns:
                    matches = re.finditer(pattern, header_text, re.IGNORECASE)
                    for match in matches:
                        title_candidates.append(match.group(0))
                
                # Extract company
                for pattern in company_indicators:
                    matches = re.finditer(pattern, header_text, re.IGNORECASE)
                    for match in matches:
                        if match.groups():
                            company_candidates.append(match.group(1))
                
                # If no company found, try to extract from the first line
                if not company_candidates and len(header_lines) > 0:
                    first_line = header_lines[0]
                    # If the first line contains a comma, the part after the comma might be the company
                    if "," in first_line:
                        parts = first_line.split(",", 1)
                        if len(parts) > 1 and parts[1].strip():
                            company_candidates.append(parts[1].strip())
                
                # Set title, company, and dates
                if title_candidates:
                    job_entry["title"] = title_candidates[0]
                elif len(header_lines) > 0:
                    # If no title found but we have a header line, use it as title
                    job_entry["title"] = header_lines[0]
                
                if company_candidates:
                    job_entry["company"] = company_candidates[0]
                
                if dates:
                    job_entry["start_date"] = dates[0]
                    if len(dates) > 1:
                        job_entry["end_date"] = dates[-1]
                    else:
                        # If only one date found, it's likely the end date
                        job_entry["end_date"] = dates[0]
                        job_entry["start_date"] = ""
                    
                    # Calculate duration
                    try:
                        job_entry["duration"] = self._calculate_duration(job_entry["start_date"], job_entry["end_date"])
                    except:
                        job_entry["duration"] = ""
                
                # Extract responsibilities from the remaining lines
                responsibility_lines = entry_lines[min(3, len(entry_lines)):]
                
                # If we have very few lines, the entire entry might be a single responsibility
                if len(responsibility_lines) == 0 and len(entry_lines) == 1:
                    responsibility_lines = entry_lines
                
                for line in responsibility_lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Remove bullet points and other markers
                    clean_line = re.sub(r'^[•·\-\*\+◦‣⁃⦿⦾⁌⁍⧫⬝⬦⬧⬨⭘⭙]|\d+\.\s*', '', line).strip()
                    
                    # Skip lines that look like headers or dates
                    if any(re.search(pattern, clean_line) for pattern in date_patterns) and len(clean_line.split()) < 5:
                        continue
                        
                    # Skip very short lines
                    if len(clean_line.split()) < 3:
                        continue
                    
                    job_entry["responsibilities"].append(clean_line)
                
                # Only add entries that have some meaningful content
                if ((job_entry["title"] and len(job_entry["title"]) > 2) or 
                    (job_entry["company"] and len(job_entry["company"]) > 2)) and (
                     job_entry["responsibilities"] or job_entry["start_date"]):
                    experience.append(job_entry)
        
        # If we couldn't extract any experience entries but have text, create a single entry
        if not experience and experience_text.strip():
            logger.info("Creating a single experience entry from available text")
            
            # Try to extract at least a title
            title = ""
            for pattern in job_title_patterns:
                match = re.search(pattern, experience_text, re.IGNORECASE)
                if match:
                    title = match.group(0)
                    break
            
            # If we still don't have a title, try to extract something that looks like a job title
            if not title:
                # Look for capitalized text that may be a job title
                for line in lines[:10]:  # Check first few lines
                    if line.strip() and line[0].isupper() and len(line.split()) <= 5:
                        title = line.strip()
                        break
            
            experience.append({
                "title": title or "Unknown Position",
                "company": "",
                "start_date": "",
                "end_date": "",
                "duration": "",
                "responsibilities": [line.strip() for line in experience_text.split('\n') if line.strip()]
            })
        
        logger.info(f"Extracted {len(experience)} experience entries")
        return experience
    
    def _extract_education_new(self, text: str, sections: Dict[str, Tuple[int, int]]) -> List[Dict]:
        """Extract education section with improved handling of non-standard formats"""
        education = []
        
        if 'education' not in sections:
            logger.info("No education section found")
            return education
            
        start, end = sections['education']
        lines = text.split('\n')[start:end]
        
        # Skip the header line
        if start + 1 < end:
            lines = lines[1:]
        
        # Join lines to process the text
        education_text = '\n'.join(lines)
        logger.info(f"Extracting education from {len(lines)} lines")
        
        # Define patterns for education extraction
        degree_patterns = [
            r'(?i)(Bachelor|Master|Ph\.?D\.?|B\.S\.|M\.S\.|B\.A\.|M\.A\.|B\.Tech|M\.Tech|B\.E\.|M\.E\.|MBA|PGDM|Associate|Diploma)',
            r'(?i)(Bachelor\'s|Master\'s|Doctorate|Doctoral|Undergraduate|Graduate|Postgraduate)',
            r'(?i)(BS|MS|BA|MA|BSc|MSc|BEng|MEng|BBA|MBA) (?:in|of)? ([A-Za-z\s]+)',
            r'(?i)(Bachelor|Master) (?:of|in) ([A-Za-z\s]+)',
            r'(?i)(High School|Secondary School|College) (Diploma|Certificate)'
        ]
        
        institution_patterns = [
            r'(?i)(University|College|Institute|School|Academy) (?:of|at)? ([A-Za-z\s\,\.]+)',
            r'(?i)([A-Z][A-Za-z0-9\s\.\,\&\-]+) (University|College|Institute|School|Academy)',
            r'(?i)([A-Z][A-Za-z0-9\s\.\,\&\-]+ University|College|Institute|School|Academy)'
        ]
        
        date_patterns = [
            r'(?i)(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s\.]+\d{4}',
            r'(?i)\d{2}/\d{2}/\d{4}',
            r'(?i)\d{2}-\d{2}-\d{4}',
            r'(?i)\d{4}'
        ]
        
        gpa_patterns = [
            r'(?i)(?:GPA|CGPA)[:\s]*([0-9]\.[0-9]+)',
            r'(?i)(?:GPA|CGPA)[:\s]*([0-9])/([0-9])',
            r'(?i)(?:Grade|Score)[:\s]*([A-Z][+-]?)'
        ]
        
        # Try to identify education entries
        potential_entries = []
        current_entry = {"lines": []}
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line looks like a degree or institution
            is_new_entry = False
            for pattern in degree_patterns + institution_patterns:
                if re.search(pattern, line):
                    # If we already have an entry, save it
                    if current_entry["lines"]:
                        potential_entries.append(current_entry)
                    
                    # Start a new entry
                    current_entry = {"lines": [line], "header_line_index": 0}
                    is_new_entry = True
                    break
            
            # If not a new entry, check if it contains a date (possible education header)
            if not is_new_entry:
                has_date = False
                for pattern in date_patterns:
                    if re.search(pattern, line):
                        has_date = True
                        break
                
                # If line has a date and looks like a header (short line)
                if has_date and len(line.split()) < 10:
                    # If we already have an entry, save it
                    if current_entry["lines"]:
                        potential_entries.append(current_entry)
                    
                    # Start a new entry
                    current_entry = {"lines": [line], "header_line_index": 0}
                else:
                    # Add to current entry
                    current_entry["lines"].append(line)
        
        # Add the last entry if not empty
        if current_entry["lines"]:
            potential_entries.append(current_entry)
        
        # If we couldn't identify separate entries, treat the entire section as one entry
        if not potential_entries:
            logger.info("Could not identify separate education entries, treating as one entry")
            potential_entries = [{"lines": lines, "header_line_index": 0}]
        
        # Process each potential entry
        for entry in potential_entries:
            entry_lines = entry["lines"]
            if not entry_lines:
                continue
                
            edu_entry = {
                "degree": "",
                "field_of_study": "",
                "institution": "",
                "location": "",
                "start_date": "",
                "end_date": "",
                "gpa": ""
            }
            
            # Combine all lines for pattern matching
            entry_text = " ".join(entry_lines)
            
            # Extract degree and field of study
            for pattern in degree_patterns:
                matches = re.finditer(pattern, entry_text, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        edu_entry["degree"] = match.group(1)
                        if len(match.groups()) > 1 and match.group(2):
                            edu_entry["field_of_study"] = match.group(2)
                        break
            
            # If no field of study found, try to extract it separately
            if not edu_entry["field_of_study"]:
                field_patterns = [
                    r'(?i)(?:in|of) ([A-Za-z\s\&]+)',
                    r'(?i)(?:degree|major) (?:in|of) ([A-Za-z\s\&]+)'
                ]
                for pattern in field_patterns:
                    matches = re.search(pattern, entry_text)
                    if matches and matches.group(1):
                        edu_entry["field_of_study"] = matches.group(1).strip()
                        break
            
            # Extract institution
            for pattern in institution_patterns:
                matches = re.finditer(pattern, entry_text, re.IGNORECASE)
                for match in matches:
                    if match.groups():
                        if "University" in match.group(0) or "College" in match.group(0) or "Institute" in match.group(0) or "School" in match.group(0) or "Academy" in match.group(0):
                            edu_entry["institution"] = match.group(0)
                        elif len(match.groups()) > 1:
                            edu_entry["institution"] = f"{match.group(1)} {match.group(2)}"
                        else:
                            edu_entry["institution"] = match.group(1)
                        break
            
            # If no institution found, try to extract from the first line
            if not edu_entry["institution"] and entry_lines:
                # If the first line doesn't look like a degree, it might be an institution
                if not any(re.search(pattern, entry_lines[0], re.IGNORECASE) for pattern in degree_patterns):
                    edu_entry["institution"] = entry_lines[0]
            
            # Extract dates
            dates = []
            for pattern in date_patterns:
                found_dates = re.findall(pattern, entry_text)
                if found_dates:
                    dates.extend(found_dates)
            
            if dates:
                edu_entry["start_date"] = dates[0]
                if len(dates) > 1:
                    edu_entry["end_date"] = dates[-1]
                else:
                    # If only one date found, it's likely the end date
                    edu_entry["end_date"] = dates[0]
                    edu_entry["start_date"] = ""
            
            # Extract GPA
            for pattern in gpa_patterns:
                match = re.search(pattern, entry_text, re.IGNORECASE)
                if match:
                    if len(match.groups()) == 2:
                        edu_entry["gpa"] = f"{match.group(1)}/{match.group(2)}"
                    else:
                        edu_entry["gpa"] = match.group(1)
                    break
            
            # Extract location (if present)
            location_patterns = [
                r'(?i)(?:located in|located at|in) ([A-Za-z\s\,\.]+)',
                r'(?i)([A-Z][A-Za-z]+, [A-Z]{2})',  # City, State format
                r'(?i)([A-Z][A-Za-z]+, [A-Za-z]+)'  # City, Country format
            ]
            
            for pattern in location_patterns:
                match = re.search(pattern, entry_text, re.IGNORECASE)
                if match:
                    edu_entry["location"] = match.group(1).strip()
                    break
            
            # Only add entries that have some meaningful content
            if edu_entry["degree"] or edu_entry["institution"]:
                education.append(edu_entry)
        
        # If we couldn't extract any education entries but have text, create a single entry
        if not education and education_text.strip():
            logger.info("Creating a single education entry from available text")
            
            # Try to extract at least some information
            degree = ""
            institution = ""
            
            for pattern in degree_patterns:
                match = re.search(pattern, education_text, re.IGNORECASE)
                if match:
                    degree = match.group(0)
                    break
            
            for pattern in institution_patterns:
                match = re.search(pattern, education_text, re.IGNORECASE)
                if match:
                    institution = match.group(0)
                    break
            
            education.append({
                "degree": degree,
                "field_of_study": "",
                "institution": institution,
                "location": "",
                "start_date": "",
                "end_date": "",
                "gpa": ""
            })
        
        logger.info(f"Extracted {len(education)} education entries")
        return education
    
    def _extract_skills_new(self, text: str, sections: Dict[str, Tuple[int, int]]) -> Dict[str, List[str]]:
        """Extract skills from resume using improved pattern matching"""
        # Initialize skills dictionary with categories
        skills_by_category = {
            'programming_languages': [],
            'web_frameworks': [],
            'databases': [],
            'cloud_platforms': [],
            'tools': [],
            'soft_skills': []
        }
        
        # Get skills section if available
        if 'skills' in sections:
            start, end = sections['skills']
            lines = text.split('\n')[start:end]
            skills_text = '\n'.join(lines)
            logger.info(f"Found dedicated skills section with {len(lines)} lines")
        else:
            # If no skills section, scan the entire document
            logger.info("No dedicated skills section found, scanning entire document")
            skills_text = text
            
        # Extract skills using multiple methods
        
        # Method 1: Look for bullet points or comma-separated lists
        skill_lines = []
        for line in skills_text.split('\n'):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
                
            # Look for bullet point lists
            if re.match(r'^[-•*]\s+', line) or re.match(r'^\d+\.\s+', line):
                skill_lines.append(line)
                
            # Look for lines with multiple commas (likely skill lists)
            elif line.count(',') >= 2 and len(line) < 200:
                skill_lines.append(line)
                
            # Look for lines with skill keywords
            elif any(keyword in line.lower() for keyword in [
                'proficient', 'experience with', 'skilled in', 'knowledge of'
            ]):
                skill_lines.append(line)
                
        logger.info(f"Found {len(skill_lines)} potential skill lines")
        
        # Process each skill line
        all_skills = set()
        for line in skill_lines:
            # Remove bullets and numbering
            line = re.sub(r'^[-•*]\s+', '', line)
            line = re.sub(r'^\d+\.\s+', '', line)
            
            # Split by common separators
            if ',' in line:
                items = [item.strip() for item in line.split(',')]
            elif '|' in line:
                items = [item.strip() for item in line.split('|')]
            elif ';' in line:
                items = [item.strip() for item in line.split(';')]
            elif '•' in line:
                items = [item.strip() for item in line.split('•')]
            else:
                # Try to split based on capitalization patterns
                items = re.findall(r'[A-Z][a-z]+(?:\s+[a-z]+)*', line)
                if not items:
                    items = [line]
            
            # Add to all skills
            all_skills.update(item for item in items if item and len(item) > 2)
        
        # Method 2: Extract skills from entire document as backup
        logger.info("Extracting skills from entire document as backup")
        doc = self.nlp(skills_text)
        
        # Define common tech skills to look for
        common_tech_skills = [
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
            'golang', 'rust', 'scala', 'perl', 'r', 'matlab', 'html', 'css', 'sql', 'nosql',
            'machine learning', 'deep learning', 'data science', 'artificial intelligence', 'ai', 'ml',
            'nlp', 'computer vision', 'cv', 'data mining', 'statistics', 'analytics', 'big data',
            'data visualization', 'etl', 'data modeling', 'data warehousing', 'data engineering',
            'full stack', 'frontend', 'backend', 'web development', 'mobile development',
            'devops', 'ci/cd', 'testing', 'qa', 'security', 'blockchain', 'iot', 'embedded systems',
            'networking', 'distributed systems', 'parallel computing', 'high performance computing',
            'algorithms', 'data structures', 'system design', 'architecture', 'microservices',
            'serverless', 'api', 'rest', 'graphql', 'soap', 'websockets', 'oauth', 'jwt',
            'authentication', 'authorization', 'encryption', 'hashing', 'cryptography'
        ]
        
        # Look for these common skills in the text
        for skill in common_tech_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', skills_text.lower()):
                all_skills.add(skill.title())  # Add with proper capitalization
        
        # Look for skill-related noun phrases (more selective)
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip()
            # Only consider chunks that are likely to be skills (3-30 chars, no punctuation)
            if (3 <= len(chunk_text) <= 30 and 
                not any(char in chunk_text for char in ',.()[]{}:;"\'')):
                
                # Check if it contains any tech skill keywords
                if any(keyword in chunk_text.lower() for keyword in [
                    'programming', 'language', 'framework', 'database', 'cloud', 'platform',
                    'tool', 'software', 'development', 'engineering', 'design', 'analysis'
                ]):
                    all_skills.add(chunk_text)
        
        # Filter out common false positives
        false_positives = [
            'bachelor', 'master', 'degree', 'university', 'college', 'school',
            'resume', 'curriculum', 'vitae', 'cv', 'contact', 'email', 'phone',
            'address', 'city', 'state', 'zip', 'country', 'name', 'date', 'birth',
            'education', 'experience', 'work', 'job', 'position', 'title',
            'company', 'organization', 'employer', 'employee', 'manager', 'director',
            'year', 'month', 'day', 'hour', 'minute', 'second', 'time', 'date'
        ]
        
        filtered_skills = set()
        for skill in all_skills:
            skill_lower = skill.lower()
            if not any(fp in skill_lower for fp in false_positives):
                # Only keep skills between 3 and 30 characters
                if 3 <= len(skill) <= 30:
                    filtered_skills.add(skill)
        
        # Categorize skills
        for skill in filtered_skills:
            skill_lower = skill.lower()
            
            # Check against skill taxonomy
            categorized = False
            for category, keywords in self.skill_taxonomy.items():
                for keyword in keywords:
                    if keyword in skill_lower or skill_lower in keyword:
                        skills_by_category[category].append(skill)
                        categorized = True
                        break
                if categorized:
                    break
            
            # If not categorized, try to determine category
            if not categorized:
                if any(lang in skill_lower for lang in ['python', 'java', 'c++', 'javascript', 'typescript', 'ruby', 'go', 'rust', 'php']):
                    skills_by_category['programming_languages'].append(skill)
                elif any(fw in skill_lower for fw in ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'express', 'rails']):
                    skills_by_category['web_frameworks'].append(skill)
                elif any(db in skill_lower for db in ['sql', 'database', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle']):
                    skills_by_category['databases'].append(skill)
                elif any(cloud in skill_lower for cloud in ['aws', 'azure', 'gcp', 'cloud', 'docker', 'kubernetes']):
                    skills_by_category['cloud_platforms'].append(skill)
                elif any(soft in skill_lower for soft in ['communication', 'leadership', 'teamwork', 'problem solving', 'analytical']):
                    skills_by_category['soft_skills'].append(skill)
                else:
                    skills_by_category['tools'].append(skill)
                    
        # Remove duplicates and sort
        for category in skills_by_category:
            skills_by_category[category] = sorted(list(set(skills_by_category[category])))
            
            # Limit to top 10 skills per category to avoid noise
            if len(skills_by_category[category]) > 10:
                skills_by_category[category] = skills_by_category[category][:10]
        
        # Log results
        total_skills = sum(len(skills) for skills in skills_by_category.values())
        logger.info(f"Extracted {total_skills} skills across {len(skills_by_category)} categories")
        
        return skills_by_category
    
    def _categorize_skill(self, skill: str, skills: Dict[str, List[str]]) -> None:
        """Categorize a skill into the appropriate category"""
        skill_lower = skill.lower()
        
        # Technical skills
        if re.search(r'(?:java|python|javascript|typescript|html|css|sql|c\+\+|react|angular|vue|node|express|django|flask|spring|hibernate|docker|kubernetes|aws|azure|gcp)', skill_lower):
            if skill not in skills['technical']:
                skills['technical'].append(skill)
                logger.debug(f"Auto-categorized '{skill}' as technical skill")
            return
        
        # Soft skills
        if re.search(r'(?:leadership|communication|teamwork|problem[- ]solving|analytical|creative|innovative|management|organization|planning)', skill_lower):
            if skill not in skills['soft']:
                skills['soft'].append(skill)
                logger.debug(f"Auto-categorized '{skill}' as soft skill")
            return
        
        # Languages
        if re.search(r'(?:english|spanish|french|german|chinese|japanese|korean|arabic|hindi|tamil|telugu)', skill_lower):
            if skill not in skills['languages']:
                skills['languages'].append(skill)
                logger.debug(f"Auto-categorized '{skill}' as language")
            return
        
        # Default to tools if not categorized
        if skill not in skills['tools']:
            skills['tools'].append(skill)
            logger.debug(f"Auto-categorized '{skill}' as tool (default)")
    
    def _extract_skills_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from any text block"""
        skills = {
            'technical': [],
            'soft': [],
            'languages': [],
            'tools': []
        }
        
        # Split text into potential skill items
        lines = text.split('\n')
        for line in lines:
            # Skip empty lines
            if not line.strip():
                continue
                
            # Split by common separators
            parts = re.split(r'[,;•\-|/]', line)
            for part in parts:
                skill = part.strip()
                if skill and len(skill) >= 2:  # Skip very short items
                    self._categorize_skill(skill, skills)
        
        return skills
    
    def _extract_potential_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract potential skills from any text"""
        skills = {
            'technical': [],
            'soft': [],
            'languages': [],
            'tools': []
        }
        
        # Common skills patterns
        technical_pattern = r'(?:java|python|javascript|typescript|html|css|sql|c\+\+|react|angular|vue|node|express|django|flask|spring|hibernate|docker|kubernetes|aws|azure|gcp)'
        soft_pattern = r'(?:leadership|communication|teamwork|problem[- ]solving|analytical|creative|innovative|management|organization|planning)'
        language_pattern = r'(?:english|spanish|french|german|chinese|japanese|korean|arabic|hindi|tamil|telugu)'
        
        # Find all potential skills
        words = re.findall(r'\b\w+(?:[-\s]\w+)*\b', text.lower())
        for word in words:
            if re.search(technical_pattern, word):
                if word not in skills['technical']:
                    skills['technical'].append(word)
            elif re.search(soft_pattern, word):
                if word not in skills['soft']:
                    skills['soft'].append(word)
            elif re.search(language_pattern, word):
                if word not in skills['languages']:
                    skills['languages'].append(word)
        
        return skills
        
    def _extract_certifications_new(self, text: str, sections: Dict[str, Tuple[int, int]]) -> List[str]:
        """Extract certifications section"""
        certifications = []
        if 'certifications' not in sections:
            return certifications
            
        start, end = sections['certifications']
        lines = text.split('\n')[start + 1:end]
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('•'):
                certifications.append(line)
                
        return certifications

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                # Create PDF reader object
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                    
            return text
            
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return ""

    def _create_artificial_sections(self, text: str) -> Dict[str, Tuple[int, int]]:
        """Create artificial sections when standard section boundaries can't be detected"""
        lines = text.split('\n')
        section_boundaries = {}
        
        # Find potential experience-related content
        job_title_keywords = [
            'engineer', 'developer', 'manager', 'director', 'analyst'
        ]
        
        date_patterns = [
            r'\d{4}\s*-\s*(?:\d{4}|present)', 
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}'
        ]
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in job_title_keywords):
                # Check if next few lines contain dates
                context = ' '.join(lines[i:min(i+5, len(lines))])
                if any(re.search(pattern, context.lower()) for pattern in date_patterns):
                    section_boundaries['experience'] = (i, min(i + 15, len(lines)))
                    break
        
        # Find potential education-related content
        education_keywords = ['degree', 'university', 'college', 'bachelor', 'master']
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in education_keywords):
                section_boundaries['education'] = (i, min(i + 10, len(lines)))
                break
        
        # Find potential skills-related content
        skills_keywords = ['skills', 'proficient', 'expertise', 'technologies']
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in skills_keywords):
                section_boundaries['skills'] = (i, min(i + 10, len(lines)))
                break
        
        return section_boundaries

    def _extract_contact_info(self, text):
        """Extract contact information from resume text"""
        contact_info = {
            'name': None,
            'email': None,
            'phone': None,
            'location': None,
            'links': []
        }
        
        # Parse the text with NLP
        doc = self.nlp(text[:1000])  # Use first 1000 chars for efficiency
        
        # Extract entities
        for ent in doc.ents:
            if ent.label_ == 'PERSON' and not contact_info['name']:
                contact_info['name'] = ent.text
            elif ent.label_ == 'GPE' and not contact_info['location']:
                contact_info['location'] = ent.text
        
        # Extract email using regex
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        if emails:
            contact_info['email'] = emails[0]
        
        # Extract phone using regex
        phone_patterns = [
            r'\b(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',  # (123) 456-7890
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890
            r'\b\+\d{1,2}\s\d{3}\s\d{3}\s\d{4}\b'  # +1 123 456 7890
        ]
        
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
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
            links = re.findall(pattern, text, re.I)
            if links:
                contact_info['links'].append({
                    'platform': platform,
                    'url': links[0]
                })
        
        return contact_info
