import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import re
import logging
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
import numpy as np
from backend.timeout import Timeout

logger = logging.getLogger(__name__)

class EnhancedJobMatcher:
    def __init__(self):
        """Initialize the job matcher with models and configurations."""
        self.logger = logging.getLogger(__name__)
        self.bert_model = None
        self.tokenizer = None
        self.skill_categories = {
            'technical': {'weight': 0.8},
            'soft': {'weight': 0.6},
            'languages': {'weight': 0.4},
            'tools': {'weight': 0.7},
            'general': {'weight': 0.5}
        }
        
        # Try to load BERT model, with robust error handling
        try:
            # Only import if not already imported
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            # Check if CUDA is available
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using device: {device}")
            
            model_name = 'best_model.pt'
            
            # Attempt to load the model with a timeout
            # Use a 30 second timeout for model loading
            try:
                with Timeout(30):
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                    self.bert_model = AutoModel.from_pretrained(model_name).to(device)
                    
                self.logger.info("Successfully loaded BERT model for job matching")
                self.model_loaded = True
                self.device = device
            except TimeoutError:
                self.logger.error("Timeout while loading BERT model")
                self.model_loaded = False
        except ImportError as e:
            self.logger.error(f"Failed to import required modules for BERT: {str(e)}")
            self.model_loaded = False
        except Exception as e:
            self.logger.error(f"Error loading BERT model: {str(e)}")
            self.model_loaded = False
            
        # Initialize NLP for fallback
        try:
            import spacy
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except:
                self.logger.warning("Could not load spaCy model, downloading...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            self.logger.error(f"Error loading spaCy: {str(e)}")
            self.nlp = None
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Initialize PCA for dimensionality reduction
        self.pca = PCA(n_components=50)
        
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from text using pattern matching and NLP."""
        if not text:
            return []
            
        # Log the text length for debugging
        logger.info(f"Extracting skills from text of length: {len(text)}")
        
        # Common technical skills patterns
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|Ruby|PHP|Swift|Kotlin|Go|Rust)\b',
            r'\b(?:React|Angular|Vue|Node\.js|Django|Flask|Spring|Laravel)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins|Git)\b',
            r'\b(?:SQL|MongoDB|PostgreSQL|Redis|Elasticsearch)\b',
            r'\b(?:Machine Learning|Deep Learning|Data Science|AI|NLP)\b',
            r'\b(?:Project Management|Agile|Scrum|Kanban)\b',
            r'\b(?:UI/UX|Design|Figma|Adobe)\b',
            r'\b(?:Sales|Marketing|Business Development|Customer Success)\b',
            r'\b(?:Excel|Word|PowerPoint|Office)\b',
            r'\b(?:HTML|CSS|XML|JSON|YAML)\b',
            r'\b(?:Linux|Unix|Windows|MacOS)\b',
            r'\b(?:REST|API|SOAP|GraphQL)\b',
            r'\b(?:CI/CD|DevOps|Automation)\b',
            r'\b(?:Testing|QA|Quality Assurance)\b',
            r'\b(?:Communication|Teamwork|Leadership)\b'
        ]
        
        # First use NLP to extract potential skill entities
        # Limit text size to avoid memory issues
        max_text_size = 100000  # 100KB
        truncated_text = text[:max_text_size] if len(text) > max_text_size else text
        
        skills = set()
        try:
            doc = self.nlp(truncated_text)
            
            # Extract entities that might be skills
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "GPE"]:
                    skills.add(ent.text)
        except Exception as e:
            logger.warning(f"Error in NLP skill extraction: {str(e)}")
        
        # Use regex patterns for known skills
        for pattern in skill_patterns:
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                skills.update(match.group() for match in matches)
            except Exception as e:
                logger.warning(f"Error in regex skill extraction: {str(e)}")
            
        # Log extracted skills for debugging
        logger.info(f"Extracted {len(skills)} skills")
        
        return list(skills)
        
    def _extract_fallback_skills(self, text: str) -> List[str]:
        """Fallback skill extraction when normal extraction fails."""
        if not text:
            return []
            
        # Simplified extraction: just look for common terms
        common_skills = [
            "Python", "Java", "JavaScript", "C++", "SQL", 
            "AWS", "Cloud", "DevOps", "Docker", "Kubernetes",
            "Machine Learning", "AI", "Data Science", "Analytics",
            "React", "Angular", "Frontend", "Backend", "Full Stack",
            "Agile", "Scrum", "Project Management", "Communication",
            "Leadership", "Teamwork", "Problem Solving"
        ]
        
        found_skills = []
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                found_skills.append(skill)
                
        # Add some generic skill categories based on job keywords
        if re.search(r'\b(software|developer|engineer|programming)\b', text, re.IGNORECASE):
            found_skills.append("Software Development")
            
        if re.search(r'\b(data|analytics|analysis|statistics)\b', text, re.IGNORECASE):
            found_skills.append("Data Analysis")
            
        if re.search(r'\b(manage|management|lead|team)\b', text, re.IGNORECASE):
            found_skills.append("Management")
            
        if re.search(r'\b(design|ui|ux|user.experience)\b', text, re.IGNORECASE):
            found_skills.append("Design")
            
        if re.search(r'\b(marketing|sales|customer)\b', text, re.IGNORECASE):
            found_skills.append("Marketing/Sales")
            
        return found_skills
    
    def _get_bert_embedding(self, text: str) -> np.ndarray:
        """Get BERT embedding for text."""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[0][0].numpy()
            return embedding
        except Exception as e:
            print(f"Error getting BERT embedding: {str(e)}")
            return np.zeros(768)  # Return zero vector as fallback
    
    def _calculate_semantic_match(self, resume_text: str, job_description: str) -> float:
        """Calculate semantic similarity between resume and job description.
        Uses BERT embeddings if available, falls back to simpler methods if not."""
        if not resume_text or not job_description:
            self.logger.warning("Empty resume text or job description for semantic matching")
            return 0.0

        # Preprocess texts
        resume_text = self._preprocess_text(resume_text)
        job_description = self._preprocess_text(job_description)
        
        try:
            # Try using BERT for semantic matching if available
            if self.model_loaded and self.bert_model and self.tokenizer:
                self.logger.info("Using BERT model for semantic matching")
                import torch
                
                # Function to get embeddings
                def get_embedding(text):
                    # Truncate text if too long to avoid token length errors
                    max_length = 512
                    if len(text.split()) > max_length:
                        # Keep first 450 words and last 50 words (most important parts)
                        words = text.split()
                        text = " ".join(words[:450] + words[-50:])
                    
                    try:
                        # Use a timeout for tokenization and inference to prevent hangs
                        with Timeout(5):  # 5 second timeout
                            # Tokenize and encode
                            inputs = self.tokenizer(text, return_tensors="pt", 
                                                   padding=True, truncation=True, max_length=512)
                            
                            # Move to same device as model
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                            
                            # Get model output
                            with torch.no_grad():
                                outputs = self.bert_model(**inputs)
                            
                            # Use CLS token embedding as sentence embedding
                            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                            return embedding
                    except TimeoutError:
                        self.logger.warning("BERT embedding timed out, falling back to alternative method")
                        return None
                    except Exception as e:
                        self.logger.error(f"Error getting BERT embedding: {str(e)}")
                        return None
                
                # Get embeddings and compute cosine similarity
                resume_embedding = get_embedding(resume_text)
                if resume_embedding is None:
                    self.logger.warning("Failed to get resume embedding, falling back to alternative method")
                    return self._calculate_fallback_similarity(resume_text, job_description)
                    
                job_embedding = get_embedding(job_description)
                if job_embedding is None:
                    self.logger.warning("Failed to get job description embedding, falling back to alternative method")
                    return self._calculate_fallback_similarity(resume_text, job_description)
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
                
                # Normalize similarity to be between 0 and 1
                similarity = max(0, min(1, similarity))
                self.logger.info(f"BERT semantic similarity: {similarity:.4f}")
                return similarity
                
            else:
                # Fall back to spaCy if available
                return self._calculate_fallback_similarity(resume_text, job_description)
                
        except Exception as e:
            self.logger.error(f"Error in semantic matching: {str(e)}")
            return self._calculate_fallback_similarity(resume_text, job_description)
            
    def _calculate_tfidf_match(self, resume_text: str, job_description: str) -> float:
        """Calculate TF-IDF based match score between resume and job description."""
        try:
            if not resume_text or not job_description:
                return 0.0
                
            # Create a corpus with both texts
            corpus = [resume_text, job_description]
            
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            # Ensure similarity is in valid range
            similarity = max(0.0, min(1.0, float(similarity)))
            
            logger.info(f"TF-IDF similarity score: {similarity:.4f}")
            return similarity
            
        except Exception as e:
            logger.error(f"Error in TF-IDF matching: {str(e)}")
            return 0.0
    
    def _calculate_skill_match(self, resume_skills, job_requirements):
        """Calculate skill match score and details between resume skills and job requirements"""
        if not resume_skills or not job_requirements:
            return 0.0, {}
        
        # Handle different input formats for resume_skills and job_requirements
        # If either is a list, use the _calculate_skill_match_simple method instead
        if isinstance(resume_skills, list) or isinstance(job_requirements, list):
            match_score, matching_skills, missing_skills = self._calculate_skill_match_simple(
                resume_skills if isinstance(resume_skills, list) else self._flatten_skills_dict(resume_skills),
                job_requirements if isinstance(job_requirements, list) else self._flatten_skills_dict(job_requirements)
            )
            # Format result similar to the dictionary version
            skill_match_details = {
                'overall': {
                    'match_percent': round(match_score * 100, 1),
                    'matched': matching_skills,
                    'missing': missing_skills
                }
            }
            return match_score, skill_match_details
        
        # Standardize resume skills format
        resume_skills_flat = []
        
        # If resume_skills is a list, process it directly
        if isinstance(resume_skills, list):
            for skill in resume_skills:
                if isinstance(skill, dict) and 'name' in skill:
                    resume_skills_flat.append(skill['name'].lower())
                elif isinstance(skill, str):
                    resume_skills_flat.append(skill.lower())
        # If resume_skills is a dict, process each category
        elif isinstance(resume_skills, dict):
            for skill_category, skills in resume_skills.items():
                if isinstance(skills, list):
                    for skill in skills:
                        if isinstance(skill, dict) and 'name' in skill:
                            resume_skills_flat.append(skill['name'].lower())
                        elif isinstance(skill, str):
                            resume_skills_flat.append(skill.lower())
                elif isinstance(skills, str):
                    resume_skills_flat.append(skills.lower())
        
        skill_match_details = {}
        total_req_score = 0
        matched_req_score = 0
        
        # Calculate match scores for each category
        for category, requirements in job_requirements.items():
            category_score = 0
            category_max = len(requirements) if requirements else 0
            matched_skills = []
            
            if category_max > 0:
                for req in requirements:
                    req_lower = req.lower()
                    total_req_score += 1
                    
                    # Check for exact match
                    if req_lower in resume_skills_flat:
                        category_score += 1
                        matched_req_score += 1
                        matched_skills.append(req)
                        continue
                    
                    # Check for partial matches
                    for skill in resume_skills_flat:
                        # If skill contains the requirement or vice versa
                        if req_lower in skill or skill in req_lower:
                            category_score += 0.7  # Partial match
                            matched_req_score += 0.7
                            matched_skills.append(req)
                            break
            
            # Store category details
            if category_max > 0:
                match_percent = (category_score / category_max) * 100
                skill_match_details[category] = {
                    'match_percent': round(match_percent, 1),
                    'matched': matched_skills,
                    'missing': [r for r in requirements if r not in matched_skills]
                }
        
        # Calculate overall match score
        if total_req_score > 0:
            overall_match = matched_req_score / total_req_score
        else:
            overall_match = 0.0
        
        return overall_match, skill_match_details
    
    def _flatten_skills_dict(self, skills_dict):
        """Convert a hierarchical skills dictionary to a flat list of skills."""
        if isinstance(skills_dict, list):
            # If it's already a list, extract string skills or skill names
            return [
                skill['name'] if isinstance(skill, dict) and 'name' in skill else skill
                for skill in skills_dict if isinstance(skill, (dict, str))
            ]
            
        flattened_skills = []
        if isinstance(skills_dict, dict):
            for category, skills in skills_dict.items():
                if isinstance(skills, list):
                    for skill in skills:
                        if isinstance(skill, dict) and 'name' in skill:
                            flattened_skills.append(skill['name'])
                        elif isinstance(skill, str):
                            flattened_skills.append(skill)
                elif isinstance(skills, str):
                    flattened_skills.append(skills)
                    
        return flattened_skills
    
    def _normalize_skills(self, skills_data):
        """Normalize skills data to a standard list format regardless of input structure.
        
        Args:
            skills_data: Skills data that could be in various formats (list, dict, etc.)
            
        Returns:
            List of skills as strings, normalized to lowercase
        """
        normalized_skills = []
        
        if not skills_data:
            return normalized_skills
            
        # Handle list of skills
        if isinstance(skills_data, list):
            for skill in skills_data:
                if isinstance(skill, dict) and 'name' in skill:
                    normalized_skills.append(skill['name'].lower())
                elif isinstance(skill, str):
                    normalized_skills.append(skill.lower())
                    
        # Handle dictionary of skills
        elif isinstance(skills_data, dict):
            for category, skills in skills_data.items():
                if isinstance(skills, list):
                    for skill in skills:
                        if isinstance(skill, dict) and 'name' in skill:
                            normalized_skills.append(skill['name'].lower())
                        elif isinstance(skill, str):
                            normalized_skills.append(skill.lower())
                elif isinstance(skills, str):
                    normalized_skills.append(skills.lower())
                    
        # Remove duplicates while preserving order
        seen = set()
        return [skill for skill in normalized_skills if not (skill in seen or seen.add(skill))]

    def analyze_job_match(self, resume_analysis: Dict[str, Any], job_description: str) -> Dict[str, Any]:
        """Analyze how well a resume matches a job description"""
        logger.info("Starting job match analysis...")
        
        if not resume_analysis or not job_description:
            logger.warning("Empty resume analysis or job description for job matching")
            return {
                'match_score': 0.0,
                'match_level': 'No Match',
                'details': {}
            }
        
        try:
            # Get resume text for semantic analysis
            resume_text = self._get_full_resume_text(resume_analysis)
            
            # Calculate semantic match
            semantic_score = self._calculate_semantic_match(resume_text, job_description)
            logger.info(f"Semantic match score: {semantic_score:.4f}")
            
            # Extract requirements from job description
            requirements = self.extract_requirements(job_description)
            
            # Get normalized resume skills - using the new standardized method
            resume_skills = self._normalize_skills(resume_analysis.get('skills', []))
            
            # If no skills found, try to extract from full text
            if not resume_skills:
                logger.info("No skills found in resume data, extracting from text")
                resume_skills = self._extract_skills(resume_text)
            
            # Flatten all job requirements into a single skills list
            job_skills = []
            for category, skills in requirements.items():
                job_skills.extend(skills)
            
            # Calculate skill match using the simple method
            skill_match_score, matching_skills, missing_skills = self._calculate_skill_match_simple(
                resume_skills,
                job_skills
            )
            
            logger.info(f"Skill match score: {skill_match_score:.4f}")
            
            # Combine scores with appropriate weights
            # Semantic match is weighted more heavily as it considers the entire resume
            combined_score = (semantic_score * 0.7) + (skill_match_score * 0.3)
            
            # Scale the combined score to 0-100 range
            match_score = min(100, max(0, combined_score * 100))
            
            # Determine match level
            match_level = "No Match"
            if match_score >= 85:
                match_level = "Excellent Match"
            elif match_score >= 70:
                match_level = "Good Match"
            elif match_score >= 50:
                match_level = "Fair Match"
            elif match_score >= 30:
                match_level = "Partial Match"
            
            # Generate recommendations based on missing skills
            recommendations = []
            if missing_skills:
                # Limit to top 5 missing skills to avoid overwhelming
                top_missing = missing_skills[:5]
                recommendations.append(f"Consider adding these skills to your resume: {', '.join(top_missing)}")
                
            if skill_match_score < 0.4:
                recommendations.append("Your resume needs more relevant skills for this job position")
            
            if semantic_score < 0.5:
                recommendations.append("Update your resume to better match the job description terminology")
            
            return {
                'match_score': round(match_score, 1),
                'match_level': match_level,
                'matching_skills': matching_skills,
                'missing_skills': missing_skills,
                'details': {
                    'semantic_match': round(semantic_score * 100, 1),
                    'skill_match': round(skill_match_score * 100, 1)
                },
                'recommendations': recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in job matching: {str(e)}", exc_info=True)
            return {
                'match_score': 18.0,
                'match_level': 'No Match',
                'details': {
                    'semantic_match': 0.0,
                    'skill_match': 0.0
                },
                'error': str(e)
            }
    
    def _calculate_skill_match_simple(self, resume_skills: List[str], job_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        """Simple skill matching that works with flat lists of skills."""
        if not resume_skills or not job_skills:
            return 0.0, [], []
            
        # Normalize all skills to lowercase for comparison
        resume_skills_lower = [s.lower() for s in resume_skills]
        job_skills_lower = [s.lower() for s in job_skills]
        
        # Find exact matches
        matching_skills = []
        missing_skills = []
        
        for job_skill in job_skills:
            job_skill_lower = job_skill.lower()
            
            # Check for exact match
            if job_skill_lower in resume_skills_lower:
                matching_skills.append(job_skill)
                continue
                
            # Check for partial matches
            found_partial = False
            for resume_skill in resume_skills_lower:
                # If resume skill contains the job skill or vice versa
                if job_skill_lower in resume_skill or resume_skill in job_skill_lower:
                    matching_skills.append(job_skill)
                    found_partial = True
                    break
                    
            if not found_partial:
                missing_skills.append(job_skill)
        
        # Calculate match score
        if job_skills:
            match_score = len(matching_skills) / len(job_skills)
        else:
            match_score = 0.0
            
        return match_score, matching_skills, missing_skills
    
    def get_tfidf_vector(self, text: str, corpus: List[str]) -> np.ndarray:
        """Get TF-IDF vector for a text"""
        if not text or not corpus:
            return np.zeros(100)  # Return empty vector
            
        try:
            # Add the text to the corpus
            all_texts = [text] + corpus
            
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            
            # Return the vector for the input text
            return tfidf_matrix[0].toarray()[0]
            
        except Exception as e:
            logger.error(f"Error getting TF-IDF vector: {str(e)}")
            return np.zeros(100)
            
    def get_hybrid_embedding(self, text: str, corpus: List[str]) -> np.ndarray:
        """Combine BERT embeddings with TF-IDF vectors for a more comprehensive representation"""
        if not text:
            # Return zero array for empty text
            return np.zeros(100)  # Default size
        
        # Get base embeddings
        bert_embedding = self._get_bert_embedding(text)
        tfidf_vector = self.get_tfidf_vector(text, corpus)
        
        # Concatenate the embeddings
        combined = np.concatenate([bert_embedding, tfidf_vector])
        
        # Apply PCA for dimensionality reduction if there's enough data
        try:
            # Set n_components adaptively based on data size
            n_samples = 1  # We're embedding a single text
            n_features = combined.shape[0]
            n_components = min(50, n_samples, n_features)
            
            if n_components < 1:
                # If we can't reduce dimensions, return the original combined vector
                return combined
                
            # Create a new PCA instance with appropriate n_components
            pca = PCA(n_components=n_components)
            reduced = pca.fit_transform(combined.reshape(1, -1))
            return reduced.flatten()
        except Exception as e:
            logger.warning(f"Error applying PCA: {str(e)}")
            # Return the original combined vector if PCA fails
            return combined
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding"""
        if not text:
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def extract_requirements(self, job_description: str) -> Dict[str, Dict[str, List[str]]]:
        """Extract detailed requirements from a job description."""
        if not job_description:
            return {}
            
        try:
            print("Extracting requirements from job description...")
            
            # Process with spaCy
            if self.nlp:
                doc = self.nlp(job_description)
            else:
                return self._extract_requirements_fallback(job_description)
            
            # Define categories of requirements
            requirement_categories = {
                'skills': {
                    'patterns': [
                        r'(?:technical\s+)?skills(?:\s+include)?:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'(?:technical\s+)?requirements:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'qualifications:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'you(?:\'ll| will) need:?\s*(.*?)(?=\n\n|\n[A-Z]|$)'
                    ],
                    'keywords': [
                        'programming', 'language', 'framework', 'tool', 'software', 'technology',
                        'development', 'database', 'cloud', 'api', 'platform', 'system'
                    ]
                },
                'experience': {
                    'patterns': [
                        r'experience:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'background:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'(?:[\d-]+\+?\s+years?|proven)\s+(?:of\s+)?experience\s+(?:in|with):?\s*(.*?)(?=\n\n|\n[A-Z]|$)'
                    ],
                    'keywords': [
                        'experience', 'year', 'background', 'worked', 'history', 'track record'
                    ]
                },
                'education': {
                    'patterns': [
                        r'education:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'qualification:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'degree:?\s*(.*?)(?=\n\n|\n[A-Z]|$)'
                    ],
                    'keywords': [
                        'degree', 'bachelor', 'master', 'phd', 'certification', 'diploma', 'university', 'college'
                    ]
                },
                'soft_skills': {
                    'patterns': [
                        r'soft\s+skills:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'interpersonal\s+skills:?\s*(.*?)(?=\n\n|\n[A-Z]|$)',
                        r'personal\s+qualities:?\s*(.*?)(?=\n\n|\n[A-Z]|$)'
                    ],
                    'keywords': [
                        'communicate', 'team', 'collaborate', 'manage', 'lead', 'problem-solving',
                        'analytical', 'organized', 'detail-oriented', 'adaptable', 'flexible'
                    ]
                }
            }
            
            # Extract requirements using patterns and NLP
            requirements = {}
            for category, config in requirement_categories.items():
                requirements[category] = []
                
                # Get patterns for this category if available, otherwise use empty list for safety
                patterns = config.get('patterns', [])
                keywords = config.get('keywords', [])
                
                # Extract using regex patterns
                for pattern in patterns:
                    matches = re.search(pattern, job_description, re.IGNORECASE | re.DOTALL)
                    if matches:
                        text = matches.group(1)
                        skills = self._extract_bullet_points(text)
                        requirements[category].extend(skills)
                
                # Extract skills from sentences containing keywords
                for sentence in doc.sents:
                    if any(keyword in sentence.text.lower() for keyword in keywords):
                        # Extract noun phrases that might be skills
                        for chunk in sentence.noun_chunks:
                            # Skip very short or very long chunks
                            if 3 <= len(chunk.text) <= 50:
                                requirements[category].append(chunk.text.strip())
            
            # Deduplicate and clean
            for category in requirements:
                requirements[category] = list(set([self._clean_skill(s) for s in requirements[category]]))
                requirements[category] = [s for s in requirements[category] if s and len(s) > 2]
            
            # Debug: print extracted requirements
            for category, skills in requirements.items():
                if skills:
                    print(f"  {category}: {', '.join(skills)}")
            
            return requirements
                
        except Exception as e:
            # Return a basic structure with empty lists on error
            return {
                'skills': [],
                'experience': [],
                'education': [],
                'soft_skills': []
            }
            
    def _extract_requirements_fallback(self, job_description: str) -> Dict[str, List[str]]:
        """Extract requirements using simple fallback methods if NLP is not available."""
        requirements = {
            'skills': [],
            'experience': [],
            'education': [],
            'soft_skills': []
        }
        
        # Simple keyword extraction
        skill_keywords = ['python', 'java', 'javascript', 'sql', 'react', 'angular', 'cloud', 'aws', 'azure']
        experience_keywords = ['years', 'experience', 'background']
        education_keywords = ['degree', 'bachelor', 'master', 'phd']
        soft_skill_keywords = ['communication', 'teamwork', 'leadership', 'problem-solving']
        
        lines = job_description.split('\n')
        for line in lines:
            line = line.strip().lower()
            
            # Extract skills
            if any(kw in line for kw in skill_keywords):
                requirements['skills'].append(line)
                
            # Extract experience
            if any(kw in line for kw in experience_keywords):
                requirements['experience'].append(line)
                
            # Extract education
            if any(kw in line for kw in education_keywords):
                requirements['education'].append(line)
                
            # Extract soft skills
            if any(kw in line for kw in soft_skill_keywords):
                requirements['soft_skills'].append(line)
                
        return requirements

    def _get_full_resume_text(self, resume_data: Dict) -> str:
        """Extract all text content from resume data structure."""
        text_parts = []
        
        # Extract contact info
        if 'contact_info' in resume_data:
            contact = resume_data['contact_info']
            if isinstance(contact, dict):
                for field in ['name', 'email', 'phone', 'location']:
                    if field in contact and contact[field]:
                        text_parts.append(str(contact[field]))
        
        # Extract summary
        if 'summary' in resume_data and resume_data['summary']:
            text_parts.append(str(resume_data['summary']))
        
        # Extract experience
        if 'experience' in resume_data:
            experience = resume_data['experience']
            if isinstance(experience, list):
                for item in experience:
                    if isinstance(item, dict):
                        for field in ['title', 'company', 'description', 'location']:
                            if field in item and item[field]:
                                text_parts.append(str(item[field]))
                    elif isinstance(item, str):
                        text_parts.append(item)
        
        # Extract education
        if 'education' in resume_data:
            education = resume_data['education']
            if isinstance(education, list):
                for item in education:
                    if isinstance(item, dict):
                        for field in ['degree', 'field', 'institution', 'date']:
                            if field in item and item[field]:
                                text_parts.append(str(item[field]))
                    elif isinstance(item, str):
                        text_parts.append(item)
        
        # Extract skills - handle multiple formats
        if 'skills' in resume_data:
            skills = resume_data['skills']
            
            # Handle case where skills is a dictionary mapping categories to skill lists
            if isinstance(skills, dict):
                for category, skill_list in skills.items():
                    if isinstance(skill_list, list):
                        for skill in skill_list:
                            if isinstance(skill, dict) and 'name' in skill:
                                text_parts.append(skill['name'])
                            elif isinstance(skill, str):
                                text_parts.append(skill)
                    elif isinstance(skill_list, str):
                        text_parts.append(skill_list)
            
            # Handle case where skills is a list of skills
            elif isinstance(skills, list):
                for skill in skills:
                    if isinstance(skill, dict) and 'name' in skill:
                        text_parts.append(skill['name'])
                    elif isinstance(skill, str):
                        text_parts.append(skill)
            
            # Handle case where skills is a string
            elif isinstance(skills, str):
                text_parts.append(skills)
        
        return " ".join(text_parts)

    def _calculate_fallback_similarity(self, text1: str, text2: str) -> float:
        """Calculate a fallback similarity score based on keyword overlap"""
        # Tokenize and process texts
        doc1 = self.nlp(text1.lower())
        doc2 = self.nlp(text2.lower())
        
        # Extract meaningful tokens (no stopwords, punctuation, etc.)
        tokens1 = [token.lemma_ for token in doc1 if not token.is_stop and not token.is_punct and len(token.text) > 2]
        tokens2 = [token.lemma_ for token in doc2 if not token.is_stop and not token.is_punct and len(token.text) > 2]
        
        # Create sets for comparison
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        # Calculate Jaccard similarity
        if not set1 or not set2:
            return 0.0
            
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Scale to 0-100 range, with a bias toward higher scores
        scaled_similarity = min(100, similarity * 85 + 15)
        
        logger.info(f"Fallback similarity score: {scaled_similarity:.2f}")
        return scaled_similarity 

    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        if not text:
            return []
            
        # Split by common bullet point markers
        bullet_patterns = [
            r'•\s*(.*?)(?=\n•|\n\n|$)',
            r'-\s*(.*?)(?=\n-|\n\n|$)',
            r'\*\s*(.*?)(?=\n\*|\n\n|$)',
            r'\d+\.\s*(.*?)(?=\n\d+\.|\n\n|$)',
            r'(?m)^[\s\t]*[▪●◦○✓✔]\s*(.*?)$'
        ]
        
        results = []
        found_bullets = False
        
        for pattern in bullet_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                found_bullets = True
                for match in matches:
                    clean_item = match.strip()
                    if clean_item and len(clean_item) > 2:
                        results.append(clean_item)
        
        # If no bullet points found, try splitting by newlines or commas
        if not found_bullets:
            # Try newlines first
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            if len(lines) > 1:
                results.extend(lines)
            else:
                # Try comma separation
                items = [item.strip() for item in text.split(',') if item.strip()]
                results.extend(items)
                
        # Clean and deduplicate
        clean_results = []
        for item in results:
            # Remove trailing punctuation and clean up
            clean_item = re.sub(r'[,.;]+$', '', item).strip()
            if clean_item and clean_item not in clean_results:
                clean_results.append(clean_item)
                
        return clean_results
        
    def _clean_skill(self, skill: str) -> str:
        """Clean up a skill string."""
        if not skill:
            return ""
            
        # Remove leading/trailing punctuation and whitespace
        skill = re.sub(r'^[^\w]+|[^\w]+$', '', skill).strip()
        
        # Remove unnecessary phrases
        skill = re.sub(r'(?i)^knowledge of\s+', '', skill)
        skill = re.sub(r'(?i)^experience (?:in|with)\s+', '', skill)
        skill = re.sub(r'(?i)^proficiency (?:in|with)\s+', '', skill)
        skill = re.sub(r'(?i)^understanding of\s+', '', skill)
        skill = re.sub(r'(?i)^ability to\s+', '', skill)
        
        return skill 