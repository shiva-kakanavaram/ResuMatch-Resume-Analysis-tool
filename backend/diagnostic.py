import sys
import os
import pathlib
from pathlib import Path
import logging
import time
import traceback
from datetime import datetime
import json
from typing import Dict, List, Optional, Union

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from fastapi import FastAPI, APIRouter, HTTPException, Body, File, UploadFile, Form, Query, status
from pydantic import BaseModel

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("diagnostic")

# Create router
router = APIRouter()

class DiagnosticResponse(BaseModel):
    """Response model for diagnostic endpoint"""
    status: str
    message: str
    execution_time: float
    details: Optional[Dict] = None

@router.get("/diagnostic/check", response_model=DiagnosticResponse)
async def check_system():
    """Basic system check endpoint."""
    start_time = time.time()
    return DiagnosticResponse(
        status="ok",
        message="Diagnostic API is operational",
        execution_time=time.time() - start_time,
        details={
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "path": sys.path,
        }
    )

@router.get("/diagnostic/test_parser", response_model=DiagnosticResponse)
async def test_parser():
    """Test the resume parser with a minimal resume."""
    start_time = time.time()
    
    try:
        from backend.ml.enhanced_resume_parser import EnhancedResumeParser
        parser = EnhancedResumeParser()
        
        # Simple test resume
        sample_text = """John Doe
Software Engineer
johndoe@example.com | (123) 456-7890 | New York, NY
LinkedIn: linkedin.com/in/johndoe

Summary
Experienced software engineer with 5 years of expertise in Python and web development.

Experience
Senior Software Engineer | ABC Tech | January 2020 - Present
- Developed and maintained web applications using Python and FastAPI
- Implemented CI/CD pipelines using GitHub Actions
- Optimized database queries resulting in 30% performance improvement

Software Engineer | XYZ Solutions | June 2018 - December 2019
- Created RESTful APIs using Flask and SQLAlchemy
- Collaborated with cross-functional teams to design and implement features

Education
Bachelor of Science in Computer Science | State University | 2018
- GPA: 3.8/4.0
- Relevant coursework: Data Structures, Algorithms, Web Development

Skills
Programming Languages: Python, JavaScript, Java
Web Frameworks: FastAPI, Flask, Django, React
Databases: PostgreSQL, MongoDB
Tools: Git, Docker, Kubernetes, AWS
"""
        
        # Parse the sample resume
        result = parser.parse_resume(sample_text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        if result:
            return DiagnosticResponse(
                status="ok",
                message=f"Parser test successful (completed in {processing_time:.2f}s)",
                execution_time=processing_time,
                details={
                    "sections_found": list(result.keys()),
                    "experience_entries": len(result.get("experience", [])),
                    "skills_found": len(result.get("skills", {}).get("technical", [])) if isinstance(result.get("skills", {}), dict) else 0
                }
            )
        else:
            return DiagnosticResponse(
                status="error",
                message="Parser returned empty result",
                execution_time=processing_time,
                details={"error": "Parser failed to extract information"}
            )
    except Exception as e:
        logger.error(f"Error testing parser: {str(e)}")
        traceback.print_exc()
        return DiagnosticResponse(
            status="error",
            message=f"Parser test failed: {str(e)}",
            execution_time=time.time() - start_time,
            details={"error": str(e), "traceback": traceback.format_exc()}
        )

@router.get("/diagnostic/test_analyzer", response_model=DiagnosticResponse)
async def test_analyzer():
    """Test the resume analyzer with a minimal resume."""
    start_time = time.time()
    
    try:
        from backend.ml.resume_analyzer import ResumeAnalyzer
        analyzer = ResumeAnalyzer()
        
        # Create a minimal resume data structure
        resume_data = {
            "contact_info": {
                "name": "John Doe",
                "email": "john@example.com",
                "phone": "(123) 456-7890",
                "location": "New York, NY"
            },
            "summary": "Experienced software engineer with 5 years of expertise in Python and web development.",
            "experience": [
                {
                    "title": "Senior Software Engineer",
                    "company": "ABC Tech",
                    "start_date": "January 2020",
                    "end_date": "Present",
                    "responsibilities": [
                        "Developed and maintained web applications using Python and FastAPI",
                        "Implemented CI/CD pipelines using GitHub Actions",
                        "Optimized database queries resulting in 30% performance improvement"
                    ]
                },
                {
                    "title": "Software Engineer",
                    "company": "XYZ Solutions",
                    "start_date": "June 2018",
                    "end_date": "December 2019",
                    "responsibilities": [
                        "Created RESTful APIs using Flask and SQLAlchemy",
                        "Collaborated with cross-functional teams to design and implement features"
                    ]
                }
            ],
            "education": [
                {
                    "degree": "Bachelor of Science in Computer Science",
                    "institution": "State University",
                    "year": "2018",
                    "gpa": "3.8/4.0"
                }
            ],
            "skills": {
                "programming_languages": ["Python", "JavaScript", "Java"],
                "web_frameworks": ["FastAPI", "Flask", "Django", "React"],
                "databases": ["PostgreSQL", "MongoDB"],
                "tools": ["Git", "Docker", "Kubernetes", "AWS"]
            }
        }
        
        # Test the analyzer with a timeout
        try:
            # First check if analyze method exists
            if hasattr(analyzer, 'analyze'):
                logger.info("Using 'analyze' method")
                result = analyzer.analyze(resume_data, timeout=5)
            elif hasattr(analyzer, 'analyze_resume'):
                logger.info("Using 'analyze_resume' method")
                result = analyzer.analyze_resume(resume_data)
            else:
                return DiagnosticResponse(
                    status="error",
                    message="No analyze method found on ResumeAnalyzer class",
                    execution_time=time.time() - start_time,
                    details={"error": "MethodNotFound"}
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            if result:
                return DiagnosticResponse(
                    status="ok",
                    message=f"Analyzer test successful (completed in {processing_time:.2f}s)",
                    execution_time=processing_time,
                    details={
                        "overall_score": result.get("overall_score", 0),
                        "section_scores": result.get("section_scores", {}),
                        "recommendation_count": len(result.get("recommendations", []))
                    }
                )
            else:
                return DiagnosticResponse(
                    status="error",
                    message="Analyzer returned empty result",
                    execution_time=processing_time,
                    details={"error": "Analyzer failed to process resume"}
                )
        except TimeoutError:
            return DiagnosticResponse(
                status="error",
                message="Analyzer test timed out",
                execution_time=time.time() - start_time,
                details={"error": "TimeoutError"}
            )
            
    except Exception as e:
        logger.error(f"Error testing analyzer: {str(e)}")
        traceback.print_exc()
        return DiagnosticResponse(
            status="error",
            message=f"Analyzer test failed: {str(e)}",
            execution_time=time.time() - start_time,
            details={"error": str(e), "traceback": traceback.format_exc()}
        )

@router.get("/diagnostic/memory", response_model=DiagnosticResponse)
async def check_memory():
    """Check memory usage."""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return DiagnosticResponse(
            status="ok",
            message="Memory check successful",
            execution_time=time.time() - start_time,
            details={
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "process_name": process.name(),
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "threads": len(process.threads())
            }
        )
    except ImportError:
        return DiagnosticResponse(
            status="error",
            message="psutil not installed",
            execution_time=time.time() - start_time,
            details={"error": "ImportError"}
        )
    except Exception as e:
        return DiagnosticResponse(
            status="error",
            message=f"Memory check failed: {str(e)}",
            execution_time=time.time() - start_time,
            details={"error": str(e), "traceback": traceback.format_exc()}
        )

def get_diagnostic_router():
    """Returns a router with diagnostic endpoints for development use only."""
    router = APIRouter(
        prefix="/diagnostic",
        tags=["diagnostic"],
        responses={404: {"description": "Not found"}},
    )
    
    @router.get("/check", response_model=DiagnosticResponse)
    async def check_api():
        """Simple health check for API."""
        start_time = time.time()
        return DiagnosticResponse(
            status="ok",
            message="API is running",
            execution_time=time.time() - start_time,
            details={
                "environment": os.environ.get("ENVIRONMENT", "development"),
                "python_version": os.environ.get("PYTHON_VERSION", "unknown")
            }
        )
    
    @router.get("/test_parser", response_model=DiagnosticResponse)
    async def test_parser():
        """Test the resume parser functionality."""
        start_time = time.time()
        
        try:
            from backend.ml.enhanced_resume_parser import EnhancedResumeParser
            
            # Create sample text to test parser
            sample_text = """
            John Doe
            Software Engineer
            
            EXPERIENCE
            Senior Developer - ABC Corp (2018-Present)
            * Developed web applications using Python and JavaScript
            * Led a team of 3 junior developers
            
            Junior Developer - XYZ Inc (2016-2018)
            * Assisted in maintaining legacy code
            * Implemented new features for mobile app
            
            EDUCATION
            Bachelor of Science in Computer Science
            University of Technology (2012-2016)
            
            SKILLS
            Python, JavaScript, React, Docker, AWS
            """
            
            # Initialize parser
            parser = EnhancedResumeParser()
            
            # Test parsing
            result = parser.parse_resume(sample_text)
            
            # Check if parsing was successful
            if not result:
                return DiagnosticResponse(
                    status="error",
                    message="Parser returned empty result",
                    execution_time=time.time() - start_time,
                    details={"error": "Parser failed to extract information"}
                )
            
            # Get sections for validation
            experience = result.get("experience", [])
            skills = result.get("skills", {})
            
            return DiagnosticResponse(
                status="ok",
                message="Parser test completed successfully",
                execution_time=time.time() - start_time,
                details={
                    "sections_found": list(result.keys()),
                    "experience_entries": len(experience),
                    "skills_found": len(skills.get("technical", [])) if isinstance(skills, dict) else 0
                }
            )
        except Exception as e:
            logger.error(f"Error testing parser: {str(e)}")
            traceback.print_exc()
            return DiagnosticResponse(
                status="error",
                message=f"Parser test failed: {str(e)}",
                execution_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    @router.get("/test_analyzer", response_model=DiagnosticResponse)
    async def test_analyzer():
        """Test the resume analyzer functionality."""
        start_time = time.time()
        
        try:
            from backend.ml.resume_analyzer import ResumeAnalyzer
            
            # Create sample resume data
            sample_resume = {
                "contact_info": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "phone": "123-456-7890"
                },
                "summary": "Experienced software engineer with 5 years of experience in web development.",
                "experience": [
                    {
                        "title": "Senior Developer",
                        "company": "ABC Corp",
                        "date": "2018-Present",
                        "responsibilities": [
                            "Developed web applications using Python and JavaScript",
                            "Led a team of 3 junior developers"
                        ]
                    },
                    {
                        "title": "Junior Developer",
                        "company": "XYZ Inc",
                        "date": "2016-2018",
                        "responsibilities": [
                            "Assisted in maintaining legacy code",
                            "Implemented new features for mobile app"
                        ]
                    }
                ],
                "education": [
                    {
                        "degree": "Bachelor of Science in Computer Science",
                        "institution": "University of Technology",
                        "date": "2012-2016"
                    }
                ],
                "skills": {
                    "technical": ["Python", "JavaScript", "React", "Docker", "AWS"],
                    "soft": ["Leadership", "Communication"]
                }
            }
            
            # Initialize analyzer
            analyzer = ResumeAnalyzer()
            
            # Test analysis
            result = analyzer.analyze(sample_resume)
            
            # Check if analysis was successful
            if not result:
                return DiagnosticResponse(
                    status="error",
                    message="Analyzer returned empty result",
                    execution_time=time.time() - start_time,
                    details={"error": "Analyzer failed to process resume"}
                )
            
            return DiagnosticResponse(
                status="ok",
                message="Analyzer test completed successfully",
                execution_time=time.time() - start_time,
                details={
                    "overall_score": result.get("overall_score", 0),
                    "section_scores": result.get("section_scores", {}),
                    "recommendation_count": len(result.get("recommendations", []))
                }
            )
        except Exception as e:
            logger.error(f"Error testing analyzer: {str(e)}")
            traceback.print_exc()
            return DiagnosticResponse(
                status="error",
                message=f"Analyzer test failed: {str(e)}",
                execution_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    @router.get("/test_experience_extraction", response_model=DiagnosticResponse)
    async def test_experience_extraction():
        """Test the experience extraction functionality specifically."""
        start_time = time.time()
        
        try:
            from backend.ml.enhanced_resume_parser import EnhancedResumeParser
            
            # Create sample text to test experience extraction
            sample_text = """
            EXPERIENCE
            
            Senior Developer - ABC Corp (2018-Present)
            * Developed web applications using Python and JavaScript
            * Led a team of 3 junior developers
            * Implemented RESTful APIs using Flask and Django
            
            Junior Developer - XYZ Inc (2016-2018)
            * Assisted in maintaining legacy code
            * Implemented new features for mobile app
            * Participated in code reviews and testing
            
            Intern - Tech Startup (Summer 2015)
            * Developed frontend components using React
            * Fixed bugs in the codebase
            """
            
            # Initialize parser
            parser = EnhancedResumeParser()
            
            # Test experience extraction specifically
            experience_entries = parser._extract_experience(sample_text)
            
            # Check if extraction was successful
            if not experience_entries:
                return DiagnosticResponse(
                    status="warning",
                    message="No experience entries extracted",
                    execution_time=time.time() - start_time,
                    details={"sample_text": sample_text}
                )
            
            return DiagnosticResponse(
                status="ok",
                message=f"Successfully extracted {len(experience_entries)} experience entries",
                execution_time=time.time() - start_time,
                details={
                    "entry_count": len(experience_entries),
                    "entries": experience_entries
                }
            )
        except Exception as e:
            logger.error(f"Error testing experience extraction: {str(e)}")
            traceback.print_exc()
            return DiagnosticResponse(
                status="error",
                message=f"Experience extraction test failed: {str(e)}",
                execution_time=time.time() - start_time,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    return router 