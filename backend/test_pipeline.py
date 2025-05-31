import sys
from pathlib import Path
import os
import json

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
sys.path.insert(0, project_root)

from backend.utils.document_processor import DocumentProcessor
from backend.ml.resume_analyzer import ResumeAnalyzer

def test_pipeline():
    try:
        print("\n" + "="*80)
        print("RESUME ANALYSIS TEST")
        print("="*80 + "\n")
        
        # Test resume path
        resume_path = Path(r"C:\Users\Varun\Downloads\sample_resume_2.pdf")
        if not resume_path.exists():
            print(f"Error: Resume file not found at {resume_path}")
            return
        
        print(f"Analyzing resume: {resume_path.name}")
        print("="*80)
        
        # Read the PDF file
        with open(resume_path, "rb") as f:
            file_bytes = f.read()
        
        # Process the PDF
        doc_processor = DocumentProcessor()
        resume_text = doc_processor.process_resume(file_bytes, str(resume_path))
        
        # Initialize analyzer
        analyzer = ResumeAnalyzer()
        
        # Sample job description for testing
        job_description = """
        Senior Software Engineer Position
        
        Requirements:
        - 5+ years of experience in Python development
        - Expert in web frameworks (Django, Flask, FastAPI)
        - Deep understanding of REST APIs and microservices
        - Strong experience with SQL and NoSQL databases
        - Expert in cloud platforms (AWS, Azure, GCP)
        - Advanced Git and CI/CD experience
        - Strong problem-solving and system design skills
        - Team leadership experience
        - Excellent communication skills
        
        Responsibilities:
        - Lead development of web applications
        - Design and implement microservices architecture
        - Mentor junior developers
        - Drive technical decisions
        - Implement best practices and standards
        - Optimize application performance
        - Write technical documentation
        """
        
        # Analyze resume
        score, details = analyzer.calculate_ats_score(resume_text, job_description)
        
        # Save results to file
        output_path = Path('analysis_results.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("RESUME ANALYSIS RESULTS\n")
            f.write("="*80 + "\n\n")
            
            # Role Classification
            f.write("ROLE CLASSIFICATION\n")
            f.write("-"*30 + "\n")
            classification = details['role_classification']
            f.write(f"Detected Role: {classification['category']}\n")
            f.write(f"Confidence: {classification['confidence']*100:.2f}%\n\n")
            
            f.write("SCORE BREAKDOWN\n")
            f.write("-"*30 + "\n")
            f.write(f"Overall Score: {score:.2f}%\n")
            for component, component_score in details['score_breakdown'].items():
                f.write(f"{component.replace('_', ' ').title()}: {component_score:.2f}%\n")
            f.write("\n")
            
            f.write("SKILL ANALYSIS\n")
            f.write("-"*30 + "\n")
            if details['skill_matches']['matched_skills']:
                f.write("\nMatched Skills:\n")
                for skill in sorted(details['skill_matches']['matched_skills']):
                    f.write(f"✓ {skill}\n")
            
            if details['skill_matches']['missing_skills']:
                f.write("\nMissing Skills:\n")
                for skill in sorted(details['skill_matches']['missing_skills']):
                    f.write(f"✗ {skill}\n")
            
            if details['skill_analysis']:
                f.write("\nSKILL ANALYSIS BY CATEGORY\n")
                f.write("-"*30 + "\n")
                for category, data in sorted(details['skill_analysis'].items()):
                    f.write(f"\n{category.upper()}:\n")
                    f.write(f"Match Rate: {data['match_percentage']:.2f}%\n")
                    if data['matched']:
                        f.write("Matched:\n")
                        for skill in sorted(data['matched']):
                            f.write(f"✓ {skill}\n")
                    if data['missing']:
                        f.write("Missing:\n")
                        for skill in sorted(data['missing']):
                            f.write(f"✗ {skill}\n")
            
            if details['improvement_suggestions']:
                f.write("\nIMPROVEMENT SUGGESTIONS\n")
                f.write("-"*30 + "\n")
                for i, suggestion in enumerate(details['improvement_suggestions'], 1):
                    f.write(f"{i}. {suggestion}\n")
        
        print(f"\nAnalysis results have been saved to: {output_path.absolute()}")
        
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    test_pipeline()
