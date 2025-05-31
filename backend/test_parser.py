from backend.ml.enhanced_resume_parser import EnhancedResumeParser
from backend.utils.enhanced_document_processor import EnhancedDocumentProcessor
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_parser():
    # Initialize the processor and parser
    processor = EnhancedDocumentProcessor()
    parser = EnhancedResumeParser()
    
    logger.info("Parser initialized successfully.")
    
    # Sample resume text
    sample_text = """John Doe
Email: john@example.com
Phone: (123) 456-7890

SUMMARY
Experienced software engineer with 5 years of experience in full-stack development.

EXPERIENCE
Senior Software Engineer - ABC Inc. (Jan 2020 - Present)
• Developed and maintained web applications using React and Node.js
• Led a team of 3 developers on a major project

Junior Developer - XYZ Corp (2018 - 2020)
• Built RESTful APIs using Python and Flask
• Implemented automated testing

EDUCATION
Bachelor of Science in Computer Science
University of Technology, 2018
GPA: 3.8

SKILLS
• Programming: Python, JavaScript, Java
• Frameworks: React, Node.js, Flask
• Tools: Git, Docker, AWS
• Soft Skills: Team leadership, Communication"""
    
    try:
        # Process the document using process_plain_text method
        logger.info("Processing document...")
        doc_data = processor.process_plain_text(sample_text)
        
        # Print sections detected
        if 'structure' in doc_data and 'sections' in doc_data['structure']:
            sections = doc_data['structure']['sections']
            logger.info(f"Detected {len(sections)} sections:")
            for section in sections:
                section_type = section.get('type', 'unknown')
                section_content = section.get('content', '')
                logger.info(f"  - {section_type}: {len(section_content.split()) if section_content else 0} words")
                logger.info(f"    Content: {section_content[:100]}...")
        else:
            logger.warning("No sections detected in document structure")
        
        # Parse the resume
        logger.info("Parsing resume...")
        result = parser.parse_resume(doc_data)
        
        # Print results
        logger.info("Parsing completed with the following results:")
        logger.info(f"- Name: {result.get('contact_info', {}).get('name')}")
        logger.info(f"- Email: {result.get('contact_info', {}).get('email')}")
        logger.info(f"- Phone: {result.get('contact_info', {}).get('phone')}")
        logger.info(f"- Summary length: {len(result.get('summary', '').split()) if result.get('summary') else 0} words")
        logger.info(f"- Experience entries: {len(result.get('experience', []))}")
        logger.info(f"- Education entries: {len(result.get('education', []))}")
        logger.info(f"- Skills: {len(result.get('skills', []))}")
        
        # Write output to file
        with open('parser_output.json', 'w') as f:
            json.dump(result, f, indent=2)
            logger.info("Output written to parser_output.json")
            
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parser() 