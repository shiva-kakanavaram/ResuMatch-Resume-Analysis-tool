import logging
import os
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

def get_document_processor():
    """Get a document processor instance."""
    from .enhanced_document_processor import EnhancedDocumentProcessor
    return EnhancedDocumentProcessor() 