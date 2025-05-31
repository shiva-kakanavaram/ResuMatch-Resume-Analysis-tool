import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import json
from pathlib import Path
import os
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)

class ResumeClassifier:
    def __init__(self, model_path: str = None, use_quantization: bool = True):
        """
        Initialize the resume classifier
        
        Args:
            model_path: Path to saved model, if None uses default BERT
            use_quantization: Whether to use quantization for inference
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load categories from mapping file
        try:
            category_mapping_path = Path(__file__).parent.parent / "data" / "category_mapping.json"
            with open(category_mapping_path, 'r') as f:
                mapping_data = json.load(f)
                self.job_categories = mapping_data["categories"]
        except Exception as e:
            logger.error(f"Error loading category mapping: {str(e)}")
            raise
        
        # Initialize tokenizer with caching
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Initialize model
        try:
            if model_path and os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device)
                logger.info(f"Loaded model from {model_path}")
            else:
                self.model = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=len(self.job_categories),
                    id2label={i: cat for i, cat in enumerate(self.job_categories)},
                    label2id={cat: i for i, cat in enumerate(self.job_categories)}
                )
            
            # Optimize model for inference
            if use_quantization and self.device == "cpu":
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear}, dtype=torch.qint8
                )
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize prediction cache
            self._prediction_cache = {}
            
            logger.info(f"Initialized ResumeClassifier with {len(self.job_categories)} categories")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> dict:
        """
        Preprocess text for classification with caching
        
        Args:
            text: Input text to preprocess
            
        Returns:
            dict: Encoded text with input_ids and attention_mask
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Tokenize and encode the text
        try:
            encoded = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoded['input_ids'].to(self.device),
                'attention_mask': encoded['attention_mask'].to(self.device)
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            raise
    
    def predict(self, text: str, return_probabilities: bool = False) -> dict:
        """
        Predict job categories for a given text
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return prediction probabilities
            
        Returns:
            dict: Predicted categories and confidence scores
        """
        # Check cache first
        cache_key = hash(text)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        
        try:
            # Preprocess text
            inputs = self.preprocess_text(text)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            # Get probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.topk(probs, k=3)
            
            # Format results
            results = {
                'categories': [
                    {
                        'category': self.job_categories[idx.item()],
                        'confidence': score.item()
                    }
                    for score, idx in zip(predictions.values[0], predictions.indices[0])
                ],
                'raw_probabilities': probs[0].cpu().numpy().tolist() if return_probabilities else None
            }
            
            # Cache results
            self._prediction_cache[cache_key] = results
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for prediction"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep important punctuation
        text = ''.join(char for char in text if char.isalnum() or char in ' .,()[]{}')
        
        return text.strip()
    
    def get_model_info(self) -> dict:
        """Get information about the model"""
        return {
            'device': str(self.device),
            'num_categories': len(self.job_categories),
            'categories': self.job_categories,
            'model_type': self.model.config.model_type,
            'quantized': hasattr(self.model, 'is_quantized'),
            'cache_info': self.preprocess_text.cache_info()._asdict()
        }
