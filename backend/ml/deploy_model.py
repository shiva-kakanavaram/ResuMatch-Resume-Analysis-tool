import os
import torch
import json
from transformers import BertTokenizer
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Initialized ModelDeployer with model directory: {model_dir}")
    
    def save_model(self, model, tokenizer, config, model_name='resume_match_model'):
        """Save model, tokenizer, and configuration"""
        try:
            # Create model directory
            model_path = os.path.join(self.model_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            logger.info(f"Created model directory: {model_path}")
            
            # Save model state dict
            logger.info("Saving model state dict...")
            torch.save(model.state_dict(), os.path.join(model_path, 'pytorch_model.bin'))
            
            # Save tokenizer
            logger.info("Saving tokenizer...")
            tokenizer.save_pretrained(model_path)
            
            # Save config
            logger.info("Saving model config...")
            with open(os.path.join(model_path, 'config.json'), 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Successfully saved model to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, model_class, model_name='resume_match_model'):
        """Load model, tokenizer, and configuration"""
        try:
            # Get model path
            model_path = os.path.join(self.model_dir, model_name)
            if not os.path.exists(model_path):
                logger.error(f"Model path does not exist: {model_path}")
                return None, None, None
            
            logger.info(f"Loading model from {model_path}")
            
            # Load config
            config_path = os.path.join(model_path, 'config.json')
            if not os.path.exists(config_path):
                logger.error(f"Config file not found: {config_path}")
                return None, None, None
                
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Initialize model with config
            model = model_class(
                bert_model=config.get('bert_model', 'bert-base-uncased'),
                hidden_size=config.get('hidden_size', 768),
                dropout=config.get('dropout', 0.1)
            )
            
            # Load model state dict
            model_path_bin = os.path.join(model_path, 'pytorch_model.bin')
            if not os.path.exists(model_path_bin):
                logger.error(f"Model weights not found: {model_path_bin}")
                return None, None, None
                
            model.load_state_dict(torch.load(model_path_bin))
            model.eval()
            
            # Load tokenizer
            tokenizer = BertTokenizer.from_pretrained(model_path)
            
            logger.info("Successfully loaded model, tokenizer, and config")
            return model, tokenizer, config
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None, None, None
    
    def optimize_for_inference(self, model, model_name='resume_match_model'):
        """Optimize model for inference"""
        try:
            logger.info("Optimizing model for inference...")
            
            # Convert to TorchScript
            model.eval()
            example_input_ids = torch.zeros((1, 512), dtype=torch.long)
            example_attention_mask = torch.ones((1, 512), dtype=torch.long)
            
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    model,
                    (example_input_ids, example_attention_mask)
                )
            
            # Save optimized model
            model_path = os.path.join(self.model_dir, model_name, 'model_optimized.pt')
            torch.jit.save(traced_model, model_path)
            
            logger.info(f"Successfully saved optimized model to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            return False
    
    def get_model_info(self, model_name='resume_match_model'):
        """Get information about the saved model"""
        try:
            model_path = os.path.join(self.model_dir, model_name)
            if not os.path.exists(model_path):
                logger.error(f"Model path does not exist: {model_path}")
                return None
            
            # Load config
            with open(os.path.join(model_path, 'config.json'), 'r') as f:
                config = json.load(f)
            
            # Get model size
            model_size = os.path.getsize(os.path.join(model_path, 'pytorch_model.bin'))
            
            info = {
                'model_name': model_name,
                'model_path': model_path,
                'model_size_mb': model_size / (1024 * 1024),
                'config': config,
                'has_optimized_version': os.path.exists(os.path.join(model_path, 'model_optimized.pt'))
            }
            
            logger.info(f"Model info: {json.dumps(info, indent=2)}")
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None
