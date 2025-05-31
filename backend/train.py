import torch
from ml.train_model import train_model
from ml.train_monitor import TrainingMonitor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    try:
        logger.info("Starting model training...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Train model
        model, summary = train_model(
            train_data_path='data/training_data.json',
            val_split=0.2,
            batch_size=16,
            num_epochs=5,
            learning_rate=2e-5,
            warmup_steps=100,
            max_grad_norm=1.0,
            save_dir='models'
        )
        
        if model is None:
            logger.error("Training failed")
            return
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
