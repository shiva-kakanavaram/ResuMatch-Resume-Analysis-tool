import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import logging
from types import SimpleNamespace
from backend.ml.deploy_model import ModelDeployer
from backend.ml.train_monitor import TrainingMonitor
from backend.ml.data_loader import KaggleDataLoader, prepare_training_data
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResumeMatchModel(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', hidden_size=768, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        
        # Freeze BERT parameters to prevent overfitting on small dataset
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # Unfreeze last few layers for fine-tuning
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Single output for binary classification
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return SimpleNamespace(logits=logits)

def create_model(device='cuda'):
    """Create and initialize the model"""
    try:
        logger.info("Creating model...")
        model = ResumeMatchModel()
        model = model.to(device)
        logger.info(f"Model created successfully and moved to {device}")
        return model
    except Exception as e:
        logger.error(f"Error creating model: {str(e)}")
        raise

def train_model(
    train_data_path='data/resume_job_pairs.json',
    save_dir='models',
    batch_size=16,
    num_epochs=10,
    learning_rate=2e-5,
    val_split=0.2,
    device=None
):
    """Train the resume matching model"""
    try:
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
            
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
            torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
        else:
            logger.info("No GPU available, using CPU")
            
        logger.info(f"Using device: {device}")
        
        # Create data loader
        data_loader = KaggleDataLoader()
        train_dataset, val_dataset = data_loader.prepare_data(val_split)
        
        # Windows-compatible data loader settings
        loader_kwargs = {
            'batch_size': batch_size,
            'pin_memory': True if torch.cuda.is_available() else False,
            'num_workers': 0  # Required for Windows
        }
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            **loader_kwargs
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            **loader_kwargs
        )
        
        # Create model
        model = create_model(device)
        
        # Create optimizer with weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
        
        # Create learning rate scheduler
        total_steps = len(train_dataloader) * num_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Initialize training monitor
        monitor = TrainingMonitor()
        monitor.start()
        
        # Train model
        best_model, best_val_accuracy = train(
            model,
            train_dataloader,
            val_dataloader,
            optimizer,
            scheduler,
            num_epochs,
            device,
            monitor,
            save_dir
        )
        
        # Save model for deployment
        deployer = ModelDeployer(save_dir)
        config = {
            'model_type': 'ResumeMatchModel',
            'bert_model': 'bert-base-uncased',
            'hidden_size': 768,
            'dropout': 0.1
        }
        success = deployer.save_model(
            model,
            BertTokenizer.from_pretrained('bert-base-uncased'),
            config,
            'resume_match_model'
        )
        
        if not success:
            logger.error("Failed to save model")
            return None, None
        
        # Get training summary
        summary = monitor.get_summary()
        logger.info(f"Training completed. Summary:\n{summary}")
        
        return best_model, best_val_accuracy
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    num_epochs,
    device,
    monitor=None,
    save_dir='models'
):
    """Train the model"""
    logger.info('-' * 60)
    logger.info('Starting Training')
    logger.info(f'Training on device: {device}')
    logger.info(f'Number of training batches: {len(train_dataloader)}')
    logger.info(f'Number of validation batches: {len(val_dataloader)}')
    logger.info('-' * 60)
    
    best_val_accuracy = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        logger.info(f'\nEpoch {epoch + 1}/{num_epochs}')
        logger.info('-' * 20)
        
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train_preds = 0
        total_train_samples = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(outputs.logits.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            preds = (torch.sigmoid(outputs.logits.squeeze()) > 0.5).float()
            correct_train_preds += (preds == labels).sum().item()
            total_train_samples += labels.size(0)
            
            if batch_idx % 1 == 0:  # Log every batch
                logger.info(f'Batch {batch_idx + 1}/{len(train_dataloader)} - Loss: {loss.item():.4f}')
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_accuracy = correct_train_preds / total_train_samples
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val_preds = 0
        total_val_samples = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = nn.BCEWithLogitsLoss()(outputs.logits.squeeze(), labels)
                
                total_val_loss += loss.item()
                preds = (torch.sigmoid(outputs.logits.squeeze()) > 0.5).float()
                correct_val_preds += (preds == labels).sum().item()
                total_val_samples += labels.size(0)
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_accuracy = correct_val_preds / total_val_samples
        
        # Log metrics
        logger.info('\nEpoch Summary:')
        logger.info(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
        logger.info(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Update monitor if provided
        if monitor:
            monitor.log_stats(
                epoch + 1,
                avg_train_loss,
                train_accuracy,
                avg_val_loss,
                val_accuracy
            )
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict().copy()
            logger.info(f'New best model saved! Validation Accuracy: {val_accuracy:.4f}')
            
            # Save the model
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': best_model,
                'val_accuracy': best_val_accuracy,
                'train_accuracy': train_accuracy
            }, model_path)
            logger.info(f'Model saved to {model_path}')
    
    logger.info('-' * 60)
    logger.info('Training completed!')
    logger.info(f'Best validation accuracy: {best_val_accuracy:.4f}')
    logger.info('-' * 60)
    
    return best_model, best_val_accuracy

if __name__ == "__main__":
    try:
        logger.info("Starting the training process...")
        logger.info("Python executable: %s", sys.executable)
        logger.info("Current working directory: %s", os.getcwd())
        
        # Train model
        best_model, best_val_accuracy = train_model()
        
        if best_model is None:
            logger.error("Training failed - model is None")
            sys.exit(1)
        
        logger.info("Training completed successfully")
        logger.info("Best validation accuracy: %.4f", best_val_accuracy)
        
    except Exception as e:
        logger.error("Training failed with error: %s", str(e))
        import traceback
        logger.error("Traceback:\n%s", traceback.format_exc())
        sys.exit(1)
