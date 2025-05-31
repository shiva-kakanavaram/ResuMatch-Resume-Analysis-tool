import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import requests
import logging
import json
import zipfile
import io
from tqdm import tqdm
import mlflow
import wandb

logger = logging.getLogger(__name__)

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(data_dir: str = None, test_size: float = 0.2):
    """
    Prepare training data with validation split
    
    Args:
        data_dir: Directory containing training data
        test_size: Fraction of data to use for validation
        
    Returns:
        tuple: Training and validation datasets
    """
    try:
        # Load and preprocess data
        if not data_dir:
            data_dir = Path(__file__).parent.parent / "data"
        
        with open(data_dir / "training_data.json", 'r') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=test_size, stratify=df['category']
        )
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        
        return train_df, val_df
        
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def train_model(
    train_df,
    val_df,
    model_save_path: str = None,
    num_epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    warmup_steps: int = 0,
    use_wandb: bool = True
):
    """
    Train the resume classifier with validation
    
    Args:
        train_df: Training data
        val_df: Validation data
        model_save_path: Where to save the model
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        warmup_steps: Number of warmup steps
        use_wandb: Whether to use Weights & Biases
    """
    try:
        # Initialize tracking
        if use_wandb:
            wandb.init(project="resumatch", config={
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "warmup_steps": warmup_steps
            })
        
        # Set up device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        # Initialize tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(train_df['category'].unique())
        )
        model.to(device)
        
        # Create datasets
        train_dataset = ResumeDataset(
            train_df['text'].values,
            train_df['category'].values,
            tokenizer
        )
        val_dataset = ResumeDataset(
            val_df['text'].values,
            val_df['category'].values,
            tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size
        )
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            model.train()
            train_loss = 0
            train_steps = 0
            
            progress_bar = tqdm(train_loader, desc="Training")
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_steps += 1
                
                progress_bar.set_postfix({'loss': train_loss / train_steps})
            
            avg_train_loss = train_loss / train_steps
            
            # Validation
            model.eval()
            val_loss = 0
            val_steps = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    val_steps += 1
                    
                    preds = torch.argmax(outputs.logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / val_steps
            
            # Calculate metrics
            classification_metrics = classification_report(
                all_labels,
                all_preds,
                output_dict=True
            )
            
            # Log metrics
            metrics = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'val_accuracy': classification_metrics['accuracy'],
                'val_f1_macro': classification_metrics['macro avg']['f1-score']
            }
            
            if use_wandb:
                wandb.log(metrics)
            
            logger.info(f"Train Loss: {avg_train_loss:.4f}")
            logger.info(f"Val Loss: {avg_val_loss:.4f}")
            logger.info(f"Val Accuracy: {metrics['val_accuracy']:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if model_save_path:
                    torch.save(model.state_dict(), model_save_path)
                    logger.info(f"Saved best model to {model_save_path}")
        
        if use_wandb:
            wandb.finish()
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Prepare data
    train_df, val_df = prepare_data()
    
    # Train model
    model_save_path = Path(__file__).parent / "models" / "best_model.pt"
    train_model(train_df, val_df, model_save_path)
