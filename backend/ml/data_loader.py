import os
import json
import torch
from torch.utils.data import Dataset, random_split
from transformers import BertTokenizer
from typing import Tuple, List, Dict
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Remove the batch dimension since DataLoader will handle batching
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

class KaggleDataLoader:
    def __init__(self, data_dir: str = 'data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.download_dataset()
        logger.info(f"Initialized KaggleDataLoader with data directory: {data_dir}")
    
    def download_dataset(self):
        """Download dataset from Kaggle if not already present"""
        try:
            # Check if files already exist
            resumes_path = os.path.join(self.data_dir, 'resumes.json')
            jobs_path = os.path.join(self.data_dir, 'job_descriptions.json')
            
            if os.path.exists(resumes_path) and os.path.exists(jobs_path):
                logger.info("Dataset files already exist, skipping download")
                return
            
            logger.info("Downloading dataset from Kaggle...")
            import kaggle
            logger.info("Using Kaggle credentials from: %s", os.path.expanduser('~/.kaggle/kaggle.json'))
            
            # Download the dataset
            kaggle.api.authenticate()
            logger.info("Kaggle authentication successful")
            
            logger.info("Downloading dataset: shreya2k3/resume-job-description-matching")
            kaggle.api.dataset_download_files(
                'shreya2k3/resume-job-description-matching',
                path=self.data_dir,
                unzip=True
            )
            logger.info("Dataset downloaded successfully to: %s", self.data_dir)
            
        except Exception as e:
            logger.error("Error downloading dataset: %s", str(e))
            logger.error("Exception type: %s", type(e).__name__)
            import traceback
            logger.error("Traceback:\n%s", traceback.format_exc())
            raise
    
    def prepare_data(self, val_split: float = 0.2) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        try:
            # Load the resume and job description data
            resumes_path = os.path.join(self.data_dir, 'resume_data.csv')
            jobs_path = os.path.join(self.data_dir, 'job_title_des.csv')
            
            logger.info("Loading resumes from %s", resumes_path)
            with open(resumes_path, 'r', encoding='utf-8') as f:
                resumes = json.load(f)
            logger.info("Loaded %d resumes", len(resumes))
            
            logger.info("Loading job descriptions from %s", jobs_path)
            with open(jobs_path, 'r', encoding='utf-8') as f:
                jobs = json.load(f)
            logger.info("Loaded %d job descriptions", len(jobs))
            
            # Print some sample data to verify structure
            if resumes:
                logger.info("Sample resume keys: %s", list(resumes[0].keys()))
            if jobs:
                logger.info("Sample job keys: %s", list(jobs[0].keys()))
            
            texts = []
            labels = []
            
            # Create positive and negative pairs
            for resume in resumes:
                resume_text = resume.get('text', '')
                resume_category = resume.get('category', '')
                
                if not resume_text or not resume_category:
                    logger.warning("Skipping resume without text or category")
                    continue
                
                logger.info("Processing resume in category: %s", resume_category)
                
                # Categorize jobs
                categorized_jobs = []
                for job in jobs:
                    title = job.get('title', '').lower()
                    requirements = job.get('requirements', '').lower()
                    description = job.get('description', '')
                    
                    if not description:
                        continue
                        
                    # Determine job category based on title and requirements
                    category = None
                    if 'software' in title or 'python' in requirements or 'javascript' in requirements:
                        category = 'Software Engineering'
                    elif 'data' in title or 'machine learning' in requirements:
                        category = 'Data Science'
                    elif 'web' in title or 'frontend' in requirements or 'react' in requirements:
                        category = 'Web Development'
                    
                    if category:
                        categorized_jobs.append((job, category))
                
                # Create positive pairs (matching jobs)
                matching_jobs = [job for job, cat in categorized_jobs if cat == resume_category]
                logger.info("Found %d matching jobs", len(matching_jobs))
                
                for job in matching_jobs[:2]:  # Use top 2 matching jobs
                    job_text = job.get('description', '')
                    if job_text:
                        combined_text = f"{resume_text} [SEP] {job_text}"
                        texts.append(combined_text)
                        labels.append(1)  # Match
                
                # Create negative pairs (non-matching jobs)
                non_matching_jobs = [job for job, cat in categorized_jobs if cat != resume_category]
                logger.info("Found %d non-matching jobs", len(non_matching_jobs))
                
                for job in non_matching_jobs[:2]:  # Use top 2 non-matching jobs
                    job_text = job.get('description', '')
                    if job_text:
                        combined_text = f"{resume_text} [SEP] {job_text}"
                        texts.append(combined_text)
                        labels.append(0)  # No match
            
            logger.info("Created %d training pairs (%d positive, %d negative)",
                       len(texts),
                       sum(labels),
                       len(labels) - sum(labels))
            
            if not texts:
                raise ValueError("No valid training pairs were created")
            
            # Create dataset
            dataset = ResumeDataset(texts, labels, self.tokenizer)
            
            # Split into train and validation
            val_size = int(len(dataset) * val_split)
            train_size = len(dataset) - val_size
            
            generator = torch.Generator().manual_seed(42)
            train_dataset, val_dataset = random_split(
                dataset,
                [train_size, val_size],
                generator=generator
            )
            
            logger.info(f"Created datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
            return train_dataset, val_dataset
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

def prepare_training_data():
    """Main function to prepare training data"""
    logger.info("Starting training data preparation...")
    loader = KaggleDataLoader()
    
    # Prepare data
    train_dataset, val_dataset = loader.prepare_data()
    
    if not train_dataset:
        logger.error("Failed to prepare datasets")
        return None, None
    
    logger.info("Training data preparation completed successfully")
    return train_dataset, val_dataset

if __name__ == "__main__":
    prepare_training_data()
