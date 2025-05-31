import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class ResumeAnalyzer(nn.Module):
    def __init__(self):
        super(ResumeAnalyzer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

# Initialize model and tokenizer
model = ResumeAnalyzer()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
