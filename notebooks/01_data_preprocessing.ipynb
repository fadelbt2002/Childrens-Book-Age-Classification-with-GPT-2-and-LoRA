import pandas as pd
import re
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class ChildrenBookDataset(Dataset):
    """
    Custom PyTorch Dataset for Children's Book Age Classification
    """
    def __init__(self, csv_path, tokenizer, max_length=128):
        """
        Initialize the dataset
        
        Args:
            csv_path (str): Path to the CSV file
            tokenizer (GPT2Tokenizer): Tokenizer for encoding text
            max_length (int): Maximum sequence length
        """
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Extract minimum age
        self.data['min_age'] = self.data['Reading_age'].apply(self._extract_min_age)
        self.data.dropna(subset=['Desc', 'min_age'], inplace=True)
    
    def _extract_min_age(self, age):
        """
        Extract minimum age from reading age column
        
        Args:
            age (str): Reading age from dataset
        
        Returns:
            int: Minimum age or None
        """
        if isinstance(age, str):
            match = re.search(r'\d+', age)
            return int(match.group()) if match else None
        return None
    
    def __len__(self):
        """
        Get dataset length
        
        Returns:
            int: Number of samples
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            dict: Tokenized sample with input_ids, attention_mask, and label
        """
        desc = self.data.iloc[idx]['Desc'] + " [cls]"
        label = int(self.data.iloc[idx]['min_age']) - 8  # Normalize age
        
        tokens = self.tokenizer(
            desc,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_datasets(csv_path, tokenizer, train_split=0.8, random_state=42):
    """
    Prepare train and test datasets
    
    Args:
        csv_path (str): Path to the CSV file
        tokenizer (GPT2Tokenizer): Tokenizer for encoding text
        train_split (float): Proportion of data for training
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: Train and test datasets
    """
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    train_size = int(len(df) * train_split)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Save processed datasets
    train_df.to_csv('data/processed/train_books.csv', index=False)
    test_df.to_csv('data/processed/test_books.csv', index=False)
    
    train_dataset = ChildrenBookDataset('data/processed/train_books.csv', tokenizer)
    test_dataset = ChildrenBookDataset('data/processed/test_books.csv', tokenizer)
    
    return train_dataset, test_dataset

def add_special_tokens(tokenizer):
    """
    Add [cls] special token to tokenizer and resize embeddings
    
    Args:
        tokenizer (GPT2Tokenizer): Tokenizer to modify
    
    Returns:
        GPT2Tokenizer: Updated tokenizer
    """
    special_token = "[cls]"
    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})
    return tokenizer
