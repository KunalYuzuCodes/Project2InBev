import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from utils.logger import logger

class EcommerceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def clean_text(text):
    """Clean text data"""
    if pd.isna(text):
        return ""
    # Convert to string if not already
    text = str(text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    return text

def load_and_preprocess_data(file_path, test_size=0.2):
    logger.info("Loading and preprocessing data...")
    
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise

    # Display initial dataset info
    logger.info(f"Initial dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Rename columns if needed
    if len(df.columns) >= 2:
        df.columns = ['category', 'text']
    else:
        raise ValueError("Dataset must have at least 2 columns")

    # Remove rows with missing values
    initial_size = len(df)
    df = df.dropna()
    logger.info(f"Removed {initial_size - len(df)} rows with missing values")

    # Clean text data
    df['text'] = df['text'].apply(clean_text)
    
    # Remove rows with empty text after cleaning
    df = df[df['text'].str.len() > 0]
    
    # Convert labels to numerical values
    label_map = {
        'Electronics': 0,
        'Household': 1,
        'Books': 2,
        'Clothing & Accessories': 3
    }
    
    # Check for unknown categories
    unknown_categories = set(df['category'].unique()) - set(label_map.keys())
    if unknown_categories:
        logger.warning(f"Found unknown categories: {unknown_categories}")
        df = df[df['category'].isin(label_map.keys())]

    # Convert categories to numerical labels
    df['label'] = df['category'].map(label_map)
    
    # Remove any rows where label mapping failed
    df = df.dropna(subset=['label'])
    
    # Convert label to int
    df['label'] = df['label'].astype(int)

    # Log class distribution
    logger.info("Class distribution:")
    for category, count in df['category'].value_counts().items():
        logger.info(f"{category}: {count}")

    # Split the data
    texts = df['text'].values
    labels = df['label'].values

    # Verify no NaN values in the final dataset
    if np.isnan(labels).any():
        raise ValueError("Labels contain NaN values after preprocessing")
    if any(pd.isna(texts)):
        raise ValueError("Texts contain NaN values after preprocessing")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
    except Exception as e:
        logger.error(f"Error in train-test split: {e}")
        raise

    logger.info(f"Training set size: {len(X_train)}")
    logger.info(f"Test set size: {len(X_test)}")

    return X_train, X_test, y_train, y_test, label_map

def load_and_preprocess_single_text(text):
    """Preprocess a single text input"""
    return clean_text(text)
