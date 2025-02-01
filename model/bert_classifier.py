# import torch
# import torch.nn as nn
# from transformers import BertModel, BertTokenizer
# from torch.utils.data import DataLoader
# from utils.logger import logger
# from model.data_processor import EcommerceDataset


# class BertClassifier(nn.Module):
#     def __init__(self, num_classes):
#         super(BertClassifier, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.dropout = nn.Dropout(0.1)
#         self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
#         # Move BERT to GPU if available
#         if torch.cuda.is_available():
#             self.bert = self.bert.cuda()
#             self.classifier = self.classifier.cuda()

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         pooled_output = outputs.pooler_output
#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
#         return logits

# class ModelTrainer:
#     def __init__(self, num_classes=4):
#         # Check CUDA availability
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         if self.device.type == 'cuda':
#             logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
#             logger.info(f"CUDA Version: {torch.version.cuda}")
#         else:
#             logger.warning("GPU not available, using CPU!")
            
#         self.model = BertClassifier(num_classes).to(self.device)
#         self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
#         # Move model to GPU if available
#         if torch.cuda.is_available():
#             self.model = self.model.cuda()

#     def train(self, train_texts, train_labels, val_texts, val_labels, 
#               batch_size=16, epochs=3, learning_rate=2e-5):
        
#         train_dataset = EcommerceDataset(train_texts, train_labels, self.tokenizer)
#         val_dataset = EcommerceDataset(val_texts, val_labels, self.tokenizer)

#         train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_loader = DataLoader(val_dataset, batch_size=batch_size)

#         optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
#         criterion = nn.CrossEntropyLoss()

#         for epoch in range(epochs):
#             self.model.train()
#             total_loss = 0
            
#             for batch in train_loader:
#                 optimizer.zero_grad()
                
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['labels'].to(self.device)

#                 outputs = self.model(input_ids, attention_mask)
#                 loss = criterion(outputs, labels)
                
#                 loss.backward()
#                 optimizer.step()
#                 total_loss += loss.item()

#             avg_train_loss = total_loss / len(train_loader)
#             val_accuracy = self.evaluate(val_loader)
            
#             logger.info(f'Epoch {epoch+1}/{epochs}:')
#             logger.info(f'Average training loss: {avg_train_loss:.4f}')
#             logger.info(f'Validation accuracy: {val_accuracy:.4f}')

#     def evaluate(self, data_loader):
#         self.model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for batch in data_loader:
#                 input_ids = batch['input_ids'].to(self.device)
#                 attention_mask = batch['attention_mask'].to(self.device)
#                 labels = batch['labels'].to(self.device)

#                 outputs = self.model(input_ids, attention_mask)
#                 _, predicted = torch.max(outputs, 1)
                
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()

#         return correct / total

#     def predict(self, text):
#         self.model.eval()
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=128,
#             truncation=True,
#             padding='max_length',
#             return_tensors='pt'
#         )

#         input_ids = encoding['input_ids'].to(self.device)
#         attention_mask = encoding['attention_mask'].to(self.device)

#         with torch.no_grad():
#             outputs = self.model(input_ids, attention_mask)
#             _, predicted = torch.max(outputs, 1)
            
#         return predicted.item()

#     def save_model(self, path):
#         torch.save(self.model.state_dict(), path)

#     def load_model(self, path):
#         self.model.load_state_dict(torch.load(path))



import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from utils.logger import logger
from model.data_processor import EcommerceDataset

class BertClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class ModelTrainer:
    def __init__(self, num_classes=4):
        # Check CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            logger.warning("GPU not available, using CPU!")
            
        self.model = BertClassifier(num_classes).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def train(self, train_texts, train_labels, val_texts, val_labels, 
              batch_size=16, epochs=3, learning_rate=2e-5):
        
        train_dataset = EcommerceDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = EcommerceDataset(val_texts, val_labels, self.tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            val_accuracy = self.evaluate(val_loader)
            
            logger.info(f'Epoch {epoch+1}/{epochs}:')
            logger.info(f'Average training loss: {avg_train_loss:.4f}')
            logger.info(f'Validation accuracy: {val_accuracy:.4f}')

    def evaluate(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def predict(self, text):
        self.model.eval()
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            
        return predicted.item()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
