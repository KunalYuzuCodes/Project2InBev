<<<<<<< HEAD
# Project-2-for-InBev
Text Classification Model for Ubuntu Customer Centre Inquiries
=======
# Project2InBev
Text Classification Model for Ubuntu Customer Centre Inquiries

# E-commerce Text Classification

This project implements a BERT-based text classification model for categorizing e-commerce products into four categories:
- Electronics
- Household
- Books
- Clothing & Accessories

# Project Structure

project/
│
├── data/
│   └── ecommerceDataset.csv
│
├── model/
│   ├── __init__.py
│   ├── bert_classifier.py
│   └── data_processor.py
│
├── utils/
│   ├── __init__.py
│   └── logger.py
    └── cuda_check.py
│
├── main.py
├── config.py
├── app.py
├── requirements.txt
└── README.md

## Setup and Installation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

## Usage
1. Training the model:

python main.py --train


2. Making predictions:

python main.py --text "iPhone 13 Pro Max with 256GB storage"


3. Running the API:

uvicorn app:app --reload



**SCREENSHOTS**

Dataset Count and Accuracy
![image](https://github.com/user-attachments/assets/430d3a31-7789-4796-9f3b-c2a9d81af7b9)

Output for Electronics
![image](https://github.com/user-attachments/assets/258c5d45-8ff6-424a-8b5c-ac1f8ebf8e2a)

Output for Household
![image](https://github.com/user-attachments/assets/76c77597-cccf-4470-82ce-c3a7c1ed4cfe)

Output for Books
![image](https://github.com/user-attachments/assets/4a980c18-0354-41f6-b170-da7dc6ae2e28)

Output for Clothing and Accessories
![image](https://github.com/user-attachments/assets/817b69f8-c56d-4713-adaf-e161a7ba243d)

Running the API 

![image](https://github.com/user-attachments/assets/e688ae55-3b86-4d05-a099-a43d7f4fc1cd)

**BONUS Task**

Swagger Screenshots - 

![image](https://github.com/user-attachments/assets/d0f45740-bab4-4c77-b236-5423266ad3c2)

![image](https://github.com/user-attachments/assets/1c477289-bbc0-41ec-b5ad-922b88e2d0cd)



>>>>>>> 4784fd5 (Complete Changes readme)
