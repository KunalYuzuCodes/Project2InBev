
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
![image](https://github.com/user-attachments/assets/a522e8f9-3b3a-4345-8ac3-ed821f23e6a8)

Output for Electronics
![image](https://github.com/user-attachments/assets/7218621f-7ab5-44bf-9dfa-c774ca67406e)

Output for Household
![image](https://github.com/user-attachments/assets/af332c01-d0af-40af-9416-eece554d5202)

Output for Books
![image](https://github.com/user-attachments/assets/fece3051-909b-4dd6-921d-044694a4cb6c)

Output for Clothing and Accessories
![image](https://github.com/user-attachments/assets/32609bfe-83a0-40a6-8c06-de1485b4ff74)

Running the API 

![image](https://github.com/user-attachments/assets/c6dec41a-e023-4804-9513-26b3aa80fbcf)

**BONUS Task**

Swagger Screenshots - 

![image](https://github.com/user-attachments/assets/f9cc1859-2f90-463c-b0fc-242cf8a953d0)

![image](https://github.com/user-attachments/assets/a2b62777-c397-4939-85dc-0e7345764b24)



