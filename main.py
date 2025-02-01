import argparse
import sys
import torch
import os
from model.data_processor import load_and_preprocess_data, load_and_preprocess_single_text
from model.bert_classifier import ModelTrainer
from utils.logger import logger

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DATA_PATH = os.path.join(DATA_DIR, 'ecommerceDataset.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.pth')

def verify_gpu():
    """Verify GPU availability and print device information"""
    if torch.cuda.is_available():
        logger.info("===============GPU Information===============")
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"PyTorch CUDA: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Current GPU Device: {torch.cuda.current_device()}")
        logger.info(f"GPU Memory Usage: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        logger.info("===========================================")
        # Set default GPU
        torch.cuda.set_device(0)
        return True
    else:
        logger.warning("No GPU available! Running on CPU.")
        return False

def train_model(model_trainer, label_map):
    """Train the model with error handling and GPU support"""
    try:
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test, label_map = load_and_preprocess_data(DATA_PATH)
        
        # Log training details
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Print initial GPU memory
        if torch.cuda.is_available():
            logger.info(f"GPU Memory before training: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        
        # Training parameters
        training_params = {
            'batch_size': 16,  # Adjust based on your GPU memory
            'epochs': 3,
            'learning_rate': 2e-5
        }
        
        logger.info("Starting training with parameters:")
        for param, value in training_params.items():
            logger.info(f"{param}: {value}")
        
        # Train the model
        model_trainer.train(
            X_train, y_train, X_test, y_test,
            batch_size=training_params['batch_size'],
            epochs=training_params['epochs'],
            learning_rate=training_params['learning_rate']
        )
        
        # Save the model
        os.makedirs(MODELS_DIR, exist_ok=True)
        model_trainer.save_model(MODEL_PATH)
        
        # Print final GPU memory
        if torch.cuda.is_available():
            logger.info(f"GPU Memory after training: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            # Clear cache
            torch.cuda.empty_cache()
        
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        return False

def predict_text(model_trainer, text, label_map):
    """Make prediction for a single text input"""
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Model file not found. Please train the model first.")
        
        # Load the model
        model_trainer.load_model(MODEL_PATH)
        
        # Preprocess the input text
        processed_text = load_and_preprocess_single_text(text)
        
        # Make prediction
        prediction = model_trainer.predict(processed_text)
        
        # Convert numerical prediction to category
        reverse_label_map = {v: k for k, v in label_map.items()}
        predicted_category = reverse_label_map[prediction]
        
        return predicted_category
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return None

def main():
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='E-commerce Text Classification')
        parser.add_argument('--text', type=str, help='Text to classify')
        parser.add_argument('--train', action='store_true', help='Train the model')
        args = parser.parse_args()

        # Verify GPU and set device
        is_gpu_available = verify_gpu()
        device = torch.device("cuda" if is_gpu_available else "cpu")
        
        # Create directories
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Check if dataset exists
        if not os.path.exists(DATA_PATH):
            logger.error(f"Dataset not found at {DATA_PATH}")
            sys.exit(1)

        # Initialize model trainer
        model_trainer = ModelTrainer(num_classes=4)  # 4 classes for our e-commerce categories
        
        # Define label map
        label_map = {
            'Electronics': 0,
            'Household': 1,
            'Books': 2,
            'Clothing & Accessories': 3
        }

        # Training mode
        if args.train:
            logger.info("=== Starting Training Mode ===")
            success = train_model(model_trainer, label_map)
            if not success:
                sys.exit(1)

        # Prediction mode
        if args.text:
            logger.info("=== Starting Prediction Mode ===")
            predicted_category = predict_text(model_trainer, args.text, label_map)
            if predicted_category:
                logger.info(f"Input text: {args.text}")
                logger.info(f"Predicted category: {predicted_category}")
            else:
                sys.exit(1)

        # If no arguments provided
        if not args.train and not args.text:
            logger.error("Please specify --train to train the model or --text to make a prediction")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
