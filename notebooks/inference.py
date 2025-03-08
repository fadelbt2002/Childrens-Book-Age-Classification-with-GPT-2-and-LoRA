import torch
from transformers import GPT2Tokenizer

class AgePredictor:
    """
    Inference class for age classification
    """
    def __init__(self, model, tokenizer, max_length=128):
        """
        Initialize predictor
        
        Args:
            model (nn.Module): Trained classification model
            tokenizer (GPT2Tokenizer): Text tokenizer
            max_length (int): Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def preprocess(self, text):
        """
        Preprocess input text
        
        Args:
            text (str): Book description
        
        Returns:
            dict: Tokenized input
        """
        # Append [cls] token
        text = text + " [cls]"
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': tokens['input_ids'].to(self.device),
            'attention_mask': tokens['attention_mask'].to(self.device)
        }
    
    def predict(self, text):
        """
        Predict age classification
        
        Args:
            text (str): Book description
        
        Returns:
            dict: Prediction details
        """
        # Preprocess input
        inputs = self.preprocess(text)
        
        # Disable gradient computation
        with torch.no_grad():
            # Forward pass
            logits = self.model(
                inputs['input_ids'], 
                inputs['attention_mask']
            )
            
            # Get prediction
            prediction = torch.argmax(logits, dim=1).item()
            probabilities = torch.softmax(logits, dim=1)
            confidence = probabilities[0][prediction].item()
        
        # Map prediction back to age
        predicted_age = prediction + 8
        
        return {
            'predicted_age': predicted_age,
            'confidence': confidence,
            'logits': logits.cpu().numpy()
        }
    
    def batch_predict(self, texts):
        """
        Predict ages for multiple texts
        
        Args:
            texts (list): List of book descriptions
        
        Returns:
            list: Predictions for each text
        """
        return [self.predict(text) for text in texts]

def load_predictor(model_path, tokenizer, model_class):
    """
    Load predictor from saved model
    
    Args:
        model_path (str): Path to saved model weights
        tokenizer (GPT2Tokenizer): Tokenizer
        model_class (type): Model class for reconstruction
    
    Returns:
        AgePredictor: Inference-ready predictor
    """
    # Reconstruct model
    model = model_class(tokenizer)
    
    # Load weights
    model.load_state_dict(torch.load(model_path))
    
    # Create predictor
    return AgePredictor(model, tokenizer)
