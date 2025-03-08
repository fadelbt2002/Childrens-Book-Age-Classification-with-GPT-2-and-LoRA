import torch
from transformers import GPT2Tokenizer

from src.data_processor import prepare_datasets, add_special_tokens
from src.model import initialize_model
from src.train import train_model, save_model
from src.inference import AgePredictor

def main():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer = add_special_tokens(tokenizer)

    # Prepare datasets
    train_dataset, test_dataset = prepare_datasets(
        'data/raw/children_books.csv', 
        tokenizer
    )

    # Initialize model
    model, tokenizer = initialize_model(
        tokenizer, 
        num_classes=7,  # Ages 8-14
        random_init=True,
        apply_lora=True
    )

    # Train model
    history = train_model(
        model, 
        train_dataset, 
        test_dataset, 
        num_epochs=5,
        batch_size=32,
        learning_rate=5e-5
    )

    # Save model
    save_model(model, 'children_book_age_classifier.pth')

    # Create predictor for inference
    predictor = AgePredictor(model, tokenizer)

    # Example predictions
    test_texts = [
        "A thrilling adventure of a young detective solving mysteries in her neighborhood.",
        "A complex novel exploring teenage relationships and personal growth.",
        "A gentle story about friendship and kindness for younger readers."
    ]

    for text in test_texts:
        prediction = predictor.predict(text)
        print(f"Text: {text}")
        print(f"Predicted Age: {prediction['predicted_age']}")
        print(f"Confidence: {prediction['confidence']:.2%}\n")

if __name__ == "__main__":
    main()
