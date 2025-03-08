# Children's Book Age Classification with GPT-2 and LoRA

## Project Overview
This project implements a novel approach to age classification for children's books using a fine-tuned GPT-2 model with Low-Rank Adaptation (LoRA). The research aims to automatically determine the appropriate reading age for book descriptions.

### Key Features
- Custom GPT-2 language model training on Tiny Stories dataset
- LoRA fine-tuning for efficient model adaptation
- Book age classification using transformer architecture
- High-accuracy age prediction for children's literature

## Project Structure
```
children-book-age-classifier/
│
├── data/
│   ├── raw/
│   │   ├── tiny_stories.csv
│   │   └── children_books.csv
│   └── processed/
│       ├── train_books.csv
│       └── test_books.csv
│
├── models/
│   ├── tiny_stories_model.pt
│   └── age_classifier_model.pt
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_language_model_training.ipynb
│   └── 03_age_classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── requirements.txt
├── README.md
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU recommended

### Setup
1. Clone the repository
```bash
git clone https://github.com/fadelbt2002/children-book-age-classifier.git
cd children-book-age-classifier
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Datasets
- **Tiny Stories**: Synthetic dataset of short stories for young children
- **Highly Rated Children Books**: Book descriptions with age recommendations

## Model Architecture
- Base Model: GPT-2 Small
- Adaptation: Low-Rank Adaptation (LoRA)
- Classification Head: Two-layer MLP

## Performance
- Training Dataset: Tiny Stories
- Fine-tuning Dataset: Highly Rated Children Books
- Test Accuracy: 99.92%

## Key Components
1. Language Model Training
2. LoRA Fine-tuning
3. Age Classification

## References
1. Eldan, R., & Li, Y. (2023). TinyStories
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models

## Contributions
Contributions are welcome! Please read the contributing guidelines before getting started.
```
