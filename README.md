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
│   │   └── children_books.csv
│   └── processed/
│       ├── train_books.csv
│       └── test_books.csv
├── models/
│   └── children_book_age_classifier.pth
│   
│
│
├── src/
│   ├── __init__.py
│   ├── data_processor.py
│   ├── model.py
│   ├── train.py
│   └── inference.py
│
├── requirements.txt
└──  README.md
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

### Tiny Stories Dataset
- **Source**: [Hugging Face - roneneldan/TinyStories-1M](https://huggingface.co/roneneldan/TinyStories-1M)
- **Description**: Synthetic dataset of short stories for young children
- **How to Download**:
  ```bash
  # Using Hugging Face Datasets library
  from datasets import load_dataset
  tiny_stories = load_dataset("roneneldan/TinyStories")
  ```

### Highly Rated Children Books Dataset
- **Source**: [Kaggle - Highly Rated Children Books and Stories](https://www.kaggle.com/datasets/thomaskonstantin/highly-rated-children-books-and-stories)
- **Description**: Book descriptions with age recommendations
- **How to Download**:
  1. Create a Kaggle account
  2. Navigate to the dataset page
  3. Download `children_books.csv`
  4. Place the file in `data/raw/children_books.csv`

### Data Preparation
```bash
# Create data directories
mkdir -p data/raw data/processed
```

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
1. Eldan, R., & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English?
2. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models
3. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. OpenAI
   - Original GPT-2 paper introducing the transformer-based language model
4. Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)

```
