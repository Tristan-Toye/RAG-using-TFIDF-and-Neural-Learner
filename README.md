# RAG Recipe Search System

A sophisticated Recipe Retrieval-Augmented Generation (RAG) system that combines TF-IDF and neural embeddings for intelligent recipe search and recommendation.

## 🍳 Overview

This project implements a hybrid information retrieval system specifically designed for recipe search, combining traditional TF-IDF techniques with modern neural embeddings and Large Language Models (LLMs) to provide intelligent recipe recommendations and answers to cooking queries.

## ✨ Features

- **Hybrid Search**: Combines sparse (TF-IDF) and dense (neural embeddings) representations
- **Intelligent Ranking**: Multiple ranking strategies with configurable parameters
- **Recipe Understanding**: Advanced text preprocessing including lemmatization and phrase extraction
- **LLM Integration**: Mistral-7B model for natural language recipe responses
- **Comprehensive Evaluation**: Multiple IR metrics including nDCG, MAP, Precision, Recall, and F1
- **Query Intent Classification**: Fine-tuned model for understanding user intent
- **Dish Name Recognition**: Specialized handling of recipe names and food entities

## 🏗️ Architecture

### Core Components

1. **TFIDF.py** - Traditional TF-IDF vectorization with LSA dimensionality reduction
2. **NeuralLearner.py** - Neural embeddings using SentenceTransformers
3. **Ranking.py** - Hybrid ranking system combining sparse and dense similarities
4. **RecipeRAG.py** - RAG system with query routing and response generation
5. **Evaluation.py** - Comprehensive evaluation metrics
6. **LLM_simple.py** - Simple LLM integration for recipe responses

### Data Flow

```
User Query → Query Preprocessing → Hybrid Search (TF-IDF + Neural) → 
Ranking & Filtering → LLM Response Generation → Formatted Answer
```

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from main import Ranking
import datasets

# Load recipe dataset
dataset = datasets.load_dataset("parquet", data_files="./irse_documents_2025_recipes.parquet")['train']
dataset = dataset.to_pandas()

# Initialize ranking system
ranking = Ranking(dataset)

# Search for recipes
query = "easy vegetarian lasagna recipe"
results, all_results = ranking.query(query)

print(results[["name", "score", "tags"]])
```

### RAG System Usage

```python
from RecipeRAG import RecipeRAG

# Initialize RAG system
rag = RecipeRAG()

# Generate response
query = "How do I make a quick pasta dish?"
response, intent, prompt = rag.generate_response(query, docs, tags, ingredients, names)
print(response)
```

## 📊 Evaluation

The system includes comprehensive evaluation metrics:

- **nDCG@10**: Normalized Discounted Cumulative Gain
- **MAP**: Mean Average Precision
- **Precision/Recall/F1**: Both macro and micro averaging
- **Custom Metrics**: Recipe-specific relevance scoring

Run evaluation:

```python
from Evaluation import Evaluate

evaluator = Evaluate()
macro_metrics, micro_metrics = evaluator.evaluate_with_relevance(queries, dataset)
print(f"nDCG@10: {macro_metrics['ndcg']:.3f}")
print(f"MAP: {macro_metrics['MAP']:.3f}")
```

## ⚙️ Configuration

Key parameters in `main.py`:

- `ALPHA`: Weight for combining sparse and dense similarities (default: 0.85)
- `MODE`: Ranking strategy ("statistical", "relative", "absolute", "derivative")
- `PARAM`: Sensitivity parameter for ranking
- `SVD_COMPONENTS`: LSA dimensionality (default: 250)
- `MIN_SCORE`: Minimum number of results to return

## 📁 Project Structure

```
RAG-using-TFIDF-and-Neural-Learner/
├── main.py                 # Main execution script
├── TFIDF.py               # TF-IDF vectorization and preprocessing
├── NeuralLearner.py       # Neural embeddings and evaluation
├── Ranking.py             # Hybrid ranking system
├── RecipeRAG.py           # RAG system with LLM integration
├── Evaluation.py          # Evaluation metrics
├── LLM_simple.py          # Simple LLM wrapper
└── storage/               # Cached models and embeddings
    ├── phrases/
    ├── lemma/
    ├── vectorizers/
    ├── matrixs/
    └── embeddings/
```

## 🔧 Advanced Features

### Query Preprocessing
- Lemmatization using spaCy
- Phrase extraction with RAKE and spaCy
- Dish name recognition and duplication
- Text normalization and canonical mapping

### Ranking Strategies
- **Statistical**: Mean + param * std threshold
- **Relative**: Max score * param threshold
- **Absolute**: Fixed threshold
- **Derivative**: Dynamic threshold based on score changes

### Neural Components
- SentenceTransformer embeddings (all-MiniLM-L6-v2)
- LSA dimensionality reduction
- Cosine similarity computation
- GPU acceleration support

## 📈 Performance

The system achieves competitive performance on recipe search tasks:
- Fast query processing with cached embeddings
- Scalable to large recipe datasets
- Configurable trade-offs between speed and accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Recipe dataset provided by KU Leuven
- Built with Hugging Face Transformers and spaCy
- Evaluation metrics from Information Retrieval standards 
