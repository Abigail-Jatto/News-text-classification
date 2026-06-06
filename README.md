
## Preprocessing

All preprocessing is implemented as modular, reusable functions:

| Function | Purpose |
|----------|---------|
| `sent_to_words()` | Tokenises raw sentences into word lists |
| `remove_stopwords()` | Removes Gensim stopword list |
| `build_ngram_models()` | Fits bigram and trigram Phraser models on training data only |
| `lemmatization()` | Reduces tokens to base form using spaCy, retaining nouns, adjectives, verbs and adverbs |
| `preprocess_pipeline()` | Orchestrates full preprocessing sequence; fitted on train, applied to val and test to prevent data leakage |

**Key design decision:** Bigram and trigram models are fitted exclusively on training data and applied to validation and test sets — preventing leakage of test distribution information into the feature engineering stage.


## Models Built and Evaluated

### 1. Multinomial Naïve Bayes
- Probabilistic baseline using TF-IDF weighted bag-of-words features
- Hyperparameter search across 24 combinations using GridSearchCV (3-fold CV)
- Parameters tuned: `ngram_range`, `max_features`, `sublinear_tf`, `alpha`
- **Why chosen:** Strong interpretable baseline for sparse high-dimensional text; establishes performance floor for comparison

### 2. Linear Support Vector Machine (LinearSVC)
- Discriminative classifier using TF-IDF features
- Hyperparameter search across 24 combinations using GridSearchCV
- Parameters tuned: `ngram_range`, `sublinear_tf`, `C`, `loss`, `dual`
- **Why chosen:** LinearSVC consistently outperforms Naïve Bayes on text classification due to its ability to find optimal decision boundaries in high-dimensional sparse feature spaces; provides the classical ML ceiling

### 3. Text CNN with Word2Vec Embeddings
- Convolutional neural network capturing local n-gram patterns
- Word2Vec embeddings trained on corpus; hyperparameter search across 8 combinations
- Parameters tuned: learning rate, dropout rate, batch size
- **Why chosen:** CNNs capture phrase-level local features missed entirely by bag-of-words approaches; computationally efficient relative to recurrent architectures

### 4. Bidirectional LSTM with TF-IDF Weighted Word2Vec Embeddings
- Sequential model capturing long-range dependencies in both directions
- Randomised hyperparameter search
- **Why chosen:** BiLSTM captures contextual meaning that depends on both preceding and following text — theoretically better suited to articles where sentence meaning is determined by full context, not just local phrases

---

## Results

| Model | Accuracy | Weighted F1 |
|-------|----------|-------------|
| Multinomial Naïve Bayes | 88% | 0.88 |
| LinearSVC | 89% | 0.89 |
| Text CNN | 91% | 0.91 |
| BiLSTM | 92% | 0.92 |

**Key finding:** Deep learning's advantage over well-tuned classical models is real but modest — approximately 3–4 percentage points at significantly greater computational cost and reduced interpretability. This finding matters: in production environments where inference speed, cost and explainability are constraints, a well-tuned LinearSVC may be the correct engineering choice even when a BiLSTM achieves marginally higher accuracy.

---

## Visualisations Produced

- Class distribution bar charts — training and test sets
- Article length histograms with mean and median overlaid
- Per-category article length boxplots
- Per-category word clouds generated from lemmatised tokens
- Training and validation accuracy/loss curves for all neural models
- Confusion matrices for all four models

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| pandas, numpy | Data manipulation |
| spaCy, Gensim | NLP preprocessing and embeddings |
| scikit-learn | Classical ML models and hyperparameter tuning |
| TensorFlow / Keras | Deep learning architectures |
| matplotlib, seaborn | Visualisation |
| WordCloud | Per category grouping |
| HuggingFace datasets | Data Collection |

---

## Path to Production

This project is currently implemented as a documented research notebook. 
A production-ready version would involve the following architectural changes:

- **Pipeline serialisation:** Fitted preprocessing objects (bigram model, 
trigram model, TF-IDF vectoriser) serialised using `joblib` or `pickle` 
for consistent inference without retraining
- **Model serving:** Best-performing model wrapped in a REST API endpoint 
(FastAPI or Flask) accepting raw text and returning predicted category 
with confidence score
- **Batch processing:** Pipeline refactored to support batch inference 
on streaming article feeds, enabling integration with live news APIs
- **Monitoring:** Prediction confidence distribution tracked over time 
to detect model drift as news language evolves
- **Retraining trigger:** Automated retraining pipeline triggered when 
classification confidence drops below threshold on incoming data

---
