# MediBot 💊

MediBot is a drug review and sentiment analysis application. It allows users and physicians to submit reviews for medications, analyzes the sentiment of those reviews using VADER sentiment analysis, and provides drug information (uses, side effects, and warnings) along with ML-based text classification.

## Features

- **Drug Reviews**: Users and physicians can submit reviews for various medications.
- **Sentiment Analysis**: Reviews are automatically scored and labeled (Positive / Neutral / Negative) using VADER sentiment analysis.
- **Drug Information Database**: Lookup uses, side effects, and warnings for medications.
- **Machine Learning**: Train a text classification model (TF-IDF + Logistic Regression) on review data to predict sentiment labels.
- **AI Chat Integration**: Connects to a local Ollama LLM for conversational features.
- **SQLite Database**: Stores and manages all submitted reviews persistently.

## Project Structure

```
.
├── app.py                     # Main application entry point
├── db.py                       # Database operations (SQLite) for reviews
├── ml.py                        # ML model training and prediction (TF-IDF + Logistic Regression)
├── sentiment.py                  # VADER sentiment analysis logic
├── utils.py                       # Utility functions (drug info loading, helpers)
├── ollama_client.py                # Client for interacting with a local Ollama LLM
├── generate_data.py                 # Script to generate synthetic drug review data
├── main.py                           # Script to generate synthetic drug info dataset
├── drug_info.csv                      # Dataset containing drug uses, side effects, and warnings
├── drug_reviews_dataset.csv            # Synthetic dataset of drug reviews
└── reviews.db                           # SQLite database storing user/physician reviews
```

## Requirements

- Python 3.9+
- pandas
- scikit-learn
- vaderSentiment
- requests
- (Optional) [Ollama](https://ollama.ai/) running locally for AI chat features

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup & Usage

1. **Clone the repository**

   ```bash
   git clone https://github.com/JAYANT5VJ/MEDIBOT.git
   cd MEDIBOT
   ```

2. **Generate sample data (optional)**

   ```bash
   python generate_data.py     # creates drug_reviews_dataset.csv
   python main.py               # creates drug_info.csv
   ```

3. **Initialize the database**

   The database is initialized automatically via `db.init_db()` when the app starts.

4. **Run the application**

   ```bash
   python app.py
   ```

5. **(Optional) Run Ollama for AI features**

   Make sure [Ollama](https://ollama.ai/) is running locally on `http://localhost:11434` if you want to use the AI chat functionality.

## Sentiment Analysis

Sentiment is calculated using VADER (`sentiment.py`):

- **Compound score ≥ 0.05** → Positive
- **Compound score ≤ -0.05** → Negative
- Otherwise → Neutral

## Machine Learning Model

`ml.py` trains a text classification pipeline:

- **Vectorizer**: TF-IDF (unigrams + bigrams)
- **Classifier**: Logistic Regression
- **Output**: Accuracy, classification report, and confusion matrix

## Database Schema

The `drug_reviews` table stores:

| Column            | Type    | Description                          |
|-------------------|---------|---------------------------------------|
| id                | INTEGER | Primary key (auto-increment)          |
| drug_name         | TEXT    | Name of the drug                      |
| role              | TEXT    | 'User' or 'Physician'                 |
| reviewer_name     | TEXT    | Name of the reviewer (optional)       |
| review_text       | TEXT    | The review content                    |
| sentiment_score   | REAL    | VADER compound sentiment score        |
| sentiment_label   | TEXT    | Positive / Neutral / Negative         |
| created_at        | TEXT    | Timestamp of the review (ISO format)  |

## Disclaimer

This project is for educational and research purposes only. It is **not** intended for actual medical advice. Always consult a qualified healthcare professional for medical concerns.

## License

This project is open-source and available under the MIT License.
