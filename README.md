# MediBot

MediBot is a drug review and sentiment analysis chatbot application. It allows users and
physicians to submit reviews for medications, analyzes the sentiment of those reviews using VADER
sentiment analysis, provides drug information (uses, side effects, warnings), and includes an
ML-based text classifier and an AI chat assistant powered by Groq.

## Live Demo

[https://drugreview-medibot.streamlit.app/](https://drugreview-medibot.streamlit.app/)

> **Note:** On the hosted version, log in with the demo credentials below, or click
> **"Skip Login (Continue as Guest)"** to explore immediately with limited access.
>
> The **AI Chatbot (Info Mode)** is powered by Groq and works on both the hosted version and
> locally. All features (Home Dashboard, Review Mode, ML Training, Review Database for Admin)
> work normally.

## Features

- **Drug Reviews** — Users, physicians, and guests can submit reviews for various medications.
- **Sentiment Analysis** — Reviews are automatically scored and labeled (Positive / Neutral /
  Negative) using VADER sentiment analysis.
- **Live Analytics Dashboard** — charts for sentiment by drug, review counts, and sentiment by
  reviewer role, updating automatically as reviews are submitted.
- **Drug Information Lookup** — uses, side effects, and warnings for medications from a dataset.
- **AI Chatbot (Info Mode)** — powered by Groq API for conversational drug information, works on
  both local and hosted versions.
- **Review Mode Chatbot** — query stored reviews for a drug using partial name search and see
  sentiment breakdowns by role. Available to Admin and Guest users.
- **Machine Learning** — train a text classification model (TF-IDF + Logistic Regression) on
  review data to predict sentiment labels (Admin only, requires minimum 15 reviews).
- **Review Database Management** — view, edit, and delete stored reviews (Admin only).
- **Role-based Access** — User, Physician, Admin, and Guest roles with different permissions.
- **SQLite Database** — stores and manages all submitted reviews persistently.

## Login Credentials (Demo)

| Username     | Password | Role      | Access                                      |
|--------------|----------|-----------|---------------------------------------------|
| `user1`      | `123`    | User      | Submit reviews, view dashboard              |
| `physician1` | `123`    | Physician | Submit reviews, view dashboard              |
| `admin`      | `admin`  | Admin     | Full access including ML Training and DB    |

Or click **"Skip Login (Continue as Guest)"** for instant access with User-level permissions plus
Review Mode lookup.

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── db.py                   # Database operations (SQLite) for reviews
├── ml.py                    # ML model training and prediction (TF-IDF + Logistic Regression)
├── sentiment.py              # VADER sentiment analysis logic
├── utils.py                   # Utility functions (drug info loading, helpers)
├── groq_client.py              # Groq API client for AI chat
├── ollama_client.py             # Legacy Ollama client (kept for reference)
├── requirements.txt              # Python dependencies
├── drug_info.csv                  # Dataset containing drug uses, side effects, and warnings
└── reviews.db                      # SQLite database storing reviews (auto-created on first run)
```

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/JAYANT5VJ/MEDIBOT.git
cd MEDIBOT
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Set up Groq API key

Get a free API key from [console.groq.com](https://console.groq.com), then create a file at
`.streamlit/secrets.toml`:

```
GROQ_API_KEY = "your_key_here"
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

The SQLite database (`reviews.db`) and tables are created automatically on first run.

## Sentiment Analysis

Sentiment is calculated using VADER (`sentiment.py`):

- **Compound score >= 0.05** → Positive
- **Compound score <= -0.05** → Negative
- Otherwise → Neutral

## Machine Learning Model

`ml.py` trains a text classification pipeline (Admin only, via the "ML Training" tab):

- **Vectorizer**: TF-IDF (unigrams + bigrams)
- **Classifier**: Logistic Regression
- **Output**: Accuracy, classification report, and confusion matrix
- **Minimum data required**: at least 15 reviews with a mix of Positive, Neutral and Negative
- Training data source: stored reviews or an uploaded CSV

## Database Schema

The `drug_reviews` table stores:

| Column            | Type    | Description                          |
|-------------------|---------|---------------------------------------|
| id                | INTEGER | Primary key (auto-increment)          |
| drug_name         | TEXT    | Name of the drug                      |
| role              | TEXT    | User, Physician, Admin, or Guest      |
| reviewer_name     | TEXT    | Name of the reviewer (optional)       |
| review_text       | TEXT    | The review content                    |
| sentiment_score   | REAL    | VADER compound sentiment score        |
| sentiment_label   | TEXT    | Positive / Neutral / Negative         |
| created_at        | TEXT    | Timestamp of the review (ISO format)  |

> **Note:** On Streamlit Community Cloud (free tier), the filesystem is temporary — `reviews.db`
> may reset when the app restarts after inactivity. For permanent storage, use an external
> database (Google Sheets integration planned).

## Tech Stack

| Category         | Tools                              |
|------------------|------------------------------------|
| Language         | Python 3                           |
| Web Framework    | Streamlit                          |
| AI Chat          | Groq API (llama-3.1-8b-instant)    |
| Sentiment        | VADER (vaderSentiment)             |
| ML               | Scikit-learn (TF-IDF + LogReg)     |
| Database         | SQLite                             |
| Data Handling    | Pandas, NumPy                      |
| Visualization    | Matplotlib                         |

## Deployment

This app is deployed for free on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push the repository to GitHub.
2. Go to share.streamlit.io → "New app".
3. Select the repo, branch (`main`), and main file (`app.py`).
4. Add your `GROQ_API_KEY` under Settings → Secrets.
5. Click "Deploy".

## Disclaimer

This project is for educational and research purposes only. It is **not** intended for actual
medical advice. Always consult a qualified healthcare professional for medical concerns.

## License

This project is open-source and available under the MIT License.
