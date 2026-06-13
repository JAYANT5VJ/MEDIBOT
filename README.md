# MediBot 💊

MediBot is a drug review and sentiment analysis chatbot application. It allows users and
physicians to submit reviews for medications, analyzes the sentiment of those reviews using VADER
sentiment analysis, provides drug information (uses, side effects, warnings), and includes an
ML-based text classifier and an optional AI chat assistant powered by Ollama.

## Live Demo

🔗 [https://drugreview-medibot.streamlit.app/](https://drugreview-medibot.streamlit.app/)

> **Note:** On the hosted version, log in with the demo credentials below, or click
> **"Skip Login (Continue as Guest)"** to explore immediately with limited access.
>
> The **AI Chatbot (Info Mode)** requires a locally running Ollama instance, so it is **not
> available** on the hosted demo — it will show a friendly notice instead. All other features
> (Home Dashboard, Review Mode, ML Training, Review Database for Admin) work normally.

## Features

- **Drug Reviews** — Users and physicians can submit reviews for various medications.
- **Sentiment Analysis** — Reviews are automatically scored and labeled (Positive / Neutral /
  Negative) using VADER sentiment analysis.
- **Live Analytics Dashboard** — charts for sentiment by drug, review counts, and sentiment by
  reviewer role, updating automatically as reviews are submitted.
- **Drug Information Lookup** — uses, side effects, and warnings for medications from a dataset.
- **AI Chatbot (Info Mode)** — connects to a local Ollama LLM for conversational drug information
  (local use only).
- **Review Mode Chatbot** — query stored reviews for a drug and see sentiment breakdowns by role.
- **Machine Learning** — train a text classification model (TF-IDF + Logistic Regression) on
  review data to predict sentiment labels (Admin only).
- **Review Database Management** — view, edit, and delete stored reviews (Admin only).
- **Role-based Access** — User, Physician, Admin, and Guest roles with different permissions.
- **SQLite Database** — stores and manages all submitted reviews.

## Login Credentials (Demo)

| Username     | Password | Role      |
|--------------|----------|-----------|
| `user1`      | `123`    | User      |
| `physician1` | `123`    | Physician |
| `admin`      | `admin`  | Admin     |

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
├── ollama_client.py             # Client for interacting with a local Ollama LLM
├── requirements.txt              # Python dependencies
├── drug_info.csv                  # Dataset containing drug uses, side effects, and warnings
└── reviews.db                      # SQLite database storing user/physician reviews (auto-created)
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

### 3. Run the application

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

The SQLite database (`reviews.db`) and tables are created automatically on first run.

### 4. (Optional) Enable the AI Chatbot with Ollama

The **Info Mode** chatbot uses [Ollama](https://ollama.ai/) to run a local LLM. To enable it:

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Pull a model (default used is `llama3`):
   ```bash
   ollama pull llama3
   ```
3. Make sure Ollama is running (it runs as a background service after installation)
4. In the app sidebar, confirm the **Ollama URL** (`http://localhost:11434`) and **Ollama Model**
   (`llama3`) match your setup

Without Ollama running, Info Mode will show a friendly "AI chat isn't available" message instead
of an error — all other features work normally.

## Sentiment Analysis

Sentiment is calculated using VADER (`sentiment.py`):

- **Compound score ≥ 0.05** → Positive
- **Compound score ≤ -0.05** → Negative
- Otherwise → Neutral

## Machine Learning Model

`ml.py` trains a text classification pipeline (Admin only, via the "ML Training" tab):

- **Vectorizer**: TF-IDF (unigrams + bigrams)
- **Classifier**: Logistic Regression
- **Output**: Accuracy, classification report, and confusion matrix
- Training data source: stored reviews (`sentiment_label`) or an uploaded CSV

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

> **Note:** On Streamlit Community Cloud (free tier), the filesystem is temporary — `reviews.db`
> may reset when the app restarts after inactivity. For permanent storage, use an external
> database.

## Deployment

This app is deployed for free on [Streamlit Community Cloud](https://share.streamlit.io):

1. Push the repository to GitHub.
2. Go to share.streamlit.io → "New app".
3. Select the repo, branch (`main`), and main file (`app.py`).
4. Click "Deploy".

## Disclaimer

This project is for educational and research purposes only. It is **not** intended for actual
medical advice. Always consult a qualified healthcare professional for medical concerns.

## License

This project is open-source and available under the MIT License.
