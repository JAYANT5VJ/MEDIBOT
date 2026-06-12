from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text: str) -> tuple[float, str, dict]:
    """
    Returns (compound_score, label, full_scores)
    label: Positive / Neutral / Negative
    """
    if not text:
        return 0.0, "Neutral", {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    scores = _analyzer.polarity_scores(text)
    compound = float(scores.get("compound", 0.0))

    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return compound, label, scores