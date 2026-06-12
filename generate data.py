import pandas as pd
import random

n = 200

drugs = [
    "Paracetamol","Ibuprofen","Amoxicillin","Metformin","Atorvastatin",
    "Amlodipine","Losartan","Omeprazole","Pantoprazole","Ciprofloxacin",
    "Azithromycin","Doxycycline","Cetirizine","Montelukast","Gabapentin",
    "Sertraline","Escitalopram","Diazepam","Insulin","Levothyroxine"
]

positive_reviews = [
    "The medication worked very well for my condition.",
    "Symptoms improved significantly after using this drug.",
    "Very effective treatment with minimal side effects.",
    "I felt better within a few days of starting the medicine.",
    "Doctor recommended this and it helped a lot."
]

neutral_reviews = [
    "The drug worked but results were moderate.",
    "Some improvement but not very significant.",
    "It helped slightly but took time to see results.",
    "Average effectiveness for my symptoms.",
    "It worked okay but nothing remarkable."
]

negative_reviews = [
    "I experienced several side effects after taking this.",
    "The medication did not help my symptoms.",
    "I had headaches and nausea while using this drug.",
    "Did not work well for me.",
    "Symptoms remained the same even after using it."
]

roles = ["User","Physician"]

rows = []

for i in range(n):

    sentiment = random.choice(["Positive","Neutral","Negative"])

    if sentiment == "Positive":
        review = random.choice(positive_reviews)
        rating = random.randint(7,10)

    elif sentiment == "Neutral":
        review = random.choice(neutral_reviews)
        rating = random.randint(4,6)

    else:
        review = random.choice(negative_reviews)
        rating = random.randint(1,3)

    rows.append({
        "drug_name": random.choice(drugs),
        "review_text": review,
        "rating": rating,
        "sentiment_label": sentiment,
        "role": random.choice(roles)
    })

df = pd.DataFrame(rows)

df.to_csv("drug_reviews_dataset.csv",index=False)

print("✅ 200 synthetic drug reviews generated → drug_reviews_dataset.csv")