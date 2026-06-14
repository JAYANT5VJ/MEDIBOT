import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import os

# Load .env for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load Groq API key from Streamlit secrets (cloud) or .env (local)
GROQ_API_KEY = ""
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

from db import init_db, insert_review, load_reviews_df, delete_review, update_review
from sentiment import vader_sentiment
from utils import load_drug_info, top_n_drugs_from_reviews
from groq_client import groq_chat
from ml import train_text_model, predict_text

st.set_page_config(page_title="Drug Review Chatbot", layout="wide")
# ==========================
# LOGIN SYSTEM
# ==========================

USERS = {
    "user1": {"password": "123", "role": "User"},
    "physician1": {"password": "123", "role": "Physician"},
    "admin": {"password": "admin", "role": "Admin"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

if not st.session_state.logged_in:

    st.title("Drug Review System Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login", use_container_width=True):

            if username in USERS and USERS[username]["password"] == password:

                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = USERS[username]["role"]

                st.success("Login successful")
                st.rerun()

            else:
                st.error("Invalid login")

    with col2:
        if st.button("Skip Login (Continue as Guest)", use_container_width=True):
            st.session_state.logged_in = True
            st.session_state.username = "Guest"
            st.session_state.role = "User"
            st.rerun()

    st.caption("Guests can submit reviews and view the dashboard with **User** access. "
               "Admin/Physician features require logging in.")

    st.stop()
# ---------- Init ----------
init_db()
drug_df = load_drug_info()
reviews_df = load_reviews_df()

fallback_drugs = drug_df["drug_name"].dropna().unique().tolist()
drug_list = top_n_drugs_from_reviews(reviews_df, fallback_drugs, n=30)

# ---------- Sidebar ----------
st.sidebar.title("⚙️ Settings")
# ==========================
# USER INFO
# ==========================

st.sidebar.markdown("---")
st.sidebar.subheader("User Session")

st.sidebar.write("User:", st.session_state.username)
st.sidebar.write("Role:", st.session_state.role)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()
groq_model = st.sidebar.selectbox("AI Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"], index=0)
temperature = st.sidebar.slider("Chat temperature", 0.0, 1.0, 0.2, 0.05)
st.sidebar.caption("ℹ️ AI Chat powered by Groq — works on both local and hosted versions.")

st.sidebar.markdown("---")
st.sidebar.caption("Safety note: This tool is for research support, not medical diagnosis.")

# ---------- Helpers ----------
def get_drug_row(drug_name: str):
    if drug_df.empty:
        return None
    m = drug_df["drug_name"].str.lower() == str(drug_name).lower()
    if m.any():
        return drug_df[m].iloc[0].to_dict()
    return None

def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str):
    fig, ax = plt.subplots()
    series.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

def plot_role_sentiment(df: pd.DataFrame):
    if df.empty:
        st.info("No reviews yet.")
        return
    pivot = df.groupby(["role"])["sentiment_score"].mean().sort_values(ascending=False)
    plot_bar(pivot, "Average Sentiment by Role", "Role", "Avg VADER Compound")

def plot_drug_sentiment(df: pd.DataFrame, top_n: int = 10):
    if df.empty:
        st.info("No reviews yet.")
        return
    avg = df.groupby("drug_name")["sentiment_score"].mean()
    cnt = df["drug_name"].value_counts()
    # show most-reviewed top_n
    top = cnt.head(top_n).index
    series = avg.loc[top].sort_values(ascending=False)
    plot_bar(series, f"Average Sentiment (Top {top_n} Most-Reviewed Drugs)", "Drug", "Avg VADER Compound")

def plot_review_counts(df: pd.DataFrame, top_n: int = 10):
    if df.empty:
        st.info("No reviews yet.")
        return
    cnt = df["drug_name"].value_counts().head(top_n)
    plot_bar(cnt, f"Review Counts (Top {top_n})", "Drug", "Count")

# ---------- Title ----------
st.title("🧪Drug Information + Review Analytics Chatbot (Streamlit + Ollama + VADER + ML)")

tabs = st.tabs(["🏠 Home Dashboard", "🤖 Chatbot", "🧠 ML Training", "🗃️ Review Database"])

# =======================
# TAB 1: HOME DASHBOARD
# =======================
with tabs[0]:
    left, right = st.columns([1.1, 1.2], gap="large")

    with left:
        st.subheader("Submit / Update Drug Review")
        role = st.session_state.role

        st.info(f"Logged in as **{role}** reviewer")
        reviewer_name = st.text_input("Name (optional)", value="")

        drug_name = st.selectbox("Select drug (top ~30)", drug_list if drug_list else ["(No drugs loaded)"])

        review_text = st.text_area("Write review", height=160, placeholder="Describe experience, effects, side effects...")

        colA, colB = st.columns(2)
        with colA:
            if st.button("✅ Submit Review", use_container_width=True):
                if not drug_name or drug_name.startswith("("):
                    st.error("Please select a valid drug.")
                elif not review_text.strip():
                    st.error("Please enter review text.")
                else:
                    score, label, full = vader_sentiment(review_text)
                    insert_review(
                        drug_name=drug_name,
                        role=role,
                        reviewer_name=reviewer_name.strip() if reviewer_name.strip() else None,
                        review_text=review_text.strip(),
                        sentiment_score=score,
                        sentiment_label=label,
                        created_at_iso=datetime.now(timezone.utc).isoformat()
                    )
                    st.success(f"Stored review ✅  Sentiment: **{label}** (compound={score:.3f})")
                    st.rerun()

        with colB:
            st.caption("Tip: physicians can submit updates too. All updates refresh charts automatically.")

        st.markdown("---")
        st.subheader("Drug Information (from dataset)")
        info = get_drug_row(drug_name) if drug_name and not drug_name.startswith("(") else None
        if info:
            st.markdown(f"**Uses:** {info.get('uses','-')}")
            st.markdown(f"**Side effects:** {info.get('side_effects','-')}")
            st.markdown(f"**Warnings:** {info.get('warnings','-')}")
        else:
            st.info("No drug info found in data/drug_info.csv for this drug.")

    with right:
        st.subheader("Live Analytics (auto updates on submit/update)")
        reviews_df = load_reviews_df()

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Reviews", int(len(reviews_df)))
        c2.metric("Unique Drugs", int(reviews_df["drug_name"].nunique()) if not reviews_df.empty else 0)
        c3.metric("Avg Sentiment", f"{reviews_df['sentiment_score'].mean():.3f}" if not reviews_df.empty else "—")

        st.markdown("### Charts")
        plot_drug_sentiment(reviews_df, top_n=10)
        plot_review_counts(reviews_df, top_n=10)
        plot_role_sentiment(reviews_df)

# =======================
# TAB 2: CHATBOT
# =======================

with tabs[1]:

    st.subheader("Drug Assistant Chatbot")

    if st.session_state.role == "Admin" or st.session_state.username == "Guest":
        mode = st.radio(
            "Select mode",
            ["Info Mode", "Review Mode"],
            horizontal=True
        )
    else:
        mode = "Info Mode"
        st.info("Review lookup available only for Admin and Guest users")

    # ---------------------
    # CHAT HISTORY
    # ---------------------

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_msg = st.chat_input("Enter medicine name...")

    if user_msg:

        st.session_state.chat_messages.append(
            {"role": "user", "content": user_msg}
        )

        with st.chat_message("user"):
            st.write(user_msg)

        # ===================================================
        # INFO MODE (OLLAMA DRUG INFORMATION)
        # ===================================================

        if mode == "Info Mode":

            drug_context = ""

            if not drug_df.empty:

                sample = drug_df.head(50).to_dict(orient="records")

                drug_context = "Drug dataset:\n" + "\n".join(
                    [
                        f"{r['drug_name']} | Uses:{r.get('uses','')} | Side effects:{r.get('side_effects','')}"
                        for r in sample
                    ]
                )

            messages = [
                {"role": "system", "content": drug_context},
                {"role": "user", "content": user_msg}
            ]

            reply = groq_chat(
                messages=messages,
                model=groq_model,
                temperature=temperature,
                api_key=GROQ_API_KEY
            )

            if reply.startswith("Groq error") or reply.startswith("Groq connection error") or reply.startswith("Groq API key not found"):
                reply = (
                    "🤖 AI chat isn't available right now. "
                    "Please check your GROQ_API_KEY is set correctly."
                )

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": reply}
            )

            with st.chat_message("assistant"):
                st.write(reply)

        # ===================================================
        # REVIEW MODE (LOCAL SENTIMENT DATABASE)
        # ===================================================

        if mode == "Review Mode":

            query = user_msg.strip().lower()

            reviews_df = load_reviews_df()

            if reviews_df.empty:

                reply = "No reviews are stored yet."

                with st.chat_message("assistant"):
                    st.write(reply)

            else:

                matched_reviews = reviews_df[
                    reviews_df["drug_name"].astype(str).str.lower().str.contains(query, na=False)
                ]

                if matched_reviews.empty:

                    reply = "No reviews found for this medication. Please search another drug."

                    with st.chat_message("assistant"):
                        st.write(reply)

                else:

                    overall_score = matched_reviews["sentiment_score"].mean()

                    user_reviews = matched_reviews[
                        matched_reviews["role"] == "User"
                    ]

                    physician_reviews = matched_reviews[
                        matched_reviews["role"] == "Physician"
                    ]

                    user_score = (
                        user_reviews["sentiment_score"].mean()
                        if not user_reviews.empty else None
                    )

                    physician_score = (
                        physician_reviews["sentiment_score"].mean()
                        if not physician_reviews.empty else None
                    )

                    with st.chat_message("assistant"):

                        st.subheader(
                            f"Review Sentiment for {user_msg.title()}"
                        )

                        c1, c2, c3 = st.columns(3)

                        c1.metric(
                            "Overall Score",
                            f"{overall_score:.3f}"
                        )

                        c2.metric(
                            "Total Reviews",
                            len(matched_reviews)
                        )

                        c3.metric(
                            "Top Sentiment",
                            matched_reviews["sentiment_label"].value_counts().idxmax()
                        )

                        st.markdown("### User vs Physician Sentiment")

                        col1, col2 = st.columns(2)

                        col1.metric(
                            "User Avg Score",
                            f"{user_score:.3f}" if user_score else "No user reviews"
                        )

                        col2.metric(
                            "Physician Avg Score",
                            f"{physician_score:.3f}" if physician_score else "No physician reviews"
                        )

                        st.markdown("### Sentiment Distribution")

                        st.bar_chart(
                            matched_reviews["sentiment_label"].value_counts()
                        )

                        st.markdown("### Recent Reviews")

                        st.dataframe(
                            matched_reviews[
                                [
                                    "drug_name",
                                    "role",
                                    "review_text",
                                    "sentiment_label",
                                    "sentiment_score",
                                    "created_at"
                                ]
                            ].sort_values("created_at", ascending=False),
                            use_container_width=True,
                            hide_index=True
                        )

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_messages = []
        st.rerun()

# =======================
# TAB 3: ML TRAINING
# =======================
with tabs[2]:

    # Admin restriction
    if st.session_state.role != "Admin":
        st.warning("⚠️ Only Admin can access ML training.")
        st.stop()


    st.subheader("ML Training (Dataset-based)")
    st.caption("Train a text classifier using your dataset. You can use DB reviews or upload your own CSV.")

    source = st.radio("Training data source", ["Use DB reviews (sentiment_label)", "Upload CSV"], horizontal=True)

    train_df = None
    if source == "Use DB reviews (sentiment_label)":
        reviews_df = load_reviews_df()
        if reviews_df.empty:
            st.warning("No reviews in DB. Add reviews first.")
        else:
            train_df = reviews_df.copy()
            st.write("Sample:", train_df.head(10))

    else:
        up = st.file_uploader("Upload CSV with columns: review_text and label", type=["csv"])
        if up:
            train_df = pd.read_csv(up)
            st.write("Preview:", train_df.head(10))

    if train_df is not None and not train_df.empty:
        if source == "Use DB reviews (sentiment_label)":
            text_col = "review_text"
            label_col = "sentiment_label"
        else:
            text_col = st.selectbox("Text column", options=train_df.columns.tolist())
            label_col = st.selectbox("Label column", options=train_df.columns.tolist(), index=min(1, len(train_df.columns)-1))

        st.markdown("---")
        if st.button("🏋️ Train Model", use_container_width=True):
            try:
                res = train_text_model(train_df, text_col=text_col, label_col=label_col)
                st.session_state.trained_model = res.model
                st.success(f"Trained ✅ Accuracy: {res.accuracy:.3f}")
                st.text("Classification report:\n" + res.report)
                st.write("Confusion matrix:", res.confusion)
            except Exception as e:
                st.error(f"Training failed: {e}")

        st.markdown("### Predict with trained model")
        test_text = st.text_area("Enter a review to predict", height=120)
        if st.button("🔮 Predict", use_container_width=True):
            if "trained_model" not in st.session_state:
                st.error("Train a model first.")
            elif not test_text.strip():
                st.error("Enter some text.")
            else:
                pred = predict_text(st.session_state.trained_model, test_text)
                st.success(f"Prediction: **{pred}**")

# =======================
# TAB 4: REVIEW DATABASE
# =======================
with tabs[3]:

    if st.session_state.role != "Admin":
        st.warning("Admin access only")
        st.stop()
    st.subheader("Stored Reviews (Edit / Delete)")
    reviews_df = load_reviews_df()

    if reviews_df.empty:
        st.info("No reviews stored yet.")
    else:
        st.dataframe(reviews_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### Edit a review (re-runs VADER + updates charts)")
        review_ids = reviews_df["id"].tolist()
        selected_id = st.selectbox("Select review ID", review_ids)

        current_row = reviews_df[reviews_df["id"] == selected_id].iloc[0]
        st.write(f"Drug: **{current_row['drug_name']}** | Role: **{current_row['role']}** | Label: **{current_row['sentiment_label']}**")

        new_text = st.text_area("Update review text", value=str(current_row["review_text"]), height=140)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save Update", use_container_width=True):
                score, label, _ = vader_sentiment(new_text)
                update_review(selected_id, new_text.strip(), score, label)
                st.success(f"Updated ✅ New sentiment: **{label}** (compound={score:.3f})")
                st.rerun()
        with c2:
            if st.button("🗑️ Delete Review", use_container_width=True):
                delete_review(int(selected_id))
                st.success("Deleted ✅")
                st.rerun()