"""ScamShield AI - Model Training Script

Loads the SMS Spam Collection dataset, builds an NLP pipeline,
evaluates on held-out test data, and saves the pipeline with joblib.
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Set up reliable paths relative to this script
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "spam_pipeline.joblib"


def load_dataset() -> pd.DataFrame:
    """Loads and cleans the dataset, falling back to sample data if missing."""
    if DATA_PATH.exists():
        print(f"[*] Reading dataset from: {DATA_PATH}")
        df = pd.read_csv(
            DATA_PATH,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8",
            on_bad_lines="skip",
        )
    else:
        print("[!] Warning: Dataset file not found at data/SMSSpamCollection.")
        print(
            "[!] Using an in-memory sample fallback for demonstration purposes."
        )
        fallback_data = {
            "label": ["ham", "spam", "ham", "spam", "ham", "spam"],
            "message": [
                "Hey, are we still meeting for lunch today at 1pm?",
                "URGENT: Your account is locked. Click http://bit.ly/fake to reset.",
                "Can you send me the lecture slides from yesterday?",
                "Congratulations! You won $10,000 cash. Reply CLAIM to receive.",
                "I will call you when I get home from work.",
                "Free entry into our holiday draw! Text WIN to 80085 now.",
            ],
        }
        df = pd.DataFrame(fallback_data)

    # 1. Clean missing or duplicate messages
    initial_count = len(df)
    df = df.dropna(subset=["message", "label"])
    df = df.drop_duplicates(subset=["message"])
    print(f"[*] Removed {initial_count - len(df)} duplicate/empty entries.")

    # 2. Standardize labels
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].copy()

    # 3. Binary encoding: ham -> 0, spam -> 1
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def train_and_evaluate():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()

    print(f"[*] Usable samples: {len(df)}")
    print(
        f"[*] Class breakdown:\n{df['label'].value_counts(normalize=True).round(3)}"
    )

    X = df["message"]
    y = df["target"]

    # Stratified split: ensures equal spam/ham distribution across train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build Pipeline: TF-IDF feature extraction -> Multinomial Naive Bayes
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=6000,
                ),
            ),
            ("nb", MultinomialNB(alpha=0.1)),
        ]
    )

    print("\n[*] Training the ML pipeline...")
    pipeline.fit(X_train, y_train)

    print("[*] Evaluating against held-out test data...")
    y_pred = pipeline.predict(X_test)

    print("\n" + "=" * 55)
    print("                EVALUATION METRICS")
    print("=" * 55)
    print(
        classification_report(
            y_test, y_pred, target_names=["Ham (Safe)", "Spam (Suspicious)"]
        )
    )

    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 55)

    # Save fitted pipeline to disk
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[+] Trained pipeline saved successfully to: {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()