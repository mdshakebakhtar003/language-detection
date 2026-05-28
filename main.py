import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("        LANGUAGE DETECTION - NLP PROJECT")
print("=" * 60)

data = pd.read_csv("language.csv")
print("\n[1] Dataset Loaded Successfully")
print(f"    Shape : {data.shape}")
print(f"\n    First 5 rows:\n{data.head()}")

# ─────────────────────────────────────────────
# 2. Check for Null Values
# ─────────────────────────────────────────────
print("\n[2] Null Values Check:")
print(data.isnull().sum())
data.dropna(inplace=True)

# ─────────────────────────────────────────────
# 3. Language Distribution
# ─────────────────────────────────────────────
print("\n[3] Language Distribution:")
print(data["language"].value_counts())

# ─────────────────────────────────────────────
# 4. Prepare Features and Labels
# ─────────────────────────────────────────────
x = np.array(data["Text"])
y = np.array(data["language"])

print(f"\n[4] Features (x) shape : {x.shape}")
print(f"    Labels  (y) shape : {y.shape}")

# ─────────────────────────────────────────────
# 5. CountVectorizer (Bag of Words)
# ─────────────────────────────────────────────
cv = CountVectorizer()
X = cv.fit_transform(x)

print(f"\n[5] CountVectorizer Applied")
print(f"    Vocabulary size     : {len(cv.vocabulary_)}")
print(f"    Transformed X shape : {X.shape}")

# ─────────────────────────────────────────────
# 6. Train / Test Split
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42
)

print(f"\n[6] Train/Test Split (test_size=0.33)")
print(f"    Training samples : {X_train.shape[0]}")
print(f"    Testing  samples : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 7. Train All Three Models
# ─────────────────────────────────────────────
print("\n[7] Training Models...")

# --- Multinomial Naive Bayes ---
print("    Training MultinomialNB        ", end="", flush=True)
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print("Done ✓")

# --- Logistic Regression ---
print("    Training LogisticRegression   ", end="", flush=True)
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
print("Done ✓")

# --- Random Forest ---
print("    Training RandomForestClassifier (may take a moment)...", end="", flush=True)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
print("Done ✓")

# ─────────────────────────────────────────────
# 8. Evaluate All Models
# ─────────────────────────────────────────────
models = {
    "MultinomialNB"         : nb_model,
    "LogisticRegression"    : lr_model,
    "RandomForestClassifier": rf_model,
}

print("\n" + "=" * 60)
print("[8] MODEL COMPARISON")
print("=" * 60)

results = {}
for name, mdl in models.items():
    y_pred = mdl.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": mdl, "acc": acc, "y_pred": y_pred}
    print(f"\n  {'─'*50}")
    print(f"  Model    : {name}")
    print(f"  Accuracy : {acc * 100:.4f}%")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))

# ─────────────────────────────────────────────
# 9. Best Model Summary
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["acc"])
best_acc  = results[best_name]["acc"]

print("=" * 60)
print("  ACCURACY SUMMARY")
print("=" * 60)
for name, res in results.items():
    marker = "  ◀ BEST" if name == best_name else ""
    bar = "█" * int(res["acc"] * 40)
    print(f"  {name:<28} {res['acc']*100:6.2f}%  {bar}{marker}")

print(f"\n  Best Model : {best_name}  ({best_acc*100:.4f}%)")
print("=" * 60)

# ─────────────────────────────────────────────
# 10. Live Prediction using Best Model
# ─────────────────────────────────────────────
print(f"\n[10] Live Prediction  (using best model: {best_name})")
print("     Type 'switch' to change model | 'quit' to exit\n")

active_model_name = best_name
active_model      = results[best_name]["model"]

while True:
    user = input("Enter a Text: ").strip()

    if user.lower() == "quit":
        print("Exiting. Goodbye!")
        break

    if user.lower() == "switch":
        print("\n  Available models:")
        for i, name in enumerate(models.keys(), 1):
            print(f"    {i}. {name}")
        choice = input("  Enter number: ").strip()
        model_list = list(models.keys())
        if choice in ["1", "2", "3"]:
            active_model_name = model_list[int(choice) - 1]
            active_model      = results[active_model_name]["model"]
            print(f"  Switched to: {active_model_name}\n")
        else:
            print("  Invalid choice.\n")
        continue

    if not user:
        print("  [!] Please enter some text.\n")
        continue

    data_input = cv.transform([user]).toarray()
    prediction = active_model.predict(data_input)[0]

    # Confidence (all 3 models support predict_proba)
    proba      = active_model.predict_proba(data_input)[0]
    confidence = proba.max() * 100
    top3_idx   = proba.argsort()[::-1][:3]
    top3_langs = active_model.classes_[top3_idx]
    top3_probs = proba[top3_idx] * 100

    print(f"\n  [{active_model_name}]")
    print(f"  Detected Language : {prediction}")
    print(f"  Confidence        : {confidence:.2f}%")
    print(f"\n  Top-3 Predictions:")
    for lang, prob in zip(top3_langs, top3_probs):
        bar = "█" * int(prob / 5)
        print(f"    {lang:<15} {prob:5.1f}%  {bar}")
    print()
