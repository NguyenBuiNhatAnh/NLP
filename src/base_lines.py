import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score
import warnings

warnings.filterwarnings("ignore")

LABEL_NAMES = ["Tiêu cực", "Trung tính", "Tích cực"]


class BaselineTrainer:
    def __init__(self, label_names=None):
        self.label_names = label_names or LABEL_NAMES

        self.vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            sublinear_tf=True,
            min_df=2,
            max_df=0.8
        )

        self.feature_pipeline = Pipeline([
            ('tfidf', self.vectorizer)
        ])
        self.models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=42, C=1.0, class_weight='balanced'
            ),
            "Naive Bayes": MultinomialNB(),
        }

    def train_and_evaluate(self, train_texts, train_labels, test_texts, test_labels):
        train_texts = [str(t) for t in train_texts]
        test_texts = [str(t) for t in test_texts]

        print("--- [Baseline] Đang áp dụng TF-IDF Feature Engineering ---")
        print("  ✓ TfidfVectorizer (N-grams 1-3) - Lên tới 15.000 chiều dữ liệu thưa")

        # Chuyển đổi dữ liệu
        X_train = self.feature_pipeline.fit_transform(train_texts, train_labels)
        X_test = self.feature_pipeline.transform(test_texts)

        print(f"[*] Kích thước ma trận TF-IDF (Sparse): {X_train.shape}")

        results = {}
        all_predictions = {}
        
        for name, model in self.models.items():
            print(f"\n[*] Đang huấn luyện {name} trên không gian thưa...")
            model.fit(X_train, train_labels)

            predictions = model.predict(X_test)
            all_predictions[name] = predictions

            print(f"\n--- BÁO CÁO KẾT QUẢ: {name.upper()} ---")
            print(classification_report(
                test_labels,
                predictions,
                target_names=self.label_names,
                zero_division=0,
            ))

            results[name] = f1_score(test_labels, predictions, average='macro')

        return results, all_predictions