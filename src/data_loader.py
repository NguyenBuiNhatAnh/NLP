import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from .preprocess import preprocess_batch

# Mapping nhãn sao (1-5) sang nhãn cảm xúc 3 lớp
STAR_TO_SENTIMENT = {
    1: 0,  # 1 sao → Tiêu cực
    2: 0,  # 2 sao → Tiêu cực
    3: 1,  # 3 sao → Trung tính
    4: 1,  # 4 sao → Trung tính 
    5: 2,  # 5 sao → Tích cực
}

LABEL_NAMES = ["Tiêu cực", "Trung tính", "Tích cực"]


class SentimentDataset:

    def __init__(
        self,
        train_path: str = "dataset/train.csv",
        test_path: str = "dataset/test.csv",
        val_size: float = 0.2,
        map_to_3_classes: bool = True,
        random_state: int = 42,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.val_size = val_size
        self.map_to_3_classes = map_to_3_classes
        self.random_state = random_state

    def _load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)

        print(f"[INFO] File '{os.path.basename(path)}': shape = {df.shape}")

        text_col = "comment"
        df = df[[text_col, "label"]].copy()
        df.rename(columns={text_col: "text"}, inplace=True)

        df.dropna(subset=["text", "label"], inplace=True)
        df = df[pd.to_numeric(df["label"], errors="coerce").notna()]
        df["label"] = df["label"].astype(int)

        # Tự động phát hiện nếu nhãn đã là 0, 1, 2 thì không cần map nữa
        unique_labels = df["label"].unique()
        if set(unique_labels).issubset({0, 1, 2}):
            pass # Giữ nguyên nhãn
        elif self.map_to_3_classes:
            # Map nhãn 1-5 → 3 lớp cảm xúc (nếu bật)
            df["label"] = df["label"].map(STAR_TO_SENTIMENT)
            df.dropna(subset=["label"], inplace=True)
            df["label"] = df["label"].astype(int)
        else:
            # Chuyển sang 0-indexed (1-5 → 0-4)
            df["label"] = df["label"] - 1

        return df

    def _apply_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        print("--- Đang tiền xử lý văn bản ---")
        df["clean_text"] = preprocess_batch(df["text"].tolist())
        return df

    def load_and_preprocess(self):
        """
        Tải và xử lý toàn bộ dataset.
        Trả về: (train_df, val_df, test_df)
        """
        print("--- Đang tải dataset từ local CSV ---")
        train_df = self._load_csv(self.train_path)
        test_df = self._load_csv(self.test_path)

        # Tách val từ train
        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_size,
            stratify=train_df["label"],
            random_state=self.random_state,
        )
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        
        print(f"\nPhân phối nhãn (sau map):")
        print(f"  Train : {dict(train_df['label'].value_counts().sort_index())}")
        print(f"  Val   : {dict(val_df['label'].value_counts().sort_index())}")
        print(f"  Test  : {dict(test_df['label'].value_counts().sort_index())}")

        # Tiền xử lý
        train_df = self._apply_preprocessing(train_df)
        val_df = self._apply_preprocessing(val_df)
        test_df = self._apply_preprocessing(test_df)


        return train_df, val_df, test_df

    def get_class_weights(self, labels):
        """
        Tính class weights để xử lý mất cân bằng nhãn.
        Trả về tensor dùng trong CrossEntropyLoss.
        """
        import torch
        from collections import Counter

        counts = Counter(labels)
        n_samples = len(labels)
        n_classes = max(counts.keys()) + 1
        weights = [n_samples / (n_classes * counts.get(i, 1)) for i in range(n_classes)]
        weights[0] = weights[0] * 1.5
        return torch.tensor(weights, dtype=torch.float)