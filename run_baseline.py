import random
import numpy as np
import pandas as pd
import torch
from src.data_loader import SentimentDataset
from src.base_lines import BaselineTrainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(42)

    label_names = ["Tiêu cực", "Trung tính", "Tích cực"]

    print("=== BẮT ĐẦU CHẠY BASELINE (TF-IDF) ===")

    # 1. Tải Dữ Liệu
    loader = SentimentDataset(
        train_path="dataset/train.csv",
        test_path="dataset/test.csv",
        val_size=0.2,
        map_to_3_classes=True,
        random_state=42,
    )
    train_df, val_df, test_df = loader.load_and_preprocess()

    train_texts = train_df["clean_text"].tolist()
    train_labels = train_df["label"].tolist()
    test_texts = test_df["clean_text"].tolist()
    test_labels = test_df["label"].tolist()

    # 2. Khởi tạo và Chạy BaselineTrainer
    trainer = BaselineTrainer(label_names=label_names)
    results, all_predictions = trainer.train_and_evaluate(train_texts, train_labels, test_texts, test_labels)

    # 3. In Bảng Xếp Hạng F1-Score
    print("\n" + "=" * 50)
    print("TỔNG KẾT F1-MACRO SCORE (TỪ CAO XUỐNG THẤP):")
    for model, f1 in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"  {model}: {f1:.4f}")
    print("=" * 50)

    # ============================================================
    # 4. Lưu kết quả dự đoán của Linear SVC (mô hình tốt nhất)
    # ============================================================
    logistic_pred = all_predictions["Logistic Regression"]

    # Tạo DataFrame chứa câu gốc, nhãn thực tế và nhãn dự đoán
    result_df = pd.DataFrame({
        "Nội dung": test_df["text"].tolist(),
        "Nhãn thực tế": [label_names[l] for l in test_labels],
        "Nhãn dự đoán (Logistic)": [label_names[p] for p in logistic_pred],
        "Đúng/Sai": ["✓" if l == p else "✗" for l, p in zip(test_labels, logistic_pred)]
    })

    # Lưu toàn bộ kết quả
    result_df.to_csv("baseline_logistic_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"\n[INFO] Đã lưu toàn bộ {len(result_df)} kết quả dự đoán → 'baseline_logistic_predictions.csv'")

    # Lọc riêng các câu dự đoán sai
    wrong_df = result_df[result_df["Đúng/Sai"] == "✗"].copy()
    wrong_df.to_csv("baseline_logistic_wrong_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"[INFO] Đã lọc ra {len(wrong_df)} câu dự đoán SAI → 'baseline_logistic_wrong_predictions.csv'")
    print(f"       Tỷ lệ sai: {len(wrong_df) / len(result_df) * 100:.2f}%")


if __name__ == "__main__":
    main()
