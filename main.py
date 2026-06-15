import random
import numpy as np
import torch
import pandas as pd
from src.data_loader import SentimentDataset
from src.model import PhoBertClassifier
from src.trainer import PhoBertTrainer
from torch.utils.data import DataLoader

def set_seed(seed: int = 42):
    """Cố định seed để kết quả có thể tái tạo (reproducible)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    SEED = 42
    set_seed(SEED)

    # Chuyển đổi cờ để đánh giá 3 lớp hay 5 lớp
    MAP_TO_3_CLASSES = True

    if MAP_TO_3_CLASSES:
        label_names = ["Tiêu cực", "Trung tính", "Tích cực"]
    else:
        label_names = ["1 Sao", "2 Sao", "3 Sao", "4 Sao", "5 Sao"]

    # ========== 1. Tải và xử lý dữ liệu ==========
    loader = SentimentDataset(
        train_path="dataset/train.csv",
        test_path="dataset/test.csv",
        val_size=0.2,
        map_to_3_classes=MAP_TO_3_CLASSES,
        random_state=SEED,
    )
    train_df, val_df, test_df = loader.load_and_preprocess()

    train_texts = train_df["clean_text"].tolist()
    train_labels = train_df["label"].tolist()

    val_texts = val_df["clean_text"].tolist()
    val_labels = val_df["label"].tolist()

    test_texts = test_df["clean_text"].tolist()
    test_labels = test_df["label"].tolist()

    # ========== 1.5 Lưu từ vựng ra file (Vocabs) ==========
    print("\n--- Đang trích xuất từ vựng ra file vocabs.txt ---")
    all_texts = train_texts + val_texts + test_texts
    unique_words = set()
    for text in all_texts:
        if isinstance(text, str):
            unique_words.update(text.split())
            
    with open("vocabs.txt", "w", encoding="utf-8") as f:
        for word in sorted(unique_words):
            f.write(word + "\n")
    print(f"[INFO] Đã lưu {len(unique_words)} từ vựng vào vocabs.txt")

    # ========== 2. Tính class weights để xử lý mất cân bằng nhãn ==========
    class_weights = loader.get_class_weights(train_labels)
    print(f"\n[INFO] Nhãn: {label_names}")
    print(f"[INFO] Class weights: {class_weights.tolist()}")

    # ========== 3. Khởi tạo Mô hình PhoBERT ==========
    phobert_wrapper = PhoBertClassifier(
        model_name="vinai/phobert-base",
        num_labels=len(label_names),
    )

    # ========== 4. Khởi tạo Trainer ==========
    trainer = PhoBertTrainer(
        model_wrapper=phobert_wrapper,
        batch_size=16,
        epochs=5,
        learning_rate=2e-5,
        class_weights=class_weights,
        label_names=label_names,
        save_dir="./saved_models",
        patience=2,             # Early stopping sau 2 epoch không cải thiện
    )

    # ========== 5. Bắt đầu huấn luyện ==========
    history = trainer.train_and_evaluate(train_texts, train_labels, val_texts, val_labels)

    # ========== 6. Đánh giá trên tập Test ==========
    print("\n" + "=" * 50)
    print("ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST")
    print("=" * 50)
    
    test_loader = trainer.prepare_dataloader(test_texts, test_labels, shuffle=False)
    test_acc, test_f1, test_preds = trainer.evaluate(test_loader)
    print(f"\n[FINAL] Test Accuracy: {test_acc:.4f} | Test F1-macro: {test_f1:.4f}")

    # ========== 7. Lưu kết quả dự đoán ra file CSV ==========

    print("\n--- Đang xuất kết quả dự đoán ra file test_predictions.csv ---")
    pred_df = pd.DataFrame({
        "Nội dung": test_texts,
        "Nhãn thực tế": [label_names[label] for label in test_labels],
        "Nhãn dự đoán": [label_names[pred] for pred in test_preds]
    })
    pred_df.to_csv("test_predictions.csv", index=False, encoding="utf-8-sig")
    print("[INFO] Đã lưu file test_predictions.csv để xem chi tiết từng câu dự đoán!")


if __name__ == "__main__":
    main()