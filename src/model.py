import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PhoBertClassifier:
    def __init__(self, model_name="vinai/phobert-base", num_labels=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_labels = num_labels
        print(f"--- Đang tải Tokenizer và Model: {model_name} trên {self.device} ---")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            # Bật warning khi model head chưa được fine-tune 
            ignore_mismatched_sizes=True,
        ).to(self.device)

    def tokenize_data(self, texts, max_length=256):
        """
        Hàm chuyển đổi văn bản tiếng Việt sang Input IDs và Attention Masks.
        PhoBERT dùng max_length=256 (tối đa của model).
        Dùng padding='longest' để không pad tất cả lên 256 (tiết kiệm VRAM).
        """
        # Đảm bảo texts là list of strings
        if hasattr(texts, "tolist"):
            texts = texts.tolist()
        texts = [str(t) if t is not None else "" for t in texts]

        return self.tokenizer(
            texts,
            padding="longest",       # Chỉ pad đến độ dài dài nhất trong batch
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )