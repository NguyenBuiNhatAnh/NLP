import os
import copy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

LABEL_NAMES = ["Tiêu cực", "Trung tính", "Tích cực"]


# 1. Chuyển đổi DataFrame sang Dataset của PyTorch
class SentimentTorchDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


# 2. Lớp quản lý Huấn luyện
class PhoBertTrainer:
    def __init__(
        self,
        model_wrapper,
        batch_size=16,
        epochs=5,
        learning_rate=2e-5,
        class_weights=None,
        label_names=None,
        save_dir="./saved_models",
        patience=2,
    ):
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = model_wrapper.device

        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.save_dir = save_dir
        self.patience = patience  # Early stopping

        self.label_names = label_names or LABEL_NAMES

        # Khởi tạo Scaler cho FP16 — chỉ dùng khi có CUDA
        self.use_amp = self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
            print("[INFO] FP16 Mixed Precision được bật (CUDA).")
        else:
            self.scaler = None
            print("[INFO] Chạy trên CPU — FP16 bị tắt.")

        # Class weights để xử lý mất cân bằng nhãn
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
            print(f"[INFO] Class weights: {self.class_weights.tolist()}")
        else:
            self.class_weights = None

        # Lịch sử loss/metric để theo dõi
        self.history = {"train_loss": [], "val_accuracy": [], "val_f1": []}

    def prepare_dataloader(self, texts, labels, max_length=256, shuffle=True):
        """Mã hóa văn bản và tạo DataLoader."""
        encodings = self.model_wrapper.tokenize_data(texts, max_length=max_length)
        dataset = SentimentTorchDataset(encodings, labels)
        return DataLoader(
            dataset,
            batch_size=self.batch_size if shuffle else self.batch_size * 2,
            shuffle=shuffle,
            pin_memory=(self.device.type == "cuda"),
            num_workers=2 if self.device.type == "cuda" else 0,
        )

    def train_and_evaluate(self, train_texts, train_labels, val_texts, val_labels):
        """
        Huấn luyện PhoBERT và đánh giá sau mỗi epoch.
        Lưu best model theo val F1-macro.
        """
        print("\n--- ĐANG CHUẨN BỊ DỮ LIỆU ---")
        train_loader = self.prepare_dataloader(train_texts, train_labels, shuffle=True)
        val_loader = self.prepare_dataloader(val_texts, val_labels, shuffle=False)

        # Loss function có class weights
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights)

        # Optimizer và Scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps,
        )

        print(f"\nBẮT ĐẦU HUẤN LUYỆN TRÊN: {self.device.type.upper()}")
        print(f"Tổng epochs: {self.epochs} | Batch size: {self.batch_size} | LR: {self.learning_rate}")

        best_val_f1 = 0.0
        best_model_state = None
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            print(f"\n{'='*50}")
            print(f"[Epoch {epoch + 1}/{self.epochs}]")

            # === Training ===
            self.model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc="  Training", leave=False)

            for batch in progress_bar:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(input_ids, attention_mask=attention_mask)
                        loss = loss_fn(outputs.logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    loss = loss_fn(outputs.logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()

                scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)
            print(f"  → Loss trung bình (train): {avg_train_loss:.4f}")

            # === Validation ===
            val_acc, val_f1, _ = self.evaluate(val_loader)
            self.history["val_accuracy"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            # === Best Model Checkpoint theo val F1 ===
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"  ✓ Best model được cập nhật! (val F1-macro: {val_f1:.4f})")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  ✗ Không cải thiện. ({epochs_no_improve}/{self.patience})")
                if epochs_no_improve >= self.patience:
                    print(f"\n[Early Stopping] Dừng sớm tại Epoch {epoch + 1}.")
                    break

        # Khôi phục best model và lưu
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.model.save_pretrained(self.save_dir)
        self.model_wrapper.tokenizer.save_pretrained(self.save_dir)
        print(f"\nHoàn tất! Best model đã được lưu vào '{self.save_dir}' (val F1: {best_val_f1:.4f})")
        return self.history

    def evaluate(self, data_loader):
        """Đánh giá mô hình và in classification report."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        print(f"  Accuracy: {acc:.4f} | F1-macro: {f1:.4f}")
        print(
            classification_report(
                all_labels,
                all_preds,
                target_names=self.label_names,
                zero_division=0,
            )
        )
        return acc, f1, all_preds