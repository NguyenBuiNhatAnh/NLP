import torch
import warnings
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.preprocess import clean_pipeline

warnings.filterwarnings("ignore")

# Cấu hình
MODEL_DIR = "./saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Tiêu cực", "Trung tính", "Tích cực"]

class SentimentPredictor:
    """Lớp hỗ trợ inference (dự đoán) các câu bình luận mới."""
    
    def __init__(self, model_dir=MODEL_DIR, device=DEVICE):
        self.device = device
        print(f"--- Đang tải mô hình từ '{model_dir}' trên {self.device.type.upper()} ---")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
            self.model.eval()
            print("[INFO] Đã tải mô hình thành công!\n")
        except Exception as e:
            print(f"[ERROR] Không tìm thấy mô hình trong thư mục '{model_dir}'. Vui lòng chạy main.py để train mô hình trước.")
            raise e

    def predict(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            # Bước 1: Tiền xử lý giống hệt lúc train (NFC, Xóa HTML, Chuẩn hóa Teencode, Tách từ PyVi)
            cleaned_text = clean_pipeline(text)
            
            # Bước 2: Mã hóa văn bản
            inputs = self.tokenizer(
                cleaned_text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)

            # Bước 3: Đưa vào mô hình dự đoán
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[0] # Tính xác suất bằng Softmax
                
                # Lấy nhãn có xác suất cao nhất
                pred_idx = torch.argmax(probs).item()
                
                # Tự động điều chỉnh nhãn nếu mô hình trước đó train theo 5 lớp (1-5 sao)
                if len(probs) == 5:
                    current_labels = ["1 Sao", "2 Sao", "3 Sao", "4 Sao", "5 Sao"]
                else:
                    current_labels = LABEL_NAMES
                    
                predicted_label = current_labels[pred_idx]
                confidence = probs[pred_idx].item() * 100

                results.append({
                    "original": text,
                    "cleaned": cleaned_text,
                    "prediction": predicted_label,
                    "confidence": confidence,
                    "all_probs": probs.cpu().numpy()
                })
                
        return results

def main():
    # Danh sách các câu bình luận "MỚI" bạn muốn thử nghiệm
    test_comments = [
        "Mik mới mua máy được 1 tuần . Mình để qua đêm thì máy hao mất 3 % . Máy mới mua mà pin đã vậy rồi .,",
        "Máy bị đơ màn_hình khi chơi pubg va liên_quân cho hoi xin doi may được không con moi thu deu on xai tốt,",       
        "Sản_phẩm không tích_hợp cảm_biến la_bàn . chạy chậm . Nói_chung so với máy Galaxy_a10 thì chẳng hơn gì ?",
        "Camera chụp không đẹp , mình hơi thất_vọng , màu nhạt và lúc chụp thì xấu . Lúc ảnh lưu vào thì đẹp hơn tí . Camera trước ảo Pin cũng dùng 1 ngày ."        
                ]

    try:
        predictor = SentimentPredictor()
    except Exception:
        return

    predictions = predictor.predict(test_comments)

    print("================ KẾT QUẢ DỰ ĐOÁN BẰNG PHO-BERT ================")
    for idx, res in enumerate(predictions):
        print(f"\n[Câu {idx + 1}]: {res['original']}")
        print(f"  → Văn bản sau xử lý : {res['cleaned']}")
        print(f"  → NHÃN DỰ ĐOÁN      : {res['prediction']} (Độ tự tin: {res['confidence']:.2f}%)")
        
        # In ra xác suất chi tiết của từng lớp để xem mô hình có đang "phân vân" không
        probs = res['all_probs']
        if len(probs) == 3:
            print(f"  → Chi tiết xác suất : [Tiêu cực: {probs[0]:.2f} | Trung tính: {probs[1]:.2f} | Tích cực: {probs[2]:.2f}]")
        elif len(probs) == 5:
            print(f"  → Chi tiết xác suất : [1*: {probs[0]:.2f} | 2*: {probs[1]:.2f} | 3*: {probs[2]:.2f} | 4*: {probs[3]:.2f} | 5*: {probs[4]:.2f}]")
            
    print("\n" + "="*63)

if __name__ == "__main__":
    main()
