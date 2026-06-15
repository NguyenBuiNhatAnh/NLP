import torch
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Union
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.preprocess import clean_pipeline   # import hàm xử lý văn bản của bạn

warnings.filterwarnings("ignore")

# Cấu hình
MODEL_DIR = "./saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_NAMES = ["Tiêu cực", "Trung tính", "Tích cực"]

# ---------- Khởi tạo model một lần khi server start ----------
print(f"--- Đang tải mô hình từ '{MODEL_DIR}' trên {DEVICE.type.upper()} ---")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    print("[INFO] Đã tải mô hình thành công!\n")
except Exception as e:
    print(f"[ERROR] Không tìm thấy mô hình trong thư mục '{MODEL_DIR}'. Vui lòng train mô hình trước.")
    raise e

# ---------- Định nghĩa request/response models ----------
class PredictRequest(BaseModel):
    text: Union[str, List[str]] = Field(..., description="Câu hoặc danh sách câu cần dự đoán")

class PredictionResult(BaseModel):
    original: str
    cleaned: str
    prediction: str
    confidence: float
    # Nếu muốn trả về toàn bộ xác suất các lớp (tuỳ chọn)
    # probabilities: List[float]

class PredictResponse(BaseModel):
    results: List[PredictionResult]

# ---------- FastAPI app ----------
app = FastAPI(title="Sentiment Analysis API", description="Dự đoán cảm xúc bình luận tiếng Việt (PhoBERT)")

# CORS để React gọi được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def predict_sentiment(texts: Union[str, List[str]]) -> List[dict]:
    """Hàm dự đoán, tách riêng để dễ test và tái sử dụng"""
    if isinstance(texts, str):
        texts = [texts]

    results = []
    for text in texts:
        # 1. Tiền xử lý
        cleaned_text = clean_pipeline(text)

        # 2. Tokenize
        inputs = tokenizer(
            cleaned_text,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)

        # 3. Dự đoán
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)[0]  # tensor shape (num_classes,)
            pred_idx = torch.argmax(probs).item()
            confidence = probs[pred_idx].item() * 100

            # Xác định nhãn dựa trên số lớp (3 hoặc 5)
            num_classes = len(probs)
            if num_classes == 5:
                current_labels = ["1 Sao", "2 Sao", "3 Sao", "4 Sao", "5 Sao"]
            else:
                current_labels = LABEL_NAMES
            predicted_label = current_labels[pred_idx]

        results.append({
            "original": text,
            "cleaned": cleaned_text,
            "prediction": predicted_label,
            "confidence": round(confidence, 2)
        })
    return results

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Endpoint nhận một câu hoặc nhiều câu, trả về kết quả phân loại.
    """
    try:
        predictions = predict_sentiment(request.text)
        return PredictResponse(results=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok", "device": str(DEVICE), "num_classes": model.config.num_labels}

# ---------- Chạy thử nếu file được gọi trực tiếp (cho debug) ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)