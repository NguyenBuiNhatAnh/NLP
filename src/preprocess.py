import re
import unicodedata
import threading
from typing import List
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from tqdm import tqdm
from pyvi import ViTokenizer

from .vi_dict import TEENCODE_MAP


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # 1. Remove URLs và HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # 2. Unicode Normalization (NFC)
    text = unicodedata.normalize("NFC", text)

    # 3. Normalize repeated characters nhẹ (Lặp > 2 lần thì quy về 1)
    text = re.sub(r'(.)\1{2,}', r'\1', text)

    # 4. Padding Emoji bằng khoảng trắng
    text = re.sub(r'([^\w\s\.,!\?\'\"@#\-%&\(\)\[\]\{\}\<\>\\/\:\;]+)', r' \1 ', text)

    # 5. Slang / Teencode normalization
    for slang, standard in TEENCODE_MAP.items():
        text = re.sub(rf'(?i)\b{re.escape(slang)}\b', standard, text)

    # 6. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


class VietnameseTextPreprocessor:
    """Lớp xử lý văn bản đơn lẻ, hỗ trợ tương thích ngược."""
    def __init__(self, *args, **kwargs):
        pass

    def preprocess(self, text: str) -> str:
        text = clean_text(text)
        if not text:
            return ""
        try:
            return ViTokenizer.tokenize(text)
        except Exception:
            return text

    def close(self):
        pass


class ParallelVietnameseTextPreprocessor:
    """Lớp xử lý đa luồng (song song) tối ưu tốc độ cho DataFrame."""
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or min(cpu_count(), 4)
        self._local = threading.local()
    
    def _get_preprocessor(self):
        if not hasattr(self._local, 'preprocessor'):
            self._local.preprocessor = VietnameseTextPreprocessor()
        return self._local.preprocessor
    
    def _preprocess_single(self, text: str) -> str:
        try:
            preprocessor = self._get_preprocessor()
            return preprocessor.preprocess(text)
        except Exception:
            return str(text)
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(tqdm(executor.map(self._preprocess_single, texts), 
                                total=len(texts), 
                                desc="Tiền xử lý (Song song)"))
        return results

    def close(self):
        pass


def preprocess_batch(texts: List[str]) -> List[str]:
    """API Pipeline để tích hợp với Data Loader."""
    parallel_preprocessor = ParallelVietnameseTextPreprocessor()
    res = parallel_preprocessor.preprocess_batch(texts)
    return res

def clean_pipeline(text: str) -> str:
    """API để test 1 câu đơn lẻ."""
    preprocessor = VietnameseTextPreprocessor()
    return preprocessor.preprocess(text)