import { useState } from 'react';
import axios from 'axios';
import "./App.css"

const API_URL = 'http://localhost:8000/predict';

const LABELS = [
  { key: 'Tích cực', barClass: 'bar-pos', icon: 'ti-mood-happy', badge: 'pos' },
  { key: 'Trung tính', barClass: 'bar-neu', icon: 'ti-mood-neutral', badge: 'neu' },
  { key: 'Tiêu cực', barClass: 'bar-neg', icon: 'ti-mood-angry', badge: 'neg' },
];

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    if (!text.trim()) { setError('Vui lòng nhập bình luận'); return; }
    setLoading(true); setError(''); setResult(null);
    try {
      const res = await axios.post(API_URL, { text });
      setResult(res.data.results[0]);
    } catch {
      setError('Không thể kết nối đến API. Hãy đảm bảo backend đang chạy ở cổng 8000.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="wrap">
      <h1>Phân tích cảm xúc bình luận</h1>
      <p className="sub">Nhập bình luận tiếng Việt, mô hình PhoBERT sẽ dự đoán cảm xúc</p>

      <textarea
        rows="5"
        placeholder="Ví dụ: Sản phẩm quá tệ, tôi thất vọng..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      />

      <button onClick={handlePredict} disabled={loading}>
        {loading ? 'Đang phân tích...' : 'Dự đoán cảm xúc'}
      </button>

      {error && <div className="error">{error}</div>}

      {result && (
        <div className="card">
          <div className="meta-row">
            <div className="meta-item">
              <span className="meta-lbl">Văn bản gốc</span>
              <span className="meta-val light">{result.original}</span>
            </div>
            <div className="meta-item">
              <span className="meta-lbl">Sau xử lý</span>
              <span className="meta-val muted">{result.cleaned}</span>
            </div>
            <div className="meta-item">
              <span className="meta-lbl">Kết quả</span>
              {LABELS.filter(l => l.key === result.prediction).map(l => (
                <span key={l.key} className={`badge ${l.badge}`}>
                  <i className={l.icon}></i> {l.key}
                </span>
              ))}
            </div>
          </div>

          <hr className="divider" />

          <div className="bar-section">
            {LABELS.map(({ key, barClass }) => {
              const pct = result.probabilities?.[key] ?? 0;
              return (
                <div className="bar-row" key={key}>
                  <span className="bar-lbl">{key}</span>
                  <div className="bar-track">
                    <div className={`bar-fill ${barClass}`} style={{ width: `${pct}%` }} />
                  </div>
                  <span className="bar-pct">{pct.toFixed(1)}%</span>
                </div>
              );
            })}
          </div>

          <div className="confidence-row">
            <span className="conf-lbl">Độ tự tin</span>
            <div className="conf-track">
              <div className="conf-fill" style={{ width: `${result.confidence}%` }} />
            </div>
            <span className="conf-val">{result.confidence}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;