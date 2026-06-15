import { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = 'http://localhost:8000/predict';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Vui lòng nhập bình luận');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await axios.post(API_URL, { text });
      const prediction = response.data.results[0];
      setResult(prediction);
    } catch (err) {
      console.error(err);
      setError('Không thể kết nối đến API. Hãy đảm bảo backend đang chạy ở cổng 8000.');
    } finally {
      setLoading(false);
    }
  };

  const getColor = (prediction) => {
    if (prediction === 'Tích cực') return '#28a745';
    if (prediction === 'Tiêu cực') return '#dc3545';
    return '#ffc107';
  };

  const getEmoji = (prediction) => {
    if (prediction === 'Tích cực') return '😊';
    if (prediction === 'Tiêu cực') return '😠';
    return '😐';
  };

  return (
    <div className="container">
      <div className="header">
        <h1>🔍 Phân tích cảm xúc bình luận</h1>
        <p>Nhập bình luận tiếng Việt, mô hình PhoBERT sẽ dự đoán cảm xúc</p>

        <textarea
          rows="5"
          placeholder="Ví dụ: Sản phẩm quá tệ, tôi thất vọng..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="textarea"
        />

        <button onClick={handlePredict} disabled={loading} className="button">
          {loading ? 'Đang phân tích...' : 'Dự đoán cảm xúc'}
        </button>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result-card">
            <h2>Kết quả phân tích</h2>
            <div className="row">
              <span className="label">Văn bản gốc:</span>
              <span>{result.original}</span>
            </div>
            <div className="row">
              <span className="label">Văn bản sau xử lý:</span>
              <span>{result.cleaned}</span>
            </div>
            <div className="row">
              <span className="label">Cảm xúc:</span>
              <span
                className="sentiment"
                style={{
                  backgroundColor: getColor(result.prediction),
                  color: 'white',
                  padding: '4px 12px',
                  borderRadius: '20px',
                  display: 'inline-block'
                }}
              >
                {getEmoji(result.prediction)} {result.prediction}
              </span>
            </div>
            <div className="row">
              <span className="label">Độ tự tin:</span>
              <span>{result.confidence}%</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;