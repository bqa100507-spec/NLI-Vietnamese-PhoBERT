# NLI Vietnamese PhoBERT: Systematic Error Analysis

## 📌 Giới thiệu (Introduction)
Dự án này nghiên cứu và triển khai mô hình Natural Language Inference (NLI) cho tiếng Việt, tinh chỉnh (fine-tuning) dựa trên kiến trúc PhoBERT-base. NLI là bài toán cốt lõi trong Xử lý ngôn ngữ tự nhiên (NLP), nhằm xác định mối quan hệ logic giữa hai câu (Tiền đề - Premise và Giả thuyết - Hypothesis) thành 3 nhãn: Entailment (Kéo theo), Contradiction (Mâu thuẫn), và Neutral (Trung lập).

Điểm khác biệt của dự án này không chỉ dừng ở việc huấn luyện mô hình, mà tập trung sâu vào Phân tích lỗi hệ thống (Systematic Error Analysis) để đánh giá năng lực suy luận thực sự của mô hình ngôn ngữ, thay vì chỉ khớp từ vựng bề mặt.

## 🗂️ Dữ liệu (Dataset)
Sử dụng bộ dữ liệu ViANLI (Vietnamese Adversarial Natural Language Inference) từ `uitnlp/ViANLI`.
Đặc điểm của ViANLI là chứa nhiều mẫu câu đối nghịch (adversarial), đòi hỏi mô hình phải có khả năng đọc hiểu ngữ nghĩa phức tạp thay vì chỉ nhận diện từ khóa trùng lặp.

## 🚀 Điểm nhấn Kỹ thuật (Technical Highlights)
Pipeline huấn luyện được xây dựng từ đầu (Custom Training Loop) bằng PyTorch nhằm tối ưu hóa và kiểm soát hoàn toàn luồng dữ liệu:

- **Tối ưu phần cứng**: Tích hợp Mixed Precision (FP16) và Gradient Accumulation, giảm thiểu 50% VRAM sử dụng mà không làm giảm tốc độ huấn luyện.
- **Auto-device Detection**: Tự động phát hiện môi trường (CPU/GPU). Khi chạy trên local (CPU), code tự động thu nhỏ kích thước batch và epoch để phục vụ debug nhanh gọn; khi deploy lên Cloud (GPU), hệ thống tự mở khóa full-mode.
- **Modular Codebase**: Xây dựng class Dataset linh hoạt (tự động ánh xạ label dictionary không cần hard-code).

## 📊 Kết quả Baseline (Baseline Results)
Mô hình PhoBERT-base đạt kết quả ban đầu trên tập Validation/Test:

- Accuracy: ~43%
- Macro F1-score: ~42%

## 🔬 Phân tích Lỗi (Key Findings in Error Analysis)
Thông qua ma trận nhầm lẫn (Confusion Matrix) và việc lọc các Dự đoán sai có độ tự tin cao (>70%), dự án đã "bắt bệnh" được 3 lỗ hổng suy luận điển hình của mô hình:

- **Lỗ hổng Toán học & Thời gian (Numerical/Temporal Blindness)**: Mô hình ngôn ngữ chỉ khớp chuỗi ký tự mà không hiểu logic tính toán (VD: Không hiểu 100g x 10 = 1kg, dẫn đến việc gán nhãn Contradiction một cách sai lệch).
- **Bẫy chồng chéo từ vựng (Lexical Overlap Trap)**: Khi hai câu lặp lại nhiều từ giống nhau (đặc biệt khi bị đảo cấu trúc Chủ - Vị), mô hình lười suy luận và lập tức gán nhãn Entailment.
- **Định kiến dán nhãn (Systematic Bias)**: Mô hình có xu hướng "sợ hãi" việc đưa ra quyết định an toàn là Neutral, luôn cố ép các câu trung lập thành Entailment hoặc Contradiction một cách khiên cưỡng.

👉 Chi tiết báo cáo đồ thị học tập, ma trận nhầm lẫn và mổ xẻ dữ liệu: `notebook/errorAnalysis.ipynb`

## 📂 Cấu trúc dự án (Project Structure)
```text
├── model/
│   ├── best_model.pth      # Trọng số mô hình tốt nhất 
│   └── history.json        # Lịch sử training (Loss, Accuracy, F1-score)
├── notebook/
│   ├── eda.ipynb           # Exploratory Data Analysis (Khám phá dữ liệu)
│   └── errorAnalysis.ipynb # Dự đoán, trực quan hóa và phân tích lỗi chuyên sâu
├── src/
│   ├── dataset.py          # Class NLIDataset, xử lý tokenization & padding
│   ├── model.py            # Hàm khởi tạo mô hình phân loại
│   └── train.py            # Vòng lặp huấn luyện Custom (Train/Valid Loop)
├── .env                    # (Bảo mật) Biến môi trường
├── .gitignore              
└── README.md               
```

## 🛠️ Cài đặt và Sử dụng (Installation)
Clone dự án về máy:
```bash
git clone https://github.com/bqa100507-spec/NLI-Vietnamese-PhoBERT.git
cd NLI-Vietnamese-PhoBERT
```

Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Khởi chạy quá trình huấn luyện:
```bash
python src/train.py
```
