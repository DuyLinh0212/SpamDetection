# Shopee Spam Comment Detection

Dự án Data Mining dùng để thu thập, xử lý và phát hiện bình luận spam trên Shopee. Pipeline hiện tại kết hợp crawl dữ liệu, trích xuất đặc trưng, gán nhãn bán tự động bằng Isolation Forest + rule-based score, chọn đặc trưng bằng ANOVA và huấn luyện mô hình XGBoost. Repo cũng có ứng dụng Tkinter để nạp model và lọc spam từ file CSV/JSON.

## Pipeline chính

```text
Shopee keywords
    -> Thu thập link sản phẩm
    -> Crawl bình luận đánh giá
    -> Feature engineering + TF-IDF
    -> Isolation Forest + rule-based score
    -> Gán nhãn spam theo top score
    -> ANOVA chọn 20 feature
    -> Train / valid / test
    -> XGBoost model
    -> App Tkinter lọc bình luận
```

## Cấu trúc thư mục

```text
Data-Mining-main/
├── application/
│   └── spam_filter_tkinter.py       # App giao diện để lọc bình luận spam
├── data/
│   ├── raw/
│   │   ├── shopee_keywords.txt      # Từ khóa tìm sản phẩm
│   │   ├── shopee_links.txt         # Link sản phẩm Shopee đã thu thập
│   │   ├── shopee_checkpoint.json   # Checkpoint khi crawl
│   │   ├── shopee_ratings.csv       # Dữ liệu đánh giá thô/phụ trợ
│   │   ├── shopee_test.csv          # Dữ liệu test thô/phụ trợ
│   │   └── data_crawl.csv           # Bình luận crawl từ Shopee
│   └── processed/
│       ├── data_cleaned.csv         # Dữ liệu numeric sau feature engineering
│       ├── data_labeled.csv         # Dữ liệu có if_score, rule_score, label
│       ├── train.csv                # Tập train cho XGBoost
│       ├── valid.csv                # Tập validation
│       └── test.csv                 # Tập test
├── models/
│   ├── xgb_spam_model.pkl           # Model XGBoost đã train
│   ├── test_predictions.csv         # Dự đoán trên tập test
│   ├── evaluation_charts.png        # ROC curve + confusion matrix
│   └── feature_importance.png       # Biểu đồ feature importance
├── reports/
│   ├── pipeline_summary.json        # Thống kê pipeline
│   ├── score_distribution_sorted.csv
│   └── score_percentiles.csv
├── src/
│   ├── shopee_collect_links.py      # Thu thập link sản phẩm theo keyword
│   ├── shopee_crawl.py              # Crawl bình luận đánh giá bằng Playwright
│   ├── shopee_feature_engineer.py   # Tạo đặc trưng numeric + TF-IDF
│   ├── spam_label_pipeline.py       # Gán nhãn, chọn feature, chia tập
│   ├── train_XGBoots.py             # Train XGBoost và lưu model
│   └── feature_extract.py           # Trích xuất 20 feature cho app dự đoán
├── requirements.txt
└── README.md
```

## Cài đặt

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

Các script crawl đang mặc định dùng Cốc Cốc với profile đã đăng nhập Shopee. Nếu máy khác đường dẫn, truyền lại `--coccoc-path`, `--coccoc-user-data` và `--coccoc-profile` khi chạy crawler.

## Chạy lại toàn bộ quy trình

### 1. Thu thập link sản phẩm

```bash
python src/shopee_collect_links.py \
  --keywords-file data/raw/shopee_keywords.txt \
  --per-keyword 15 \
  --out data/raw/shopee_links.txt
```

### 2. Crawl bình luận Shopee

```bash
python src/shopee_crawl.py \
  --links data/raw/shopee_links.txt \
  --total-pages 15 \
  --out data/raw/data_crawl.csv \
  --checkpoint data/raw/shopee_checkpoint.json
```

### 3. Tạo đặc trưng

```bash
python src/shopee_feature_engineer.py \
  --input data/raw/data_crawl.csv \
  --output-clean data/processed/data_cleaned.csv \
  --tfidf-max-features 25
```

### 4. Gán nhãn và chia dữ liệu

```bash
python src/spam_label_pipeline.py \
  --input data/processed/data_cleaned.csv \
  --output-dir data/processed \
  --report-dir reports \
  --spam-ratio 0.07 \
  --anova-k 20 \
  --random-state 42
```

Kết quả chính:

- `data/processed/data_labeled.csv`
- `data/processed/train.csv`
- `data/processed/valid.csv`
- `data/processed/test.csv`
- `reports/pipeline_summary.json`

### 5. Train XGBoost

```bash
python src/train_XGBoots.py \
  --data-dir data/processed \
  --output-dir models \
  --random-state 42
```

Script tự kiểm tra GPU qua `nvidia-smi`. Nếu không có GPU, mô hình sẽ train bằng CPU.

Kết quả chính:

- `models/xgb_spam_model.pkl`
- `models/test_predictions.csv`
- `models/evaluation_charts.png`
- `models/feature_importance.png`

## Chạy ứng dụng lọc spam

```bash
python application/spam_filter_tkinter.py
```

Ứng dụng sẽ tự nạp `models/xgb_spam_model.pkl` nếu file tồn tại. Có thể mở file CSV hoặc JSON chứa bình luận, chỉnh ngưỡng spam, lọc các dòng bị dự đoán là spam và lưu kết quả ra JSON.

Các trường text được app nhận diện tự động gồm `comment_text`, `content`, `text`, `comment`, `message`, `body`. Trường thời gian có thể là `CreatedAt`, `created_at`, `comment_time`, `time`, `timestamp`, `date`.

## Thống kê dữ liệu hiện tại

Theo `reports/pipeline_summary.json`:

- Tổng số comment: `22,933`
- Số feature đầu vào: `43`
- Số feature được chọn bằng ANOVA: `20`
- Tỷ lệ gán nhãn spam: `7%`
- Số spam: `1,606`
- Số bình luận thường: `21,327`
- Split dữ liệu: `16,053` train, `3,440` valid, `3,440` test
- Isolation Forest: `300` estimators, contamination `0.10`

## Ghi chú

- Nhãn spam trong project được tạo bán tự động từ anomaly score và rule score, không phải nhãn thủ công hoàn toàn.
- `models/` đang được `.gitignore`; nếu clone repo không có model, hãy chạy lại bước train trước khi dùng app.
- `data/coccoc_profile/` và các file debug lớn không nên commit vì có thể chứa profile trình duyệt hoặc dữ liệu tạm.
- Shopee có thể thay đổi giao diện hoặc cơ chế chống bot, nên selector trong crawler có thể cần cập nhật khi crawl dữ liệu mới.
