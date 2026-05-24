# Spam Detection API

## Run

```powershell
py -m pip install -r requirements.txt
py -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Endpoints

- `GET /health`: check model path and selected features.
- `POST /predict-csv`: upload CSV and return spam predictions.

Example:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict-csv" `
  -F "file=@data/raw/data_crawl.csv" `
  -F "threshold=0.5"
```

The CSV should contain a comment column named one of:

```text
comment_text, Content, content, text, comment, message, body
```

Optional time columns:

```text
CreatedAt, created_at, comment_time, time, timestamp, date
```

Optional rating columns:

```text
rating_star, rating, stars, star, score
```
