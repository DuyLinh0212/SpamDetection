import argparse
from pathlib import Path

import pandas as pd

from feature_extract import extract_feature_dict


BASE_COLS = [
    "length_char",
    "length_word",
    "num_uppercase",
    "uppercase_ratio",
    "num_exclamation",
    "num_question",
    "num_special_chars",
    "num_digits",
    "num_links",
    "repeat_word_ratio",
    "duplicate_char_ratio",
    "contains_promo_words",
    "contains_brand",
    "contains_call_action",
    "rating",
    "time_comment",
    "is_night",
    "num_repeated_phrases",
]


def safe_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def pick_series(df: pd.DataFrame, *names: str, default: object = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series([default] * len(df), index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    text = pick_series(df, "comment_text", "Content", "content", "text").apply(safe_text)
    created_at = pick_series(df, "comment_time", "CreatedAt", "created_at", "time", "timestamp", "date").fillna("")
    rating = pd.to_numeric(pick_series(df, "rating_star", "rating", "stars", "star", "score", default=0), errors="coerce").fillna(0)

    rows = [
        extract_feature_dict(comment, str(time_value), float(rating_value))
        for comment, time_value, rating_value in zip(text, created_at, rating)
    ]

    return pd.DataFrame(rows, columns=BASE_COLS).fillna(0)


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for Shopee comments")
    parser.add_argument("--input", default="data/raw/data_crawl.csv", help="Input CSV from crawl")
    parser.add_argument("--output-features", default="data/processed/data_features.csv", help="Output features CSV")
    parser.add_argument("--output-clean", default="data/processed/data_cleaned.csv", help="Output numeric feature CSV")
    args = parser.parse_args()

    print("[1/3] Load data")
    df = pd.read_csv(args.input)

    print("[2/3] Extract features with shared feature_extract.py logic")
    cleaned = build_features(df)

    print("[3/3] Save numeric features")
    out_features = Path(args.output_features)
    out_features.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_features, index=False, encoding="utf-8-sig")

    out_clean = Path(args.output_clean)
    out_clean.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(out_clean, index=False, encoding="utf-8-sig")

    print(f"Saved features: {out_features}")
    print(f"Saved cleaned: {out_clean}")


if __name__ == "__main__":
    main()
