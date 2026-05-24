import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROMO_WORDS = [
    "sale",
    "free",
    "miễn phí",
    "giảm giá",
    "khuyến mãi",
    "ưu đãi",
    "deal",
    "flash sale",
    "xả kho",
    "voucher",
    "mã giảm giá",
    "mua 1 tặng 1",
    "freeship",
    "kiếm tiền",
    "làm giàu",
    "trúng thưởng",
    "hoàn tiền",
]
BRAND_WORDS = [
    "iphone",
    "samsung",
    "xiaomi",
    "oppo",
    "vivo",
    "nokia",
    "asus",
    "msi",
    "dell",
    "hp",
    "lenovo",
    "apple",
]
CALL_ACTION_WORDS = [
    "mua ngay",
    "click",
    "nhấn",
    "inbox",
    "đặt hàng",
    "order",
    "ib",
    "liên hệ",
    "zalo",
    "call",
    "contact",
    "nhắn tin",
    "nhắn ngay",
    "vào link",
    "gọi",
]


def safe_text(x: str) -> str:
    return x if isinstance(x, str) else ""


def count_links(text: str) -> int:
    return len(re.findall(r"(http://|https://|www\.)", text, flags=re.IGNORECASE))


def count_special_chars(text: str) -> int:
    return len(re.findall(r"[^A-Za-z0-9À-ỹ\s]", text))


def repeat_word_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    total = len(words)
    uniq = len(set(words))
    return (total - uniq) / total


def duplicate_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    runs = re.findall(r"(.)\1{2,}", text)
    # count chars that are part of repeated runs
    repeated_chars = 0
    for m in re.finditer(r"(.)\1{2,}", text):
        repeated_chars += len(m.group(0))
    return repeated_chars / len(text)


def contains_any(text: str, keywords: list[str]) -> int:
    t = text.lower()
    return 1 if any(k in t for k in keywords) else 0


def num_repeated_phrases(words: list[str]) -> int:
    # count repeated consecutive bigrams (e.g., "mua ngay mua ngay")
    if len(words) < 4:
        return 0
    count = 0
    for i in range(len(words) - 3):
        if words[i] == words[i + 2] and words[i + 1] == words[i + 3]:
            count += 1
    return count


def extract_hour_decimal(comment_time: str) -> float:
    # formats like "2022-05-16 16:13" or "16:13"
    if not comment_time:
        return np.nan
    m = re.search(r"(\d{1,2}):(\d{2})", comment_time)
    if not m:
        return np.nan
    hour = int(m.group(1))
    minute = int(m.group(2))
    return round(hour + minute / 100, 2)


def is_night_hour(hour_dec: float) -> int:
    if np.isnan(hour_dec):
        return 0
    hour = int(hour_dec)
    return 1 if (hour >= 23 or hour <= 5) else 0


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for Shopee comments")
    parser.add_argument("--input", default="data/raw/data_crawl.csv", help="Input CSV from crawl")
    parser.add_argument("--output-features", default="data/processed/data_features.csv", help="Output features CSV")
    parser.add_argument("--output-clean", default="data/processed/data_cleaned.csv", help="Output numeric feature CSV")
    args = parser.parse_args()

    print("[1/4] Load data")
    df = pd.read_csv(args.input)

    print("[2/4] Basic text features")
    text = df.get("comment_text", pd.Series(dtype=str)).apply(safe_text)
    words = text.str.lower().str.findall(r"[A-Za-zÀ-ỹ0-9]+")

    df["length_char"] = text.str.len()
    df["length_word"] = words.apply(len)
    df["num_uppercase"] = text.apply(lambda t: sum(1 for c in t if c.isupper()))
    df["uppercase_ratio"] = df["num_uppercase"] / df["length_char"].replace(0, np.nan)
    df["num_exclamation"] = text.str.count("!")
    df["num_question"] = text.str.count(r"\?")
    df["num_special_chars"] = text.apply(count_special_chars)
    df["num_digits"] = text.str.count(r"\d")
    df["num_links"] = text.apply(count_links)
    df["repeat_word_ratio"] = words.apply(repeat_word_ratio)
    df["duplicate_char_ratio"] = text.apply(duplicate_char_ratio)
    df["contains_promo_words"] = text.apply(lambda t: contains_any(t, PROMO_WORDS))
    df["contains_brand"] = text.apply(lambda t: contains_any(t, BRAND_WORDS))
    df["contains_call_action"] = text.apply(lambda t: contains_any(t, CALL_ACTION_WORDS))
    df["num_repeated_phrases"] = words.apply(num_repeated_phrases)

    print("[3/4] Time + identity features")
    df["time_comment"] = df.get("comment_time", "").astype(str).apply(extract_hour_decimal)
    df["is_night"] = df["time_comment"].apply(is_night_hour)
    df["rating"] = df.get("rating_star", df.get("rating", np.nan))
    df["user_id"] = df.get("userId", df.get("user_id", ""))
    df["product_id"] = df.get("productID", df.get("product_id", ""))

    print("[4/4] Save numeric features")
    base_cols = [
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
    cleaned = df[base_cols].reset_index(drop=True)

    # Drop any remaining non-numeric columns (no strings allowed)
    cleaned = cleaned.select_dtypes(exclude=["object"])

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
