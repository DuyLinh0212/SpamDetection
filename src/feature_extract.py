import re

import numpy as np


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
CALL_ACTIONS = [
    "mua ngay",
    "click",
    "nhấn",
    "inbox",
    "ib",
    "đặt hàng",
    "order",
    "liên hệ",
    "zalo",
    "call",
    "contact",
    "nhắn tin",
    "nhắn ngay",
    "vào link",
    "gọi",
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

LINK_RE = re.compile(r"(http://|https://|www\.)", re.IGNORECASE)
SPECIAL_RE = re.compile(r"[^A-Za-z0-9À-ỹ\s]")
WORD_RE = re.compile(r"[A-Za-zÀ-ỹ0-9]+")

# This list must match data/processed/train.csv, excluding the label column.
# It is updated after running src/spam_label_pipeline.py.
SELECTED_FEATURES = [
    "num_links",
    "contains_call_action",
    "num_uppercase",
    "contains_promo_words",
    "num_digits",
    "num_repeated_phrases",
    "num_special_chars",
    "num_exclamation",
    "repeat_word_ratio",
    "length_char",
    "length_word",
    "uppercase_ratio",
    "duplicate_char_ratio",
    "num_question",
    "is_night",
    "rating",
    "contains_brand",
    "time_comment",
]


def contains_any(text: str, keywords: list[str]) -> int:
    lower = text.lower()
    return int(any(k in lower for k in keywords))


def duplicate_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    repeated_chars = 0
    for match in re.finditer(r"(.)\1{2,}", text):
        repeated_chars += len(match.group(0))
    return repeated_chars / len(text)


def num_repeated_phrases(words: list[str]) -> int:
    if len(words) < 4:
        return 0
    count = 0
    for i in range(len(words) - 3):
        if words[i] == words[i + 2] and words[i + 1] == words[i + 3]:
            count += 1
    return count


def extract_hour_decimal(created_at: str) -> float:
    match = re.search(r"(\d{1,2}):(\d{2})", str(created_at or ""))
    if not match:
        return 0.0
    hour = int(match.group(1))
    minute = int(match.group(2))
    return round(hour + minute / 100, 2)


def is_night_hour(hour_dec: float) -> int:
    if np.isnan(hour_dec):
        return 0
    hour = int(hour_dec)
    return int(hour >= 23 or hour <= 5)


def extract_feature_dict(text: str, created_at: str = "", rating: float = 0.0) -> dict[str, float]:
    text = text if isinstance(text, str) else ""
    words = WORD_RE.findall(text.lower())
    length_char = len(text)
    num_uppercase = sum(1 for c in text if c.isupper())
    time_comment = extract_hour_decimal(created_at)

    return {
        "length_char": length_char,
        "length_word": len(words),
        "num_uppercase": num_uppercase,
        "uppercase_ratio": num_uppercase / length_char if length_char > 0 else 0.0,
        "num_exclamation": text.count("!"),
        "num_question": text.count("?"),
        "num_special_chars": len(SPECIAL_RE.findall(text)),
        "num_digits": sum(1 for c in text if c.isdigit()),
        "num_links": len(LINK_RE.findall(text)),
        "repeat_word_ratio": (len(words) - len(set(words))) / len(words) if words else 0.0,
        "duplicate_char_ratio": duplicate_char_ratio(text),
        "contains_promo_words": contains_any(text, PROMO_WORDS),
        "contains_brand": contains_any(text, BRAND_WORDS),
        "contains_call_action": contains_any(text, CALL_ACTIONS),
        "rating": rating,
        "time_comment": time_comment,
        "is_night": is_night_hour(time_comment),
        "num_repeated_phrases": num_repeated_phrases(words),
    }


def extract_features(text: str, created_at: str, rating: float = 0.0) -> list[float]:
    features = extract_feature_dict(text, created_at, rating)
    return [float(features.get(name, 0.0)) for name in SELECTED_FEATURES]
