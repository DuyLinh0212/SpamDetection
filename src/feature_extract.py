import re
import unicodedata

import numpy as np


PROMO_WORDS = [
    "sale",
    "free",
    "miễn phí",
    "giam gia",
    "giảm giá",
    "giamgia",
    "khuyến mãi",
    "khuyen mai",
    "ưu đãi",
    "uu dai",
    "deal",
    "flash sale",
    "xả kho",
    "xa kho",
    "voucher",
    "mã giảm giá",
    "ma giam gia",
    "mua 1 tặng",
    "mua 1 tang",
    "freeship",
    "kiếm tiền",
    "kiem tien",
    "làm giàu",
    "lam giau",
    "trúng thưởng",
    "trung thuong",
    "hoàn tiền",
    "hoan tien",
    "giá sỉ",
    "gia si",
    "giá xưởng",
    "gia xuong",
    "nhận xu",
    "nhan xu",
    "tặng xu",
    "tang xu",
    "nhận quà",
    "nhan qua",
    "giveaway",
    "giá rẻ",
    "gia re",
    "giảm sâu",
    "giam sau",
    "hoàn xu",
    "hoan xu",
    "quà tặng",
    "qua tang",
    "mã độc quyền",
    "ma doc quyen",
    "săn sale",
    "san sale",
    "săn mã",
    "san ma",
    "hoa hồng",
    "hoa hong",
    "cộng tác viên",
    "cong tac vien",
    "không cần vốn",
    "khong can von",
]

CALL_ACTIONS = [
    "mua ngay",
    "click",
    "nhấn",
    "nhan",
    "inbox",
    "ib",
    "đặt hàng",
    "dat hang",
    "order",
    "liên hệ",
    "lien he",
    "zalo",
    "call",
    "contact",
    "nhắn tin",
    "nhan tin",
    "nhắn ngay",
    "nhan ngay",
    "vào link",
    "vao link",
    "truy cập",
    "truy cap",
    "gọi",
    "goi",
    "ghé shop",
    "ghe shop",
    "follow",
    "tham gia",
    "nhanh tay",
    "để lại",
    "de lai",
    "xem bio",
    "bio",
    "hồ sơ",
    "ho so",
    "kênh mình",
    "kenh minh",
    "trang mình",
    "trang minh",
    "trang cá nhân",
    "trang ca nhan",
    "livestream",
    "chốt đơn",
    "chot don",
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

LINK_RE = re.compile(
    r"(https?://|www\.|[\w.-]+\.(?:com|vn|net|org|info|shop|xyz|site|test|me|ly|io|co)(?:/\S*)?)",
    re.IGNORECASE,
)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?84|0)(?:[\s.-]?\d){8,10}(?!\d)")
SPECIAL_RE = re.compile(r"[^\w\s]", re.UNICODE)
WORD_RE = re.compile(r"[\w]+", re.UNICODE)

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


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    decomposed = unicodedata.normalize("NFD", text)
    without_accents = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return without_accents.replace("đ", "d")


def contains_any(text: str, keywords: list[str]) -> int:
    lower = str(text or "").lower()
    normalized = normalize_text(text)
    return int(any(k.lower() in lower or normalize_text(k) in normalized for k in keywords))


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


def rule_based_spam_score(text: str, created_at: str = "", rating: float = 0.0) -> float:
    features = extract_feature_dict(text, created_at, rating)
    normalized = normalize_text(text)
    score = 0.0

    has_link = features["num_links"] > 0
    has_phone = PHONE_RE.search(text or "") is not None
    has_promo = features["contains_promo_words"] == 1
    has_cta = features["contains_call_action"] == 1

    if has_link:
        score += 0.45
    if has_phone:
        score += 0.45
    if has_promo:
        score += 0.28
    if has_cta:
        score += 0.25
    if has_promo and has_cta:
        score += 0.22
    if has_link and (has_promo or has_cta):
        score += 0.20
    if has_phone and (has_promo or has_cta):
        score += 0.20
    if features["repeat_word_ratio"] >= 0.28:
        score += 0.35
    if features["num_repeated_phrases"] >= 1:
        score += 0.25
    if features["num_exclamation"] >= 3:
        score += 0.15
    if features["num_digits"] >= 8:
        score += 0.15
    if rating <= 2 and (has_promo or has_cta or has_link or has_phone):
        score += 0.15

    spam_phrases = [
        "khong lien quan",
        "khong biet san pham",
        "khong mua san pham",
        "shop minh",
        "shop em",
        "qua shop",
        "ghe shop",
        "trang minh",
        "trang ca nhan",
        "ho so minh",
        "kenh minh",
        "livestream",
        "chot don",
        "nguoi theo doi",
        "xin tuong tac",
        "tra tuong tac",
        "follow cheo",
        "follow lai",
        "gia si",
        "gia xuong",
        "gia huy diet",
        "hang xach tay",
        "hang cao cap",
        "hang theo yeu cau",
        "order hang",
        "nhan ma",
        "san ma",
        "san sale",
        "san deal",
        "nhom san deal",
        "link san deal",
        "ma rieng",
        "ma noi bo",
        "ma doc quyen",
        "ma gioi thieu",
        "link gioi thieu",
        "nhan voucher",
        "nhan qua",
        "qua tang",
        "lay xu",
        "nhan xu",
        "tang xu",
        "lay diem nhiem vu",
        "du nhiem vu",
        "lam nhiem vu",
        "comment nhan tien",
        "danh gia 5 sao",
        "danh gia ho",
        "danh gia san pham so luong lon",
        "binh luan tu dong",
        "copy tu dong",
        "comment de",
        "cham mot cai",
        "gioi thieu shop",
        "muc dich chi de kiem thu",
        "kiem thu he thong",
        "spam spam",
        "seeding",
        "tang uy tin",
        "tang danh gia",
        "tang follow",
        "tang don",
        "tang inbox",
        "tang luot thich",
        "tang mat livestream",
        "dich vu",
        "quang cao",
        "chay quang cao",
        "len top",
        "bang gia",
        "nguon hang",
        "gom don",
        "combo ben minh",
        "mua qua minh",
        "ben minh",
        "ban hang online",
        "ban si le",
        "dropship",
        "loi nhuan",
        "app hoan tien",
        "tuyen cong tac vien",
    ]
    if any(phrase in normalized for phrase in spam_phrases):
        score += 0.55

    return min(score, 1.0)


def extract_features(text: str, created_at: str, rating: float = 0.0) -> list[float]:
    features = extract_feature_dict(text, created_at, rating)
    return [float(features.get(name, 0.0)) for name in SELECTED_FEATURES]
