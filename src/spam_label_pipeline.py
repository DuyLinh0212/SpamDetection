"""
Pipeline: cleaned numeric data -> Isolation Forest -> rule score -> label
-> ANOVA feature selection -> train/valid/test split.

Usage:
    python src/spam_label_pipeline.py \
        --input data/processed/data_cleaned.csv \
        --output-dir data/processed \
        --report-dir reports \
        --spam-ratio 0.07 \
        --anova-k 20 \
        --random-state 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split


SYNTHETIC_SPAM_REGEX = (
    r"example\.com|bit\.ly/fake-sale|SPAM\d+|DEAL\d+|"
    r"Ai can .* gia re .* Zalo 0|Ai cần .* giá rẻ .* Zalo 0|"
    r"Can cong tac vien .* hoa hong cao|Cần cộng tác viên .* hoa hồng cao|"
    r"Kho moi ve .* sieu re sieu dep|Kho mới về .* siêu rẻ siêu đẹp|"
    r"gia huy diet|giá hủy diệt|"
    r"(?:kiem tien online|nhan hoa hong|viec nhe luong cao|dang ky nhan qua|"
    r"trung thuong 100%|hoan tien cuc lon) tai nha, khong can von|"
    r"(?:kiếm tiền online|nhận hoa hồng|việc nhẹ lương cao|đăng ký nhận quà|"
    r"trúng thưởng 100%|hoàn tiền cực lớn) tại nhà, không cần vốn|"
    r"Mua hang nhan qua mien phi|Mua hàng nhận quà miễn phí|"
    r"Ban da duoc chon nhan qua|Bạn đã được chọn nhận quà|"
    r"ORDER NGAY .* GIA SI|ORDER NGAY .* GIÁ SỈ|"
    r"Khuyen mai khuyen mai|Khuyến mãi khuyến mãi|"
    r"Ho tro vay mua hang|Hỗ trợ vay mua hàng|"
    r"Nhom san ma giam gia|Nhóm săn mã giảm giá"
)


def minmax_norm(values: np.ndarray) -> np.ndarray:
    v = values.astype(float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def run_isolation_forest(X: pd.DataFrame, contamination: float, random_state: int):
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)
    raw = -model.score_samples(X)
    return raw, model


def analyze_anomalies(df: pd.DataFrame, if_score: np.ndarray, top_ratio: float = 0.10):
    n_top = max(1, int(len(df) * top_ratio))
    order = np.argsort(-if_score)
    top_idx = order[:n_top]
    rest_idx = order[n_top:]

    top_mean = df.iloc[top_idx].mean(numeric_only=True)
    rest_mean = df.iloc[rest_idx].mean(numeric_only=True)
    diff = (top_mean - rest_mean).sort_values(ascending=False)

    summary = pd.DataFrame(
        {
            "top_anomaly_mean": top_mean,
            "normal_mean": rest_mean,
            "diff_top_minus_normal": diff,
        }
    ).sort_values("diff_top_minus_normal", ascending=False)
    return summary, top_idx


def get_rule_inputs(df: pd.DataFrame) -> dict[str, pd.Series]:
    def col(name: str, default: float = 0.0) -> pd.Series:
        s = df.get(name, None)
        if isinstance(s, pd.Series):
            return s.fillna(default)
        return pd.Series(default, index=df.index, dtype=float)

    lc = col("length_char")
    lw = col("length_word")
    rating = col("rating")
    links = col("num_links")
    special = col("num_special_chars")
    repeated_phrases = col("num_repeated_phrases")
    dup_char_ratio = col("duplicate_char_ratio")
    repeat_word_ratio = col("repeat_word_ratio")
    upper_ratio = col("uppercase_ratio")
    exclam = col("num_exclamation")
    digits = col("num_digits")
    promo = col("contains_promo_words")
    cta = col("contains_call_action")
    brand = col("contains_brand")
    is_night = col("is_night")
    questions = col("num_question")

    char_len = lc.replace(0, np.nan)
    digit_ratio = (digits / char_len).fillna(0)
    special_ratio = (special / char_len).fillna(0)
    phone_like = (digits >= 9) & (digits <= 13)
    digit_heavy = (digits >= 14) | ((digit_ratio > 0.12) & (lc >= 40))
    num_uppercase = upper_ratio * lc

    # These TF-IDF columns exist in the current cleaned data and are mostly
    # review-like context, so they are used only for score discounts.
    review_terms = (
        col("tfidf_giao") +
        col("tfidf_mua") +
        col("tfidf_nhanh") +
        col("tfidf_shop")
    )

    strong_contact = (links > 0) | phone_like | digit_heavy
    has_ad_words = (promo == 1) | (cta == 1) | (brand == 1)

    return {
        "lc": lc,
        "lw": lw,
        "rating": rating,
        "links": links,
        "special": special,
        "repeated_phrases": repeated_phrases,
        "dup_char_ratio": dup_char_ratio,
        "repeat_word_ratio": repeat_word_ratio,
        "upper_ratio": upper_ratio,
        "exclam": exclam,
        "digits": digits,
        "promo": promo,
        "cta": cta,
        "brand": brand,
        "is_night": is_night,
        "questions": questions,
        "digit_ratio": digit_ratio,
        "special_ratio": special_ratio,
        "phone_like": phone_like,
        "digit_heavy": digit_heavy,
        "num_uppercase": num_uppercase,
        "review_terms": review_terms,
        "strong_contact": strong_contact,
        "has_ad_words": has_ad_words,
    }


def build_rule_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    r = get_rule_inputs(df)

    return {
        "link_present": r["links"] > 0,
        "many_links": r["links"] >= 3,
        "phone_like_digits": r["phone_like"],
        "digit_heavy": r["digit_heavy"],
        "short_digit_burst": (r["digits"] >= 6) & (r["lc"] <= 80),
        "promo_word": r["promo"] == 1,
        "call_to_action": r["cta"] == 1,
        "brand_word": r["brand"] == 1,
        "promo_and_cta": (r["promo"] == 1) & (r["cta"] == 1),
        "contact_with_ad_words": r["strong_contact"] & ((r["promo"] == 1) | (r["cta"] == 1)),
        "link_with_phone": (r["links"] > 0) & r["phone_like"],
        "link_with_digits": (r["links"] > 0) & (r["digits"] >= 4),
        "short_cta_high_rating": (r["cta"] == 1) & (r["rating"] >= 4) & (r["lc"] < 120),
        "long_comment": r["lc"] > 220,
        "very_long_comment": r["lc"] > 320,
        "extreme_long_comment": r["lc"] > 450,
        "many_words": r["lw"] > 90,
        "long_with_contact": (r["lc"] > 220) & r["strong_contact"],
        "long_with_ad_words": (r["lc"] > 320) & r["has_ad_words"],
        "empty_comment": r["lc"] == 0,
        "very_short_comment": (r["lc"] > 0) & (r["lc"] < 5),
        "short_five_star": (r["lw"] <= 2) & (r["rating"] >= 5),
        "short_excited_five_star": (r["lc"] < 12) & (r["rating"] >= 5) & (r["exclam"] > 0),
        "many_special_chars": r["special"] > 12,
        "extreme_special_chars": r["special"] > 28,
        "high_special_ratio": (r["special_ratio"] > 0.18) & (r["lc"] > 30),
        "many_exclamation": r["exclam"] > 2,
        "extreme_exclamation": r["exclam"] > 8,
        "uppercase_ratio_high": (r["upper_ratio"] > 0.12) & (r["lc"] > 20),
        "uppercase_ratio_extreme": (r["upper_ratio"] > 0.25) & (r["lc"] > 20),
        "many_uppercase_chars": r["num_uppercase"] > 30,
        "extreme_uppercase_chars": r["num_uppercase"] > 80,
        "repeated_phrase": r["repeated_phrases"] > 0,
        "many_repeated_phrases": r["repeated_phrases"] > 3,
        "duplicate_char_run": r["dup_char_ratio"] > 0.08,
        "extreme_duplicate_char_run": r["dup_char_ratio"] > 0.20,
        "repeat_word_high": r["repeat_word_ratio"] > 0.30,
        "repeat_word_very_high": r["repeat_word_ratio"] > 0.50,
        "repeat_word_extreme": r["repeat_word_ratio"] > 0.70,
        "many_questions": r["questions"] >= 2,
        "question_with_ad_words": (r["questions"] >= 1) & ((r["promo"] == 1) | (r["cta"] == 1)),
        "question_with_contact": (r["questions"] >= 1) & r["strong_contact"],
        "night_comment": r["is_night"] == 1,
        "night_with_contact": (r["is_night"] == 1) & r["strong_contact"],
        "low_rating_with_contact": (r["rating"] <= 2) & r["strong_contact"],
        "review_like_no_contact": (
            (r["review_terms"] > 0.20) &
            (r["links"] == 0) &
            (~r["phone_like"]) &
            (~r["digit_heavy"])
        ),
        "standalone_promo": (
            (r["promo"] == 1) &
            (r["links"] == 0) &
            (~r["phone_like"]) &
            (~r["digit_heavy"]) &
            (r["repeat_word_ratio"] < 0.40)
        ),
        "standalone_repeat": (
            (r["repeat_word_ratio"] > 0.50) &
            (r["links"] == 0) &
            (~r["phone_like"]) &
            (~r["digit_heavy"]) &
            (r["promo"] == 0) &
            (r["cta"] == 0)
        ),
        "standalone_special": (
            (r["special"] > 12) &
            (r["links"] == 0) &
            (~r["phone_like"]) &
            (~r["digit_heavy"]) &
            (r["promo"] == 0) &
            (r["cta"] == 0)
        ),
        "natural_short_review": (
            (r["lc"] >= 15) &
            (r["lc"] <= 220) &
            (r["links"] == 0) &
            (~r["phone_like"]) &
            (~r["digit_heavy"]) &
            (r["promo"] == 0) &
            (r["cta"] == 0) &
            (r["repeat_word_ratio"] < 0.30) &
            (r["dup_char_ratio"] < 0.20) &
            (r["upper_ratio"] < 0.25) &
            (r["exclam"] <= 2) &
            (r["special"] <= 10) &
            (r["digits"] <= 4)
        ),
        "natural_positive_review": (
            (r["rating"] >= 4) &
            (r["links"] == 0) &
            (~r["phone_like"]) &
            (~r["digit_heavy"]) &
            (r["promo"] == 0) &
            (r["cta"] == 0) &
            (r["repeat_word_ratio"] < 0.20) &
            (r["questions"] <= 2)
        ),
    }


RULE_WEIGHTS = {
    "link_present": 4.8,
    "many_links": 3.5,
    "phone_like_digits": 4.2,
    "digit_heavy": 2.0,
    "short_digit_burst": 1.2,
    "promo_word": 0.35,
    "call_to_action": 0.9,
    "brand_word": 0.25,
    "promo_and_cta": 0.9,
    "contact_with_ad_words": 2.2,
    "link_with_phone": 2.8,
    "link_with_digits": 1.4,
    "short_cta_high_rating": 0.7,
    "long_comment": 1.5,
    "very_long_comment": 2.0,
    "extreme_long_comment": 2.5,
    "many_words": 1.0,
    "long_with_contact": 2.4,
    "long_with_ad_words": 1.4,
    "empty_comment": 0.4,
    "very_short_comment": 1.2,
    "short_five_star": 0.8,
    "short_excited_five_star": 0.8,
    "many_special_chars": 0.8,
    "extreme_special_chars": 1.2,
    "high_special_ratio": 1.0,
    "many_exclamation": 0.5,
    "extreme_exclamation": 1.2,
    "uppercase_ratio_high": 0.6,
    "uppercase_ratio_extreme": 1.0,
    "many_uppercase_chars": 1.0,
    "extreme_uppercase_chars": 1.5,
    "repeated_phrase": 1.0,
    "many_repeated_phrases": 1.4,
    "duplicate_char_run": 1.2,
    "extreme_duplicate_char_run": 1.0,
    "repeat_word_high": 0.8,
    "repeat_word_very_high": 1.2,
    "repeat_word_extreme": 1.5,
    "many_questions": 0.5,
    "question_with_ad_words": 0.8,
    "question_with_contact": 0.8,
    "night_comment": 0.1,
    "night_with_contact": 0.35,
    "low_rating_with_contact": 0.8,
    "review_like_no_contact": -0.8,
    "standalone_promo": -0.9,
    "standalone_repeat": -0.8,
    "standalone_special": -0.5,
    "natural_short_review": -1.8,
    "natural_positive_review": -1.2,
}


def build_rule_score(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    masks = build_rule_masks(df)

    for name, weight in RULE_WEIGHTS.items():
        score += np.where(masks[name], weight, 0.0)

    return score.clip(lower=0.0)


def summarize_rule_hits(df: pd.DataFrame) -> pd.DataFrame:
    masks = build_rule_masks(df)
    rows = []
    for name, weight in RULE_WEIGHTS.items():
        hits = int(masks[name].sum())
        rows.append(
            {
                "rule": name,
                "weight": weight,
                "n_hits": hits,
                "hit_rate": hits / len(df) if len(df) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["weight", "n_hits"], ascending=[False, False])


def load_manual_spam_mask(text_input: str | None, n_rows: int, regex: str) -> np.ndarray | None:
    if not text_input:
        return None

    text_df = pd.read_csv(text_input, usecols=["comment_text"])
    if len(text_df) != n_rows:
        raise ValueError(
            f"manual spam text input has {len(text_df)} rows, expected {n_rows}: {text_input}"
        )

    text = text_df["comment_text"].fillna("").astype(str)
    return text.str.contains(regex, case=False, regex=True, na=False).to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/data_cleaned.csv")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--spam-ratio", type=float, default=0.07,
                        help="Ratio of comments labeled as spam, e.g. 0.07 = top 7%.")
    parser.add_argument("--if-contamination", type=float, default=0.10,
                        help="Isolation Forest contamination.")
    parser.add_argument("--anova-k", type=int, default=20,
                        help="Number of features kept after ANOVA.")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--valid-size", type=float, default=0.15)
    parser.add_argument(
        "--manual-spam-text-input",
        default=None,
        help="Optional raw CSV with comment_text. Matching rows are forced to label=1.",
    )
    parser.add_argument(
        "--manual-spam-regex",
        default=SYNTHETIC_SPAM_REGEX,
        help="Regex used with --manual-spam-text-input for forced spam labels.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    rep_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/8] Load data: {args.input}")
    df = pd.read_csv(args.input)
    df = df.select_dtypes(include=[np.number]).copy()
    df = df.fillna(0)
    print(f"  -> shape: {df.shape}")

    feature_cols = df.columns.tolist()

    print(f"[2/8] Isolation Forest (contamination={args.if_contamination})")
    if_score, _ = run_isolation_forest(
        df[feature_cols], args.if_contamination, args.random_state
    )

    print("[3/8] Analyze top anomalies (top 10% by IF) -> reports")
    anomaly_summary, _ = analyze_anomalies(df, if_score, top_ratio=0.10)
    anomaly_summary.to_csv(rep_dir / "anomaly_feature_summary.csv",
                           encoding="utf-8-sig")
    print("  Top 10 features with largest anomaly-normal difference:")
    print(anomaly_summary.head(10).to_string())

    print("[4/8] Compute expanded rule-based score")
    rule_score = build_rule_score(df).values
    rule_hit_summary = summarize_rule_hits(df)
    rule_hit_summary.to_csv(rep_dir / "rule_hit_summary.csv",
                            index=False, encoding="utf-8-sig")
    print("  Top rule hit counts:")
    print(rule_hit_summary.head(12).to_string(index=False))

    print("[5/8] Combine IF score + rule score (min-max -> sum)")
    if_norm = minmax_norm(if_score)
    rule_norm = minmax_norm(rule_score)
    final_score = if_norm + rule_norm

    print("[5.5/8] Analyze score distribution -> CSV")
    score_df = pd.DataFrame({
        "if_score": if_score,
        "rule_score": rule_score,
        "final_score": final_score,
    })

    score_df_sorted = score_df.sort_values("final_score", ascending=False)
    score_df_sorted.to_csv(rep_dir / "score_distribution_sorted.csv",
                           index=False, encoding="utf-8-sig")

    percentiles = np.linspace(0, 1, 21)
    quantiles = np.quantile(final_score, percentiles)
    percentile_df = pd.DataFrame({
        "percentile": percentiles,
        "score_threshold": quantiles,
    })
    percentile_df.to_csv(rep_dir / "score_percentiles.csv",
                         index=False, encoding="utf-8-sig")
    print(percentile_df)

    print(f"[6/8] Label top {args.spam_ratio * 100:.1f}% as spam")
    threshold = np.quantile(final_score, 1 - args.spam_ratio)
    label = (final_score >= threshold).astype(int)
    manual_spam_mask = load_manual_spam_mask(
        args.manual_spam_text_input,
        n_rows=len(df),
        regex=args.manual_spam_regex,
    )
    manual_spam_count = 0
    if manual_spam_mask is not None:
        manual_spam_count = int(manual_spam_mask.sum())
        before_override = int(label.sum())
        label[manual_spam_mask] = 1
        print(
            f"  -> manual spam override: {manual_spam_count} rows "
            f"(spam before={before_override}, after={int(label.sum())})"
        )
    print(f"  -> threshold={threshold:.4f}, "
          f"#spam={label.sum()} / {len(label)} ({label.mean() * 100:.2f}%)")

    df_labeled = df.copy()
    df_labeled["if_score"] = if_score
    df_labeled["rule_score"] = rule_score
    df_labeled["final_score"] = final_score
    df_labeled["label"] = label
    df_labeled.to_csv(out_dir / "data_labeled.csv",
                      index=False, encoding="utf-8-sig")
    print(f"  -> saved: {out_dir / 'data_labeled.csv'}")

    print(f"[7/8] ANOVA F-test, keep top {args.anova_k} features")
    X = df[feature_cols].values
    y = label

    k = min(args.anova_k, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    f_scores = selector.scores_
    p_values = selector.pvalues_
    anova_df = (
        pd.DataFrame({"feature": feature_cols, "f_score": f_scores, "p_value": p_values})
        .sort_values("f_score", ascending=False)
        .reset_index(drop=True)
    )
    anova_df.to_csv(rep_dir / "anova_feature_ranking.csv",
                    index=False, encoding="utf-8-sig")
    selected_features = anova_df.head(k)["feature"].tolist()
    print("  -> selected:")
    for feature in selected_features:
        print(f"     - {feature.encode('ascii', errors='backslashreplace').decode('ascii')}")

    train_size = 1 - args.test_size - args.valid_size
    print(f"[8/8] Split stratified train/valid/test "
          f"({train_size:.0%}/{args.valid_size:.0%}/{args.test_size:.0%})")
    X_sel = df[selected_features].copy()
    X_sel["label"] = label

    df_trainval, df_test = train_test_split(
        X_sel,
        test_size=args.test_size,
        stratify=X_sel["label"],
        random_state=args.random_state,
    )
    relative_valid = args.valid_size / (1 - args.test_size)
    df_train, df_valid = train_test_split(
        df_trainval,
        test_size=relative_valid,
        stratify=df_trainval["label"],
        random_state=args.random_state,
    )

    df_train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8-sig")
    df_valid.to_csv(out_dir / "valid.csv", index=False, encoding="utf-8-sig")
    df_test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8-sig")

    info = {
        "n_total": int(len(df)),
        "n_features_in": len(feature_cols),
        "n_features_selected": k,
        "selected_features": selected_features,
        "spam_ratio": args.spam_ratio,
        "label_threshold": float(threshold),
        "n_spam": int(label.sum()),
        "n_normal": int((1 - label).sum()),
        "split": {
            "train": {"n": int(len(df_train)),
                      "n_spam": int(df_train["label"].sum())},
            "valid": {"n": int(len(df_valid)),
                      "n_spam": int(df_valid["label"].sum())},
            "test": {"n": int(len(df_test)),
                     "n_spam": int(df_test["label"].sum())},
        },
        "rule_score": {
            "n_rules": len(RULE_WEIGHTS),
            "report": str(rep_dir / "rule_hit_summary.csv"),
        },
        "manual_spam_override": {
            "text_input": args.manual_spam_text_input,
            "n_rows": manual_spam_count,
        },
        "isolation_forest": {
            "n_estimators": 300,
            "contamination": args.if_contamination,
        },
        "random_state": args.random_state,
    }
    with open(rep_dir / "pipeline_summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n=== DONE ===")
    print(f"train: {out_dir / 'train.csv'}  ({len(df_train)} rows)")
    print(f"valid: {out_dir / 'valid.csv'}  ({len(df_valid)} rows)")
    print(f"test : {out_dir / 'test.csv'}   ({len(df_test)} rows)")
    print(f"summary: {rep_dir / 'pipeline_summary.json'}")
    print("\nSuggested model training:")
    print("  py src/train_XGBoots.py --data-dir data/processed --output-dir models")


if __name__ == "__main__":
    main()
