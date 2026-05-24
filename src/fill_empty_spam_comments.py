import argparse
import random
from pathlib import Path

import pandas as pd


SYNTHETIC_SPAM_REGEX = (
    r"example\.com|bit\.ly/fake-sale|SPAM\d+|DEAL\d+|"
    r"(?:kiem tien online|nhan hoa hong|viec nhe luong cao|dang ky nhan qua|"
    r"trung thuong 100%|hoan tien cuc lon) tai nha, khong can von|"
    r"(?:kiếm tiền online|nhận hoa hồng|việc nhẹ lương cao|đăng ký nhận quà|"
    r"trúng thưởng 100%|hoàn tiền cực lớn) tại nhà, không cần vốn|"
    r"Ai can .* gia re .* Zalo 0|Ai cần .* giá rẻ .* Zalo 0|"
    r"Can cong tac vien .* hoa hong cao|Cần cộng tác viên .* hoa hồng cao|"
    r"Kho moi ve .* sieu re sieu dep|Kho mới về .* siêu rẻ siêu đẹp|"
    r"gia huy diet|giá hủy diệt|"
    r"Mua hang nhan qua mien phi|Mua hàng nhận quà miễn phí|"
    r"Ban da duoc chon nhan qua|Bạn đã được chọn nhận quà|"
    r"ORDER NGAY .* GIA SI|ORDER NGAY .* GIÁ SỈ|"
    r"Khuyen mai khuyen mai|Khuyến mãi khuyến mãi|"
    r"Ho tro vay mua hang|Hỗ trợ vay mua hàng|"
    r"Nhom san ma giam gia|Nhóm săn mã giảm giá"
)


def build_spam_comments(n_comments: int, seed: int) -> list[str]:
    random.seed(seed)

    products = [
        "ốp lưng", "tai nghe", "đồng hồ", "mỹ phẩm", "giày sneaker",
        "áo thun", "đèn led", "sạc nhanh", "nước hoa", "kem dưỡng",
    ]
    promos = [
        "SALE SỐC", "GIẢM GIÁ 70%", "FREESHIP TOÀN QUỐC", "MUA 1 TẶNG 1",
        "FLASH SALE", "XẢ KHO", "ƯU ĐÃI ĐỘC QUYỀN",
    ]
    ctas = [
        "click ngay", "ib shop", "nhắn Zalo", "inbox liền", "đặt hàng ngay",
        "vào link nhận mã", "liên hệ ngay",
    ]
    links = [
        "https://example.com/deal", "www.example.com/voucher",
        "bit.ly/fake-sale", "https://example.com/freeship",
        "www.example.com/flash",
    ]
    phones = ["0901234567", "0912345678", "0987654321", "0933333333", "0876543210", "0899999999"]
    money = [
        "kiếm tiền online", "nhận hoa hồng", "việc nhẹ lương cao",
        "đăng ký nhận quà", "trúng thưởng 100%", "hoàn tiền cực lớn",
    ]
    shipping = [
        "giao trong 2h", "ship COD toàn quốc", "miễn phí đổi trả",
        "cam kết chính hãng", "bảo hành 12 tháng",
    ]
    noise = ["!!!", "*** HOT ***", "NHANH TAY", "SỐ LƯỢNG CÓ HẠN", "DEAL HOT"]

    patterns = [
        lambda: f"{random.choice(promos)} {random.choice(products)} chỉ hôm nay "
                f"{random.choice(ctas)} {random.choice(links)} {random.choice(noise)}",
        lambda: f"Ai cần {random.choice(products)} giá rẻ thì {random.choice(ctas)}, "
                f"Zalo {random.choice(phones)}, {random.choice(shipping)}!",
        lambda: f"Mã giảm giá {random.randint(100, 999)}K, {random.choice(promos)}, "
                f"nhận tại {random.choice(links)}",
        lambda: f"{random.choice(money)} tại nhà, không cần vốn, liên hệ "
                f"{random.choice(phones)} để được hướng dẫn ngay!!!",
        lambda: f"Kho mới về {random.choice(products)} siêu rẻ siêu đẹp, "
                f"{random.choice(promos)}, {random.choice(ctas)} kẻo hết hàng",
        lambda: f"TẶNG VOUCHER {random.randint(50, 500)}K cho khách mới, vào "
                f"{random.choice(links)} nhập mã SPAM{random.randint(1000, 9999)}",
        lambda: f"Cần cộng tác viên bán {random.choice(products)}, hoa hồng cao, "
                f"ib ngay {random.choice(phones)} *** HOT ***",
        lambda: f"Mua hàng nhận quà miễn phí miễn phí miễn phí, {random.choice(ctas)}, "
                f"link: {random.choice(links)}",
        lambda: f"{random.choice(products)} chính hãng giá hủy diệt, số lượng có hạn, "
                f"gọi {random.choice(phones)} hoặc {random.choice(ctas)}",
        lambda: f"Đừng bỏ lỡ {random.choice(promos)} {random.choice(products)} "
                f"{random.choice(shipping)} {random.choice(noise)} {random.choice(links)}",
        lambda: f"Bạn đã được chọn nhận quà, xác nhận thông tin tại "
                f"{random.choice(links)} hoặc gọi {random.choice(phones)}",
        lambda: f"ORDER NGAY {random.choice(products).upper()} GIÁ SỈ, IB NHANH, "
                f"DEAL HOT {random.randint(10, 99)}K !!!",
        lambda: f"Khuyến mãi khuyến mãi khuyến mãi, mua ngay mua ngay mua ngay, "
                f"{random.choice(links)}",
        lambda: f"Hỗ trợ vay mua hàng trả góp 0%, duyệt nhanh trong 5 phút, "
                f"liên hệ {random.choice(phones)}",
        lambda: f"Nhóm săn mã giảm giá mới mở, tham gia tại {random.choice(links)}, "
                f"số lượng slot giới hạn!!!",
    ]

    comments = []
    for i in range(n_comments):
        comment = patterns[i % len(patterns)]()
        if i % 7 == 0:
            comment += f" MÃ DEAL{random.randint(10000, 99999)}"
        if i % 11 == 0:
            comment += " !!!! nhanh tay nhanh tay"
        if i % 17 == 0:
            comment += f" {random.choice(products)} {random.choice(products)} {random.choice(products)}"
        comments.append(comment)

    random.shuffle(comments)
    return comments


def main():
    parser = argparse.ArgumentParser(description="Fill empty Shopee comments with synthetic spam comments.")
    parser.add_argument("--input", default="data/raw/data_crawl.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--replace-synthetic",
        action="store_true",
        help="Replace existing synthetic spam comments generated by this script.",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    df = pd.read_csv(in_path)
    text = df["comment_text"].fillna("").astype(str)
    empty = text.str.strip().eq("")
    synthetic = text.str.contains(SYNTHETIC_SPAM_REGEX, case=False, regex=True, na=False)
    target = empty | synthetic if args.replace_synthetic else empty
    target_idx = df.index[target].tolist()

    comments = build_spam_comments(len(target_idx), args.seed)
    df.loc[target_idx, "comment_text"] = comments

    remaining = int((df["comment_text"].isna() | df["comment_text"].astype(str).str.strip().eq("")).sum())
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"filled_or_replaced_rows={len(target_idx)}")
    print(f"remaining_empty_rows={remaining}")
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
