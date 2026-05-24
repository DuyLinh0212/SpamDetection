import argparse
import re
import time
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def normalize_product_url(url: str) -> str:
    # Keep canonical product URL without query params
    m = re.search(r"(https?://shopee\.vn/[^?]+-i\.\d+\.\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"(https?://shopee\.vn/[^?]+/i\.\d+\.\d+)", url)
    if m:
        return m.group(1)
    return ""


def extract_links_from_page(page) -> list[str]:
    links = page.eval_on_selector_all(
        "a[href*='-i.'], a[href*='/i.']",
        "els => els.map(e => e.href)",
    )
    cleaned = []
    for href in links:
        norm = normalize_product_url(href)
        if norm:
            cleaned.append(norm)
    return cleaned


def main():
    parser = argparse.ArgumentParser(description="Collect Shopee product links by keyword search")
    parser.add_argument("--keywords", nargs="*", help="List of keywords")
    parser.add_argument("--keywords-file", default="data/raw/shopee_keywords.txt", help="File with keywords")
    parser.add_argument("--per-keyword", type=int, default=15, help="Links per keyword")
    parser.add_argument("--delay-seconds", type=int, default=10, help="Delay between keywords to reduce bot detection")
    parser.add_argument("--delay-per-link", type=float, default=1.0, help="Delay after collecting each new product link")
    parser.add_argument("--delay-per-scroll", type=float, default=0.8, help="Delay after each scroll")
    parser.add_argument("--out", default="data/raw/shopee_links.txt", help="Output links file")
    parser.add_argument(
        "--coccoc-path",
        default=r"C:\Program Files\CocCoc\Browser\Application\browser.exe",
        help="Path to CocCoc browser.exe",
    )
    parser.add_argument(
        "--coccoc-user-data",
        default=r"C:\Users\duytn\AppData\Local\CocCoc\Browser\User Data",
        help="Coc Coc User Data folder",
    )
    parser.add_argument(
        "--coccoc-profile",
        default="Profile 2",
        help="Coc Coc profile directory name (e.g. Default, Profile 1)",
    )
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    args = parser.parse_args()

    keywords = []
    if args.keywords:
        keywords.extend([k.strip() for k in args.keywords if k.strip()])
    else:
        kpath = Path(args.keywords_file)
        if kpath.exists():
            keywords.extend([k.strip() for k in kpath.read_text(encoding="utf-8").splitlines() if k.strip()])

    if not keywords:
        raise SystemExit("Không có keyword. Truyền --keywords hoặc tạo data/raw/shopee_keywords.txt")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    collected = []
    seen = set(out_path.read_text(encoding="utf-8").splitlines()) if out_path.exists() else set()

    def flush_links():
        if seen:
            out_path.write_text("\n".join(sorted(seen)), encoding="utf-8")

    print("Lưu ý: hãy đóng toàn bộ Cốc Cốc trước khi chạy để Playwright dùng đúng profile đã login.")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=args.coccoc_user_data,
            executable_path=args.coccoc_path,
            headless=args.headless,
            user_agent=USER_AGENT,
            args=[f"--profile-directory={args.coccoc_profile}"],
        )
        page = context.new_page()

        try:
            for kw in keywords:
                target = f"https://shopee.vn/search?keyword={quote(kw)}"
                page.goto(target, wait_until="domcontentloaded", timeout=60000)
                time.sleep(1.0)

                kw_links = []
                tries = 0
                while len(kw_links) < args.per_keyword and tries < 30:
                    links = extract_links_from_page(page)
                    for link in links:
                        if link not in seen and link not in kw_links:
                            kw_links.append(link)
                            time.sleep(args.delay_per_link)
                    page.mouse.wheel(0, 1400)
                    time.sleep(args.delay_per_scroll)
                    tries += 1

                print(f"{kw}: {len(kw_links)} links")
                for link in kw_links:
                    if link not in seen:
                        collected.append(link)
                        seen.add(link)
                flush_links()
                time.sleep(args.delay_seconds)
        finally:
            context.close()

    if collected:
        flush_links()
        print(f"Saved: {out_path} (+{len(collected)} new)")
    else:
        print("Không có link mới.")


if __name__ == "__main__":
    main()
