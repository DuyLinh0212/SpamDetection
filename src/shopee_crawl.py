import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, sync_playwright


DEFAULT_URL = (
    "https://shopee.vn/Combo-Acne-Trio-ch%C4%83m-da-d%E1%BA%A7u-m%E1%BB%A5n-cho-nam-"
    "Men-Stay-Simplicity-S%E1%BB%AFa-r%E1%BB%A7a-m%E1%BA%B7t-100g-Serum-Calm-30ml-"
    "Kem-d%C6%B0%E1%BB%A1ng-%E1%BA%A9m-80g-i.118768792.11535989197"
    "?extraParams=%7B%22display_model_id%22%3A27352080358%2C%22model_selection_logic%22%3A3%7D"
)

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)


def extract_ids_from_url(url: str):
    m = re.search(r"-i\.(\d+)\.(\d+)", url)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def parse_reviews_from_html(html: str, product_id: str, seen_ids: set):
    soup = BeautifulSoup(html, "html.parser")
    rows = []

    for item in soup.select("div.q2b7Oq"):
        cmtid = item.get("data-cmtid", "")
        if cmtid and cmtid in seen_ids:
            continue

        user_el = item.select_one("a.InK5kS") or item.select_one("div.InK5kS")
        user = user_el.get_text(strip=True) if user_el else ""

        time_el = item.select_one(".XYk98l")
        comment_time = time_el.get_text(strip=True) if time_el else ""

        rating_star = len(item.select("svg.icon-rating-solid"))

        comment_el = item.select_one(".YNedDV")
        comment_text = comment_el.get_text(" ", strip=True) if comment_el else ""

        has_media = 1 if item.select_one(".rating-media-list__image-wrapper") or item.select_one("video") else 0

        helpful_el = item.select_one(".shopee-product-rating__like-count")
        helpful_text = helpful_el.get_text(strip=True) if helpful_el else "0"
        helpful_votes = int(re.sub(r"\D", "", helpful_text) or 0)

        rows.append(
            {
                "userId": user,
                "comment_time": comment_time,
                "rating_star": rating_star,
                "has_media": has_media,
                "comment_text": comment_text,
                "productID": product_id,
                "helpful_votes": helpful_votes,
            }
        )
        if cmtid:
            seen_ids.add(cmtid)

    return rows


def extract_sold_count_from_text(text: str) -> int:
    # Examples: "Đã bán 1,2k", "Đã bán 120", "Đã bán 1.2k", "Đã Bán45", "đã bán 3 nghìn"
    pattern = re.compile(r"đã bán\s*([0-9]+(?:[.,][0-9]+)?)\s*(k|nghìn)?", re.IGNORECASE)
    m = pattern.search(text or "")
    if not m:
        return 0
    num = m.group(1).replace(",", ".")
    try:
        val = float(num)
    except Exception:
        return 0
    unit = (m.group(2) or "").lower()
    if unit in ("k", "nghìn"):
        return int(val * 1000)
    return int(val)


def extract_sold_count(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    candidates = []
    # Ưu tiên các đoạn text ngắn có chứa "Đã bán" (khu vực thống kê)
    for el in soup.select("span, div, p"):
        txt = el.get_text(" ", strip=True)
        if txt and "đã bán" in txt.lower():
            candidates.append(txt)

    # Fallback: toàn bộ text trang
    candidates.append(soup.get_text(" ", strip=True))

    for text in candidates:
        val = extract_sold_count_from_text(text)
        if val:
            return val
    return 0


def extract_sold_count_from_page(page) -> int:
    # Ưu tiên lấy từ DOM đã render (text selector), fallback sang HTML
    try:
        locator = page.locator("text=/đã\\s*bán/i")
        cnt = min(locator.count(), 6)
        for i in range(cnt):
            txt = locator.nth(i).inner_text()
            val = extract_sold_count_from_text(txt)
            if val:
                return val
    except Exception:
        pass
    try:
        return extract_sold_count(page.content())
    except Exception:
        return 0


def apply_rating_filter(page, star: int, delay: float):
    # Try to click the rating filter button (e.g., "5 Sao")
    btn = page.locator(f"button:has-text('{star} Sao')").first
    if btn.count() == 0:
        # Fallback: any element containing text
        btn = page.locator(f":text('{star} Sao')").first
    if btn.count() > 0:
        first_id = page.locator("div.q2b7Oq").first.get_attribute("data-cmtid")
        try:
            btn.click()
        except Exception:
            # Fallback: force click if element is covered/animated
            btn.click(force=True)
        time.sleep(delay)
        if first_id:
            try:
                page.wait_for_function(
                    """(fid) => {
                        const el = document.querySelector('div.q2b7Oq');
                        return el && el.getAttribute('data-cmtid') !== fid;
                    }""",
                    arg=first_id,
                    timeout=15000,
                )
            except PlaywrightTimeoutError:
                # Some filters don't change the first review (or list is short)
                # so don't fail the whole run.
                print(f"Timeout khi chờ đổi danh sách sau khi lọc {star} Sao, tiếp tục...")
    else:
        print(f"Không tìm thấy nút lọc {star} Sao, dùng trạng thái hiện tại.")


def crawl_pages(page, product_id: str, max_pages: int, seen_ids: set, delay: float):
    crawled = 0
    collected = []
    for _ in range(max_pages):
        html = page.content()
        rows = parse_reviews_from_html(html, product_id, seen_ids)
        if rows:
            collected.extend(rows)
        crawled += 1

        next_btn = page.locator("nav.product-ratings__page-controller button.shopee-icon-button--right").first
        if next_btn.count() == 0:
            print("Het trang (khong co nut Next), chuyen sang link khac.")
            break
        if "disabled" in (next_btn.get_attribute("class") or ""):
            print("Het trang (Next bi disable), chuyen sang link khac.")
            break

        first_id = page.locator("div.q2b7Oq").first.get_attribute("data-cmtid")
        try:
            next_btn.click(timeout=10000)
        except PlaywrightTimeoutError:
            # Có overlay che (wAMdpk), thử dọn rồi force click
            try:
                page.keyboard.press("Escape")
            except Exception:
                pass
            try:
                page.locator("body").click(position={"x": 5, "y": 5}, timeout=2000)
            except Exception:
                pass
            try:
                next_btn.click(force=True, timeout=10000)
            except PlaywrightTimeoutError:
                print("Không bấm được nút trang tiếp theo (bị che), dừng crawl page.")
                break
        time.sleep(delay)
        if first_id:
            try:
                page.wait_for_function(
                    """(fid) => {
                        const el = document.querySelector('div.q2b7Oq');
                        return el && el.getAttribute('data-cmtid') !== fid;
                    }""",
                    arg=first_id,
                    timeout=15000,
                )
            except PlaywrightTimeoutError:
                # Nếu click mà list không đổi, coi như hết trang
                same_id = page.locator("div.q2b7Oq").first.get_attribute("data-cmtid")
                if same_id == first_id:
                    print("Timeout va list khong doi -> coi nhu het trang, chuyen link.")
                    break
                print("Timeout khi chờ chuyển trang đánh giá, tiếp tục...")
    return collected, crawled


def crawl_until_target(
    page,
    product_id: str,
    target_count: int,
    seen_ids: set,
    delay: float,
    max_pages: int,
    only_star: int | None,
    progress_prefix: str,
):
    total = 0
    crawled = 0
    collected = []
    for _ in range(max_pages):
        html = page.content()
        rows = parse_reviews_from_html(html, product_id, seen_ids)
        if only_star is not None:
            rows = [r for r in rows if r.get("rating_star") == only_star]
        if rows:
            collected.extend(rows)
            total += len(rows)
        crawled += 1
        print(f"{progress_prefix} Đã thu: {total}/{target_count} (page {crawled}/{max_pages})")
        if total >= target_count:
            break

        next_btn = page.locator("nav.product-ratings__page-controller button.shopee-icon-button--right").first
        if next_btn.count() == 0:
            break
        if "disabled" in (next_btn.get_attribute("class") or ""):
            break

        first_id = page.locator("div.q2b7Oq").first.get_attribute("data-cmtid")
        next_btn.click()
        time.sleep(delay)
        if first_id:
            try:
                page.wait_for_function(
                    """(fid) => {
                        const el = document.querySelector('div.q2b7Oq');
                        return el && el.getAttribute('data-cmtid') !== fid;
                    }""",
                    arg=first_id,
                    timeout=15000,
                )
            except PlaywrightTimeoutError:
                print("Timeout khi chờ chuyển trang đánh giá, tiếp tục...")
    return collected, crawled


def load_checkpoint(checkpoint_path: Path):
    if checkpoint_path.exists():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            return int(data.get("last_index", -1)), data.get("last_url", "")
        except Exception:
            return -1, ""
    return -1, ""


def save_checkpoint(checkpoint_path: Path, last_index: int, total: int, last_url: str):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "last_index": last_index,
        "total": total,
        "last_url": last_url,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    checkpoint_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_rows(out_path: Path, rows: list[dict]):
    if not rows:
        return 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not out_path.exists()
    df.to_csv(out_path, index=False, encoding="utf-8-sig", mode="a", header=header)
    return len(df)


def append_link(out_path: Path, url: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(url + "\n")


def page_has_captcha(page) -> bool:
    try:
        html = page.content().lower()
    except Exception:
        return False
    keywords = ["captcha", "xác minh", "verify", "bot", "human"]
    return any(k in html for k in keywords)


def main():
    parser = argparse.ArgumentParser(description="Crawl Shopee ratings to CSV (Playwright + BeautifulSoup)")
    parser.add_argument("--url", default=DEFAULT_URL, help="Product URL (single)")
    parser.add_argument("--links", default="data/raw/shopee_links.txt", help="Path to file containing product links")
    parser.add_argument("--total-pages", type=int, default=15, help="Số page đánh giá sẽ crawl mỗi sản phẩm")
    parser.add_argument("--out", default="data/raw/data_crawl.csv", help="Output CSV (combined)")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--skip-login", action="store_true", help="Skip login check")
    parser.add_argument("--delay-between-pages", type=float, default=5.0, help="Delay between rating pages")
    parser.add_argument("--delay-load-page", type=float, default=7.0, help="Chờ page load xong trước khi crawl")
    parser.add_argument("--delay-between-links", type=float, default=5.0, help="Delay between product links")
    parser.add_argument("--delay-jitter", type=float, default=2.0, help="Jitter ng?u nhi?n th?m (gi?y)")
    parser.add_argument("--checkpoint", default="data/raw/shopee_checkpoint.json", help="Checkpoint file")
    parser.add_argument("--no-resume", action="store_true", help="Khong resume tu checkpoint")
    parser.add_argument("--pause-on-captcha", action="store_true", help="Pause neu gap captcha")
    parser.add_argument(
        "--min-sold",
        type=int,
        default=200,
        help="Bo qua san pham da ban < N (0 de tat)",
    )
    parser.add_argument("--filter-links-min-sold", type=int, help="Loc link theo so luong da ban >= N")
    parser.add_argument(
        "--links-out",
        default="data/raw/shopee_links_filtered.txt",
        help="Output file cho links da loc",
    )
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
    args = parser.parse_args()

    links_path = Path(args.links)
    if links_path.exists():
        urls = [line.strip() for line in links_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        urls = [args.url]

    checkpoint_path = Path(args.checkpoint)
    start_index = 0
    if not args.no_resume:
        last_index, last_url = load_checkpoint(checkpoint_path)
        if last_index >= 0:
            start_index = last_index + 1
            print(f"Resume tu checkpoint: {start_index}/{len(urls)} (last: {last_url})")

    if args.filter_links_min_sold is not None:
        links_out = Path(args.links_out)
        if not args.no_resume and links_out.exists() and start_index == 0:
            # tránh ghi chồng khi resume
            print(f"Output links: {links_out}")

    rows_written = 0

    print("Lưu ý: hãy đóng toàn bộ Cốc Cốc trước khi chạy để Playwright dùng đúng profile đã login.")

    with sync_playwright() as p:
        try:
            context = p.chromium.launch_persistent_context(
                user_data_dir=args.coccoc_user_data,
                executable_path=args.coccoc_path,
                headless=args.headless,
                user_agent=USER_AGENT,
                args=[f"--profile-directory={args.coccoc_profile}"],
            )
        except Exception as exc:
            raise RuntimeError(
                "Không mở được Cốc Cốc bằng profile hiện tại. "
                "Hãy đóng hết Cốc Cốc hoặc thử profile khác (Profile 1, Profile 2...)."
            ) from exc

        print(f"Tổng số link: {len(urls)}")
        for idx, url in enumerate(urls):
            if idx < start_index:
                continue
            shopid, itemid = extract_ids_from_url(url)
            if not shopid or not itemid:
                print(f"Skip (không tìm thấy shopid/itemid): {url}")
                continue

            product_id = f"{shopid}.{itemid}"

            page = context.new_page()
            def ensure_page_loaded(retries: int = 3):
                for attempt in range(1, retries + 1):
                    page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    time.sleep(args.delay_load_page + random.uniform(0, args.delay_jitter))
                    try:
                        page.wait_for_selector(
                            "div.product-briefing, div.product-ratings__list, h1",
                            timeout=30000,
                        )
                        return True
                    except Exception:
                        print(f"[{product_id}] Trang trắng/không tải đủ, thử lại {attempt}/{retries}...")
                        try:
                            page.reload(wait_until="domcontentloaded", timeout=60000)
                        except Exception:
                            pass
                return False

            if not ensure_page_loaded():
                print(f"Skip (không load được trang): {url}")
                page.close()
                continue

            if args.filter_links_min_sold is None and args.min_sold and args.min_sold > 0:
                sold = extract_sold_count_from_page(page)
                if sold < args.min_sold:
                    print(f"[{product_id}] Skip (da ban {sold} < {args.min_sold})")
                    save_checkpoint(checkpoint_path, idx, len(urls), url)
                    page.close()
                    time.sleep(args.delay_between_links + random.uniform(0, args.delay_jitter))
                    continue

            if args.filter_links_min_sold is not None:
                sold = extract_sold_count_from_page(page)
                if sold >= args.filter_links_min_sold:
                    append_link(Path(args.links_out), url)
                    print(f"[{product_id}] Keep (da ban {sold})")
                else:
                    print(f"[{product_id}] Drop (da ban {sold})")
                save_checkpoint(checkpoint_path, idx, len(urls), url)
                page.close()
                time.sleep(args.delay_between_links + random.uniform(0, args.delay_jitter))
                continue

            if args.pause_on_captcha and page_has_captcha(page):
                print(f"[{product_id}] Phat hien captcha. Hay xu ly trong Coc Coc roi bam Enter de tiep tuc...")
                save_checkpoint(checkpoint_path, idx - 1, len(urls), url)
                input()
                if not ensure_page_loaded():
                    print(f"Skip (không load được trang sau captcha): {url}")
                    page.close()
                    continue

            if not args.skip_login and "buyer/login" in page.url:
                print("Shopee yêu cầu đăng nhập. Vui lòng đăng nhập trên cửa sổ Cốc Cốc, rồi quay lại đây bấm Enter.")
                input()
                if not ensure_page_loaded():
                    print(f"Skip (không load được trang sau login): {url}")
                    page.close()
                    continue

            # Scroll nh? ?? gi? h?nh vi ng??i d?ng
            try:
                page.mouse.wheel(0, 600)
                time.sleep(1.0 + random.uniform(0, args.delay_jitter))
                page.mouse.wheel(0, -300)
            except Exception:
                pass

            try:
                page.locator("div.product-ratings, div.product-ratings__list").first.scroll_into_view_if_needed(timeout=25000)
                page.wait_for_selector("div.product-ratings__list", timeout=40000)
            except Exception:
                debug_dir = Path("reports")
                debug_dir.mkdir(parents=True, exist_ok=True)
                (debug_dir / "shopee_debug.html").write_text(page.content(), encoding="utf-8")
                try:
                    page.screenshot(path=str(debug_dir / "shopee_debug.png"), full_page=True)
                except Exception:
                    pass
                print(f"Skip (không thấy đánh giá): {url}")
                page.close()
                continue

            seen_ids = set()
            max_pages = args.total_pages

            # Lọc bình thường, không phân biệt sao: crawl N pages
            new_rows, crawled = crawl_pages(
                page,
                product_id,
                max_pages=max_pages,
                seen_ids=seen_ids,
                delay=args.delay_between_pages + random.uniform(0, args.delay_jitter),
            )
            wrote = append_rows(Path(args.out), new_rows)
            rows_written += wrote
            print(f"[{product_id}] Da them {wrote} dong (crawl {crawled} pages). Tong: {rows_written}")
            save_checkpoint(checkpoint_path, idx, len(urls), url)
            page.close()
            time.sleep(args.delay_between_links + random.uniform(0, args.delay_jitter))

        context.close()

    out_path = Path(args.out)
    print(f"Saved: {out_path} (da ghi {rows_written} rows)")


if __name__ == "__main__":
    main()
