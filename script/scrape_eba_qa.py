"""
EBA Single Rulebook Q&A Scraper
================================
Scrapes all Q&As from https://www.eba.europa.eu/single-rule-book-qa/all

Usage:
    python script/scrape_eba_qa.py
    python script/scrape_eba_qa.py --output output/eba_qa_web.json
    python script/scrape_eba_qa.py --status final   # only Final Q&As (default: all)
    python script/scrape_eba_qa.py --concurrency 5  # parallel requests (default: 5)

Output JSON fields per entry:
    id, status, legal_act, topic, subject_matter,
    question, background, final_answer, published_date, answer_source, url
"""

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

BASE_URL = "https://www.eba.europa.eu"
ALL_QA_URL = BASE_URL + "/single-rule-book-qa/all"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; EBA-QA-Scraper/1.0; research purposes)"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_text(tag) -> str:
    """Extract clean text from a BS4 tag, stripping HTML."""
    if tag is None:
        return ""
    return tag.get_text(separator="\n", strip=True)


def parse_listing_page(html: str) -> list[str]:
    """Return list of Q&A relative URLs from a listing page."""
    soup = BeautifulSoup(html, "lxml")
    links = soup.find_all("a", href=re.compile(r"/single-rule-book-qa/qna/view/publicId/"))
    seen = set()
    urls = []
    for a in links:
        href = a["href"]
        if href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def get_last_page(html: str) -> int:
    """Parse the last page number from the pager."""
    soup = BeautifulSoup(html, "lxml")
    last = soup.find("a", title="Go to last page")
    if last and last.get("href"):
        m = re.search(r"page=(\d+)", last["href"])
        if m:
            return int(m.group(1))
    # fallback: find max page= in pager
    pages = re.findall(r"page=(\d+)", str(soup.find(class_="pager__items") or ""))
    return max(int(p) for p in pages) if pages else 0


def parse_qa_page(html: str, url: str) -> dict | None:
    """Parse a single Q&A detail page into a dict."""
    soup = BeautifulSoup(html, "lxml")

    def field(name: str) -> str:
        tag = soup.find(class_=f"field--name-{name}")
        return parse_text(tag) if tag else ""

    def meta_after(label: str) -> str:
        dt = soup.find("dt", string=re.compile(rf"^\s*{re.escape(label)}\s*$"))
        if dt:
            dd = dt.find_next_sibling("dd")
            if dd:
                return parse_text(dd)
        return ""

    qa_id = field("qa-question-id")
    if not qa_id:
        return None

    return {
        "id": qa_id,
        "status": field("qa-status"),
        "legal_act": meta_after("Legal act"),
        "topic": meta_after("Topic"),
        "subject_matter": field("qa-subject-matter"),
        "question": field("qa-question"),
        "background": field("qa-question-background"),
        "final_answer": field("qa-final-answer"),
        "published_date": field("qa-final-publishing-date"),
        "answer_source": field("qa-answer-source"),
        "url": url,
    }


async def fetch(session: aiohttp.ClientSession, url: str, semaphore: asyncio.Semaphore, retries: int = 3) -> str | None:
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    log.warning("HTTP %d for %s", resp.status, url)
                    if resp.status in (429, 503):
                        await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                log.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return None


async def collect_qa_urls(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, status_filter: str) -> list[str]:
    """Fetch all listing pages and return unique Q&A URLs."""
    listing_base = ALL_QA_URL
    if status_filter == "final":
        listing_base = BASE_URL + "/single-rule-book-qa/search"
    elif status_filter == "review":
        listing_base = BASE_URL + "/single-rule-book-qa/under-review"
    elif status_filter == "rejected":
        listing_base = BASE_URL + "/single-rule-book-qa/rejected"
    elif status_filter == "archive":
        listing_base = BASE_URL + "/single-rule-book-qa/archive"

    log.info("Fetching first listing page: %s", listing_base)
    first_html = await fetch(session, listing_base, semaphore)
    if not first_html:
        log.error("Failed to fetch listing page")
        return []

    last_page = get_last_page(first_html)
    log.info("Total listing pages: %d", last_page + 1)

    all_urls = parse_listing_page(first_html)

    # Fetch remaining listing pages concurrently
    page_urls = [f"{listing_base}?page={p}" for p in range(1, last_page + 1)]
    tasks = [fetch(session, u, semaphore) for u in page_urls]

    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        html = await coro
        if html:
            all_urls.extend(parse_listing_page(html))
        if i % 20 == 0:
            log.info("Listing pages fetched: %d/%d", i, len(page_urls))

    # Deduplicate preserving order
    seen = set()
    unique = []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    log.info("Collected %d unique Q&A URLs", len(unique))
    return unique


async def scrape_all(args) -> list[dict]:
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency, ssl=False)

    async with aiohttp.ClientSession(connector=connector) as session:
        qa_urls = await collect_qa_urls(session, semaphore, args.status)
        if not qa_urls:
            return []

        log.info("Fetching %d Q&A detail pages...", len(qa_urls))
        results = []
        failed = []

        full_urls = [BASE_URL + u if u.startswith("/") else u for u in qa_urls]

        async def fetch_and_parse(url: str) -> dict | None:
            html = await fetch(session, url, semaphore)
            if html is None:
                failed.append(url)
                return None
            return parse_qa_page(html, url)

        batch_size = 200
        for batch_start in range(0, len(full_urls), batch_size):
            batch = full_urls[batch_start: batch_start + batch_size]
            batch_results = await asyncio.gather(*[fetch_and_parse(u) for u in batch])
            for r in batch_results:
                if r:
                    results.append(r)
            log.info(
                "Progress: %d/%d done, %d collected, %d failed",
                min(batch_start + batch_size, len(full_urls)),
                len(full_urls),
                len(results),
                len(failed),
            )

    if failed:
        log.warning("%d pages failed to fetch", len(failed))
        for u in failed[:10]:
            log.warning("  Failed: %s", u)

    return results


def main():
    parser = argparse.ArgumentParser(description="Scrape EBA Single Rulebook Q&As from the web.")
    parser.add_argument("--output", default="output/eba_qa_web.json", help="Output JSON file")
    parser.add_argument(
        "--status",
        choices=["all", "final", "review", "rejected", "archive"],
        default="final",
        help="Which Q&A category to scrape (default: final)",
    )
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent requests (default: 5)")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Scraping EBA Q&As (status=%s, concurrency=%d)", args.status, args.concurrency)
    start = time.time()

    results = asyncio.run(scrape_all(args))

    log.info("Scraped %d Q&A entries in %.1fs", len(results), time.time() - start)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()
