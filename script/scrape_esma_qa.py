"""
ESMA Q&A Scraper
=================
Scrapes Q&As from https://www.esma.europa.eu/esma-qa-search-page/all

Usage:
    python script/scrape_esma_qa.py
    python script/scrape_esma_qa.py --output output/esma_qa_web.json
    python script/scrape_esma_qa.py --status final     # Final Q&As only (default: all)
    python script/scrape_esma_qa.py --concurrency 5    # parallel requests (default: 5)

Output JSON fields per entry:
    id, status, level1_regulation, topic, subject_matter,
    question, answer, response_date, published_date, url

Requirements:
    pip install aiohttp beautifulsoup4 lxml
"""

import argparse
import asyncio
import json
import logging
import re
import time
from pathlib import Path

import aiohttp
from bs4 import BeautifulSoup

BASE_URL = "https://www.esma.europa.eu"

STATUS_PATHS = {
    "all":      "/esma-qa-search-page/all",
    "final":    "/esma-qa-search-page/final",
    "rejected": "/esma-qa-search-page/rejected",
    "waiting":  "/esma-qa-search-page/waiting",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ESMA-QA-Scraper/1.0; research purposes)"
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


def field_text(soup: BeautifulSoup, css_class: str) -> str:
    tag = soup.find(class_=css_class)
    return tag.get_text(separator="\n", strip=True) if tag else ""


def parse_listing_page(html: str) -> list[str]:
    """Return list of relative Q&A URLs from a listing page."""
    soup = BeautifulSoup(html, "lxml")
    seen, urls = set(), []
    for a in soup.find_all("a", class_="question__header--number"):
        href = a.get("href", "")
        if href and href not in seen:
            seen.add(href)
            urls.append(href)
    return urls


def get_last_page(html: str) -> int:
    """Parse the last page index from the Drupal pager."""
    soup = BeautifulSoup(html, "lxml")
    last = soup.find("a", title=re.compile(r"last page", re.I))
    if last and last.get("href"):
        m = re.search(r"page=(\d+)", last["href"])
        if m:
            return int(m.group(1))
    pages = re.findall(r"page=(\d+)", str(soup.find(class_="pager__items") or ""))
    return max(int(p) for p in pages) if pages else 0


def parse_qa_page(html: str, url: str, status_label: str) -> dict | None:
    """Parse a single ESMA Q&A detail page into a dict."""
    soup = BeautifulSoup(html, "lxml")

    qa_id = field_text(soup, "field--name-title")
    if not qa_id or not qa_id.startswith("ESMA_QA"):
        return None

    # Strip the label prefix added by Drupal ("Subject MatterXxx" → "Xxx")
    def strip_label(css_class: str, label: str) -> str:
        text = field_text(soup, css_class)
        return text.removeprefix(label).strip()

    subject_matter = strip_label("field--name-field-qa-subject-matter", "Subject Matter")
    level1_regulation = strip_label("field--name-field-qa-level1", "Level 1 Regulation")
    topic = strip_label("field--name-field-lqa-ist-of-topics", "Topic")
    question = field_text(soup, "field--name-field-qa-question")
    published_date = field_text(soup, "question__header--date")

    # Answer — take the most recent block (last one in the list)
    answer_blocks = soup.find_all(class_="field--name-field-qa-esma-response")
    answer = answer_blocks[-1].get_text(separator="\n", strip=True) if answer_blocks else ""

    date_blocks = soup.find_all(class_="field--name-field-qa-esma-response-date")
    response_date = date_blocks[-1].get_text(strip=True) if date_blocks else ""

    return {
        "id": qa_id,
        "status": status_label,
        "level1_regulation": level1_regulation,
        "topic": topic,
        "subject_matter": subject_matter,
        "question": question,
        "answer": answer,
        "response_date": response_date,
        "published_date": published_date,
        "url": url,
    }


async def fetch(
    session: aiohttp.ClientSession,
    url: str,
    semaphore: asyncio.Semaphore,
    retries: int = 3,
) -> str | None:
    async with semaphore:
        for attempt in range(retries):
            try:
                async with session.get(
                    url, headers=HEADERS, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    log.warning("HTTP %d for %s", resp.status, url)
                    if resp.status in (429, 503):
                        await asyncio.sleep(5 * (attempt + 1))
                    elif resp.status >= 500:
                        await asyncio.sleep(2 ** attempt)
            except Exception as e:
                log.warning("Attempt %d failed for %s: %s", attempt + 1, url, e)
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
        return None


async def collect_qa_urls(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    status: str,
) -> list[str]:
    listing_base = BASE_URL + STATUS_PATHS[status]

    log.info("Fetching first listing page: %s", listing_base)
    first_html = await fetch(session, listing_base, semaphore)
    if not first_html:
        log.error("Failed to fetch listing page")
        return []

    last_page = get_last_page(first_html)
    log.info("Total listing pages: %d", last_page + 1)

    all_urls = parse_listing_page(first_html)

    page_urls = [f"{listing_base}?page={p}" for p in range(1, last_page + 1)]
    tasks = [fetch(session, u, semaphore) for u in page_urls]

    done = 0
    for coro in asyncio.as_completed(tasks):
        html = await coro
        done += 1
        if html:
            all_urls.extend(parse_listing_page(html))
        if done % 50 == 0:
            log.info("Listing pages fetched: %d/%d", done, len(page_urls))

    seen, unique = set(), []
    for u in all_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    log.info("Collected %d unique Q&A URLs", len(unique))
    return unique


async def scrape_all(args) -> list[dict]:
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency, ssl=False)

    status_labels = {
        "all":      "All",
        "final":    "Final",
        "rejected": "Rejected",
        "waiting":  "Waiting",
    }
    status_label = status_labels[args.status]

    async with aiohttp.ClientSession(connector=connector) as session:
        qa_urls = await collect_qa_urls(session, semaphore, args.status)
        if not qa_urls:
            return []

        log.info("Fetching %d Q&A detail pages...", len(qa_urls))
        results, failed = [], []

        full_urls = [BASE_URL + u if u.startswith("/") else u for u in qa_urls]

        async def fetch_and_parse(url: str) -> dict | None:
            html = await fetch(session, url, semaphore)
            if html is None:
                failed.append(url)
                return None
            return parse_qa_page(html, url, status_label)

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
    parser = argparse.ArgumentParser(description="Scrape ESMA Q&As from the web.")
    parser.add_argument("--output", default="output/esma_qa_web.json", help="Output JSON file")
    parser.add_argument(
        "--status",
        choices=list(STATUS_PATHS.keys()),
        default="final",
        help="Which Q&A category to scrape (default: final)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Max concurrent requests (default: 5)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Scraping ESMA Q&As (status=%s, concurrency=%d)", args.status, args.concurrency)
    start = time.time()

    results = asyncio.run(scrape_all(args))

    log.info("Scraped %d Q&A entries in %.1fs", len(results), time.time() - start)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    log.info("Saved to: %s", output_path)


if __name__ == "__main__":
    main()
