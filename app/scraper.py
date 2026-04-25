"""
Scraper for Maryland housing & rental law sources.

Sources (all verified live):
  1. Maryland AG — landlord/tenant disputes guide
  2. Maryland DHCD — state Office of Tenant and Landlord Affairs
  3. Montgomery County DHCA — county-level handbook & rights pages
  4. Baltimore County — Circuit Court Law Library + DHCD renter resources
  5. Baltimore City DHCD — renter resources
  6. Prince George's County DHCD — tenant resources, rent stabilization
  7. People's Law Library (community legal resource)
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from app.config import settings


SEEDS: dict[str, list[str]] = {
    "md_ag": [
        "https://oag.maryland.gov/i-need-to/Pages/landlord-tenant-disputes.aspx",
    ],
    "md_dhcd": [
        "https://dhcd.maryland.gov/Tenant-Landlord-Affairs/pages/default.aspx",
    ],
    "montgomery_dhca": [
        "https://www.montgomerycountymd.gov/dhca/housing/landlordtenant/",
        "https://www.montgomerycountymd.gov/DHCA/housing/landlordtenant/tenant_rights_responsibilities.html",
        "https://www.montgomerycountymd.gov/DHCA/housing/landlordtenant/handbook.html",
    ],
    "baltimore_county": [
        "https://www.baltimorecountymd.gov/departments/circuit/library/landlord-tenant",
        "https://www.baltimorecountymd.gov/departments/housing/landlords",
    ],
    "baltimore_city": [
        "https://www.baltimorecity.gov/dhcd/our-work/resources-for-renters",
    ],
    "prince_georges": [
        "https://www.princegeorgescountymd.gov/departments-offices/housing-community-development/resources/tenant-resources",
        "https://www.princegeorgescountymd.gov/departments-offices/housing-community-development/permanent-rent-stabilization-and-protection-act-2024",
    ],
    "peoples_law": [
        "https://www.peoples-law.org/cat/landlord-tenant",
        "https://www.peoples-law.org/cat/housing",
        "https://www.peoples-law.org/baltimore-city-rental-and-housing-laws",
        "https://www.peoples-law.org/prince-georges-county-rental-and-housing-laws",
    ],
}

KEYWORDS = re.compile(
    r"(tenant|landlord|lease|rent|evict|security\s*deposit|housing|habitab|"
    r"repair|notice\s*to\s*quit|foreclos|bill\s*of\s*rights|escrow|"
    r"discriminat|disput|stabiliz)",
    re.IGNORECASE,
)

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class ScrapedPage:
    url: str
    source: str
    title: str
    text: str
    fetched_at: float

    @property
    def doc_id(self) -> str:
        return hashlib.md5(self.url.encode()).hexdigest()[:12]


def _clean_html(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "form"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text


def _fetch(url: str, session: requests.Session) -> str | None:
    try:
        r = session.get(url, timeout=20)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "html" not in ctype.lower():
            return None
        return r.text
    except requests.RequestException as e:
        print(f"  [skip] {url} — {e}")
        return None


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    base_host = urlparse(base_url).netloc
    out: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0]
        if urlparse(href).netloc != base_host:
            continue
        if any(href.lower().endswith(ext) for ext in (".pdf", ".jpg", ".png", ".zip", ".doc", ".docx")):
            continue
        anchor = a.get_text(" ", strip=True)
        if KEYWORDS.search(href) or KEYWORDS.search(anchor):
            out.add(href)
    return sorted(out)


_PAGE_CACHE: dict[str, str] = {}


def scrape_all(max_per_source: int = 40) -> list[ScrapedPage]:
    session = requests.Session()
    session.headers.update({
        "User-Agent": BROWSER_UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    results: list[ScrapedPage] = []
    global_seen: set[str] = set()

    for source, seed_urls in SEEDS.items():
        print(f"\n[{source}] starting …")

        frontier: list[str] = list(seed_urls)
        collected: list[str] = []

        for hop in range(2):
            next_frontier: list[str] = []
            for url in frontier:
                if url in global_seen or len(collected) >= max_per_source:
                    continue
                global_seen.add(url)
                html = _fetch(url, session)
                time.sleep(settings.request_delay)
                if not html:
                    continue
                collected.append(url)
                _PAGE_CACHE[url] = html
                if hop == 0:
                    next_frontier.extend(_extract_links(html, url))
            frontier = next_frontier
            if len(collected) >= max_per_source:
                break

        collected = collected[:max_per_source]

        kept = 0
        for url in tqdm(collected, desc=f"  {source}", leave=False):
            html = _PAGE_CACHE.get(url)
            if not html:
                continue
            title, text = _clean_html(html)
            if len(text) < 300:
                continue
            results.append(
                ScrapedPage(
                    url=url,
                    source=source,
                    title=title,
                    text=text,
                    fetched_at=time.time(),
                )
            )
            kept += 1

        print(f"  [{source}] kept {kept} pages")

    return results


def save_raw(pages: Iterable[ScrapedPage]) -> Path:
    out_path = Path(settings.raw_dir) / "pages.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
    return out_path


if __name__ == "__main__":
    pages = scrape_all()
    path = save_raw(pages)
    print(f"\nSaved {len(pages)} pages → {path}")