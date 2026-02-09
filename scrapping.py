from __future__ import annotations

import json
import random
import time
import os
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional
from urllib.parse import urljoin

import cloudscraper
from bs4 import BeautifulSoup


# =========================
# CONFIG
# =========================

START_FORUM = "https://www.goldenskate.com/forum/forums/general-chat-and-questions.979/"
HOME_URL = "https://www.goldenskate.com/"

MIN_SLEEP = 3.0
MAX_SLEEP = 7.0
TIMEOUT = 25

MAX_FORUM_PAGES = 61          # forum pages to walk
MAX_THREAD_PAGES = 50         # per-thread safety cap

CACHE_DIR = "html_cache"
CHECKPOINT_FILE = "checkpoint.json"
OUTPUT_FILE = "goldenskate_generalQA.json"


# =========================
# UTIL
# =========================

def polite_sleep():
    time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))


def ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def cache_path(url: str) -> str:
    return os.path.join(CACHE_DIR, url_hash(url) + ".html")


def load_from_cache(url: str) -> Optional[str]:
    path = cache_path(url)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def save_to_cache(url: str, html: str):
    path = cache_path(url)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def canonical_thread_url(url: str) -> str:
    url = url.split("/page-")[0]
    url = url.replace("/latest", "")
    return url.rstrip("/")


def make_session():
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "darwin",
            "desktop": True,
        }
    )
    scraper.get(HOME_URL, timeout=TIMEOUT)  # warm cookies once
    return scraper


def fetch_html(session, url: str) -> Optional[str]:
    cached = load_from_cache(url)
    if cached is not None:
        return cached

    try:
        r = session.get(url, timeout=TIMEOUT)
        if r.status_code in (401, 403, 429):
            print(f"[SKIP] Blocked {r.status_code}: {url}")
            return None

        r.raise_for_status()
        save_to_cache(url, r.text)
        polite_sleep()
        return r.text

    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return None


def get_next_page(soup: BeautifulSoup, current_url: str) -> Optional[str]:
    link = soup.find("link", attrs={"rel": "next"})
    if link and link.get("href"):
        return urljoin(current_url, link["href"])

    a = soup.select_one("a.pageNav-jump--next")
    if a and a.get("href"):
        return urljoin(current_url, a["href"])

    return None


# =========================
# DATA MODELS
# =========================

@dataclass
class ThreadItem:
    title: str
    url: str


@dataclass
class PostItem:
    author: str
    datetime_iso: Optional[str]
    content_text: str
    post_url: Optional[str]


@dataclass
class ThreadWithPosts:
    thread_title: str
    thread_url: str
    posts: list[PostItem]


# =========================
# PARSERS
# =========================

def parse_threads_from_forum_page(html: str, page_url: str) -> list[ThreadItem]:
    soup = BeautifulSoup(html, "lxml")
    threads: dict[str, ThreadItem] = {}

    for a in soup.select('a[href*="/threads/"]'):
        href = a.get("href")
        title = a.get_text(" ", strip=True)
        if not href or not title:
            continue

        full = canonical_thread_url(urljoin(page_url, href))
        threads.setdefault(full, ThreadItem(title=title, url=full))

    return list(threads.values())


def parse_posts_from_thread_page(html: str, page_url: str) -> tuple[str, list[PostItem]]:
    soup = BeautifulSoup(html, "lxml")

    h1 = soup.find("h1")
    thread_title = h1.get_text(" ", strip=True) if h1 else page_url

    posts: list[PostItem] = []

    for msg in soup.select("article.message"):
        author = (msg.get("data-author") or "UNKNOWN").strip()

        time_tag = msg.find("time")
        dt = None
        if time_tag:
            dt = time_tag.get("datetime") or time_tag.get("data-time")

        body = msg.select_one("div.bbWrapper")
        text = body.get_text("\n", strip=True) if body else ""

        link = msg.select_one('a[href*="/post-"]')
        post_url = urljoin(page_url, link["href"]) if link else None

        if text:
            posts.append(PostItem(author, dt, text, post_url))

    return thread_title, posts


# =========================
# CHECKPOINTING
# =========================

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"threads_done": []}


def save_checkpoint(data):
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# =========================
# CRAWLER
# =========================

def crawl_forum():
    ensure_dirs()
    session = make_session()
    checkpoint = load_checkpoint()
    done = set(checkpoint["threads_done"])

    all_threads: list[ThreadItem] = []
    all_threads_with_posts: list[ThreadWithPosts] = []

    forum_url = START_FORUM

    # ---- forum pagination ----
    for page_idx in range(MAX_FORUM_PAGES):
        print(f"\n[FORUM PAGE {page_idx + 1}] {forum_url}")

        html = fetch_html(session, forum_url)
        if not html:
            print("[WARN] Skipping forum page due to fetch failure")
            polite_sleep()
            continue

        threads = parse_threads_from_forum_page(html, forum_url)
        all_threads.extend(threads)

        soup = BeautifulSoup(html, "lxml")
        next_url = get_next_page(soup, forum_url)
        if not next_url:
            break

        forum_url = next_url

    # ---- thread expansion ----
    for t in all_threads:
        if t.url in done:
            continue

        print(f"\n[THREAD] {t.title}")

        thread_posts: list[PostItem] = []
        page_url = t.url

        for _ in range(MAX_THREAD_PAGES):
            html = fetch_html(session, page_url)
            if not html:
                break

            title, posts = parse_posts_from_thread_page(html, page_url)
            thread_posts.extend(posts)

            soup = BeautifulSoup(html, "lxml")
            next_url = get_next_page(soup, page_url)
            if not next_url:
                break

            page_url = next_url

        print(f"  -> {len(thread_posts)} posts")

        all_threads_with_posts.append(
            ThreadWithPosts(title, t.url, thread_posts)
        )

        done.add(t.url)
        save_checkpoint({"threads_done": list(done)})

    return all_threads, all_threads_with_posts


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    threads, threads_with_posts = crawl_forum()

    output = {
        "threads": [asdict(t) for t in threads],
        "threads_with_posts": [asdict(twp) for twp in threads_with_posts],
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {OUTPUT_FILE}")
