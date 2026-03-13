"""
Context Enrichment Task — §4.F Module F

Asynchronous web scraping via Celery workers using patchright
(patched Chromium) for external metadata enrichment.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from celery import Task

from services.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(  # type: ignore[untyped-decorator]
    bind=True,
    max_retries=2,
    default_retry_delay=5,
)
def scrape_context(
    self: Task, target_url: str, scrape_type: str, timeout_seconds: int = 30
) -> dict[str, Any]:
    """
    §4.F.1 — Execute a scraping job using patchright.

    Launches an ephemeral headless browser instance with patched
    Chromium automation fingerprints (§4.F.1).

    Failure modes (§4.F contract):
      - CAPTCHA or HTTP blocking → return empty result
      - Browser crash → retry (max 2x per §12 Worker crash F)
      - Timeout → terminate browser
      - Broker outage → Celery reconnection

    Args:
        target_url: URL to scrape.
        scrape_type: Type of scrape operation.
        timeout_seconds: Maximum browser session duration.

    Returns:
        JSON dict with source_url, scraped_at_utc, extracted data.
    """
    from patchright.sync_api import sync_playwright

    browser = None
    result: dict[str, Any] = {
        "source_url": target_url,
        "scraped_at_utc": datetime.now(UTC).isoformat(),
        "scrape_type": scrape_type,
        "data": {},
        "success": False,
    }

    try:
        with sync_playwright() as pw:
            # §4.F.1 — Launch headless Chromium with patched fingerprints
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context()

            # §4.F contract — Timeout: terminate browser after timeout_seconds
            page = context.new_page()
            page.set_default_timeout(timeout_seconds * 1000)

            try:
                # §4.F.1 — Navigate and wait for content
                response = page.goto(target_url, wait_until="domcontentloaded")

                if response is None:
                    # §4.F contract — HTTP blocking: empty result
                    logger.warning("No response from %s", target_url)
                    return result

                status_code = response.status

                # §4.F contract — CAPTCHA or HTTP blocking: empty result
                if status_code == 403 or status_code == 429:
                    logger.warning(
                        "HTTP %d (blocked/CAPTCHA) for %s", status_code, target_url
                    )
                    return result

                if status_code >= 400:
                    logger.warning(
                        "HTTP %d for %s", status_code, target_url
                    )
                    return result

                # §4.F.1 — Extract page content
                title: str = page.title()
                text_content: str = page.inner_text("body")

                # §4.F.1 — Extract structured metadata
                meta_tags: list[dict[str, str | None]] = page.evaluate("""
                    () => Array.from(document.querySelectorAll('meta')).map(m => ({
                        name: m.getAttribute('name'),
                        property: m.getAttribute('property'),
                        content: m.getAttribute('content')
                    }))
                """)

                result["data"] = {
                    "title": title,
                    "text_content": text_content[:10000],  # Cap extracted text
                    "meta_tags": meta_tags,
                    "status_code": status_code,
                }
                result["success"] = True
                result["scraped_at_utc"] = datetime.now(UTC).isoformat()

            except Exception as page_exc:
                # §4.F contract — Timeout or navigation error
                logger.warning(
                    "Page error for %s: %s", target_url, page_exc
                )

            finally:
                context.close()
                browser.close()

    except Exception as exc:
        # §4.F contract / §12 Worker crash F — Browser crash: retry 2x
        logger.error("Browser crash for %s: %s", target_url, exc)
        try:
            self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            # §12 Worker crash F — max retries exhausted
            logger.error(
                "Max retries exhausted for scrape_context: %s", target_url
            )

    return result
