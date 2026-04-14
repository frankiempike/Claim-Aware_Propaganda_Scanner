"""
Periodically pings a URL defined by the PING_URL environment variable.

Environment variables:
  PING_URL      (required) URL to ping
  PING_INTERVAL (optional) Seconds between pings, default 60
  PING_TIMEOUT  (optional) Request timeout in seconds, default 10
"""

import os
import time
import urllib.request
import urllib.error

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


def ping_once(url: str, timeout: int) -> bool:
    """Send a single GET request to *url*. Returns True on HTTP 2xx/3xx."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = response.status
            logger.info(f"[OK] {url} → HTTP {status}")
            return True
    except urllib.error.HTTPError as exc:
        logger.warning(f"[HTTP ERROR] {url} → HTTP {exc.code} {exc.reason}")
        return False
    except urllib.error.URLError as exc:
        logger.error(f"[UNREACHABLE] {url} → {exc.reason}")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[UNEXPECTED] {url} → {exc}")
        return False


def main() -> None:
    url = os.environ.get("PING_URL")
    if not url:
        raise SystemExit("PING_URL environment variable is not set.")

    interval = int(os.environ.get("PING_INTERVAL", "60"))
    timeout = int(os.environ.get("PING_TIMEOUT", "10"))

    logger.info(f"Starting ping loop: url={url!r}  interval={interval}s  timeout={timeout}s")

    while True:
        ping_once(url, timeout)
        time.sleep(interval)


if __name__ == "__main__":
    main()
