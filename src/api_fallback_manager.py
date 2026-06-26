import time
import random
import requests

class ApiFallbackManager:
    def __init__(self, openaq_keys=None, max_retries=3, base_backoff=2.0):
        self.openaq_keys = openaq_keys or []
        self.current_key_idx = 0
        self.max_retries = max_retries
        self.base_backoff = base_backoff

    def request_with_fallback(self, url, params=None, headers=None, is_openaq=False):
        """Execute a request with key rotation and exponential backoff + jitter."""
        for attempt in range(self.max_retries):
            # 1. Key Rotation Logic for OpenAQ
            current_headers = headers.copy() if headers else {}
            if is_openaq and self.openaq_keys:
                current_key = self.openaq_keys[self.current_key_idx]
                current_headers['X-API-Key'] = current_key

            try:
                resp = requests.get(url, params=params, headers=current_headers, timeout=15)
                
                if resp.status_code == 200:
                    return resp.json()
                    
                # If OpenAQ hits a rate limit (429), immediately rotate the key and continue
                if is_openaq and resp.status_code == 429:
                    print(f"  ⚠️ OpenAQ 429 Limit Hit on Key {self.current_key_idx}. Rotating...")
                    self.current_key_idx = (self.current_key_idx + 1) % len(self.openaq_keys)
                    # For OpenAQ 429s, we should still wait a tiny bit before immediately hammering with new key
                    time.sleep(1.0 + random.uniform(0, 0.5))
                    continue

                resp.raise_for_status()

            except (requests.exceptions.RequestException, ValueError) as e:
                if attempt == self.max_retries - 1:
                    # Final Kill Switch
                    raise RuntimeError(f"🚨 API Exhaustion. All {self.max_retries} retries failed for {url}. Error: {str(e)}")
                
                # 2. IP-Based Defense: Exponential Backoff & Jitter
                # e.g., wait 2s, then 5s, then ~11s
                backoff_time = (self.base_backoff ** (attempt + 1)) + random.uniform(0, 1)
                print(f"  ⏳ API Failure (Attempt {attempt+1}/{self.max_retries}). Backing off for {backoff_time:.2f}s...")
                time.sleep(backoff_time)
