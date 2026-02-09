#!/usr/bin/env python3
"""
AIRR Data Commons (ADC) Repertoire Download Script
Downloads TCR repertoire metadata and optionally rearrangement sequences
from the federated AIRR Data Commons network (~19 repository nodes).

The ADC API specification: https://docs.airr-community.org/en/stable/api/adc_api.html

Output format is compatible with quest.parsers.airr.database_parser._parse_json_ireceptor
which reads {"Repertoire": [...]} via ijson.items(f, 'Repertoire.item').

Usage:
    # Default: human repertoire metadata from all repos
    python scripts/data_download/download_adc_repertoires.py

    # Include rearrangement sequences (large download)
    python scripts/data_download/download_adc_repertoires.py --download-rearrangements

    # Specific repos only
    python scripts/data_download/download_adc_repertoires.py --repos ipa1.ireceptor.org vdjserver.org
"""

import os
import sys
import json
import time
import logging
import argparse
import threading
import concurrent.futures
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Default output directory
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "databases" / "adc"

# ADC API version prefix
API_PREFIX = "/airr/v1"

# Registry URL for discovering ADC nodes
REGISTRY_URL = (
    "https://raw.githubusercontent.com/airr-community/adc-registry/master/ADC-registry.tsv"
)

# Fallback list of known ADC repository base URLs
FALLBACK_REPOS = [
    "https://covid19-1.ireceptor.org",
    "https://covid19-2.ireceptor.org",
    "https://covid19-3.ireceptor.org",
    "https://covid19-4.ireceptor.org",
    "https://ipa1.ireceptor.org",
    "https://ipa2.ireceptor.org",
    "https://ipa3.ireceptor.org",
    "https://ipa4.ireceptor.org",
    "https://ipa5.ireceptor.org",
    "https://ipa6.ireceptor.org",
    "https://vdjserver.org",
    "https://scireptor.dkfz.de",
    "https://airr-seq.vdjbase.org",
    "https://roche-airr.ireceptor.org",
    "https://t1d-1.ireceptor.org",
    "https://t1d-2.ireceptor.org",
    "https://agschwab.uni-muenster.de",
    "https://hpap.ireceptor.org",
    "https://greifflab-1.ireceptor.org",
]

logger = logging.getLogger(__name__)


@dataclass
class ADCRepository:
    """Represents a single ADC repository node."""

    base_url: str
    hostname: str = ""
    available: bool = False
    api_version: str = ""
    max_size: int = 1000
    info: Optional[Dict] = None

    def __post_init__(self):
        if not self.hostname:
            self.hostname = urlparse(self.base_url).hostname or self.base_url


class ThreadSafeThrottle:
    """Thread-safe throttle that enforces minimum delay between request initiations."""

    def __init__(self, delay: float):
        self.delay = delay
        self._lock = threading.Lock()
        self._last_request_time = 0.0

    def wait(self):
        """Block until at least `delay` seconds have passed since the last call."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self.delay:
                time.sleep(self.delay - elapsed)
            self._last_request_time = time.time()


@dataclass
class DownloadCheckpoint:
    """Tracks per-repo download progress for resume capability."""

    discovered_repos: Dict[str, str] = field(default_factory=dict)  # hostname -> status
    repertoire_counts: Dict[str, int] = field(default_factory=dict)  # hostname -> count
    downloaded_rearrangements: Dict[str, List[str]] = field(
        default_factory=dict
    )  # hostname -> [repertoire_ids]
    failed_rearrangements: Dict[str, List[str]] = field(
        default_factory=dict
    )  # hostname -> [repertoire_ids]
    last_updated: str = ""

    def save(self, path: Path):
        self.last_updated = datetime.now().isoformat()
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DownloadCheckpoint":
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(**data)
        return cls()


class ADCClient:
    """HTTP client for a single ADC repository node."""

    def __init__(
        self,
        repo: ADCRepository,
        timeout: int = 120,
        rearrangement_timeout: int = 600,
        retries: int = 3,
        delay: float = 1.0,
        verify_ssl: bool = True,
        throttle: Optional["ThreadSafeThrottle"] = None,
        pool_maxsize: int = 10,
    ):
        self.repo = repo
        self.timeout = timeout
        self.rearrangement_timeout = rearrangement_timeout
        self.delay = delay
        self._throttle_impl = throttle
        self._last_request_time = 0.0

        self.session = requests.Session()
        self.session.verify = verify_ssl
        if not verify_ssl:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        retry_strategy = Retry(
            total=retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=pool_maxsize)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )

    def _throttle(self):
        """Enforce minimum delay between requests to the same server."""
        if self._throttle_impl is not None:
            self._throttle_impl.wait()
            return
        elapsed = time.time() - self._last_request_time
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_time = time.time()

    def _url(self, endpoint: str) -> str:
        return f"{self.repo.base_url}{API_PREFIX}/{endpoint}"

    def get_info(self) -> Optional[Dict]:
        """GET /airr/v1/info to check availability and capabilities."""
        try:
            self._throttle()
            resp = self.session.get(self._url("info"), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            self.repo.available = True
            self.repo.info = data
            self.repo.api_version = str(data.get("api", {}).get("version", ""))
            self.repo.max_size = data.get("max_size", 1000)
            return data
        except Exception as e:
            logger.debug("Info request failed for %s: %s", self.repo.hostname, e)
            self.repo.available = False
            return None

    def get_repertoires(
        self, species: Optional[str] = None, page_size: int = 0
    ) -> List[Dict]:
        """
        POST /airr/v1/repertoire to get repertoire metadata.
        Returns list of repertoire dicts.
        """
        if page_size <= 0:
            page_size = min(self.repo.max_size, 1000)

        # Build filters
        filters = None
        if species:
            filters = {
                "op": "=",
                "content": {
                    "field": "subject.species.id",
                    "value": species,
                },
            }

        all_repertoires = []
        offset = 0

        while True:
            body = {
                "size": page_size,
                "from": offset,
            }
            if filters:
                body["filters"] = filters

            try:
                self._throttle()
                resp = self.session.post(
                    self._url("repertoire"),
                    json=body,
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(
                    "Repertoire request failed for %s (offset=%d): %s",
                    self.repo.hostname,
                    offset,
                    e,
                )
                break

            repertoires = data.get("Repertoire", [])
            if not repertoires:
                break

            all_repertoires.extend(repertoires)
            logger.debug(
                "%s: fetched %d repertoires (total: %d)",
                self.repo.hostname,
                len(repertoires),
                len(all_repertoires),
            )

            if len(repertoires) < page_size:
                break

            offset += len(repertoires)

        return all_repertoires

    def download_rearrangement(
        self, repertoire_id: str, output_path: Path
    ) -> bool:
        """
        POST /airr/v1/rearrangement for a single repertoire, streaming TSV to disk.
        Uses atomic write via .tmp rename.
        """
        body = {
            "filters": {
                "op": "=",
                "content": {
                    "field": "repertoire_id",
                    "value": repertoire_id,
                },
            },
            "format": "tsv",
        }

        tmp_path = output_path.with_suffix(".tmp")
        try:
            self._throttle()
            resp = self.session.post(
                self._url("rearrangement"),
                json=body,
                timeout=self.rearrangement_timeout,
                stream=True,
            )
            resp.raise_for_status()

            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            tmp_path.rename(output_path)
            return True
        except Exception as e:
            logger.warning(
                "Rearrangement download failed for %s/%s: %s",
                self.repo.hostname,
                repertoire_id,
                e,
            )
            if tmp_path.exists():
                tmp_path.unlink()
            return False


class ADCDownloader:
    """Orchestrates discovery and download across all ADC nodes."""

    def __init__(
        self,
        output_dir: Path,
        species: Optional[str] = "NCBITAXON:9606",
        repos_filter: Optional[List[str]] = None,
        download_rearrangements: bool = False,
        workers: int = 4,
        rearrangement_workers: int = 8,
        timeout: int = 120,
        rearrangement_timeout: int = 600,
        retries: int = 3,
        delay: float = 0.2,
        verify_ssl: bool = True,
    ):
        self.output_dir = output_dir
        self.species = species
        self.repos_filter = repos_filter
        self.download_rearrangements = download_rearrangements
        self.workers = workers
        self.rearrangement_workers = rearrangement_workers
        self.timeout = timeout
        self.rearrangement_timeout = rearrangement_timeout
        self.retries = retries
        self.delay = delay
        self.verify_ssl = verify_ssl

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = self.output_dir / "checkpoint.json"
        self.checkpoint = DownloadCheckpoint.load(self.checkpoint_path)

    def discover_repos(self) -> List[str]:
        """Fetch repository URLs from the ADC registry, fallback to hardcoded list."""
        logger.info("Discovering ADC repositories from registry...")
        repo_urls = []

        try:
            resp = requests.get(REGISTRY_URL, timeout=30)
            resp.raise_for_status()
            lines = resp.text.strip().split("\n")
            if len(lines) > 1:
                # TSV: parse header to find URL column
                header = lines[0].split("\t")
                url_idx = None
                for i, col in enumerate(header):
                    if "url" in col.lower() or "base" in col.lower():
                        url_idx = i
                        break

                if url_idx is not None:
                    for line in lines[1:]:
                        fields = line.split("\t")
                        if len(fields) > url_idx and fields[url_idx].startswith("http"):
                            url = fields[url_idx].rstrip("/")
                            repo_urls.append(url)

            if repo_urls:
                logger.info(
                    "Found %d repositories from registry", len(repo_urls)
                )
                return repo_urls
        except Exception as e:
            logger.warning("Failed to fetch registry: %s", e)

        logger.info("Using fallback repository list (%d repos)", len(FALLBACK_REPOS))
        return list(FALLBACK_REPOS)

    def _probe_repo(self, base_url: str) -> ADCRepository:
        """Probe a single repository for availability."""
        repo = ADCRepository(base_url=base_url)
        client = ADCClient(
            repo,
            timeout=min(self.timeout, 30),
            retries=1,
            delay=0,
            verify_ssl=self.verify_ssl,
        )
        client.get_info()
        return repo

    def probe_repos(self, repo_urls: List[str]) -> List[ADCRepository]:
        """Probe all repositories in parallel to check availability."""
        logger.info("Probing %d repositories...", len(repo_urls))
        repos = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_url = {
                executor.submit(self._probe_repo, url): url for url in repo_urls
            }
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    repo = future.result()
                    repos.append(repo)
                    status = "AVAILABLE" if repo.available else "UNAVAILABLE"
                    logger.info("  %s: %s", repo.hostname, status)
                    self.checkpoint.discovered_repos[repo.hostname] = status
                except Exception as e:
                    hostname = urlparse(url).hostname or url
                    logger.warning("  %s: PROBE ERROR - %s", hostname, e)
                    self.checkpoint.discovered_repos[hostname] = "ERROR"

        self.checkpoint.save(self.checkpoint_path)

        available = [r for r in repos if r.available]
        logger.info(
            "%d/%d repositories available", len(available), len(repos)
        )
        return repos

    def _download_repo_repertoires(
        self, repo: ADCRepository
    ) -> tuple:
        """Download repertoire metadata from a single repo. Returns (repo, repertoires)."""
        client = ADCClient(
            repo,
            timeout=self.timeout,
            retries=self.retries,
            delay=self.delay,
            verify_ssl=self.verify_ssl,
        )
        repertoires = client.get_repertoires(species=self.species)
        return repo, repertoires

    def download_repertoires(self, repos: List[ADCRepository]) -> Dict[str, List[Dict]]:
        """Download repertoire metadata from all available repos."""
        available = [r for r in repos if r.available]
        if not available:
            logger.warning("No available repositories to query")
            return {}

        logger.info(
            "Downloading repertoire metadata from %d repositories...",
            len(available),
        )

        all_repertoires = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self._download_repo_repertoires, repo): repo
                for repo in available
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Repositories",
                unit="repo",
            ):
                repo = futures[future]
                try:
                    _, repertoires = future.result()
                    all_repertoires[repo.hostname] = repertoires
                    self.checkpoint.repertoire_counts[repo.hostname] = len(repertoires)
                    logger.info(
                        "  %s: %d repertoires", repo.hostname, len(repertoires)
                    )
                except Exception as e:
                    logger.warning(
                        "  %s: FAILED - %s", repo.hostname, e
                    )
                    all_repertoires[repo.hostname] = []
                    self.checkpoint.repertoire_counts[repo.hostname] = 0

        self.checkpoint.save(self.checkpoint_path)
        return all_repertoires

    def save_repertoires(
        self, repos: List[ADCRepository], all_repertoires: Dict[str, List[Dict]]
    ):
        """Save repertoire metadata and repo info to disk."""
        for repo in repos:
            if not repo.available:
                continue

            repo_dir = self.output_dir / repo.hostname
            repo_dir.mkdir(parents=True, exist_ok=True)

            # Save info response
            if repo.info:
                info_path = repo_dir / "info.json"
                with open(info_path, "w") as f:
                    json.dump(repo.info, f, indent=2)

            # Save repertoires in AIRR format compatible with _parse_json_ireceptor
            repertoires = all_repertoires.get(repo.hostname, [])
            if repertoires:
                repertoires_path = repo_dir / "repertoires.json"
                with open(repertoires_path, "w") as f:
                    json.dump({"Repertoire": repertoires}, f, indent=2)
                logger.info(
                    "Saved %d repertoires to %s",
                    len(repertoires),
                    repertoires_path,
                )

    def download_all_rearrangements(
        self, repos: List[ADCRepository], all_repertoires: Dict[str, List[Dict]]
    ):
        """Download rearrangement TSVs for all repertoires, parallel within each repo."""
        for repo in repos:
            if not repo.available:
                continue

            repertoires = all_repertoires.get(repo.hostname, [])
            if not repertoires:
                continue

            repo_dir = self.output_dir / repo.hostname / "rearrangements"
            repo_dir.mkdir(parents=True, exist_ok=True)

            # Get already-downloaded IDs from checkpoint
            done_ids: Set[str] = set(
                self.checkpoint.downloaded_rearrangements.get(repo.hostname, [])
            )
            failed_ids: Set[str] = set(
                self.checkpoint.failed_rearrangements.get(repo.hostname, [])
            )

            # Extract repertoire IDs
            rep_ids = []
            for rep in repertoires:
                rid = rep.get("repertoire_id")
                if rid:
                    rep_ids.append(str(rid))

            to_download = [rid for rid in rep_ids if rid not in done_ids]
            skipped = len(rep_ids) - len(to_download)
            if skipped > 0:
                logger.info(
                    "%s: skipping %d already-downloaded rearrangements",
                    repo.hostname,
                    skipped,
                )

            if not to_download:
                continue

            logger.info(
                "%s: downloading rearrangements for %d repertoires (%d workers)...",
                repo.hostname,
                len(to_download),
                self.rearrangement_workers,
            )

            # Shared throttle ensures minimum delay between request initiations
            shared_throttle = ThreadSafeThrottle(self.delay)
            # Thread-local storage for per-thread ADCClient instances
            thread_local = threading.local()
            # Lock protects done_ids, failed_ids mutations and checkpoint writes
            ids_lock = threading.Lock()
            completions_since_save = 0

            def _get_client() -> ADCClient:
                """Get or create a thread-local ADCClient."""
                if not hasattr(thread_local, "client"):
                    thread_local.client = ADCClient(
                        repo,
                        timeout=self.timeout,
                        rearrangement_timeout=self.rearrangement_timeout,
                        retries=self.retries,
                        delay=self.delay,
                        verify_ssl=self.verify_ssl,
                        throttle=shared_throttle,
                        pool_maxsize=self.rearrangement_workers,
                    )
                return thread_local.client

            def _download_one(rid: str) -> tuple:
                """Download a single rearrangement, returns (rid, success)."""
                output_path = repo_dir / f"{rid}.tsv"
                if output_path.exists():
                    return rid, True
                client = _get_client()
                success = client.download_rearrangement(rid, output_path)
                return rid, success

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.rearrangement_workers
            ) as executor:
                futures = {
                    executor.submit(_download_one, rid): rid
                    for rid in to_download
                }

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=f"  {repo.hostname}",
                    unit="repertoire",
                ):
                    rid = futures[future]
                    try:
                        _, success = future.result()
                        with ids_lock:
                            if success:
                                done_ids.add(rid)
                            else:
                                failed_ids.add(rid)
                            completions_since_save += 1

                            # Batch checkpoint saves every 10 completions
                            if completions_since_save >= 10:
                                self.checkpoint.downloaded_rearrangements[repo.hostname] = list(done_ids)
                                self.checkpoint.failed_rearrangements[repo.hostname] = list(failed_ids)
                                self.checkpoint.save(self.checkpoint_path)
                                completions_since_save = 0
                    except Exception as e:
                        logger.warning(
                            "Unexpected error for %s/%s: %s",
                            repo.hostname,
                            rid,
                            e,
                        )
                        with ids_lock:
                            failed_ids.add(rid)
                            completions_since_save += 1

            # Final checkpoint save for this repo
            self.checkpoint.downloaded_rearrangements[repo.hostname] = list(done_ids)
            self.checkpoint.failed_rearrangements[repo.hostname] = list(failed_ids)
            self.checkpoint.save(self.checkpoint_path)

    def run(self):
        """Execute the full discovery and download pipeline."""
        start_time = time.time()

        # Step 1: Discover repositories
        repo_urls = self.discover_repos()

        # Filter to specific repos if requested
        if self.repos_filter:
            filtered = []
            for url in repo_urls:
                hostname = urlparse(url).hostname or url
                if hostname in self.repos_filter:
                    filtered.append(url)
            # Add any filter hostnames not found in registry as direct URLs
            found_hostnames = {urlparse(u).hostname for u in filtered}
            for h in self.repos_filter:
                if h not in found_hostnames:
                    filtered.append(f"https://{h}")
            repo_urls = filtered
            logger.info("Filtered to %d repositories", len(repo_urls))

        if not repo_urls:
            logger.error("No repositories to query")
            return

        # Step 2: Probe repositories
        repos = self.probe_repos(repo_urls)

        # Step 3: Download repertoire metadata
        all_repertoires = self.download_repertoires(repos)

        # Step 4: Save to disk
        self.save_repertoires(repos, all_repertoires)

        # Step 5: Optionally download rearrangements
        if self.download_rearrangements:
            self.download_all_rearrangements(repos, all_repertoires)

        # Step 6: Print summary
        elapsed = time.time() - start_time
        self._print_summary(repos, all_repertoires, elapsed)

    def _print_summary(
        self,
        repos: List[ADCRepository],
        all_repertoires: Dict[str, List[Dict]],
        elapsed: float,
    ):
        """Print download summary."""
        available = [r for r in repos if r.available]
        unavailable = [r for r in repos if not r.available]
        total_repertoires = sum(len(v) for v in all_repertoires.values())

        print(f"\n{'=' * 60}")
        print("ADC DOWNLOAD SUMMARY")
        print(f"{'=' * 60}")
        print(f"Repositories probed:     {len(repos)}")
        print(f"  Available:             {len(available)}")
        print(f"  Unavailable:           {len(unavailable)}")
        print(f"Total repertoires:       {total_repertoires:,}")
        if self.species:
            print(f"Species filter:          {self.species}")
        else:
            print(f"Species filter:          (none)")
        print(f"Elapsed time:            {elapsed:.1f}s")
        print(f"Output directory:        {self.output_dir}")

        if all_repertoires:
            print(f"\nPer-repository counts:")
            for hostname in sorted(all_repertoires.keys()):
                count = len(all_repertoires[hostname])
                print(f"  {hostname}: {count:,}")

        if unavailable:
            print(f"\nUnavailable repositories:")
            for r in unavailable:
                print(f"  {r.hostname}")

        if self.download_rearrangements:
            total_downloaded = sum(
                len(v) for v in self.checkpoint.downloaded_rearrangements.values()
            )
            total_failed = sum(
                len(v) for v in self.checkpoint.failed_rearrangements.values()
            )
            print(f"\nRearrangements downloaded: {total_downloaded:,}")
            if total_failed:
                print(f"Rearrangements failed:    {total_failed:,}")

        print(f"\nCheckpoint: {self.checkpoint_path}")
        print(f"{'=' * 60}")


def setup_logging(output_dir: Path, verbose: bool = False):
    """Configure dual logging: console (INFO) + file (DEBUG)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # File handler - DEBUG level
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s %(message)s")
    )
    root_logger.addHandler(fh)

    # Console handler - INFO level (or DEBUG if verbose)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    root_logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser(
        description="Download repertoire data from the AIRR Data Commons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: human repertoire metadata from all repos
  python scripts/data_download/download_adc_repertoires.py

  # All species
  python scripts/data_download/download_adc_repertoires.py --species all

  # Include rearrangement sequences
  python scripts/data_download/download_adc_repertoires.py --download-rearrangements

  # Specific repos only
  python scripts/data_download/download_adc_repertoires.py --repos ipa1.ireceptor.org vdjserver.org

  # Custom parallelism and timeouts
  python scripts/data_download/download_adc_repertoires.py --workers 8 --timeout 180 --delay 2.0

  # Resume interrupted download (automatic via checkpoint.json)
  python scripts/data_download/download_adc_repertoires.py
        """,
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: data/databases/adc/)",
    )
    parser.add_argument(
        "--species",
        default="NCBITAXON:9606",
        help="Species filter, e.g. NCBITAXON:9606 for human. Use 'all' for no filter (default: NCBITAXON:9606)",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        metavar="HOSTNAME",
        help="Specific repo hostnames to query (default: all)",
    )
    parser.add_argument(
        "--download-rearrangements",
        action="store_true",
        help="Also download rearrangement TSV files (large!)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers for repo-level operations (default: 4)",
    )
    parser.add_argument(
        "--rearrangement-workers",
        type=int,
        default=8,
        help="Parallel workers for rearrangement downloads within each repo (default: 8)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for metadata queries (default: 120)",
    )
    parser.add_argument(
        "--rearrangement-timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds for rearrangement downloads (default: 600)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry attempts per failed request (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Min seconds between requests to same server (default: 0.2)",
    )
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable SSL certificate verification. WARNING: This makes connections "
        "vulnerable to man-in-the-middle attacks. Only use when repositories have "
        "known certificate issues (e.g. self-signed or incomplete chains).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) console logging",
    )

    args = parser.parse_args()

    # Handle species filter
    species = args.species if args.species != "all" else None

    # Setup logging
    setup_logging(args.output_dir, verbose=args.verbose)

    logger.info("AIRR Data Commons Repertoire Downloader")
    logger.info("=" * 60)
    logger.info("Output directory:        %s", args.output_dir)
    logger.info("Species filter:          %s", species or "(none)")
    logger.info("Download rearrangements: %s", args.download_rearrangements)
    logger.info("Workers:                 %d", args.workers)
    logger.info("Rearrangement workers:   %d", args.rearrangement_workers)
    logger.info("Timeout:                 %ds", args.timeout)
    logger.info("Rearrangement timeout:   %ds", args.rearrangement_timeout)
    logger.info("Retries:                 %d", args.retries)
    logger.info("Delay:                   %.1fs", args.delay)
    if args.repos:
        logger.info("Repos filter:            %s", ", ".join(args.repos))
    if args.no_verify_ssl:
        logger.warning("SSL verification:        DISABLED (--no-verify-ssl)")
    logger.info("")

    downloader = ADCDownloader(
        output_dir=args.output_dir,
        species=species,
        repos_filter=args.repos,
        download_rearrangements=args.download_rearrangements,
        workers=args.workers,
        rearrangement_workers=args.rearrangement_workers,
        timeout=args.timeout,
        rearrangement_timeout=args.rearrangement_timeout,
        retries=args.retries,
        delay=args.delay,
        verify_ssl=not args.no_verify_ssl,
    )

    downloader.run()


if __name__ == "__main__":
    main()
