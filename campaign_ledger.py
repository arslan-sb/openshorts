"""Which clip went to which campaign, account and post, and what is overdue.

Content Rewards counts one post for one campaign, wants the link submitted
soon after posting, and some briefs reject the same clip on several accounts.
The ledger is a small JSON file (campaigns/ledger.json, git-ignored) that
remembers all of that so you do not have to.

Clips are identified by the SHA-256 of the file, so a renamed copy is still
recognised as the same clip.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from jsonfile import write_json_atomic

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "campaigns", "ledger.json")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class LedgerError(ValueError):
    pass


class Ledger:
    def __init__(self, path: str = DEFAULT_PATH, now=_now):
        self.path = path
        self._now = now
        self.entries: List[dict] = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.entries = json.load(f).get("entries", [])

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        write_json_atomic(self.path, {"entries": self.entries}, indent=2)

    def get(self, entry_id: str) -> dict:
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        raise LedgerError(f"no ledger entry '{entry_id}' (see: python campaign.py status)")

    # -------------------------------------------------------------- events

    def add_prepared(self, campaign, clip_path: str,
                     sha: Optional[str] = None) -> Tuple[str, Optional[str]]:
        """Record a clip prepared for a campaign. Returns (id, warning)."""
        sha = sha or file_sha256(clip_path)
        for e in self.entries:
            if e["sha256"] == sha and e["campaign"] == campaign.slug:
                e["file"] = clip_path
                return e["id"], None
        warning = None
        other = sorted({e["campaign"] for e in self.entries if e["sha256"] == sha})
        if other:
            warning = (f"this exact clip is already in campaign(s) {', '.join(other)}: "
                       f"one post only counts for one campaign")
        entry = {
            "id": secrets.token_hex(3),
            "campaign": campaign.slug,
            "campaign_name": campaign.name,
            "file": clip_path,
            "sha256": sha,
            "prepared_at": _iso(self._now()),
            "submit_within_minutes": campaign.submit_within_minutes,
            "one_account_per_clip": campaign.one_account_per_clip,
            "analytics_after_views": campaign.analytics_recording_after_views,
            "posts": [],
        }
        self.entries.append(entry)
        return entry["id"], warning

    def mark_posted(self, entry_id: str, platform: str, account: str,
                    url: str = "") -> Tuple[dict, List[str]]:
        entry = self.get(entry_id)
        warnings = []
        if url:
            for e in self.entries:
                for p in e["posts"]:
                    if p.get("url") == url:
                        raise LedgerError(f"{url} is already recorded on entry "
                                          f"{e['id']} ({e['campaign']})")
        accounts = {p["account"].lower() for p in entry["posts"]}
        if entry.get("one_account_per_clip") and accounts - {account.lower()}:
            warnings.append(f"already posted from {', '.join(sorted(accounts))}: this "
                            f"brief rejects the same clip on several accounts")
        post = {"platform": platform, "account": account, "url": url,
                "posted_at": _iso(self._now()), "submitted_at": None,
                "analytics_recorded": False}
        entry["posts"].append(post)
        return post, warnings

    def _post(self, entry_id: str, platform: Optional[str]) -> dict:
        entry = self.get(entry_id)
        posts = [p for p in entry["posts"] if platform in (None, p["platform"])]
        if not posts:
            raise LedgerError(f"entry {entry_id} has no post"
                              + (f" on {platform}" if platform else "")
                              + ": run `posted` first")
        return posts[-1]

    def mark_submitted(self, entry_id: str, platform: Optional[str] = None) -> dict:
        post = self._post(entry_id, platform)
        post["submitted_at"] = _iso(self._now())
        return post

    def mark_analytics(self, entry_id: str, platform: Optional[str] = None) -> dict:
        post = self._post(entry_id, platform)
        post["analytics_recorded"] = True
        return post

    # -------------------------------------------------------------- status

    def todo(self) -> List[str]:
        """Human-readable actions, most urgent first."""
        now = self._now()
        urgent, normal = [], []
        for e in self.entries:
            tag = f"[{e['id']}] {e['campaign_name']} · {os.path.basename(e['file'])}"
            if not e["posts"]:
                normal.append(f"{tag}: ready, not posted yet")
                continue
            for p in e["posts"]:
                where = f"{p['platform']} {p['account']}"
                if not p["submitted_at"]:
                    posted = datetime.fromisoformat(p["posted_at"])
                    minutes = (now - posted).total_seconds() / 60
                    limit = e.get("submit_within_minutes", 30)
                    if minutes > limit:
                        urgent.append(f"{tag}: SUBMIT NOW on {where} — posted "
                                      f"{minutes:.0f} min ago (limit {limit})")
                    else:
                        urgent.append(f"{tag}: submit on {where} within "
                                      f"{limit - minutes:.0f} min")
                elif e.get("analytics_after_views") and not p["analytics_recorded"]:
                    normal.append(f"{tag}: screen-record analytics on {where} once it "
                                  f"passes {e['analytics_after_views']:,} views")
        return urgent + normal
