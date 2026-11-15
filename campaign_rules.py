"""Campaign rules for paid clipping (Content Rewards and similar marketplaces).

OpenShorts picks and renders clips; it knows nothing about the brief of the
campaign you are clipping for. This module sits after a finished job:

    campaign file (JSON)  +  output/<job>/  ->  campaign_out/<campaign>/<job>/
                                                  clip_N.mp4          (brief applied)
                                                  clip_N.<platform>.txt (caption)
                                                  report.md           (what passed, what not)

What it enforces, per clip:

- length: too short is rejected (it cannot be fixed); too long is trimmed to
  ``max_seconds``;
- banned words: in the clip's own transcript, title or descriptions;
- ad disclosure: burned into the first seconds of the clip;
- audio: kept, or removed when the brief asks you to add its sound in-app;
- caption: the AI's description plus the required @mentions and hashtags,
  cut to each platform's limit without ever dropping a required tag.

The manual checklist (``checklist`` in the campaign file) is printed into the
report, because some rules can only be checked by a person.

Text from the campaign file and from clip metadata is data: the disclosure is
handed to ffmpeg through ``drawtext=textfile=``, never pasted into the filter
graph (see the "Hard rules" in CLAUDE.md).
"""
from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from jsonfile import write_json_atomic

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_FILE = os.path.join(REPO_DIR, "fonts", "Montserrat-ExtraBold.ttf")

PLATFORMS = ("tiktok", "instagram", "youtube", "x")

# Caption limits per platform (characters). YouTube's is the Shorts title.
CAPTION_LIMITS = {"tiktok": 2200, "instagram": 2200, "youtube": 100, "x": 280}

# Which AI-written field seeds the caption on each platform.
CAPTION_SOURCE = {
    "tiktok": "video_description_for_tiktok",
    "instagram": "video_description_for_instagram",
    "youtube": "video_title_for_youtube_short",
    "x": "video_description_for_tiktok",
}

_SAFE_PATH = re.compile(r"^[\w./-]+$")


class CampaignError(ValueError):
    """The campaign file is unusable as written."""


@dataclass
class Disclosure:
    text: str
    seconds: float = 3.0
    position: str = "top"  # top | bottom


@dataclass
class Campaign:
    name: str
    slug: str
    platforms: List[str]
    min_seconds: float = 15.0
    max_seconds: float = 60.0
    required_mentions: List[str] = field(default_factory=list)
    hashtags: List[str] = field(default_factory=list)
    disclosure: Optional[Disclosure] = None
    audio: str = "original"  # original | mute
    banned_words: List[str] = field(default_factory=list)
    one_account_per_clip: bool = True
    submit_within_minutes: int = 30
    analytics_recording_after_views: int = 0  # 0 = the brief does not ask
    rate_per_1k: Optional[float] = None
    max_payout_per_clip: Optional[float] = None
    brief_url: str = ""
    checklist: List[str] = field(default_factory=list)


@dataclass
class Issue:
    level: str  # "error" (clip is skipped) | "fix" (applied) | "warn"
    message: str


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:60] or "campaign"


def _str_list(raw, key) -> List[str]:
    value = raw.get(key) or []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise CampaignError(f"'{key}' must be a list of strings")
    return [v.strip() for v in value if v.strip()]


def _number(raw, key, default, lo, hi) -> float:
    value = raw.get(key, default)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignError(f"'{key}' must be a number")
    if not lo <= value <= hi:
        raise CampaignError(f"'{key}' must be between {lo} and {hi}")
    return float(value)


def parse_campaign(raw: dict) -> Campaign:
    """Validate a campaign dict (the JSON file's content) into a Campaign."""
    if not isinstance(raw, dict):
        raise CampaignError("campaign file must hold a JSON object")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise CampaignError("'name' is required")

    platforms = [p.lower() for p in _str_list(raw, "platforms")] or ["tiktok"]
    unknown = [p for p in platforms if p not in PLATFORMS]
    if unknown:
        raise CampaignError(f"unknown platform(s) {unknown}; use {list(PLATFORMS)}")

    min_s = _number(raw, "min_seconds", 15, 1, 600)
    max_s = _number(raw, "max_seconds", 60, 2, 600)
    if max_s <= min_s:
        raise CampaignError("'max_seconds' must be greater than 'min_seconds'")

    mentions = _str_list(raw, "required_mentions")
    bad = [m for m in mentions if not m.startswith("@") or " " in m]
    if bad:
        raise CampaignError(f"required_mentions must look like @handle: {bad}")
    hashtags = [h if h.startswith("#") else f"#{h}" for h in _str_list(raw, "hashtags")]
    if any(" " in h for h in hashtags):
        raise CampaignError("hashtags cannot contain spaces")

    disclosure = None
    d = raw.get("disclosure")
    if d:
        if isinstance(d, str):
            d = {"text": d}
        if not isinstance(d, dict) or not str(d.get("text") or "").strip():
            raise CampaignError("'disclosure' needs a 'text'")
        position = str(d.get("position") or "top").lower()
        if position not in ("top", "bottom"):
            raise CampaignError("disclosure position must be 'top' or 'bottom'")
        text = str(d["text"]).strip()
        if len(text) > 60:
            raise CampaignError("disclosure text must be 60 characters or fewer")
        disclosure = Disclosure(text=text,
                                seconds=_number(d, "seconds", 3, 0.5, 30),
                                position=position)

    audio = str(raw.get("audio") or "original").lower()
    if audio not in ("original", "mute"):
        raise CampaignError("'audio' must be 'original' or 'mute'")

    return Campaign(
        name=name,
        slug=slugify(raw.get("slug") or name),
        platforms=platforms,
        min_seconds=min_s,
        max_seconds=max_s,
        required_mentions=mentions,
        hashtags=hashtags,
        disclosure=disclosure,
        audio=audio,
        banned_words=[w.lower() for w in _str_list(raw, "banned_words")],
        one_account_per_clip=bool(raw.get("one_account_per_clip", True)),
        submit_within_minutes=int(_number(raw, "submit_within_minutes", 30, 1, 1440)),
        analytics_recording_after_views=int(
            _number(raw, "analytics_recording_after_views", 0, 0, 10_000_000)),
        rate_per_1k=_number(raw, "rate_per_1k", None, 0, 1000),
        max_payout_per_clip=_number(raw, "max_payout_per_clip", None, 0, 1_000_000),
        brief_url=str(raw.get("brief_url") or ""),
        checklist=_str_list(raw, "checklist"),
    )


def load_campaign(path: str) -> Campaign:
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise CampaignError(f"{path}: not valid JSON ({e})") from e
    return parse_campaign(raw)


# ---------------------------------------------------------------- per clip

def clip_transcript_text(transcript: Optional[dict], start: float, end: float) -> str:
    """The words spoken inside [start, end) of the source."""
    words = []
    for seg in (transcript or {}).get("segments") or []:
        if seg.get("end", 0) <= start or seg.get("start", 0) >= end:
            continue
        seg_words = seg.get("words") or []
        if seg_words:
            words += [w.get("word", "") for w in seg_words
                      if start <= w.get("start", 0) < end]
        else:
            words.append(" " + seg.get("text", ""))
    return "".join(words).strip()


def _banned_hits(campaign: Campaign, texts: List[str]) -> List[str]:
    hay = " ".join(t for t in texts if t).lower()
    hits = []
    for word in campaign.banned_words:
        if re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", hay):
            hits.append(word)
    return hits


def check_clip(campaign: Campaign, clip: dict, transcript: Optional[dict]) -> List[Issue]:
    """What the brief says about this clip. Errors mean: do not post it."""
    issues: List[Issue] = []
    duration = float(clip["end"]) - float(clip["start"])
    if duration < campaign.min_seconds:
        issues.append(Issue("error", f"{duration:.1f}s is shorter than the "
                                     f"{campaign.min_seconds:g}s minimum"))
    elif duration > campaign.max_seconds:
        issues.append(Issue("fix", f"{duration:.1f}s trimmed to "
                                   f"{campaign.max_seconds:g}s"))

    spoken = clip_transcript_text(transcript, float(clip["start"]), float(clip["end"]))
    written = [clip.get(k, "") for k in set(CAPTION_SOURCE.values())] + [
        clip.get("viral_hook_text", "")]
    hits = _banned_hits(campaign, [spoken] + written)
    if hits:
        issues.append(Issue("error", "contains banned word(s): " + ", ".join(hits)))

    if campaign.disclosure:
        issues.append(Issue("fix", f"disclosure \"{campaign.disclosure.text}\" burned "
                                   f"into the first {campaign.disclosure.seconds:g}s"))
    if campaign.audio == "mute":
        issues.append(Issue("fix", "original audio removed: add the campaign's "
                                   "sound in the app before posting"))
    if transcript is None or not (transcript.get("segments") or []):
        issues.append(Issue("warn", "no transcript: banned words were only checked "
                                    "in the title and descriptions"))
    return issues


def build_caption(campaign: Campaign, clip: dict, platform: str) -> str:
    """AI caption + required mentions + hashtags, inside the platform limit.

    The tags are never cut: when the text does not fit, the AI part is
    shortened (on a word boundary) to make room for them.
    """
    limit = CAPTION_LIMITS[platform]
    base = " ".join(str(clip.get(CAPTION_SOURCE[platform]) or "").split())
    present = base.lower()
    tags = [t for t in campaign.required_mentions + campaign.hashtags
            if t.lower() not in present]
    suffix = " ".join(tags)
    room = limit - (len(suffix) + 1 if suffix else 0)
    if room < 0:
        raise CampaignError(f"required mentions and hashtags alone exceed the "
                            f"{limit}-character {platform} limit")
    if len(base) > room:
        cut = base[:max(room - 1, 0)]
        base = (cut.rsplit(" ", 1)[0] if " " in cut else cut).rstrip(" ,.;:-") + "…"
        if len(base) > room:
            base = ""
    return " ".join(p for p in (base, suffix) if p)


# ---------------------------------------------------------------- rendering

def _ffmpeg_path_arg(path: str) -> str:
    """A path safe to place in an ffmpeg filter option without escaping."""
    if not _SAFE_PATH.match(path):
        raise CampaignError(f"unsafe path for ffmpeg filter: {path!r}")
    return path


def build_ffmpeg_cmd(campaign: Campaign, src: str, dst: str, duration: float,
                     textfile: Optional[str]) -> List[str]:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", src]
    if duration > campaign.max_seconds:
        cmd += ["-t", f"{campaign.max_seconds:.3f}"]
    if campaign.disclosure:
        if not textfile:
            raise CampaignError("disclosure needs its text file")
        d = campaign.disclosure
        y = "h*0.06" if d.position == "top" else "h*0.86"
        vf = (f"drawtext=fontfile={_ffmpeg_path_arg(FONT_FILE)}"
              f":textfile={_ffmpeg_path_arg(textfile)}:expansion=none"
              f":fontsize=h*0.032:fontcolor=white:box=1:boxcolor=black@0.55"
              f":boxborderw=18:x=(w-text_w)/2:y={y}"
              f":enable=lt(t\\,{d.seconds:.3f})")
        cmd += ["-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p"]
    else:
        cmd += ["-c:v", "copy"]
    if campaign.audio == "mute":
        cmd += ["-an"]
    else:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    cmd += ["-movflags", "+faststart", dst]
    return cmd


def render_for_campaign(campaign: Campaign, src: str, dst: str, duration: float) -> None:
    tmp_dir = tempfile.mkdtemp(prefix="campaign-")
    try:
        textfile = None
        if campaign.disclosure:
            textfile = os.path.join(tmp_dir, "disclosure.txt")
            with open(textfile, "w", encoding="utf-8") as f:
                f.write(campaign.disclosure.text)
        cmd = build_ffmpeg_cmd(campaign, src, dst, duration, textfile)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------- a whole job

def find_metadata(job_dir: str) -> str:
    found = sorted(glob.glob(os.path.join(job_dir, "*_metadata.json")))
    if not found:
        raise FileNotFoundError(f"no *_metadata.json in {job_dir}: is the job finished?")
    return found[0]


def final_clip_file(job_dir: str, base_name: str, index: int) -> Optional[str]:
    """The clip OpenShorts would serve: the newest captioned/hooked/recut
    version, else the clean reframe. Watermarked copies are never used."""
    clean = f"{base_name}_clip_{index + 1}.mp4"
    derived = []
    for pattern in (f"subtitled_*_{clean}", f"recut_*_{clean}",
                    f"hooked_*_{clean}", f"hook_{clean}"):
        derived += glob.glob(os.path.join(job_dir, pattern))
    if derived:
        return max(derived, key=os.path.getmtime)
    path = os.path.join(job_dir, clean)
    return path if os.path.exists(path) else None


def prepare_job(campaign: Campaign, job_dir: str, out_root: str,
                ledger=None, render=render_for_campaign) -> dict:
    """Apply the campaign to every clip of a finished job.

    Returns the manifest that is also written next to the clips.
    """
    meta_path = find_metadata(job_dir)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    base_name = os.path.basename(meta_path)[: -len("_metadata.json")]
    transcript = meta.get("transcript")
    job_id = os.path.basename(os.path.normpath(job_dir))
    out_dir = os.path.join(out_root, campaign.slug, job_id)
    os.makedirs(out_dir, exist_ok=True)

    clips_out = []
    for i, clip in enumerate(meta.get("shorts") or []):
        entry: Dict = {"index": i + 1,
                       "title": clip.get("video_title_for_youtube_short", ""),
                       "score": clip.get("predicted_score")}
        src = final_clip_file(job_dir, base_name, i)
        if not src:
            entry.update(status="missing", issues=[{"level": "error",
                         "message": "clip file was never rendered"}])
            clips_out.append(entry)
            continue
        issues = check_clip(campaign, clip, transcript)
        entry["source"] = os.path.relpath(src, job_dir)
        entry["issues"] = [vars(x) for x in issues]
        if any(x.level == "error" for x in issues):
            entry["status"] = "rejected"
            clips_out.append(entry)
            continue

        dst = os.path.join(out_dir, f"clip_{i + 1}.mp4")
        render(campaign, src, dst, float(clip["end"]) - float(clip["start"]))
        entry["file"] = os.path.relpath(dst, out_root)
        entry["captions"] = {}
        for platform in campaign.platforms:
            caption = build_caption(campaign, clip, platform)
            with open(os.path.join(out_dir, f"clip_{i + 1}.{platform}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(caption + "\n")
            entry["captions"][platform] = caption
        entry["status"] = "ready"
        if ledger is not None:
            entry["ledger_id"], warning = ledger.add_prepared(campaign, dst)
            if warning:
                entry["issues"].append({"level": "warn", "message": warning})
        clips_out.append(entry)

    manifest = {"campaign": campaign.name, "slug": campaign.slug, "job": job_id,
                "source_metadata": os.path.basename(meta_path), "clips": clips_out}
    write_json_atomic(os.path.join(out_dir, "manifest.json"), manifest, indent=2)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(campaign, manifest))
    manifest["out_dir"] = out_dir
    return manifest


def render_report(campaign: Campaign, manifest: dict) -> str:
    ready = [c for c in manifest["clips"] if c.get("status") == "ready"]
    lines = [f"# {campaign.name} — job {manifest['job']}", ""]
    if campaign.brief_url:
        lines += [f"Brief: {campaign.brief_url}", ""]
    if campaign.rate_per_1k:
        lines += [f"Rate: ${campaign.rate_per_1k:g} per 1,000 views"
                  + (f", max ${campaign.max_payout_per_clip:g} per clip"
                     if campaign.max_payout_per_clip else ""), ""]
    lines += [f"**{len(ready)} of {len(manifest['clips'])} clips are ready to post.**", ""]
    for c in manifest["clips"]:
        mark = {"ready": "✅", "rejected": "⛔", "missing": "❔"}.get(c["status"], "•")
        lines.append(f"## {mark} Clip {c['index']}: {c.get('title') or '(untitled)'}")
        if c.get("ledger_id"):
            lines.append(f"Ledger id: `{c['ledger_id']}`")
        for issue in c.get("issues", []):
            lines.append(f"- **{issue['level']}**: {issue['message']}")
        for platform, caption in (c.get("captions") or {}).items():
            lines += ["", f"{platform}:", "", "```", caption, "```"]
        lines.append("")
    lines += ["## Before you post each clip", ""]
    rules = [f"Post on: {', '.join(campaign.platforms)}",
             f"Submit the post link within {campaign.submit_within_minutes} minutes "
             f"of posting, then run `python campaign.py submitted <ledger id>`"]
    if campaign.one_account_per_clip:
        rules.append("Post each clip on ONE account only, for ONE campaign only")
    if campaign.analytics_recording_after_views:
        rules.append(f"Screen-record the clip's own analytics after "
                     f"{campaign.analytics_recording_after_views:,} views")
    rules.append("Never buy views, likes or followers (permanent ban)")
    lines += [f"- [ ] {r}" for r in rules + campaign.checklist]
    return "\n".join(lines) + "\n"
