#!/usr/bin/env python3
"""Campaign tools for paid clipping. Run `python campaign.py -h`.

  new NAME                       create campaigns/<name>.json from the template
  check CAMPAIGN                 validate a campaign file
  env CAMPAIGN                   clip-length settings to use for the OpenShorts job
  prepare CAMPAIGN JOB_DIR       apply the brief to a finished job's clips
  posted ID PLATFORM ACCOUNT [URL]
  submitted ID [--platform P]
  analytics ID [--platform P]
  status                         what to post, submit or record next

CAMPAIGN is a path (campaigns/foo.json) or just the name (foo).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import campaign_ledger
import campaign_rules

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
CAMPAIGNS_DIR = os.path.join(REPO_DIR, "campaigns")
TEMPLATE = os.path.join(CAMPAIGNS_DIR, "example.json")
OUT_ROOT = os.path.join(REPO_DIR, "campaign_out")


def resolve_campaign_path(value: str) -> str:
    if os.path.exists(value):
        return value
    candidate = os.path.join(CAMPAIGNS_DIR, f"{campaign_rules.slugify(value)}.json")
    if os.path.exists(candidate):
        return candidate
    raise SystemExit(f"campaign not found: {value} (looked for {candidate})")


def cmd_new(args):
    path = os.path.join(CAMPAIGNS_DIR, f"{campaign_rules.slugify(args.name)}.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists")
    with open(TEMPLATE, encoding="utf-8") as f:
        raw = json.load(f)
    raw["name"] = args.name
    raw.pop("slug", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Created {os.path.relpath(path, REPO_DIR)}: fill it in from the campaign brief.")


def cmd_check(args):
    c = campaign_rules.load_campaign(resolve_campaign_path(args.campaign))
    print(f"✅ {c.name} ({c.slug})")
    print(f"   platforms: {', '.join(c.platforms)} · length {c.min_seconds:g}-{c.max_seconds:g}s"
          f" · audio: {c.audio}")
    if c.disclosure:
        print(f"   disclosure: \"{c.disclosure.text}\" for {c.disclosure.seconds:g}s "
              f"({c.disclosure.position})")
    print(f"   tags: {' '.join(c.required_mentions + c.hashtags) or '(none)'}")
    for platform in c.platforms:
        sample = campaign_rules.build_caption(
            c, {campaign_rules.CAPTION_SOURCE[platform]: "Example caption"}, platform)
        print(f"   {platform} caption looks like: {sample}")


def cmd_env(args):
    c = campaign_rules.load_campaign(resolve_campaign_path(args.campaign))
    # Ask the picker for clips that already fit: a too-short clip is lost.
    print(f"CLIP_MIN_SECONDS={c.min_seconds:g}")
    print(f"CLIP_MAX_SECONDS={c.max_seconds:g}")


def cmd_prepare(args):
    c = campaign_rules.load_campaign(resolve_campaign_path(args.campaign))
    ledger = campaign_ledger.Ledger()
    manifest = campaign_rules.prepare_job(c, args.job_dir, OUT_ROOT, ledger=ledger)
    ledger.save()
    for clip in manifest["clips"]:
        mark = {"ready": "✅", "rejected": "⛔", "missing": "❔"}.get(clip["status"], "•")
        line = f"{mark} clip {clip['index']}: {clip['status']}"
        if clip.get("ledger_id"):
            line += f" (ledger id {clip['ledger_id']})"
        print(line)
        for issue in clip.get("issues", []):
            if issue["level"] != "fix":
                print(f"     {issue['level']}: {issue['message']}")
    print(f"\nOutput: {os.path.relpath(manifest['out_dir'], REPO_DIR)}/ "
          f"(open report.md for captions and the checklist)")


def _ledger_cmd(fn):
    def run(args):
        ledger = campaign_ledger.Ledger()
        try:
            fn(ledger, args)
        except campaign_ledger.LedgerError as e:
            raise SystemExit(f"❌ {e}")
        ledger.save()
    return run


@_ledger_cmd
def cmd_posted(ledger, args):
    if args.platform not in campaign_rules.PLATFORMS:
        raise campaign_ledger.LedgerError(f"platform must be one of {campaign_rules.PLATFORMS}")
    post, warnings = ledger.mark_posted(args.id, args.platform, args.account, args.url or "")
    for w in warnings:
        print(f"⚠️  {w}")
    entry = ledger.get(args.id)
    print(f"Recorded. Submit the link on the campaign within "
          f"{entry['submit_within_minutes']} minutes, then: "
          f"python campaign.py submitted {args.id}")


@_ledger_cmd
def cmd_submitted(ledger, args):
    ledger.mark_submitted(args.id, args.platform)
    print("Recorded as submitted.")


@_ledger_cmd
def cmd_analytics(ledger, args):
    ledger.mark_analytics(args.id, args.platform)
    print("Recorded: analytics screen recording done.")


def cmd_status(args):
    todo = campaign_ledger.Ledger().todo()
    print("\n".join(todo) if todo else "Nothing to do: every clip is posted and submitted.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new"); p.add_argument("name"); p.set_defaults(fn=cmd_new)
    p = sub.add_parser("check"); p.add_argument("campaign"); p.set_defaults(fn=cmd_check)
    p = sub.add_parser("env"); p.add_argument("campaign"); p.set_defaults(fn=cmd_env)
    p = sub.add_parser("prepare"); p.add_argument("campaign"); p.add_argument("job_dir")
    p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("posted"); p.add_argument("id"); p.add_argument("platform")
    p.add_argument("account"); p.add_argument("url", nargs="?"); p.set_defaults(fn=cmd_posted)
    for name, fn in (("submitted", cmd_submitted), ("analytics", cmd_analytics)):
        p = sub.add_parser(name); p.add_argument("id"); p.add_argument("--platform")
        p.set_defaults(fn=fn)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except campaign_rules.CampaignError as e:
        raise SystemExit(f"❌ campaign file: {e}")


if __name__ == "__main__":
    sys.exit(main())
