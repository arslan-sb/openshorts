#!/usr/bin/env python3
"""Compare local models on hook/title/caption writing for a finished job.

Every model gets the SAME moments (the shortlist windows the job's clips came
from) and the exact detail prompt the pipeline sends, so the only thing that
changes is the writing. Run it in the backend container, where the repo is
/app and Ollama is reachable:

  docker exec openshorts-backend python local/bench_hooks.py \
      output/<job-id> gemma4:12b gpt-oss:20b --temps 0.2 0.7

Writes local/results/hooks-<job>.md (git-ignored) and prints a summary.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

import gemini_worker  # noqa: E402
import llm_backend  # noqa: E402
from clip_selection import build_transcript_windows, clip_count_targets  # noqa: E402

# The example hooks written in the detail prompt's HOOK PLAYBOOK. A model that
# returns one of these verbatim is copying, not writing.
PLAYBOOK = re.findall(r'^- [^:]+: "([^"]+)"', gemini_worker.DETAIL_PROMPT_TEMPLATE, re.M)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def load_job(job_dir):
    meta_path = glob.glob(os.path.join(job_dir, "*_metadata.json"))[0]
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def shortlist_from_job(meta):
    """Rebuild the job's windows and return the ones its clips came from."""
    transcript = meta["transcript"]
    duration = float(transcript["segments"][-1]["end"])
    wanted = {s.get("source_window_id") for s in meta["shorts"]}
    longest = max(float(s["end"]) - float(s["start"]) for s in meta["shorts"])
    # The pipeline sizes windows from CLIP_MAX_SECONDS; find the size that
    # reproduces the ids the job recorded (each clip inside its own window).
    for window_seconds in range(90, 271):
        windows = build_transcript_windows(transcript, duration, window_seconds=window_seconds)
        by_id = {w["id"]: w for w in windows}
        if all(s.get("source_window_id") in by_id
               and by_id[s["source_window_id"]]["start"] - 2 <= float(s["start"])
               and float(s["end"]) <= by_id[s["source_window_id"]]["end"] + 2
               for s in meta["shorts"]):
            max_secs = max(60.0, round(window_seconds / 1.5))
            shortlist = sorted((by_id[i] for i in wanted), key=lambda w: w["start"])
            return shortlist, duration, max_secs, transcript.get("language") or "en"
    raise SystemExit("could not rebuild this job's windows")


def detail_prompt(shortlist, duration, max_secs, language):
    lo, hi = clip_count_targets(len(shortlist))
    payload = [{"id": w["id"], "start": w["start"], "end": w["end"], "text": w["text"]}
               for w in shortlist]
    return gemini_worker.DETAIL_PROMPT_TEMPLATE.format(
        video_duration=duration, language=language, min_clips=lo, max_clips=hi,
        min_secs=15, max_secs=max_secs, windows_json=json.dumps(payload, ensure_ascii=False))


def ollama_root():
    return re.sub(r"/v1/?$", "", llm_backend.base_url())


def gpu_share(model):
    """Fraction of the loaded model that sits in VRAM (Ollama /api/ps)."""
    try:
        for m in httpx.get(f"{ollama_root()}/api/ps", timeout=10).json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return m.get("size_vram", 0) / max(m.get("size", 1), 1), m.get("size", 0)
    except Exception:
        pass
    return None, None


def unload(model):
    try:
        httpx.post(f"{ollama_root()}/api/generate", json={"model": model, "keep_alive": 0},
                   timeout=60)
    except Exception:
        pass


def run(model, prompt, temperature):
    for attempt in (1, 2):
        t0 = time.time()
        try:
            parsed, cost = llm_backend.generate_json(
                prompt, gemini_worker.DetailResponse, model=model, temperature=temperature,
                detail=True)
            return parsed["shorts"], time.time() - t0, cost, None
        except Exception as e:  # noqa: BLE001 - reported in the table
            err = f"{type(e).__name__}: {str(e)[:160]}"
            if attempt == 2:
                return [], time.time() - t0, None, err


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir")
    ap.add_argument("models", nargs="+")
    ap.add_argument("--temps", nargs="+", type=float, default=[0.2, 0.7])
    args = ap.parse_args()

    meta = load_job(args.job_dir)
    shortlist, duration, max_secs, language = shortlist_from_job(meta)
    prompt = detail_prompt(shortlist, duration, max_secs, language)
    print(f"{len(shortlist)} windows, clips up to {max_secs:g}s, prompt {len(prompt)} chars")

    rows = []   # (model, temp, seconds, gpu, shorts, err)
    for model in args.models:
        unload_others = [m for m in args.models if m != model]
        for other in unload_others:
            unload(other)
        for temp in args.temps:
            shorts, secs, cost, err = run(model, prompt, temp)
            gpu, size = gpu_share(model)
            out_toks = (cost or {}).get("output_tokens") or 0
            rows.append({"model": model, "temp": temp, "secs": secs, "gpu": gpu,
                         "size": size, "shorts": shorts, "err": err, "out_tokens": out_toks})
            print(f"  {model:28s} t={temp:<4g} {secs:6.1f}s  clips={len(shorts)}"
                  + (f"  GPU {gpu:.0%}" if gpu is not None else "")
                  + (f"  ERROR {err}" if err else ""))
        unload(model)

    job = os.path.basename(os.path.normpath(args.job_dir))
    os.makedirs(os.path.join(os.path.dirname(__file__), "results"), exist_ok=True)
    path = os.path.join(os.path.dirname(__file__), "results", f"hooks-{job[:8]}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report(meta, shortlist, rows))
    print(f"\nReport: {os.path.relpath(path)}")


def report(meta, shortlist, rows):
    playbook = {_norm(p) for p in PLAYBOOK}
    out = ["# Hook comparison", "",
           "Same moments and the same prompt for every model; only the writing differs.", "",
           "| Model | Temp | Time | In VRAM | Clips | Copied playbook hooks | Hooks > 10 words |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        hooks = [s.get("viral_hook_text", "") for s in r["shorts"]]
        copied = sum(_norm(h) in playbook for h in hooks)
        long_ = sum(len(h.split()) > 10 for h in hooks)
        vram = f"{r['gpu']:.0%}" if r["gpu"] is not None else "?"
        out.append(f"| {r['model']} | {r['temp']:g} | {r['secs']:.0f}s | {vram} | "
                   f"{len(r['shorts'])}{' ⚠️ ' + r['err'] if r['err'] else ''} | {copied} | {long_} |")
    original = {s.get("source_window_id"): s for s in meta["shorts"]}
    for w in shortlist:
        old = original.get(w["id"], {})
        out += ["", f"## {w['id']} ({w['start']:.0f}s–{w['end']:.0f}s)", "",
                f"> {w['text'][:400]}{'…' if len(w['text']) > 400 else ''}", "",
                "| Model | Temp | Hook | Title | TikTok caption |", "|---|---|---|---|---|",
                f"| *current job* (qwen2.5:14b) | 0.2 | {old.get('viral_hook_text', '')} | "
                f"{old.get('video_title_for_youtube_short', '')} | "
                f"{old.get('video_description_for_tiktok', '')} |"]
        for r in rows:
            for s in r["shorts"]:
                if s.get("source_window_id") == w["id"]:
                    cells = [s.get("viral_hook_text", ""), s.get("video_title_for_youtube_short", ""),
                             s.get("video_description_for_tiktok", "")]
                    cells = [c.replace("|", "/").replace("\n", " ") for c in cells]
                    out.append(f"| {r['model']} | {r['temp']:g} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
