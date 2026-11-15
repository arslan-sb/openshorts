"""Campaign rules + ledger (campaign_rules.py, campaign_ledger.py, campaign.py).

A clip that breaks the brief earns nothing, so these pin what is enforced:
length, banned words, captions that always carry the required tags, the
disclosure going to ffmpeg as a text FILE (never pasted into the filter
graph), and the ledger's one-post-one-campaign / submit-in-time rules.
"""
import json
import os
import shutil
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

import campaign
import campaign_ledger
import campaign_rules as cr

BASE = {
    "name": "Test Pod",
    "platforms": ["tiktok", "youtube"],
    "min_seconds": 15,
    "max_seconds": 40,
    "required_mentions": ["@creator"],
    "hashtags": ["podcast"],
    "disclosure": {"text": "Ad: it's \"paid\" 100%: x\\y", "seconds": 2},
    "banned_words": ["casino"],
}


def make(**overrides):
    raw = dict(BASE)
    raw.update(overrides)
    return cr.parse_campaign(raw)


def transcript(*words):
    """words: (text, start)."""
    ws = [{"word": " " + t, "start": s, "end": s + 0.4} for t, s in words]
    return {"segments": [{"start": ws[0]["start"], "end": ws[-1]["end"],
                          "text": "", "words": ws}]}


CLIP = {"start": 10.0, "end": 35.0, "video_title_for_youtube_short": "Big idea",
        "video_description_for_tiktok": "He explains the big idea",
        "video_description_for_instagram": "insta", "viral_hook_text": "Wait for it"}


# ---------------------------------------------------------------- parsing

def test_parse_normalises_hashtags_and_slug():
    c = make()
    assert c.hashtags == ["#podcast"]
    assert c.slug == "test-pod"
    assert c.disclosure.seconds == 2


@pytest.mark.parametrize("bad", [
    {"name": ""},
    {"platforms": ["myspace"]},
    {"min_seconds": 50, "max_seconds": 40},
    {"required_mentions": ["creator"]},
    {"audio": "loud"},
    {"disclosure": {"text": "x" * 61}},
    {"min_seconds": "15"},
])
def test_parse_rejects_bad_campaigns(bad):
    with pytest.raises(cr.CampaignError):
        make(**bad)


def test_example_campaign_file_is_valid():
    c = cr.load_campaign(os.path.join(campaign.CAMPAIGNS_DIR, "example.json"))
    assert c.platforms


# ---------------------------------------------------------------- checks

def test_too_short_is_an_error_too_long_is_a_fix():
    c = make()
    short = dict(CLIP, end=20.0)
    assert any(i.level == "error" for i in cr.check_clip(c, short, None))
    long_ = dict(CLIP, end=70.0)
    issues = cr.check_clip(c, long_, None)
    assert not any(i.level == "error" for i in issues)
    assert any("trimmed to 40s" in i.message for i in issues)


def test_banned_word_in_the_clip_transcript_only():
    c = make()
    inside = transcript(("best", 12.0), ("casino", 13.0))
    assert any("casino" in i.message for i in cr.check_clip(c, CLIP, inside)
               if i.level == "error")
    # Said outside the clip's span: not this clip's problem.
    outside = transcript(("casino", 2.0), ("fine", 12.0))
    assert not any(i.level == "error" for i in cr.check_clip(c, CLIP, outside))


def test_banned_word_matches_whole_words():
    c = make(banned_words=["bet"])
    t = transcript(("better", 12.0), ("alphabet", 13.0))
    assert not any(i.level == "error" for i in cr.check_clip(c, CLIP, t))


def test_banned_word_in_title():
    c = make()
    clip = dict(CLIP, video_title_for_youtube_short="Casino night")
    assert any(i.level == "error" for i in cr.check_clip(c, clip, None))


# ---------------------------------------------------------------- captions

def test_caption_appends_tags_once():
    c = make()
    assert cr.build_caption(c, CLIP, "tiktok") == "He explains the big idea @creator #podcast"
    clip = dict(CLIP, video_description_for_tiktok="Great #Podcast moment")
    assert cr.build_caption(c, clip, "tiktok") == "Great #Podcast moment @creator"


def test_caption_shortens_text_never_tags():
    c = make()
    clip = dict(CLIP, video_title_for_youtube_short="word " * 60)
    caption = cr.build_caption(c, clip, "youtube")
    assert len(caption) <= cr.CAPTION_LIMITS["youtube"]
    assert caption.endswith("@creator #podcast")
    assert "…" in caption


def test_caption_fails_when_tags_alone_do_not_fit():
    c = make(hashtags=["#" + "a" * 120])
    with pytest.raises(cr.CampaignError):
        cr.build_caption(c, CLIP, "youtube")


# ---------------------------------------------------------------- ffmpeg

def test_disclosure_text_never_enters_the_filter_graph(tmp_path):
    c = make()
    textfile = str(tmp_path / "disclosure.txt")
    cmd = cr.build_ffmpeg_cmd(c, "in.mp4", "out.mp4", 25.0, textfile)
    vf = cmd[cmd.index("-vf") + 1]
    assert f"textfile={textfile}" in vf
    assert "paid" not in vf and "100%" not in vf
    # Without it drawtext reads "%" in the text as an expansion and fails.
    assert ":expansion=none" in vf


def test_trim_and_mute_flags():
    c = make(disclosure=None, audio="mute")
    cmd = cr.build_ffmpeg_cmd(c, "in.mp4", "out.mp4", 55.0, None)
    assert cmd[cmd.index("-t") + 1] == "40.000"
    assert "-an" in cmd and "-vf" not in cmd
    cmd = cr.build_ffmpeg_cmd(c, "in.mp4", "out.mp4", 30.0, None)
    assert "-t" not in cmd


def test_unsafe_filter_path_is_refused():
    c = make()
    with pytest.raises(cr.CampaignError):
        cr.build_ffmpeg_cmd(c, "in.mp4", "out.mp4", 25.0, "/tmp/a:b/x.txt")


def _fake_job(job_dir, clips, durations):
    os.makedirs(job_dir)
    meta = {"shorts": clips, "transcript": transcript(("hello", 11.0), ("casino", 52.0))}
    with open(os.path.join(job_dir, "My_Video_metadata.json"), "w") as f:
        json.dump(meta, f)
    for i, seconds in enumerate(durations):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                        f"color=c=blue:s=180x320:d={seconds}", "-f", "lavfi", "-i",
                        f"sine=f=440:d={seconds}", "-shortest", "-c:v", "libx264",
                        "-c:a", "aac",
                        os.path.join(job_dir, f"My_Video_clip_{i + 1}.mp4")], check=True)


def _probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(out.stdout)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_prepare_job_end_to_end(tmp_path):
    c = make(max_seconds=20, min_seconds=5)
    job = str(tmp_path / "job1")
    clips = [dict(CLIP, start=10.0, end=36.0),   # too long -> trimmed
             dict(CLIP, start=50.0, end=60.0),   # says "casino" -> rejected
             dict(CLIP, start=70.0, end=72.0)]   # too short -> rejected
    _fake_job(job, clips, [26, 10, 2])
    ledger = campaign_ledger.Ledger(str(tmp_path / "ledger.json"))
    manifest = cr.prepare_job(c, job, str(tmp_path / "out"), ledger=ledger)

    status = [x["status"] for x in manifest["clips"]]
    assert status == ["ready", "rejected", "rejected"]
    ready = os.path.join(str(tmp_path / "out"), manifest["clips"][0]["file"])
    assert _probe_duration(ready) == pytest.approx(20, abs=0.3)
    out_dir = manifest["out_dir"]
    assert open(os.path.join(out_dir, "clip_1.tiktok.txt")).read().strip().endswith(
        "@creator #podcast")
    report = open(os.path.join(out_dir, "report.md")).read()
    assert "1 of 3 clips are ready" in report and "casino" in report
    assert manifest["clips"][0]["ledger_id"] == ledger.entries[0]["id"]


# ---------------------------------------------------------------- ledger

class Clock:
    def __init__(self):
        self.t = datetime(2026, 11, 15, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t


@pytest.fixture
def clip_file(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"same bytes")
    return str(p)


def test_same_clip_in_two_campaigns_warns(tmp_path, clip_file):
    ledger = campaign_ledger.Ledger(str(tmp_path / "l.json"))
    first, warn = ledger.add_prepared(make(), clip_file)
    assert warn is None
    again, warn = ledger.add_prepared(make(), clip_file)
    assert again == first and warn is None          # same campaign: same entry
    _, warn = ledger.add_prepared(make(name="Other"), clip_file)
    assert "test-pod" in warn


def test_submit_deadline_and_second_account(tmp_path, clip_file):
    clock = Clock()
    ledger = campaign_ledger.Ledger(str(tmp_path / "l.json"), now=clock)
    entry_id, _ = ledger.add_prepared(make(), clip_file)
    assert "not posted yet" in ledger.todo()[0]

    ledger.mark_posted(entry_id, "tiktok", "@me", "https://t.example/1")
    clock.t += timedelta(minutes=10)
    assert "within 20 min" in ledger.todo()[0]
    clock.t += timedelta(minutes=25)
    assert "SUBMIT NOW" in ledger.todo()[0]

    with pytest.raises(campaign_ledger.LedgerError):
        ledger.mark_posted(entry_id, "tiktok", "@me", "https://t.example/1")
    _, warnings = ledger.mark_posted(entry_id, "tiktok", "@alt")
    assert warnings and "several accounts" in warnings[0]

    ledger.mark_submitted(entry_id)
    ledger.save()
    reloaded = campaign_ledger.Ledger(str(tmp_path / "l.json"), now=clock)
    # First post still unsubmitted, second one submitted.
    assert sum("SUBMIT NOW" in line for line in reloaded.todo()) == 1


def test_analytics_reminder(tmp_path, clip_file):
    ledger = campaign_ledger.Ledger(str(tmp_path / "l.json"))
    entry_id, _ = ledger.add_prepared(make(analytics_recording_after_views=1000), clip_file)
    ledger.mark_posted(entry_id, "youtube", "@me")
    ledger.mark_submitted(entry_id)
    assert "screen-record" in ledger.todo()[0]
    ledger.mark_analytics(entry_id)
    assert ledger.todo() == []


def test_cli_new_and_check(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(campaign, "CAMPAIGNS_DIR", str(tmp_path))
    shutil.copy(os.path.join(os.path.dirname(campaign.TEMPLATE), "example.json"),
                tmp_path / "example.json")
    monkeypatch.setattr(campaign, "TEMPLATE", str(tmp_path / "example.json"))
    campaign.main(["new", "My Brand Deal"])
    assert (tmp_path / "my-brand-deal.json").exists()
    campaign.main(["check", "my brand deal"])
    assert "My Brand Deal" in capsys.readouterr().out
    campaign.main(["env", "my-brand-deal"])
    assert "CLIP_MAX_SECONDS=60" in capsys.readouterr().out
