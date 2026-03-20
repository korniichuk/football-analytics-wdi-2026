"""Shared helpers for Germany national-team tactical notebooks (Sportmonks Football API v3)."""

from __future__ import annotations

import time
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

BASE = "https://api.sportmonks.com/v3/football"

# Euro 2024, 2025 UEFA Nations League, WC Qualification Europe
LEAGUE_IDS: frozenset[int] = frozenset({1326, 1538, 720})

# Sportmonks: full-time finished fixture (verify against your subscription if needed)
FINISHED_STATE_ID = 5

DEFAULT_PER_PAGE = 50


def date_chunks(start: date, end: date, max_days: int = 100) -> list[tuple[date, date]]:
    """Split [start, end] into windows of at most max_days (Sportmonks between-endpoint limit)."""
    chunks: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=max_days - 1), end)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return chunks


def _get_json(token: str, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p: dict[str, Any] = dict(params or {})
    p["api_token"] = token
    r = requests.get(url, params=p, timeout=90)
    if r.status_code == 429:
        time.sleep(2.5)
        r = requests.get(url, params=p, timeout=90)
    r.raise_for_status()
    return r.json()


def fetch_fixtures_between_team(
    token: str,
    team_id: int,
    start_date: str,
    end_date: str,
    *,
    sleep_s: float = 0.12,
) -> list[dict[str, Any]]:
    """Paginate GET /fixtures/between/{start}/{end}/{team_id}."""
    url = f"{BASE}/fixtures/between/{start_date}/{end_date}/{team_id}"
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        data = _get_json(token, url, {"per_page": DEFAULT_PER_PAGE, "page": page})
        batch = data.get("data") or []
        out.extend(batch)
        pag = data.get("pagination") or {}
        if not pag.get("has_more"):
            break
        page += 1
        time.sleep(sleep_s)
    return out


def search_teams(token: str, query: str, *, sleep_s: float = 0.12) -> list[dict[str, Any]]:
    """GET /teams/search/{query} (first page; extend if you hit pagination)."""
    url = f"{BASE}/teams/search/{query}"
    time.sleep(sleep_s)
    data = _get_json(token, url, {"per_page": DEFAULT_PER_PAGE, "page": 1})
    return list(data.get("data") or [])


def get_fixture(
    token: str,
    fixture_id: int,
    include: str | None = None,
    *,
    sleep_s: float = 0.1,
) -> dict[str, Any]:
    url = f"{BASE}/fixtures/{fixture_id}"
    params: dict[str, Any] = {}
    if include:
        params["include"] = include
    time.sleep(sleep_s)
    data = _get_json(token, url, params)
    return data["data"]


def get_fixture_with_include_fallback(
    token: str,
    fixture_id: int,
    include_candidates: list[str],
    *,
    sleep_s: float = 0.1,
) -> tuple[dict[str, Any], str]:
    """Try `include` strings in order; on **403** (plan / subscription) try the next candidate.

    Use this when optional includes such as ``xGFixture`` or ``participants`` may not be on your plan.
    """
    last_err: BaseException | None = None
    for inc in include_candidates:
        try:
            data = get_fixture(token, fixture_id, inc, sleep_s=sleep_s)
            return data, inc
        except requests.HTTPError as e:
            resp = getattr(e, "response", None)
            if resp is not None and resp.status_code == 403:
                last_err = e
                continue
            raise
    if last_err is not None:
        raise last_err
    raise RuntimeError("include_candidates must be non-empty")


def normalize_fixture_statistics(data: dict[str, Any]) -> pd.DataFrame:
    stats = data.get("statistics")
    if not stats:
        return pd.DataFrame()
    return pd.json_normalize(stats)


def opponent_participant_id(stats_df: pd.DataFrame, team_id: int) -> int | None:
    if stats_df.empty or "participant_id" not in stats_df.columns:
        return None
    ids = pd.unique(stats_df["participant_id"].dropna())
    others = [int(i) for i in ids if int(i) != int(team_id)]
    return others[0] if len(others) == 1 else None


def stats_dict_for_team(stats_df: pd.DataFrame, team_id: int) -> dict[str, float]:
    """Map type.code -> numeric value for one participant (last row wins if duplicates)."""
    if stats_df.empty:
        return {}
    col_part = "participant_id"
    col_val = "data.value"
    code_col = "type.code" if "type.code" in stats_df.columns else None
    dev_col = "type.developer_name" if "type.developer_name" in stats_df.columns else None
    if col_part not in stats_df.columns or col_val not in stats_df.columns:
        return {}
    m = stats_df[stats_df[col_part] == team_id]
    out: dict[str, float] = {}
    for _, row in m.iterrows():
        key = None
        if code_col and pd.notna(row.get(code_col)):
            key = str(row[code_col])
        elif dev_col and pd.notna(row.get(dev_col)):
            key = str(row[dev_col])
        if not key:
            continue
        v = row[col_val]
        try:
            out[key] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def filter_germany_corpus(
    fixtures: list[dict[str, Any]],
    *,
    league_ids: frozenset[int] | None = None,
    years: tuple[int, ...] = (2024, 2025),
    finished_only: bool = True,
) -> list[dict[str, Any]]:
    league_ids = league_ids or LEAGUE_IDS
    kept: list[dict[str, Any]] = []
    for fx in fixtures:
        if fx.get("league_id") not in league_ids:
            continue
        started = fx.get("starting_at")
        if not started:
            continue
        try:
            y = int(str(started)[:4])
        except ValueError:
            continue
        if y not in years:
            continue
        if finished_only and fx.get("state_id") != FINISHED_STATE_ID:
            continue
        kept.append(fx)
    return kept


def mean_pressure_for_participant(data: dict[str, Any], team_id: int) -> float | None:
    """Average pressure index for a participant when `pressure` include is present."""
    raw = data.get("pressure")
    if not raw:
        return None
    df = pd.json_normalize(raw)
    if df.empty:
        return None
    if "participant_id" in df.columns:
        df = df[df["participant_id"] == team_id]
    val_col = None
    for c in df.columns:
        if c in ("pressure", "value", "data.value"):
            val_col = c
            break
    if val_col is None:
        for c in df.columns:
            if c.endswith(".value") or c.endswith("pressure"):
                val_col = c
                break
    if val_col is None:
        return None
    s = pd.to_numeric(df[val_col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def pick_germany_national_team(search_results: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick Germany men's national team from search results when possible."""
    if not search_results:
        return None
    national = [t for t in search_results if t.get("national_team") is True]
    if national:
        germany = [t for t in national if "germany" in (t.get("name") or "").lower()]
        if len(germany) == 1:
            return germany[0]
        if germany:
            return min(germany, key=lambda x: int(x.get("id") or 0))
        return min(national, key=lambda x: int(x.get("id") or 0))
    for t in search_results:
        n = (t.get("name") or "").strip().lower()
        if n == "germany":
            return t
    return search_results[0]
