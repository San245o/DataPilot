"""Read-only counts that can be resolved exactly from the current dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetCount:
    answer: str
    evidence: str


_TEAM_COUNT = re.compile(
    r"(?:how many|count) (?:matches|games) (?:has|have|did) (.+?) "
    r"(won|win|lost|lose|played|play) (?:at|in|from) (.+)"
)
_VENUE_COUNT = re.compile(
    r"(?:how many|count) (?:matches|games)(?: (?:were |have been )?played)? (?:at|in|from) (.+)"
)
_BOTH_COUNTS = re.compile(
    r"(?:how many|count) (?:matches|games)(?: (?:were |have been )?played)? (?:at|in|from) (.+?) "
    r"and how many (?:has|have|did) (.+?) (?:won|win) (?:there|at that venue)"
)
_FOLLOW_UP = re.compile(
    r"(?:and )?how many(?: (?:matches|games))? (?:has|have|did) (?:they|it|that team) "
    r"(won|win|lost|lose|played|play)(?: (?:there|at that venue))?"
)


def _normalize(value: Any) -> str:
    return " ".join(re.findall(r"[^\W_]+", str(value if value is not None else "").casefold()))


def _match_intent(text: str) -> tuple[str, str, str] | None:
    if match := _BOTH_COUNTS.fullmatch(text):
        return match[2], "both", match[1]
    if match := _TEAM_COUNT.fullmatch(text):
        return match[1], match[2], match[3]
    if match := _VENUE_COUNT.fullmatch(text):
        return "", "all", match[1]
    return None


def _contains(value: str, term: str) -> bool:
    return f" {term} " in f" {value} "


def _column(columns: dict[str, str], *names: str) -> str | None:
    return next((columns[name] for name in names if name in columns), None)


def _team_aliases(term: str, values: set[str]) -> set[str]:
    names = {
        value for value in values
        if len(value.split()) > 1 and (
            value == term or "".join(word[0] for word in value.split()) == term
        )
    }
    if len(names) > 1:
        return {term} if term in values and len(term.split()) > 1 else set()
    if names:
        name = next(iter(names))
        acronym = "".join(word[0] for word in name.split())
        return {name, acronym}
    return {term} if term in values else set()


def calculate_dataset_count(
    prompt: str, rows: list[dict[str, Any]], history: list[dict[str, str]],
) -> DatasetCount | None:
    text = _normalize(prompt)
    if re.fullmatch(r"(?:how many|count) (?:rows|records)(?: are there)?(?: in (?:this|the) (?:data|dataset|sheet|spreadsheet))?", text):
        return DatasetCount(f"The dataset contains {len(rows):,} rows.", "Counted every row in the active dataset.")
    intent = _match_intent(text)
    if not intent and (follow_up := _FOLLOW_UP.fullmatch(text)):
        previous = next(
            (_match_intent(_normalize(turn.get("content"))) for turn in reversed(history) if turn.get("role") == "user"),
            None,
        )
        if previous and previous[0]:
            intent = previous[0], follow_up[1], previous[2]
    if not intent or not rows:
        return None

    columns = {_normalize(key): key for row in rows for key in row}
    venue = _column(columns, "venue", "stadium", "ground", "match venue")
    winner = _column(columns, "winner", "winning team", "match winner")
    home = _column(columns, "team1", "team 1", "home team", "first team")
    away = _column(columns, "team2", "team 2", "away team", "second team")
    if not venue or not (winner or (home and away)):
        return None
    # A repeated match ID can indicate ball-by-ball data or duplicate records.
    match_id = _column(columns, "match id", "id")
    if match_id:
        ids = [_normalize(row.get(match_id)) for row in rows]
        if not all(ids) or len(ids) != len(set(ids)):
            return None

    team, outcome, venue_term = intent
    venue_term = re.sub(r"^(?:the )| (?:stadium|ground|arena)$", "", venue_term).strip()
    if venue_term in {"", "stadium", "ground", "arena", "there"}:
        return None
    matches = [row for row in rows if _contains(_normalize(row.get(venue)), venue_term)]
    if not matches:
        return None

    venue_label = str(matches[0][venue]).strip()
    total = len(matches)
    if outcome == "all":
        return DatasetCount(
            f"The uploaded dataset contains **{total:,}** {'match' if total == 1 else 'matches'} at {venue_label}.",
            f"Counted records whose {venue} includes “{venue_term}”.",
        )

    values = {_normalize(row.get(column)) for row in rows for column in (winner, home, away) if column}
    aliases = _team_aliases(team, values)
    if not aliases:
        return None
    if outcome in {"lost", "lose", "played", "play"} and not (home and away):
        return None
    if outcome not in {"played", "play"} and not winner:
        return None

    if outcome in {"played", "play"}:
        count = sum(bool({_normalize(row.get(home)), _normalize(row.get(away))} & aliases) for row in matches)
        verb, evidence = "played", f"Matched the team in {home} or {away}."
    elif outcome in {"lost", "lose"}:
        count = 0
        for row in matches:
            participants = {_normalize(row.get(home)), _normalize(row.get(away))}
            winning_team = _normalize(row.get(winner))
            if participants & aliases and winning_team and winning_team in participants - aliases:
                count += 1
        verb, evidence = "lost", f"Counted matches where the team participated and {winner} named the opponent."
    else:
        count = sum(_normalize(row.get(winner)) in aliases for row in matches)
        verb, evidence = "won", f"Matched the team in {winner}."
    label = team.upper() if len(team) <= 5 else team.title()
    noun = "match" if count == 1 else "matches"
    answer = f"In the uploaded dataset, {label} {verb} **{count:,}** {noun} at {venue_label}."
    if outcome == "both":
        answer = f"The uploaded dataset contains **{total:,}** {'match' if total == 1 else 'matches'} at {venue_label}. {label} won **{count:,}**."
    return DatasetCount(answer, f"Filtered {venue} for “{venue_term}”. {evidence}")
