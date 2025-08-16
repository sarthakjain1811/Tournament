import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import streamlit.components.v1 as components

st.set_page_config(page_title="Doubles Tournament Manager", layout="wide")

# ==========================
# Compat helpers
# ==========================
def do_rerun():
    """Streamlit version-safe rerun."""
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

def get_query_params():
    """Streamlit version-safe query params."""
    if hasattr(st, "query_params"):
        return st.query_params
    return st.experimental_get_query_params()

# ==========================
# Data Models
# ==========================
@dataclass
class Team:
    id: str
    name: str
    player1: Optional[str] = None
    player2: Optional[str] = None

@dataclass
class Match:
    id: str
    stage: str  # 'GROUP' or 'KO'
    group: Optional[str]  # 'A'..'H' for group stage
    home: str  # team id
    away: str  # team id
    # GROUP & QF: single game to 11
    score_home: Optional[int] = None
    score_away: Optional[int] = None
    # SF / F: best of 3 to 11
    ko_g1h: Optional[int] = None
    ko_g1a: Optional[int] = None
    ko_g2h: Optional[int] = None
    ko_g2a: Optional[int] = None
    ko_g3h: Optional[int] = None
    ko_g3a: Optional[int] = None
    leg: Optional[str] = None  # QF/SF/F

# ==========================
# Live snapshot for Public View
# ==========================
LIVE_PATH = "live_state.json"
AUTO_REFRESH_SECONDS = 10  # public page auto-refresh interval

def export_state() -> Dict:
    ss = st.session_state
    return {
        "tournament_name": ss.tournament_name,
        "teams": {tid: asdict(t) for tid, t in ss.teams.items()},
        "groups": ss.groups,
        "matches": {mid: asdict(m) for mid, m in ss.matches.items()},
        "ko_bracket": ss.ko_bracket,
        "seed_rules": ss.seed_rules,
    }

def import_state(data: Dict):
    ss = st.session_state
    ss.tournament_name = data.get("tournament_name", ss.tournament_name)
    ss.teams = {tid: Team(**t) for tid, t in data.get("teams", {}).items()}
    ss.groups = {g: list(tids) for g, tids in data.get("groups", {}).items()}
    ss.matches = {mid: Match(**m) for mid, m in data.get("matches", {}).items()}
    ss.ko_bracket = data.get("ko_bracket", ss.ko_bracket)
    ss.seed_rules = data.get("seed_rules", ss.seed_rules)

def save_live_state():
    try:
        Path(LIVE_PATH).write_text(json.dumps(export_state(), indent=2), encoding="utf-8")
    except Exception as e:
        st.warning(f"Could not write live state: {e}")

def load_live_state() -> bool:
    try:
        p = Path(LIVE_PATH)
        if not p.exists():
            return False
        data = json.loads(p.read_text(encoding="utf-8"))
        import_state(data)
        return True
    except Exception as e:
        st.warning(f"Could not load live state: {e}")
        return False

# ==========================
# App State
# ==========================
def init_state():
    ss = st.session_state
    if 'teams' not in ss:
        ss.teams: Dict[str, Team] = {}
    if 'groups' not in ss:
        ss.groups: Dict[str, List[str]] = {g: [] for g in list("ABCDEFGH")}  # 8 groups always
    if 'matches' not in ss:
        ss.matches: Dict[str, Match] = {}
    if 'ko_bracket' not in ss:
        ss.ko_bracket: Dict[str, List[str]] = {"QF": [], "SF": [], "F": []}
    if 'seed_rules' not in ss:
        ss.seed_rules = (
            "8 groups; group winners auto-qualify to Quarterfinals (8 teams). "
            "Seeding by group performance: 1 vs 8, 4 vs 5, 3 vs 6, 2 vs 7."
        )
    if 'tournament_name' not in ss:
        ss.tournament_name = "Doubles Cup"
    if 'admin_mode' not in ss:
        ss.admin_mode = True
    if 'public_view' not in ss:
        q = get_query_params()
        view_val = q.get("view")
        if isinstance(view_val, list):
            view_val = view_val[0] if view_val else ""
        ss.public_view = (str(view_val).lower() == "public")
    if 'teams_last_import_hash' not in ss:
        ss.teams_last_import_hash = None

    # In public view, hydrate from live snapshot (read-only fixtures)
    if ss.public_view:
        load_live_state()

init_state()

# ==========================
# Utilities
# ==========================
def make_id(prefix: str, i: int) -> str:
    return f"{prefix}_{i:04d}"

def pair_label(team: Team) -> str:
    p1 = (team.player1 or "").strip()
    p2 = (team.player2 or "").strip()
    if p1 or p2:
        return " & ".join([x for x in [p1, p2] if x])
    return "(No players set)"

def pair_label_by_id(tid: str) -> str:
    t = st.session_state.teams.get(tid)
    if not t:
        return tid
    return pair_label(t)

def team_row(team: Team) -> Dict:
    return {
        "id": team.id,
        "Pair": pair_label(team),
        "Player 1": team.player1 or "",
        "Player 2": team.player2 or "",
        # Team name intentionally omitted from UI per requirement
    }

def round_robin_pairings(team_ids: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """Standard circle method; supports odd sizes by inserting a bye (None)."""
    teams = team_ids.copy()
    n = len(teams)
    bye = None
    if n % 2 == 1:
        teams.append(bye)
        n += 1

    pairings = []
    for r in range(n - 1):
        first = teams[:n//2]
        second = teams[n//2:][::-1]
        for i in range(n // 2):
            h, a = first[i], second[i]
            if h is not None and a is not None:
                if r % 2 == 1:
                    h, a = a, h
                pairings.append((h, a))
        # rotate (keep first fixed)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return pairings

def create_group_matches(groups: Dict[str, List[str]]):
    ss = st.session_state
    # clear existing group matches
    for mid in list(ss.matches.keys()):
        if ss.matches[mid].stage == 'GROUP':
            del ss.matches[mid]
    midx = 1
    for g, tids in groups.items():
        if len(tids) < 2:
            continue
        pairings = round_robin_pairings(tids)
        for (h, a) in pairings:
            if h is None or a is None:
                continue  # skip bye
            mid = make_id("G", midx)
            ss.matches[mid] = Match(
                id=mid, stage='GROUP', group=g, home=h, away=a
            )
            midx += 1

# ==========================
# CSV Helpers
# ==========================
def normalize_team_df(uploaded_bytes: bytes) -> pd.DataFrame:
    """Return a DataFrame with columns exactly: Team, Player 1, Player 2 (Team kept internally only)."""
    df = pd.read_csv(io.BytesIO(uploaded_bytes))

    def norm(s: str) -> str:
        return str(s).strip().lower().replace("_", " ").replace("-", " ")
    df.columns = [norm(c) for c in df.columns]

    def first_match(cands: List[str]) -> Optional[str]:
        for c in cands:
            if c in df.columns:
                return c
        return None

    team_col = first_match(["team", "team name", "name"])  # optional for UI
    p1_col = first_match(["player 1", "player1", "p1", "player one"])
    p2_col = first_match(["player 2", "player2", "p2", "player two"])

    if not (p1_col or p2_col):
        raise ValueError("CSV must include at least one player column: 'Player 1'/'Player1' or 'Player 2'/'Player2'.")

    def clean_cell(x):
        if pd.isna(x):
            return ""
        s = str(x).strip()
        return "" if s.lower() in ("nan", "none", "null") else s

    out = pd.DataFrame()
    # keep team name internally if present, but UI won't show it
    out["Team"] = df[team_col].map(clean_cell) if team_col else ""
    out["Player 1"] = df[p1_col].map(clean_cell) if p1_col else ""
    out["Player 2"] = df[p2_col].map(clean_cell) if p2_col else ""
    return out

def checksum(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()

# ==========================
# Standings / Tiebreak
# ==========================
def standings_for_group(group_key: str) -> pd.DataFrame:
    ss = st.session_state
    tids = ss.groups[group_key]
    rows: Dict[str, Dict[str, int]] = {
        tid: {"TeamID": tid, "Team": pair_label_by_id(tid), "P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "GD": 0, "Pts": 0}
        for tid in tids
    }

    for m in ss.matches.values():
        if m.stage != 'GROUP' or m.group != group_key:
            continue
        if m.score_home is None or m.score_away is None:
            continue
        h, a = m.home, m.away
        sh, sa = m.score_home, m.score_away

        rows[h]["P"] += 1; rows[a]["P"] += 1
        rows[h]["GF"] += sh; rows[h]["GA"] += sa
        rows[a]["GF"] += sa; rows[a]["GA"] += sh

        if sh > sa:
            rows[h]["W"] += 1; rows[a]["L"] += 1
            rows[h]["Pts"] += 3
        elif sa > sh:
            rows[a]["W"] += 1; rows[h]["L"] += 1
            rows[a]["Pts"] += 3
        else:
            rows[h]["D"] += 1; rows[a]["D"] += 1
            rows[h]["Pts"] += 1; rows[a]["Pts"] += 1

    for r in rows.values():
        r["GD"] = r["GF"] - r["GA"]

    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return df
    df = df.sort_values(["Pts", "GD", "GF", "Team"], ascending=[False, False, False, True]).reset_index(drop=True)

    # Head-to-head swap for 2-way ties on Pts/GD/GF
    i = 0
    while i < len(df) - 1:
        a = df.loc[i]
        j = i
        while j + 1 < len(df):
            b = df.loc[j + 1]
            if (a["Pts"], a["GD"], a["GF"]) == (b["Pts"], b["GD"], b["GF"]):
                j += 1
            else:
                break
        if j == i + 1:
            ta, tb = df.loc[i, "TeamID"], df.loc[j, "TeamID"]
            h2h_pts_a, h2h_pts_b = head_to_head_points(group_key, ta, tb)
            if h2h_pts_b > h2h_pts_a:
                df.iloc[i], df.iloc[j] = df.iloc[j].copy(), df.iloc[i].copy()
        i = j + 1
    return df

def head_to_head_points(group_key: str, ta: str, tb: str) -> Tuple[int, int]:
    ss = st.session_state
    pts_a = pts_b = 0
    for m in ss.matches.values():
        if m.stage != 'GROUP' or m.group != group_key:
            continue
        if {m.home, m.away} != {ta, tb}:
            continue
        if m.score_home is None or m.score_away is None:
            continue
        sh, sa = m.score_home, m.score_away
        if m.home == ta:
            if sh > sa: pts_a += 3
            elif sa > sh: pts_b += 3
            else: pts_a += 1; pts_b += 1
        else:
            if sh > sa: pts_b += 3
            elif sa > sh: pts_a += 3
            else: pts_a += 1; pts_b += 1
    return pts_a, pts_b

def all_groups_done() -> bool:
    for m in st.session_state.matches.values():
        if m.stage == 'GROUP' and (m.score_home is None or m.score_away is None):
            return False
    return True

def rank_rows(rows: List[pd.Series]) -> List[pd.Series]:
    return sorted(rows, key=lambda r: (-int(r["Pts"]), -int(r["GD"]), -int(r["GF"]), r["Team"]))

# ==========================
# Seeding & Knockouts
# ==========================
def valid_game(gh, ga) -> bool:
    if gh is None or ga is None: return False
    if gh == ga: return False
    return max(gh, ga) >= 11

def seed_quarterfinals():
    ss = st.session_state
    group_tables = {g: standings_for_group(g) for g in ss.groups.keys()}
    winners = []
    for g, df in group_tables.items():
        if len(df) > 0:
            winners.append(df.iloc[0])
    if len(winners) < 8:
        st.warning("Need 8 group winners to seed Quarterfinals. Complete all groups first.")
        return
    winners = rank_rows(winners)[:8]

    # Clear existing KO matches
    for mid in list(ss.matches.keys()):
        if ss.matches[mid].stage == 'KO':
            del ss.matches[mid]
    ss.ko_bracket = {"QF": [], "SF": [], "F": []}

    # QF pairs: 1v8, 4v5, 3v6, 2v7
    pairs = [(1,8),(4,5),(3,6),(2,7)]
    mapping_seed_to_tid = {i+1: str(winners[i]["TeamID"]) for i in range(8)}

    midx = 1
    for s1, s2 in pairs:
        t1 = mapping_seed_to_tid[s1]
        t2 = mapping_seed_to_tid[s2]
        mid = make_id("K", midx)
        ss.matches[mid] = Match(id=mid, stage='KO', group=None, home=t1, away=t2, leg='QF')
        ss.ko_bracket["QF"].append(mid)
        midx += 1

def calc_sets(h: int, a: int) -> Tuple[int, int]:
    return (1, 0) if h > a else (0, 1)

def ko_match_winner(m: Match) -> Optional[str]:
    # Best-of-3 winner for SF/F; only count games where the winner reached 11.
    sets_h = sets_a = 0
    games = [(m.ko_g1h, m.ko_g1a), (m.ko_g2h, m.ko_g2a), (m.ko_g3h, m.ko_g3a)]
    for gh, ga in games:
        if not valid_game(gh, ga):
            continue
        sh, sa = calc_sets(gh, ga)
        sets_h += sh; sets_a += sa
    if sets_h >= 2: return m.home
    if sets_a >= 2: return m.away
    return None

def advance_knockouts():
    ss = st.session_state
    rounds = ["QF", "SF", "F"]
    for i in range(len(rounds) - 1):
        cur, nxt = rounds[i], rounds[i+1]
        ss.ko_bracket[nxt] = []
        winners = []
        for mid in ss.ko_bracket[cur]:
            m = ss.matches[mid]
            if m.leg == 'QF':
                if m.score_home is None or m.score_away is None: continue
                if m.score_home == m.score_away: continue
                if max(m.score_home, m.score_away) < 11: continue
                winners.append(m.home if m.score_home > m.score_away else m.away)
            else:
                wt = ko_match_winner(m)
                if wt: winners.append(wt)
        midx = 1000 * (i+2)
        for j in range(0, len(winners), 2):
            if j+1 >= len(winners): break
            mid = make_id("K", midx)
            ss.matches[mid] = Match(id=mid, stage='KO', group=None,
                                    home=winners[j], away=winners[j+1], leg=nxt)
            ss.ko_bracket[nxt].append(mid)
            midx += 1

# ==========================
# Group Distribution (8 groups, no duplicates)
# ==========================
def distribute_into_8_groups_min3(team_ids: List[str], shuffle: bool = True) -> Dict[str, List[str]]:
    groups = {g: [] for g in list("ABCDEFGH")}
    ids = team_ids.copy()
    if shuffle:
        np.random.shuffle(ids)

    n = len(ids)
    if n <= 0:
        return groups

    if n >= 24:
        targets = [3]*8
        r = n - 24
    else:
        targets = [0]*8
        r = n

    gi = 0
    while r > 0:
        targets[gi % 8] += 1
        gi += 1
        r -= 1

    idx = 0
    for g_idx, g in enumerate(groups.keys()):
        size = targets[g_idx]
        groups[g] = ids[idx:idx+size]
        idx += size
    return groups

def validate_no_duplicates(groups: Dict[str, List[str]]) -> List[str]:
    from collections import Counter
    all_ids = [tid for tids in groups.values() for tid in tids]
    ctr = Counter(all_ids)
    return [tid for tid, c in ctr.items() if c > 1]

# ==========================
# UI (Admin vs Public View)
# ==========================
def sidebar():
    ss = st.session_state
    with st.sidebar:
        st.header("⚙️ Settings")
        ss.tournament_name = st.text_input("Tournament name", ss.tournament_name, disabled=ss.public_view)
        st.markdown("**Groups:** 8 groups (A–H). No duplicates allowed. Round-robin, then QF → SF → Final.")
        st.markdown(":blue[Seeding:] " + ss.seed_rules)
        st.markdown("---")

        if ss.public_view:
            st.success("Public View (read-only)")
            components.html(f"<script>setTimeout(()=>window.location.reload(), {AUTO_REFRESH_SECONDS*1000});</script>", height=0)
        else:
            ss.admin_mode = st.toggle("Admin Mode", value=ss.admin_mode)

        st.markdown("**Share Live Draw**")
        components.html("""
            <div id="publicLink" style="font-family: var(--font);">
              <code id="plink"></code>
              <script>
                const url = location.origin + location.pathname + '?view=public';
                const el = document.getElementById('plink');
                el.textContent = url;
              </script>
            </div>
        """, height=30)
        if st.button("🔗 Open Public View in new tab"):
            components.html("<script>window.open(location.origin + location.pathname + '?view=public','_blank');</script>", height=0)
        st.markdown("---")

        if not ss.public_view:
            if st.button("💾 Export save (JSON)"):
                data = export_state()
                st.download_button(
                    label="Download tournament.json",
                    data=json.dumps(data, indent=2),
                    file_name="tournament.json",
                    mime="application/json",
                )
            uploaded = st.file_uploader("Import save (JSON)", type=["json"], key="save_json")
            if uploaded is not None:
                try:
                    data = json.loads(uploaded.read())
                    import_state(data)
                    st.success("Save imported!")
                except Exception as e:
                    st.error(f"Import failed: {e}")

# ---------- Admin Tabs ----------
def tab_teams():
    st.subheader("1) Teams")
    st.caption("Upload or add pairs (any number ≥ 24). Team names are ignored in the UI.")
    disabled = st.session_state.public_view or not st.session_state.admin_mode
    ss = st.session_state

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader(
            "Upload CSV (columns like: Player 1 / Player1 / P1, Player 2 / Player2 / P2; Team optional)",
            type=["csv"], key="teams_csv", disabled=disabled
        )
        import_mode = st.radio("Import mode", ["Replace existing teams", "Append to existing"], index=0, horizontal=True, disabled=disabled)
        if st.button("📥 Import CSV", disabled=disabled or (uploaded is None)):
            try:
                content = uploaded.getvalue()
                ch = checksum(content)
                if ss.teams_last_import_hash == ch:
                    st.info("This CSV was already imported; skipping.")
                else:
                    df = normalize_team_df(content)
                    if import_mode == "Replace existing teams":
                        ss.teams = {}
                        ss.groups = {g: [] for g in list("ABCDEFGH")}
                        ss.matches = {}
                        ss.ko_bracket = {"QF": [], "SF": [], "F": []}

                    start_idx = len(ss.teams) + 1
                    for i, row in df.iterrows():
                        def to_val(x):
                            s = str(x).strip()
                            return None if s == "" or s.lower() in ("nan", "none", "null") else s
                        tid = make_id("T", start_idx + i)
                        ss.teams[tid] = Team(
                            id=tid,
                            name=str(row["Team"]).strip() if "Team" in row else "",
                            player1=to_val(row["Player 1"]),
                            player2=to_val(row["Player 2"]),
                        )
                    ss.teams_last_import_hash = ch
                    st.success(f"Imported {len(df)} pairs ({import_mode.lower()}).")
            except Exception as e:
                st.error(f"Import failed: {e}")

    with c2:
        st.write("Add single pair")
        p1 = st.text_input("Player 1", key="add_team_p1", disabled=disabled)
        p2 = st.text_input("Player 2", key="add_team_p2", disabled=disabled)
        if st.button("➕ Add pair", disabled=disabled):
            if not (p1 or p2):
                st.error("Enter at least one player name")
            else:
                tid = make_id("T", len(ss.teams) + 1)
                ss.teams[tid] = Team(id=tid, name="", player1=p1.strip() or None, player2=p2.strip() or None)
                st.success(f"Added {pair_label_by_id(tid)}")

        if st.button("🧽 Clear ALL pairs", disabled=disabled):
            ss.teams = {}
            ss.groups = {g: [] for g in list("ABCDEFGH")}
            ss.matches = {}
            ss.ko_bracket = {"QF": [], "SF": [], "F": []}
            ss.teams_last_import_hash = None
            st.success("All pairs and fixtures cleared.")

    if len(ss.teams) > 0:
        df = pd.DataFrame([team_row(t) for t in ss.teams.values()])
        st.dataframe(df.set_index("id"), use_container_width=True)

def tab_groups():
    st.subheader("2) Groups (A–H)")
    st.caption("Randomize or assign pairs. No duplicates across groups.")
    ss = st.session_state
    all_team_ids = list(ss.teams.keys())
    disabled = ss.public_view or not ss.admin_mode

    c1, c2 = st.columns([1,1])
    with c1:
        if st.button("🎲 Randomize groups", disabled=disabled):
            if len(all_team_ids) < 8:
                st.error("Need at least 8 pairs to randomize across 8 groups.")
            else:
                ss.groups = distribute_into_8_groups_min3(all_team_ids, shuffle=True)
                create_group_matches(ss.groups)
                st.success("Groups randomized and fixtures created!")

    with c2:
        if st.button("🧹 Clear groups", disabled=disabled):
            ss.groups = {g: [] for g in list("ABCDEFGH")}
            create_group_matches(ss.groups)

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.write("**Unassigned pairs**")
        assigned = set([tid for tids in ss.groups.values() for tid in tids])
        unassigned = [tid for tid in all_team_ids if tid not in assigned]

        if len(unassigned) > 0:
            un_df = pd.DataFrame(
                [{"PairID": tid,
                  "Pair": pair_label_by_id(tid),
                  "Player 1": ss.teams[tid].player1 or "",
                  "Player 2": ss.teams[tid].player2 or ""} for tid in unassigned]
            )
            st.dataframe(un_df.set_index("PairID"), use_container_width=True, height=300)
        else:
            st.info("No unassigned pairs.")
            empty_df = pd.DataFrame(columns=["PairID", "Pair", "Player 1", "Player 2"]).set_index("PairID")
            st.dataframe(empty_df, use_container_width=True, height=120)

    with right:
        dupes = validate_no_duplicates(ss.groups)
        if dupes:
            names = ", ".join(pair_label_by_id(tid) for tid in dupes if tid in ss.teams)
            st.error(f"Duplicate assignment detected (fix before seeding): {names}")

        for g in ss.groups.keys():
            st.write(f"### Group {g}")
            current = ss.groups[g]
            assigned_now = set([tid for tids in ss.groups.values() for tid in tids])
            add_options = [None] + [tid for tid in all_team_ids if tid not in assigned_now]
            add_tid = st.selectbox(
                f"Add pair to Group {g}",
                options=add_options,
                format_func=lambda x: "-- choose --" if x is None else pair_label_by_id(x),
                key=f"add_{g}",
                disabled=disabled or len(add_options) <= 1,
            )
            if st.button(f"Add to {g}", key=f"btn_add_{g}", disabled=disabled):
                if add_tid is None:
                    st.warning("Choose a pair first")
                else:
                    ss.groups[g].append(add_tid)
                    st.success(f"Added {pair_label_by_id(add_tid)} to Group {g}")

            rows = [{"PairID": tid,
                     "Pair": pair_label_by_id(tid),
                     "Player 1": ss.teams[tid].player1 or "",
                     "Player 2": ss.teams[tid].player2 or ""} for tid in current]
            st.table(pd.DataFrame(rows))

            for tid in list(current):
                if st.button(f"Remove {pair_label_by_id(tid)} from {g}", key=f"rm_{g}_{tid}", disabled=disabled):
                    ss.groups[g] = [x for x in ss.groups[g] if x != tid]
                    do_rerun()

    if st.button("✅ Create / Refresh group fixtures", disabled=disabled):
        any_group_has_2 = any(len(tids) >= 2 for tids in ss.groups.values())
        if not any_group_has_2:
            st.error("Each group needs at least 2 pairs to create fixtures.")
        else:
            create_group_matches(ss.groups)
            st.success("Fixtures created/refreshed.")

def tab_group_stage():
    st.subheader("3) Group Stage – Enter Results (Best of 1 to 11)")
    st.caption("Each match is a single game to 11 points. No draws—enter a winner.")
    ss = st.session_state
    tabs = st.tabs([f"Group {g}" for g in ss.groups.keys()])

    for idx, g in enumerate(ss.groups.keys()):
        with tabs[idx]:
            group_matches = [m for m in ss.matches.values() if m.stage == 'GROUP' and m.group == g]
            if len(group_matches) == 0:
                st.info("No fixtures yet. Go to the Groups tab and create fixtures.")
                continue
            for mid in sorted([m.id for m in group_matches]):
                m = ss.matches[mid]
                c1, c2, c3, c4, c5 = st.columns([3,1,1,1,3])
                with c1:
                    st.write(f"{pair_label_by_id(m.home)}  vs  {pair_label_by_id(m.away)}")
                with c2:
                    m.score_home = st.number_input("Home pts", min_value=0, max_value=11, step=1,
                                                   value=m.score_home if m.score_home is not None else 0,
                                                   key=f"sh_{m.id}",
                                                   disabled=ss.public_view or not ss.admin_mode)
                with c3:
                    m.score_away = st.number_input("Away pts", min_value=0, max_value=11, step=1,
                                                   value=m.score_away if m.score_away is not None else 0,
                                                   key=f"sa_{m.id}",
                                                   disabled=ss.public_view or not ss.admin_mode)
                with c4:
                    reset = st.button("↩︎ Reset", key=f"rst_{m.id}",
                                      disabled=ss.public_view or not ss.admin_mode)
                    if reset:
                        m.score_home = None
                        m.score_away = None
                        ss.matches[mid] = m  # persist
                        do_rerun()
                with c5:
                    if (m.score_home is not None and m.score_away is not None):
                        if m.score_home == m.score_away:
                            st.warning("No draws. Adjust to declare a winner (to 11).")
                        elif max(m.score_home, m.score_away) < 11:
                            st.info("Winning side should have 11.")
                # persist any edits on each rerun
                ss.matches[mid] = m

    st.markdown("---")
    if st.button("🏁 Finalize groups & Seed QF", disabled=ss.public_view or not ss.admin_mode):
        if not all_groups_done():
            st.error("Complete all group matches first.")
        else:
            seed_quarterfinals()
            st.success("Quarterfinals seeded from group winners.")

def tab_knockouts():
    st.subheader("4) Knockouts – QF (Bo1 to 11), SF & Final (Bo3 to 11)")
    st.caption("QF is a single game to 11. SF/Final are best-of-3, each game to 11.")
    ss = st.session_state
    rounds = ["QF", "SF", "F"]

    for rnd in rounds:
        st.write(f"## {rnd}")
        mids = ss.ko_bracket.get(rnd, [])
        if len(mids) == 0:
            st.info(f"No {rnd} fixtures yet.")
            continue
        for mid in mids:
            m = ss.matches[mid]
            t1, t2 = pair_label_by_id(m.home), pair_label_by_id(m.away)
            st.write(f"**{t1} vs {t2}**")
            if m.leg == 'QF':
                c1, c2, c3 = st.columns([1,1,6])
                with c1:
                    m.score_home = st.number_input("Home pts", min_value=0, max_value=11, step=1,
                                                   value=m.score_home if m.score_home is not None else 0,
                                                   key=f"qfh_{m.id}",
                                                   disabled=ss.public_view or not ss.admin_mode)
                with c2:
                    m.score_away = st.number_input("Away pts", min_value=0, max_value=11, step=1,
                                                   value=m.score_away if m.score_away is not None else 0,
                                                   key=f"qfa_{m.id}",
                                                   disabled=ss.public_view or not ss.admin_mode)
                with c3:
                    st.caption("Single game to 11; no draws.")
                # persist
                ss.matches[mid] = m

                if (m.score_home is not None and m.score_away is not None and
                    m.score_home != m.score_away and max(m.score_home, m.score_away) >= 11):
                    st.success(f"Winner: {t1 if m.score_home > m.score_away else t2}")
                else:
                    st.info("Enter scores until one side reaches 11 and wins.")
            else:
                # SF / Final Bo3
                for gi in [1, 2, 3]:
                    c1, c2, c3 = st.columns([1, 1, 6])
                    gh_key = f"g{gi}h_{m.id}"; ga_key = f"g{gi}a_{m.id}"
                    with c1:
                        valh = getattr(m, f"ko_g{gi}h")
                        setattr(m, f"ko_g{gi}h", st.number_input(f"G{gi} Home", min_value=0, max_value=11, step=1,
                                                                  value=valh if valh is not None else 0,
                                                                  key=gh_key,
                                                                  disabled=ss.public_view or not ss.admin_mode))
                    with c2:
                        vala = getattr(m, f"ko_g{gi}a")
                        setattr(m, f"ko_g{gi}a", st.number_input(f"G{gi} Away", min_value=0, max_value=11, step=1,
                                                                  value=vala if vala is not None else 0,
                                                                  key=ga_key,
                                                                  disabled=ss.public_view or not ss.admin_mode))
                    with c3:
                        st.caption("Game to 11; no draws.")
                # persist SF/F edits
                ss.matches[mid] = m

                winner_tid = ko_match_winner(m)
                if winner_tid:
                    st.success(f"Winner: {pair_label_by_id(winner_tid)}")
                else:
                    st.info("Enter results until someone wins 2 games (each winning game must reach 11).")

    if st.button("➡️ Build Next Rounds", disabled=ss.public_view or not ss.admin_mode):
        advance_knockouts()
        st.success("Next rounds generated (where winners available).")

def tab_roster_view():
    st.subheader("Pairs Roster (Read-only)")
    ss = st.session_state
    if not ss.teams:
        st.info("No pairs yet. Import on the Teams tab.")
        return
    df = pd.DataFrame([team_row(t) for t in ss.teams.values()]).drop(columns=["id"])
    st.dataframe(df, use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download roster CSV", data=csv, file_name="roster.csv", mime="text/csv")

def tab_help():
    st.subheader("How to Use & Share")
    st.markdown(
        """
        **Quick start**

        1. **Teams**: Upload a CSV (e.g., `Player 1`, `Player 2`; `Team` optional) and click **Import CSV**.
           Choose **Replace** to start fresh, or **Append** to add.
        2. **Groups**: Click **Randomize groups** (8 groups; start at 3 per group, distribute extras evenly), or assign manually.
        3. **Group Stage**: Each group match is **best-of-1 to 11**. Enter scores; standings update (Pts → GD → GF → 2-way H2H).
        4. **Seed QF**: When all group matches have results, click **🏁 Finalize groups & Seed QF**.
        5. **Knockouts**: **QF is best-of-1 to 11**; **SF/Final are best-of-3 to 11**. Only games that reach 11 count.
        6. **Share**: Use the sidebar button to open the public draw (`?view=public`).

        **Notes**
        - Import only runs when you press **Import CSV** (prevents duplicates/reruns).
        - UI shows **player names only** (pairs). Team names are ignored for display.
        """
    )

# ---------- Public (read-only) Draw ----------
def render_group_fixtures_readonly():
    ss = st.session_state
    tabs = st.tabs([f"Group {g}" for g in ss.groups.keys()])
    for idx, g in enumerate(ss.groups.keys()):
        with tabs[idx]:
            st.write(f"### Group {g}")
            group_matches = [m for m in ss.matches.values() if m.stage == 'GROUP' and m.group == g]
            if not group_matches:
                st.info("No fixtures yet.")
                continue
            rows = []
            for m in group_matches:
                home = pair_label_by_id(m.home)
                away = pair_label_by_id(m.away)
                if m.score_home is None or m.score_away is None:
                    score = "—"
                else:
                    score = f"{m.score_home}–{m.score_away}"
                rows.append({"Fixture": f"{home} vs {away}", "Score": score})
            st.table(pd.DataFrame(rows))

def render_knockouts_readonly():
    ss = st.session_state
    st.write("### Knockouts")
    for rnd in ["QF", "SF", "F"]:
        st.write(f"**{rnd}**")
        mids = ss.ko_bracket.get(rnd, [])
        if not mids:
            st.info(f"No {rnd} fixtures yet.")
            continue
        rows = []
        for mid in mids:
            m = ss.matches[mid]
            home = pair_label_by_id(m.home)
            away = pair_label_by_id(m.away)
            if rnd == "QF":
                score = "—" if (m.score_home is None or m.score_away is None) else f"{m.score_home}–{m.score_away}"
            else:
                g1 = f"{m.ko_g1h}–{m.ko_g1a}" if (m.ko_g1h is not None and m.ko_g1a is not None) else "—"
                g2 = f"{m.ko_g2h}–{m.ko_g2a}" if (m.ko_g2h is not None and m.ko_g2a is not None) else "—"
                g3 = f"{m.ko_g3h}–{m.ko_g3a}" if (m.ko_g3h is not None and m.ko_g3a is not None) else "—"
                score = f"G1 {g1} | G2 {g2} | G3 {g3}"
            rows.append({"Fixture": f"{home} vs {away}", "Score(s)": score})
        st.table(pd.DataFrame(rows))

# ==========================
# Main
# ==========================
def main():
    sidebar()
    st.title(st.session_state.tournament_name)

    if st.session_state.public_view:
        st.caption("🟢 Public View (read-only) • Live fixtures/draw")
        tab = st.tabs(["Draw / Fixtures"])[0]
        with tab:
            render_group_fixtures_readonly()
            st.markdown("---")
            render_knockouts_readonly()
        return

    # Admin UI
    mode_badge = "🛠️ Admin" if st.session_state.admin_mode else "🔒 Admin disabled"
    st.caption(mode_badge)
    tabs = st.tabs(["Teams", "Groups", "Group Stage", "Knockouts", "Roster", "Help"])
    with tabs[0]: tab_teams()
    with tabs[1]: tab_groups()
    with tabs[2]: tab_group_stage()
    with tabs[3]: tab_knockouts()
    with tabs[4]: tab_roster_view()
    with tabs[5]: tab_help()

    # Persist snapshot for public viewers on every admin rerun
    save_live_state()

if __name__ == "__main__":
    main()
