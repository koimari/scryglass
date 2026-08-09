"""Team / champion name hygiene for OE ↔ Leaguepedia joins."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# OE / book short names → canonical Leaguepedia-ish names used in this repo
TEAM_ALIASES: dict[str, str] = {
    "gen.g": "Gen.G",
    "geng": "Gen.G",
    "gen g": "Gen.G",
    "gen.g esports": "Gen.G",
    "t1": "T1",
    "skt t1": "T1",
    "skt": "T1",
    "hanwha life esports": "Hanwha Life Esports",
    "hle": "Hanwha Life Esports",
    "dplus kia": "Dplus Kia",
    "dwg kia": "Dplus Kia",
    "damwon kia": "Dplus Kia",
    "dk": "Dplus Kia",
    "karmine corp": "Karmine Corp",
    "karmine": "Karmine Corp",
    "kc": "Karmine Corp",
    "kt rolster": "KT Rolster",
    "kt": "KT Rolster",
    "brion": "BRION",
    "fredit brion": "BRION",
    "oksavingsbank brion": "BRION",
    "nongshim redforce": "Nongshim RedForce",
    "nongshim red force": "Nongshim RedForce",
    "ns": "Nongshim RedForce",
    "bnk fearx": "BNK FEARX",
    "fearx": "BNK FEARX",
    "dn freecs": "DN Freecs",
    "kwangdong freecs": "DN Freecs",
    "drx": "DRX",
    "g2 esports": "G2 Esports",
    "g2": "G2 Esports",
    "fnatic": "Fnatic",
    "fnc": "Fnatic",
    "team vitality": "Team Vitality",
    "vitality": "Team Vitality",
    "movistar koi": "Movistar KOI",
    "mad lions koi": "Movistar KOI",
    "mad lions": "Movistar KOI",
    "team liquid": "Team Liquid",
    "cloud9": "Cloud9",
    "flyquest": "FlyQuest",
    "lyon": "LYON",
    "lyon (2024 american team)": "LYON",
    "bilibili gaming": "Bilibili Gaming",
    "blg": "Bilibili Gaming",
    "top esports": "Top Esports",
    "tes": "Top Esports",
    "jd gaming": "JD Gaming",
    "jdg": "JD Gaming",
    "beijing jdg esports": "JD Gaming",
    "weibo gaming": "Weibo Gaming",
    "wbg": "Weibo Gaming",
    "lng esports": "LNG Esports",
    "lng": "LNG Esports",
    "anyone's legend": "Anyone's Legend",
    "al": "Anyone's Legend",
    "edward gaming": "Edward Gaming",
    "edg": "Edward Gaming",
    "invictus gaming": "Invictus Gaming",
    "ig": "Invictus Gaming",
    "funplus phoenix": "FunPlus Phoenix",
    "fpx": "FunPlus Phoenix",
    "rare atom": "Rare Atom",
    "ra": "Rare Atom",
    "thundertalk gaming": "ThunderTalk Gaming",
    "thunder talk gaming": "ThunderTalk Gaming",
    "tt": "ThunderTalk Gaming",
    "ultra prime": "Ultra Prime",
    "up": "Ultra Prime",
    "lgd gaming": "LGD Gaming",
    "lgd": "LGD Gaming",
    "team we": "Team WE",
    "we": "Team WE",
    "oh my god": "Oh My God",
    "omg": "Oh My God",
    "royal never give up": "Royal Never Give Up",
    "rng": "Royal Never Give Up",
    "ninjas in pyjamas": "Ninjas in Pyjamas",
    "nip": "Ninjas in Pyjamas",
    "weibogaming": "Weibo Gaming",
    "suzhou lng esports": "LNG Esports",
    "shenzhen ninjas in pajamas": "Ninjas in Pyjamas",
    "xi'an team we": "Team WE",
    "detonation focusme": "DetonatioN FocusMe",
    "dfm": "DetonatioN FocusMe",
    "ctbc flying oyster": "CTBC Flying Oyster",
    "cfo": "CTBC Flying Oyster",
    "fukuoka softbank hawks gaming": "Fukuoka SoftBank HAWKS gaming",
    "shg": "Fukuoka SoftBank HAWKS gaming",
    "team secret whales": "Team Secret Whales",
    "tsw": "Team Secret Whales",
    "relove deep cross gaming": "Relove Deep Cross Gaming",
    "rdcg": "Relove Deep Cross Gaming",
    "vit": "Team Vitality",
    "pain gaming": "Pain Gaming",
    "png": "Pain Gaming",
    "løs": "LØS",
    "los": "LØS",
    "mibr.los": "LØS",
    "mibr los": "LØS",
    "los grandes": "LØS",
}

CHAMP_ALIASES: dict[str, str] = {
    "kaisa": "Kai'Sa",
    "kai'sa": "Kai'Sa",
    "mel": "Mel",
    "jarvan": "Jarvan IV",
    "jarvan iv": "Jarvan IV",
    "j4": "Jarvan IV",
    "monkey king": "Wukong",
    "wukong": "Wukong",
    "ksante": "K'Sante",
    "k'sante": "K'Sante",
    "renata": "Renata Glasc",
    "nunu": "Nunu & Willump",
    "dr mundo": "Dr. Mundo",
    "mundo": "Dr. Mundo",
    "reksai": "Rek'Sai",
    "belveth": "Bel'Veth",
    "kogmaw": "Kog'Maw",
    "chogath": "Cho'Gath",
    "velkoz": "Vel'Koz",
    "khazix": "Kha'Zix",
    "mf": "Miss Fortune",
    "miss fortune": "Miss Fortune",
    "tf": "Twisted Fate",
    "lee": "Lee Sin",
    "xin": "Xin Zhao",
    "locke": "Locke",
    "corvin locke": "Locke",
}

_WS_RE = re.compile(r"\s+")

# key → canonical display name (aliases + canon self-keys), built once
_TEAM_LOOKUP: dict[str, str] | None = None


def _team_lookup() -> dict[str, str]:
    global _TEAM_LOOKUP
    if _TEAM_LOOKUP is None:
        table: dict[str, str] = {}
        for alias, canon in TEAM_ALIASES.items():
            table[alias] = canon
            table[_norm_key_uncached(canon)] = canon
        _TEAM_LOOKUP = table
    return _TEAM_LOOKUP


def _norm_key_uncached(name: str) -> str:
    s = unicodedata.normalize("NFKC", name or "").strip().lower()
    s = s.replace("’", "'").replace("`", "'")
    s = _WS_RE.sub(" ", s)
    return s


@lru_cache(maxsize=16384)
def _norm_key(name: str) -> str:
    return _norm_key_uncached(name)


@lru_cache(maxsize=8192)
def normalize_team(name: str) -> str:
    key = _norm_key(name)
    hit = _team_lookup().get(key)
    if hit is not None:
        return hit
    return (name or "").strip()


@lru_cache(maxsize=4096)
def normalize_champ(name: str) -> str:
    key = _norm_key(name)
    if key in CHAMP_ALIASES:
        return CHAMP_ALIASES[key]
    return (name or "").strip()


def fuzzy_team_match(a: str, b: str) -> bool:
    return _norm_key(normalize_team(a)) == _norm_key(normalize_team(b))
