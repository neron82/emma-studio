"""Grapheme -> viseme timeline for German text, timed by Whisper word stamps.

Whisper liefert pro Wort [t0, t1]; wir verteilen diese Spanne auf die
Lautfolge des Wortes (Gewichte: Vokale lang, Friktive kurz, Verschlüsse
kurz-closed). Ergebnis: flache Liste [{v, t0, t1}] in Sekunden relativ zum
Chunk-Beginn — der Client setzt sie 1:1 auf die Expression-Kanäle.

Visem-Labels entsprechen den VRM-Presets, die glb_to_vrm.py bindet:
aa, i, e, o, u — plus 'mbp' (= Lippen geschlossen, für Labiale/Verschlußlaute).
"""
from __future__ import annotations

# Einzellaute: Graphem -> (Label, relatives Gewicht)
_SINGLE = {
    "a": ("aa", 1.0), "e": ("e", 0.75), "i": ("i", 0.65),
    "o": ("o", 0.85), "u": ("u", 0.85), "y": ("i", 0.5),
    "ü": ("i", 0.55), "ö": ("o", 0.65), "ä": ("e", 0.7),
    # Friktive/Nasalen: leicht geöffnet, hell
    "f": ("i", 0.28), "v": ("i", 0.28), "s": ("i", 0.28), "ß": ("i", 0.28),
    "z": ("i", 0.28), "w": ("u", 0.42), "j": ("i", 0.38),
    "n": ("i", 0.24), "m": ("mbp", 0.30), "l": ("aa", 0.26), "r": ("o", 0.26),
    # Verschlüsse: kurz geschlossen
    "b": ("mbp", 0.30), "p": ("mbp", 0.30), "t": ("mbp", 0.26),
    "d": ("mbp", 0.26), "k": ("mbp", 0.26), "g": ("mbp", 0.26), "c": ("mbp", 0.20),
    "h": None,  # nur Längungs-/Aspirationszeichen — kein eigener Laut
}

# Digraphe zuerst matchen
_DIGRAPHS = {
    "sch": [("i", 0.32)], "ch": [("i", 0.26)],
    "ei": [("aa", 0.75), ("i", 0.70)], "ai": [("aa", 0.75), ("i", 0.70)],
    "au": [("aa", 0.75), ("o", 0.75)], "eu": [("o", 0.65), ("i", 0.65)],
    "äu": [("o", 0.65), ("i", 0.65)], "ie": [("i", 1.0)],
    "ee": [("e", 1.0)], "oo": [("u", 1.0)],
    "ah": [("aa", 1.1)], "eh": [("e", 0.9)], "ih": [("i", 0.85)],
    "oh": [("o", 1.0)], "uh": [("u", 1.0)], "äh": [("e", 0.75)],
    "öh": [("o", 0.7)], "üh": [("i", 0.55)],
}


def _word_tokens(word: str) -> list[tuple[str, float]]:
    """Wort -> [(label, gewicht)] laut Näherungsregeln."""
    w = word.lower().replace("—", "").replace("-", "")
    out: list[tuple[str, float]] = []
    i = 0
    while i < len(w):
        ch = w[i]
        if ch.isdigit():
            out.append(("e", 0.6))  # gesprochene Zahlen: grob 'e'-lastig
            i += 1
            continue
        if not ch.isalpha():
            i += 1
            continue
        tri = w[i:i + 3]
        di = w[i:i + 2]
        if di == "qu":
            out.append(("k", 0.2)); out.append(("u", 0.5))
            i += 2
            continue
        if tri in _DIGRAPHS:
            out.extend(_DIGRAPHS[tri]); i += 3; continue
        if di in _DIGRAPHS:
            out.extend(_DIGRAPHS[di]); i += 2; continue
        tok = _SINGLE.get(ch)
        if tok:
            out.append(tok)
        i += 1
    return out or [("aa", 0.5)]


def word_to_visemes(word: str, t0: float, t1: float,
                    *, min_seg: float = 0.03) -> list[dict]:
    """Verteilt die Wortspanne [t0,t1] auf seine Lautfolge."""
    tokens = _word_tokens(word)
    total_w = sum(wgt for _, wgt in tokens) or 1.0
    span = max(0.05, t1 - t0)
    out = []
    cursor = t0
    for label, wgt in tokens:
        dur = max(min_seg, span * wgt / total_w)
        seg = {"v": label, "t0": round(cursor, 3), "t1": round(cursor + dur, 3)}
        cursor += dur
        out.append(seg)
    # letzte Segmente ggf. auf t1 stauchen (Rundungsdreck entfernen)
    overflow = out[-1]["t1"] - (t1 + 0.05)
    if overflow > 0:
        shrink = (out[-1]["t1"] - out[-1]["t0"]) - overflow
        if shrink > min_seg:
            out[-1]["t1"] = round(out[-1]["t0"] + shrink, 3)
    return out


def build_viseme_timeline(words: list[dict]) -> list[dict]:
    """Whisper-Wörter [{'word','start','end'}] -> flache Visem-Timeline."""
    tl: list[dict] = []
    for w in words:
        token = (w.get("word") or "").strip()
        t0, t1 = w.get("start"), w.get("end")
        if not token or t0 is None or t1 is None or t1 <= t0:
            continue
        tl.extend(word_to_visemes(token, float(t0), float(t1)))
    return tl
