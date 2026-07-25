#!/usr/bin/env python3
"""Generate FURIA vs G2 void-grubs situation brief PDF for public posting."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "pdf" / "furia_g2_voidgrubs_situation_brief.pdf"

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#555555")
RULE = colors.HexColor("#cccccc")
SOFT = colors.HexColor("#f4f4f4")
ACCENT = colors.HexColor("#2c4a6e")


def styles():
    base = getSampleStyleSheet()
    s = {
        "title": ParagraphStyle(
            "title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            textColor=INK,
            spaceAfter=6,
            alignment=TA_CENTER,
        ),
        "sub": ParagraphStyle(
            "sub",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=10,
            leading=13,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            textColor=ACCENT,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=14,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            textColor=INK,
            leftIndent=12,
            spaceAfter=3,
        ),
        "note": ParagraphStyle(
            "note",
            parent=base["Normal"],
            fontName="Times-Italic",
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
        "cell": ParagraphStyle(
            "cell",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=8.5,
            leading=11,
            textColor=INK,
        ),
        "cellh": ParagraphStyle(
            "cellh",
            parent=base["Normal"],
            fontName="Times-Bold",
            fontSize=8.5,
            leading=11,
            textColor=INK,
        ),
    }
    return s


def p(text: str, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def table(data, col_widths, sty):
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SOFT),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build():
    s = styles()
    story = []

    story.append(p("FURIA vs G2 — Void Grubs Situation Brief", s["title"]))
    story.append(
        p(
            "Match case study · Live esports map-state feed · Decision window ~7:30–9:00<br/>"
            "For drafting a public post · Not a coaching mandate for a single callout",
            s["sub"],
        )
    )

    story.append(p("1. What this note is answering", s["h1"]))
    story.append(
        p(
            "The useful question is not only “should FURIA take the river fight once they are "
            "already there?” It is whether they needed to be there at all. From about 7:30, "
            "before a real pit contest exists, they can stay on safer gold and let the camp go. "
            "This brief lays out roster fundamentals, the gold path, reachability of real "
            "resources (waves / camps / plates), and how that sits against common arguments "
            "about scrap value and “having combat items.”",
            s["body"],
        )
    )
    story.append(
        p(
            "Limits, stated up front: this uses a Riot Live Stats event dump for one professional "
            "map (positions about once per second, items, gold, objective kills). It is not a "
            "full .rofl combat decode. Ability cast events here do not list AoE hit targets, so "
            "this note does not claim a solved teamfight damage model. Gold and geography are "
            "the hard evidence; kit math is supporting context only.",
            s["note"],
        )
    )

    story.append(p("2. Match fundamentals", s["h1"]))
    story.append(
        p(
            "Side convention in the feed: team 100 = FURIA (Order / blue side), team 200 = G2 "
            "(Chaos / red side). Patch family 16.13. Void Grubs taken by G2 at roughly 8:43–8:59 "
            "(all three).",
            s["body"],
        )
    )

    story.append(p("2.1 Roster", s["h2"]))
    roster = [
        [p("Side", s["cellh"]), p("Role", s["cellh"]), p("Player", s["cellh"]), p("Champion", s["cellh"])],
        [p("FURIA", s["cell"]), p("Top", s["cell"]), p("Guigo", s["cell"]), p("Ambessa", s["cell"])],
        [p("FURIA", s["cell"]), p("Jungle", s["cell"]), p("Tatu", s["cell"]), p("Lee Sin", s["cell"])],
        [p("FURIA", s["cell"]), p("Middle", s["cell"]), p("Tutsz", s["cell"]), p("Akali", s["cell"])],
        [p("FURIA", s["cell"]), p("Bottom", s["cell"]), p("Ayu", s["cell"]), p("Jhin", s["cell"])],
        [p("FURIA", s["cell"]), p("Support", s["cell"]), p("JoJo", s["cell"]), p("Camille", s["cell"])],
        [p("G2", s["cell"]), p("Top", s["cell"]), p("BrokenBlade", s["cell"]), p("Gnar", s["cell"])],
        [p("G2", s["cell"]), p("Jungle", s["cell"]), p("SkewMond", s["cell"]), p("Naafiri", s["cell"])],
        [p("G2", s["cell"]), p("Middle", s["cell"]), p("Caps", s["cell"]), p("Galio", s["cell"])],
        [p("G2", s["cell"]), p("Bottom", s["cell"]), p("Hans Sama", s["cell"]), p("Syndra", s["cell"])],
        [p("G2", s["cell"]), p("Support", s["cell"]), p("Labrov", s["cell"]), p("Leona", s["cell"])],
    ]
    story.append(table(roster, [28 * mm, 28 * mm, 40 * mm, 40 * mm], s))
    story.append(Spacer(1, 6))
    story.append(
        p(
            "Composition note that matters for river priority: G2 mid/bot (Galio / Syndra) are "
            "structurally strong first-movers into a grouped river. FURIA support Camille is "
            "still low-level in this window (level 4 around the collapse) and later loses Flash "
            "for a long timer. That is part of why “we have a Tunneler” is not the whole fight.",
            s["body"],
        )
    )

    story.append(p("3. Core claim", s["h1"]))
    story.append(
        p(
            "<b>At 7:30 there is no fight FURIA is forced to take.</b> Gold lead is about "
            "<b>+1,247</b>. The map has them in lanes and jungle away from the pit. JoJo first "
            "enters a 2,200-unit pit radius around <b>7:55</b>; Tatu around <b>8:05</b>. They "
            "walk into the skirmish geography. The later 4v5 is a consequence of that choice, "
            "not an unavoidable spawn-state.",
            s["body"],
        )
    )

    story.append(p("3.1 Where everyone is at ~7:30", s["h2"]))
    pos730 = [
        [p("Player", s["cellh"]), p("Location / job", s["cellh"]), p("Nearest real resource", s["cellh"])],
        [
            p("Guigo (Ambessa)", s["cell"]),
            p("Top side", s["cell"]),
            p("Top wave ~3.3s · pit ~10s", s["cell"]),
        ],
        [
            p("Tutsz (Akali)", s["cell"]),
            p("Mid", s["cell"]),
            p("Mid wave ~0.6s · pit ~12s", s["cell"]),
        ],
        [
            p("Tatu (Lee Sin)", s["cell"]),
            p("Bot-side jungle (near Krugs)", s["cell"]),
            p("Blue buff UP · pit ~27s", s["cell"]),
        ],
        [
            p("Ayu (Jhin)", s["cell"]),
            p("Bot lane", s["cell"]),
            p("Bot farm · pit ~30s", s["cell"]),
        ],
        [
            p("JoJo (Camille)", s["cell"]),
            p("Bot lane", s["cell"]),
            p("With ADC · pit ~27s", s["cell"]),
        ],
    ]
    story.append(table(pos730, [42 * mm, 55 * mm, 55 * mm], s))
    story.append(
        p(
            "Travel times use effective movement speed including Boots where owned "
            "(Ambessa/Jhin/Camille +25). Lee Sin dashes make jungle paths shorter than pure "
            "walking ETAs — that strengthens the “take Blue/Gromp instead of river” option, "
            "not the contest.",
            s["note"],
        )
    )

    story.append(p("4. Gold path (level + direction, no exponential)", s["h1"]))
    story.append(
        p(
            "Treat strength here as gold position and the lead’s recent direction. "
            "L = FURIA total gold − G2 total gold. Direction = ordinary least-squares slope "
            "of L over trailing windows. No exponential forecast — short slopes are warnings, "
            "not doom curves.",
            s["body"],
        )
    )
    gold = [
        [p("Window", s["cellh"]), p("Lead change rate", s["cellh"]), p("Read", s["cellh"])],
        [p("7:00–7:30", s["cell"]), p("+509 g/min", s["cell"]), p("Still building the lead", s["cell"])],
        [p("7:30–8:00", s["cell"]), p("−81 g/min", s["cell"]), p("Rotate begins / tempo softens", s["cell"])],
        [p("8:00–8:21", s["cell"]), p("−350 g/min", s["cell"]), p("Commitment cost shows up", s["cell"])],
        [p("8:21–8:44", s["cell"]), p("−2,392 g/min", s["cell"]), p("Fight collapses the lead", s["cell"])],
    ]
    story.append(table(gold, [40 * mm, 40 * mm, 70 * mm], s))
    story.append(Spacer(1, 6))
    milestones = [
        [p("Clock", s["cellh"]), p("Lead L", s["cellh"]), p("Note", s["cellh"])],
        [p("7:30", s["cell"]), p("+1,247g", s["cell"]), p("Peak-ish pre-rotate", s["cell"])],
        [p("8:00", s["cell"]), p("+1,208g", s["cell"]), p("JoJo already near river", s["cell"])],
        [p("8:21", s["cell"]), p("+1,183g", s["cell"]), p("JoJo ~16% HP, Flash down", s["cell"])],
        [p("8:35", s["cell"]), p("+577g", s["cell"]), p("Tutsz collapsing in pit", s["cell"])],
        [p("8:43–8:44", s["cell"]), p("+49g", s["cell"]), p("First Void Grub to G2", s["cell"])],
    ]
    story.append(table(milestones, [35 * mm, 30 * mm, 85 * mm], s))
    story.append(
        p(
            "At 8:21 specifically: medium trend (3 min) still about <b>+347 g/min</b>, but the "
            "last 60s already about <b>−149 g/min</b>. That matches local setup pressure "
            "(Camille dying into the river), not a story that FURIA was behind on gold.",
            s["body"],
        )
    )

    story.append(PageBreak())
    story.append(p("5. What “leave” actually buys from these positions", s["h1"]))
    story.append(
        p(
            "Public charts sometimes price “2 waves + a plate” as a generic ADC leave package. "
            "That package assumes the ADC is already in lane. From mid-map / river, bot plates "
            "are often too far for the grub window. The realistic menu is whatever is in range "
            "on movement time.",
            s["body"],
        )
    )
    story.append(p("5.1 Mechanical scrap vs farm (article fundamentals)", s["h2"]))
    story.append(
        p(
            "Void Grub cash scrap is <b>30g × 3 = 90g</b> (plus XP / Touch pressure as separate "
            "bounds). Mapped through an early gold→win association, a preferred scrap bundle "
            "(90g + brief Touch) is on the order of <b>~+1.9 percentage points</b> of map win "
            "probability — not a huge bar by itself. Early wave gold is about <b>~121g</b> per "
            "average wave; one outer plate is <b>120g</b> when actually taken.",
            s["body"],
        )
    )
    leave = [
        [
            p("Package", s["cellh"]),
            p("Gold", s["cellh"]),
            p("~pp @ even (gold map)", s["cellh"]),
            p("This game @7:30–8:45?", s["cellh"]),
        ],
        [
            p("1 laner × 1 wave", s["cell"]),
            p("121g", s["cell"]),
            p("+2.0", s["cell"]),
            p("Yes — Guigo or Tutsz", s["cell"]),
        ],
        [
            p("2 laners × 1 wave", s["cell"]),
            p("241g", s["cell"]),
            p("+4.0", s["cell"]),
            p("Yes if mid+top stay", s["cell"]),
        ],
        [
            p("2 waves + 1 plate", s["cell"]),
            p("361g", s["cell"]),
            p("+6.0", s["cell"]),
            p("Not as a bot-ADC package from river", s["cell"]),
        ],
        [
            p("Blue + Gromp (Tatu)", s["cell"]),
            p("~185g", s["cell"]),
            p("~+3.1", s["cell"]),
            p("Yes — Blue UP; Gromp ready ~7:54", s["cell"]),
        ],
        [
            p("3 Grubs scrap (gift)", s["cell"]),
            p("90g", s["cell"]),
            p("~+1.9", s["cell"]),
            p("What G2 received", s["cell"]),
        ],
    ]
    story.append(table(leave, [45 * mm, 22 * mm, 40 * mm, 50 * mm], s))
    story.append(Spacer(1, 6))
    story.append(
        p(
            "A coordinated leave from the 7:30 map (top wave + mid wave + Blue/Gromp, bot stays "
            "bot) is roughly <b>~300–400g</b> of real farm without needing a plate fantasy — "
            "already larger than scrap, and it avoids walking a +1.2k lead into a skirmish.",
            s["body"],
        )
    )

    story.append(p("5.2 Item completion if they take the near gold", s["h2"]))
    story.append(
        p(
            "Around 8:21 (for illustration of build direction once the rotate has started), "
            "next-item progress as (owned recipe components + purse) / item cost:",
            s["body"],
        )
    )
    items = [
        [p("Player", s["cellh"]), p("Building toward", s["cellh"]), p("Now", s["cellh"]), p("With leave gold", s["cellh"])],
        [p("Tatu", s["cell"]), p("Sundered Sky (3,100g)", s["cell"]), p("~87.5%", s["cell"]), p("~93% via Blue+Gromp", s["cell"])],
        [p("Tutsz", s["cell"]), p("Hextech Gunblade (3,000g)", s["cell"]), p("~69%", s["cell"]), p("~73% via mid wave", s["cell"])],
        [p("Ayu", s["cell"]), p("Youmuu's (2,800g)", s["cell"]), p("~74%", s["cell"]), p("~78% via bot/near farm", s["cell"])],
        [p("Guigo", s["cell"]), p("Eclipse (2,900g)", s["cell"]), p("~62%", s["cell"]), p("~66% via top wave", s["cell"])],
        [p("JoJo", s["cell"]), p("Sundered Sky (later)", s["cell"]), p("~43%", s["cell"]), p("0 leave EV (functionally out)", s["cell"])],
    ]
    story.append(table(items, [28 * mm, 48 * mm, 22 * mm, 55 * mm], s))
    story.append(
        p(
            "Build intentions are taken from what they actually completed later in the game "
            "(not guessed). Tutsz finished <b>Hextech Gunblade</b> at 12:30 (Alternator path), "
            "not Rocketbelt. Guigo finished <b>Eclipse</b> at 12:80. Ayu finished <b>Youmuu's</b> "
            "at 11:44 then Collector at 18:56. Tatu finished <b>Sundered Sky</b> at 9:90 then "
            "Black Cleaver at 22:95. JoJo finished Sundered much later (18:84).",
            s["note"],
        )
    )

    story.append(p("5.3 Gold position vs fighting strength @8:21", s["h1"]))
    story.append(
        p(
            "Scoreboard lead (+1,183g) overstates river readiness. Valuing inventories at shop "
            "prices (excluding potions/wards/quests): FURIA held about <b>10,850g</b> in "
            "items on the board vs G2's <b>8,875g</b> (inventory lead ~+1,975g) while G2 held "
            "more unspent purse (~−407g purse lead for FURIA). So FURIA had converted more "
            "economy into components — Tunneler, Alternator, Dirk, Rectrix, Pickaxe, etc. — "
            "than the raw lead alone shows.",
            s["body"],
        )
    )
    story.append(
        p(
            "But JoJo's Camille is level 4, ~16% HP, Flash down, with ~1,800g of that inventory "
            "functionally absent. Discount her and the participating combat-item pools are "
            "about <b>9,050g vs 8,875g</b> — nearly even. The global lead is real; the river "
            "fight is not a +1.2k-advantage fight.",
            s["body"],
        )
    )

    story.append(p("5.4 Better paths from 7:30 (before the fight exists)", s["h1"]))
    story.append(
        p(
            "The decision that matters starts before 8:21. At 7:30 FURIA are not forced into "
            "a pit contest. Alternative paths that are more valuable in gold:",
            s["body"],
        )
    )
    paths = [
        [p("Path", s["cellh"]), p("Who does what", s["cellh"]), p("Gold / lead effect", s["cellh"])],
        [
            p("A. Full leave (best)", s["cell"]),
            p(
                "Guigo stays top waves; Tutsz stays mid waves; Tatu takes Blue (UP) then "
                "Gromp (~7:54); Ayu+JoJo stay bot (Ayu was printing ~950 gpm just before).",
                s["cell"],
            ),
            p(
                "~300–500g real farm in the window + keep the ~+1.2k lead trajectory "
                "instead of the −1,134 collapse.",
                s["cell"],
            ),
        ],
        [
            p("B. Soft leave", s["cell"]),
            p(
                "Never send JoJo first (~7:55). Lanes freeze/farm; only Tatu takes Blue/Gromp "
                "if free.",
                s["cell"],
            ),
            p("~185g jungle package + avoid creating the skirmish.", s["cell"]),
        ],
        [
            p("C. What they did", s["cell"]),
            p("JoJo walks pit ~7:55 → group follows → contest.", s["cell"]),
            p("Lead +1,208 @8:00 → +49 at first grub; G2 3–0 grubs.", s["cell"]),
        ],
    ]
    story.append(table(paths, [32 * mm, 70 * mm, 50 * mm], s))

    story.append(p("5.5 How leave gold converts into their real builds", s["h2"]))
    story.append(
        p(
            "Using each player's observed gold/min from 8:21 until their first legendary, "
            "banking the leave package shifts completion only modestly on the clock — the "
            "bigger conversion is not losing the team lead that funds everything after:",
            s["body"],
        )
    )
    builds = [
        [
            p("Player", s["cellh"]),
            p("Intended first item (actual)", s["cellh"]),
            p("Actual finish", s["cellh"]),
            p("If leave gold banked", s["cellh"]),
        ],
        [
            p("Tatu", s["cell"]),
            p("Sundered Sky", s["cell"]),
            p("9:90", s["cell"]),
            p("~9:41 (−29s) via +185g", s["cell"]),
        ],
        [
            p("Ayu", s["cell"]),
            p("Youmuu's Ghostblade", s["cell"]),
            p("11:44", s["cell"]),
            p("~11:10 (−21s) via +121g", s["cell"]),
        ],
        [
            p("Tutsz", s["cell"]),
            p("Hextech Gunblade", s["cell"]),
            p("12:30", s["cell"]),
            p("~11:92 (−23s) via +121g", s["cell"]),
        ],
        [
            p("Guigo", s["cell"]),
            p("Eclipse", s["cell"]),
            p("12:80", s["cell"]),
            p("~12:32 (−29s) via +121g", s["cell"]),
        ],
        [
            p("JoJo", s["cell"]),
            p("Sundered Sky", s["cell"]),
            p("18:84", s["cell"]),
            p("unchanged in leave EV; better path is not dying @8:21", s["cell"]),
        ],
    ]
    story.append(table(builds, [28 * mm, 42 * mm, 28 * mm, 55 * mm], s))
    story.append(
        p(
            "Second items that actually appeared later: Ayu Collector (18:56), Guigo Mercury's "
            "then more components toward a second fighter item, Tatu Black Cleaver (22:95), "
            "Tutsz Haunting Guise (20:34) toward a pen item. The leave decision mainly protects "
            "the economy that pays for those spikes; the 20–30s first-item pull-forward is "
            "real but secondary to not lighting −1.1k lead on fire.",
            s["note"],
        )
    )

    story.append(p("6. What actually happened on the contest", s["h1"]))
    story.append(
        p(
            "G2 takes all three Void Grubs (~8:43–8:59). Peak bodies near pit around 8:35 look "
            "like a messy multi-man clash (FURIA already down Camille’s effective presence). "
            "Deaths in the window include JoJo and Tutsz. Team gold lead moves from about "
            "<b>+1,208 at 8:00</b> to <b>+49 at first grub</b> (on the order of <b>−1,150g</b> "
            "of lead). That is the realized cost of the path.",
            s["body"],
        )
    )
    story.append(
        p(
            "Leave EV in the same gold-lead units (illustrative): keep ~+1,183 at 8:21, bank "
            "~+350g reachable farm, gift ~90g scrap → on the order of <b>+260 net</b> to the "
            "lead if the camp walks cleanly. Contest realized about <b>−1,134</b> lead from "
            "8:21 to first grub. Even generous uncertainty on exact farm gold does not close "
            "that gap.",
            s["body"],
        )
    )

    story.append(p("7. How this speaks to the public arguments", s["h1"]))
    story.append(p("7.1 “Contest is rational because Lee has Tunneler”", s["h2"]))
    story.append(
        p(
            "Tunneler is purchased around <b>7:55</b> — the same minute JoJo first reaches pit "
            "radius. It does not justify starting the rotate from 7:30 while bot is ~27s away "
            "and mid/top have free waves. On the math: early Tunneler is a small combat bump "
            "(on the order of +15 AD and +150 HP in the usual reading). Against armor, that is "
            "not a clean flip of a bad river fight into a good one.",
            s["body"],
        )
    )
    story.append(p("7.2 “Grubs are huge / must deny”", s["h2"]))
    story.append(
        p(
            "Cash scrap is 90g. Touch-of-the-Void pressure is real but bounded; sandbox-style "
            "extremes still show early plate burn is modest unless you invent inhuman hit "
            "counts. Empirical map-win associations for taking all three are multi-pp when "
            "you condition on early state — and largely run through the later gold/plate "
            "tempo path — but that still does not pay for opting into a clearly worse "
            "skirmish when safer gold is on the ground behind you.",
            s["body"],
        )
    )
    story.append(p("7.3 “Doing nothing isn’t a plan”", s["h2"]))
    story.append(
        p(
            "From 7:30, “do nothing” on the river is not idle: it is actively taking top/mid "
            "waves, Blue/Gromp, and bot farm while the lead is still expanding (+509 g/min in "
            "the prior half-minute window). Preserving a +1.2k lead into a better timer is a "
            "plan. Walking that lead into Galio/Syndra river priority with a level-4 Flashless "
            "Camille is also a plan — a worse one on these numbers.",
            s["body"],
        )
    )

    story.append(p("8. Posting takeaways (short)", s["h1"]))
    for line in [
        "1. Rosters: FURIA Ambessa/Lee/Akali/Jhin/Camille vs G2 Gnar/Naafiri/Galio/Syndra/Leona.",
        "2. Separate scoreboard lead (L) from fighting strength — JoJo's ~1.8k inventory was not in the fight.",
        "3. At 7:30 (+1,247g) no forced contest; bot ~30s away. JoJo opts in ~7:55.",
        "4. Better path: top/mid waves + Blue/Gromp + bot stays (~300–500g) vs lighting the lead.",
        "5. Intended spikes (actual): Sundered 9:90, Youmuu's 11:44, Gunblade 12:30, Eclipse 12:80.",
        "6. Leave gold pulls those ~20–30s earlier; the main EV is not losing −1.1k lead.",
        "7. Scrap ~90g is small next to that collapse; Tunneler at 7:55 does not justify the 7:30 rotate.",
    ]:
        story.append(p(line, s["bullet"]))

    story.append(Spacer(1, 12))
    story.append(
        p(
            "Closing line: FURIA were not behind on gold. They converted a real lead into a "
            "river fight after short-term momentum had already flipped, with Camille already "
            "functionally removed — and they could have taken more valuable gold from 7:30 "
            "without creating that fight at all.",
            s["body"],
        )
    )
    story.append(
        p(
            "Numbers are from one Live Stats professional map replay plus wiki/item constants "
            "for wave/plate/scrap. Treat kit-damage rhetoric as secondary to gold and "
            "geography unless a full combat log is available.",
            s["note"],
        )
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 12 * mm, A4[0] - 18 * mm, 12 * mm)
        canvas.setFont("Times-Roman", 8)
        canvas.setFillColor(MUTED)
        canvas.drawCentredString(A4[0] / 2, 7 * mm, f"FURIA vs G2 void-grubs brief · {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=18 * mm,
        title="FURIA vs G2 — Void Grubs Situation Brief",
        author="parlay-risk-sim",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    print(OUT)


if __name__ == "__main__":
    build()
