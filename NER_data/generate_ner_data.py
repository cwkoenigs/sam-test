#!/usr/bin/env python3
"""
Synthetic call transcripts with character-offset entity annotations for
zero-shot entity extraction (GLiNER) and PCI / PII identification.

Same `agent:` / `customer:` turn format as auth_data and class_data. Every
record carries gold spans (start, end, text, label) for:

  PERSON_NAME       customer and agent names           PII
  PROVIDER_NAME     doctor / facility names
  PROVIDER_NPI      10-digit NPI (Luhn-valid with the 80840 prefix)
  DATE_OF_BIRTH     spoken / numeric DOB               PII
  ADDRESS           full street address                PII
  SSN               social security number             PII (sensitive)
  CARD_NUMBER       Luhn-valid PAN, 15-16 digits       PCI
  CARD_CVV          3-4 digit security code            PCI
  CARD_EXPIRATION   card expiry                        PCI
  MEMBER_ID         plan member id (distractor identifier)
  PHONE_NUMBER      10-digit phone                     PII

Entities are rendered the way speech-to-text would emit them (raw digits,
grouped digits, spoken digit words with the odd filler, dates as words or
numbers) and the rest of the transcript carries light ASR noise that never
touches an entity span, so offsets are exact. Card numbers, NPIs and SSNs
follow the real checksum / structure rules so a PCI detector can be tested on
realistic values. Nothing here is real data.

Usage:
    uv run python NER_data/generate_ner_data.py            # 100k calls
    uv run python NER_data/generate_ner_data.py --n 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# entity schema
# --------------------------------------------------------------------------- #

ENTITY_TYPES: Dict[str, Dict] = {
    "PERSON_NAME":     {"gliner": "person name", "pii": True, "pci": False},
    "PROVIDER_NAME":   {"gliner": "healthcare provider name", "pii": False, "pci": False},
    "PROVIDER_NPI":    {"gliner": "provider npi number", "pii": False, "pci": False},
    "DATE_OF_BIRTH":   {"gliner": "date of birth", "pii": True, "pci": False},
    "ADDRESS":         {"gliner": "street address", "pii": True, "pci": False},
    "SSN":             {"gliner": "social security number", "pii": True, "pci": False},
    "CARD_NUMBER":     {"gliner": "credit card number", "pii": False, "pci": True},
    "CARD_CVV":        {"gliner": "card security code", "pii": False, "pci": True},
    "CARD_EXPIRATION": {"gliner": "card expiration date", "pii": False, "pci": True},
    "MEMBER_ID":       {"gliner": "member id number", "pii": False, "pci": False},
    "PHONE_NUMBER":    {"gliner": "phone number", "pii": True, "pci": False},
}
PCI_TYPES = [k for k, v in ENTITY_TYPES.items() if v["pci"]]

SCENARIO_WEIGHTS = [("card_payment", 0.30), ("ssn_verification", 0.17), ("provider_lookup", 0.15),
                    ("address_update", 0.12), ("claim_inquiry", 0.14), ("payment_and_provider", 0.08),
                    ("general_no_pii", 0.04)]
NOISE_LEVEL_WEIGHTS = [("low", 0.40), ("medium", 0.40), ("high", 0.20)]

# --------------------------------------------------------------------------- #
# vocab
# --------------------------------------------------------------------------- #

FIRST = ("james mary robert patricia john jennifer michael linda david elizabeth william barbara richard susan "
         "joseph jessica thomas sarah charles karen lisa daniel nancy matthew betty anthony margaret mark sandra "
         "donald ashley steven kimberly paul emily andrew donna joshua michelle kenneth carol kevin amanda brian "
         "dorothy george melissa deborah ronald stephanie edward rebecca jason sharon jeffrey laura ryan cynthia "
         "jacob kathleen gary amy nicholas angela eric shirley jonathan anna stephen brenda larry pamela justin "
         "emma scott nicole brandon helen benjamin samantha samuel katherine gregory christine priya wei omar "
         "fatima diego mei amir rosa luis yuki kofi aisha carlos lin ana marcus elena tomas nina ravi").split()
LAST = ("smith johnson williams brown jones garcia miller davis rodriguez martinez hernandez lopez gonzalez wilson "
        "anderson thomas taylor moore jackson martin lee perez thompson white harris sanchez clark ramirez lewis "
        "robinson walker young allen king wright scott torres nguyen hill flores green adams nelson baker hall "
        "rivera campbell mitchell carter roberts gomez phillips evans turner diaz parker cruz edwards collins reyes "
        "stewart morris morales murphy cook rogers gutierrez ortiz morgan cooper peterson bailey reed kelly howard "
        "ramos kim cox ward richardson watson brooks chavez wood bennett gray mendoza ruiz hughes price alvarez "
        "castillo sanders patel myers long ross foster jimenez powell jenkins perry russell sullivan bell coleman "
        "butler henderson barnes fisher vasquez simmons romero jordan patterson alexander hamilton graham reynolds "
        "griffin wallace moreno west cole hayes bryant herrera gibson ellis tran medina aguilar stevens murray ford "
        "castro marshall owens harrison fernandez mcdonald woods washington kennedy wells vargas henry chen freeman "
        "webb tucker guzman burns crawford olson simpson porter hunter gordon mendez silva shaw snyder mason dixon "
        "munoz hunt hicks holmes palmer wagner black robertson boyd rose stone salazar fox warren mills meyer rice "
        "schmidt garza daniels ferguson nichols stephens soto weaver gardner payne grant dunn kelley spencer hawkins "
        "arnold pierce hansen peters santos hart bradley knight elliott cunningham duncan armstrong hudson carroll "
        "lane riley andrews alvarado ray delgado berry perkins hoffman johnston matthews pena richards contreras "
        "willis carpenter lawrence sandoval guerrero george chapman rios estrada ortega watkins greene nunez wheeler "
        "valdez harper burke larson santiago maldonado morrison franklin carlson austin dominguez carr lawson jacobs "
        "obrien lynch singh vega bishop montgomery oliver jensen harvey williamson gilbert dean sims espinoza howell "
        "li wong reid hanson le mccoy garrett burton fuller wang weber welch rojas lucas marquez fields park yang "
        "little banks padilla day walsh bowman schultz luna fowler mejia okafor haddad novak nakamura").split()
AGENTS = ("alex jordan taylor morgan casey riley sam jamie dana chris pat lee robin kim devon quinn avery skyler "
          "priya marcus elena tomas nina kwame sofia ravi denise brittany carla marco").split()
COMPANY = ["evergreen health plan", "harbor health", "meridian health plan", "northstar care", "bluewater health",
           "summit health plan", "lakeside community health", "cascade health"]
DOCTORS = ["patel", "nguyen", "okafor", "rivera", "chen", "johansson", "adeyemi", "kowalski", "schmidt", "haddad",
           "morales", "yamamoto", "singh", "oconnor", "fischer", "dubois", "nakamura", "alvarez"]
FACILITIES = ["lakeside orthopedics", "riverside imaging center", "northgate family medicine", "summit physical therapy",
              "harbor view cardiology", "pinecrest dermatology", "valley behavioral health", "oakwood dialysis center",
              "metro urgent care", "county general hospital", "westside womens health", "central pediatrics",
              "bayview internal medicine", "mountain ridge clinic", "sunrise eye care", "cedar park pharmacy"]
STREETS = "maple oak pine cedar elm walnut birch main park lake hill river ridge valley forest meadow sunset highland washington lincoln jefferson madison church school mill spring summit prospect union center grove orchard".split()
STYPES = ["street", "avenue", "road", "drive", "lane", "court", "boulevard", "way", "place", "circle"]
CITIES = [("springfield", "illinois"), ("riverside", "california"), ("franklin", "tennessee"), ("greenville", "south carolina"),
          ("fairview", "oregon"), ("madison", "wisconsin"), ("georgetown", "texas"), ("arlington", "virginia"),
          ("clinton", "iowa"), ("salem", "oregon"), ("bristol", "connecticut"), ("dayton", "ohio"), ("lexington", "kentucky"),
          ("burlington", "vermont"), ("milford", "massachusetts"), ("dover", "delaware"), ("hudson", "new york"),
          ("troy", "new york"), ("aurora", "colorado"), ("mesa", "arizona"), ("boulder", "colorado"), ("reno", "nevada"),
          ("tacoma", "washington"), ("eugene", "oregon"), ("tulsa", "oklahoma"), ("wichita", "kansas"), ("omaha", "nebraska"),
          ("boise", "idaho"), ("fargo", "north dakota"), ("savannah", "georgia"), ("tampa", "florida"), ("charlotte", "north carolina")]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",
        "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth",
       10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth", 16: "sixteenth",
       17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth", 30: "thirtieth"}
FILLERS = ["um", "uh", "like", "you know", "so", "okay", "hmm", "well", "yeah"]
TAGS = ["[inaudible]", "[crosstalk]", "[background noise]", "[static]", "[hold music]"]
ASR_CONFUSIONS = {"to": ["too", "two"], "for": ["four"], "there": ["their"], "our": ["are"], "your": ["you're"],
                  "know": ["no"], "right": ["write"], "card": ["cart", "car"], "code": ["coat"], "number": ["numbers"],
                  "social": ["so shall"], "security": ["secure it he"], "expiration": ["expire ation"], "address": ["a dress"],
                  "provider": ["provide her"], "claim": ["clam"], "member": ["remember"], "verify": ["very fy"],
                  "birth": ["berth"], "premium": ["premier"], "payment": ["pay mint"], "phone": ["fone"]}
NOISE = {"low": dict(sub=0.01, drop=0.01, filler=0.02, tag=0.02, entity_filler=0.05),
         "medium": dict(sub=0.04, drop=0.03, filler=0.05, tag=0.05, entity_filler=0.12),
         "high": dict(sub=0.08, drop=0.05, filler=0.09, tag=0.10, entity_filler=0.20)}

# --------------------------------------------------------------------------- #
# number helpers (checksums are real so detectors can be tested on valid values)
# --------------------------------------------------------------------------- #


def luhn_check_digit(partial: str) -> str:
    total = 0
    for i, ch in enumerate(reversed(partial)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def luhn_valid(num: str) -> bool:
    return luhn_check_digit(num[:-1]) == num[-1]


def gen_pan(rng: random.Random) -> Tuple[str, str]:
    brand = rng.choice(["visa", "visa", "mastercard", "mastercard", "amex", "discover"])
    if brand == "visa":
        prefix, length = "4", 16
    elif brand == "mastercard":
        prefix, length = rng.choice(["51", "52", "53", "54", "55"]), 16
    elif brand == "amex":
        prefix, length = rng.choice(["34", "37"]), 15
    else:
        prefix, length = "6011", 16
    body = prefix + "".join(str(rng.randint(0, 9)) for _ in range(length - len(prefix) - 1))
    return body + luhn_check_digit(body), brand


def gen_npi(rng: random.Random) -> str:
    body = rng.choice("12") + "".join(str(rng.randint(0, 9)) for _ in range(8))
    return body + luhn_check_digit("80840" + body)


def gen_ssn(rng: random.Random) -> str:
    area = rng.choice([a for a in range(1, 900) if a != 666])
    return f"{area:03d}{rng.randint(1, 99):02d}{rng.randint(1, 9999):04d}"


def two_digit_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t] if o == 0 else f"{TENS[t]} {ONES[o]}"


def year_words(y: int) -> str:
    if y < 2000:
        return f"nineteen {two_digit_words(y - 1900)}"
    if y == 2000:
        return "two thousand"
    if y < 2010:
        return f"two thousand {ONES[y - 2000]}"
    return f"twenty {two_digit_words(y - 2000)}"


def ordinal_words(d: int) -> str:
    if d in ORD:
        return ORD[d]
    t, o = divmod(d, 10)
    return f"{TENS[t]} {ORD[o]}"


def spoken_digits(s: str, rng: random.Random, filler_p: float = 0.0) -> str:
    out = []
    i = 0
    while i < len(s):
        d = s[i]
        run = 1
        while i + run < len(s) and s[i + run] == d:
            run += 1
        if run >= 2 and rng.random() < 0.4:
            out.append(f"{'double' if run == 2 or rng.random() < 0.5 else 'triple'} {ONES[int(d)]}")
            i += 2 if out[-1].startswith("double") else 3
        else:
            out.append(rng.choice(["oh", "zero"]) if d == "0" else ONES[int(d)])
            i += 1
        if filler_p and i < len(s) and rng.random() < filler_p:
            out.append(rng.choice(["um", "uh", "hold on", "let me see"]))
    return " ".join(out)


def grouped(s: str, rng: random.Random, sizes: Optional[Sequence[int]] = None, sep: str = " ") -> str:
    if sizes:
        parts, i = [], 0
        for k in sizes:
            parts.append(s[i:i + k])
            i += k
        return sep.join(parts)
    parts, i = [], 0
    while i < len(s):
        k = min(rng.choice([3, 4, 4, 2]), len(s) - i)
        parts.append(s[i:i + k])
        i += k
    return sep.join(parts)


# --------------------------------------------------------------------------- #
# entity renderers -> (text, format)
# --------------------------------------------------------------------------- #


def render_pan(pan: str, rng: random.Random, filler_p: float) -> Tuple[str, str]:
    sizes = [4, 6, 5] if len(pan) == 15 else [4, 4, 4, 4]
    r = rng.random()
    if r < 0.30:
        return grouped(pan, rng, sizes), "grouped_digits"
    if r < 0.45:
        return pan, "raw_digits"
    if r < 0.55:
        return grouped(pan, rng, sizes, sep="-"), "dashed_digits"
    if r < 0.85:
        return spoken_digits(pan, rng, filler_p), "spoken_words"
    cut = rng.choice([4, 8])
    return grouped(pan[:cut], rng, [4] * (cut // 4)) + " " + spoken_digits(pan[cut:], rng, filler_p), "mixed"


def render_cvv(cvv: str, rng: random.Random) -> Tuple[str, str]:
    return (cvv, "raw_digits") if rng.random() < 0.55 else (spoken_digits(cvv, rng), "spoken_words")


def render_expiry(m: int, y: int, rng: random.Random) -> Tuple[str, str]:
    r = rng.random()
    if r < 0.30:
        return f"{m:02d}/{y % 100:02d}", "mm/yy"
    if r < 0.45:
        return f"{m:02d} {y % 100:02d}", "mm yy"
    if r < 0.60:
        return f"{MONTHS[m - 1]} {y}", "month yyyy"
    if r < 0.80:
        return f"{MONTHS[m - 1]} {year_words(y)}", "month words"
    if r < 0.90:
        return f"{m:02d}{y % 100:02d}", "mmyy"
    return f"{spoken_digits(f'{m:02d}', rng)} {spoken_digits(f'{y % 100:02d}', rng)}", "spoken_words"


def render_ssn(ssn: str, rng: random.Random, filler_p: float) -> Tuple[str, str]:
    r = rng.random()
    if r < 0.30:
        return grouped(ssn, rng, [3, 2, 4], sep="-"), "dashed"
    if r < 0.55:
        return grouped(ssn, rng, [3, 2, 4]), "grouped_digits"
    if r < 0.65:
        return ssn, "raw_digits"
    return spoken_digits(ssn, rng, filler_p), "spoken_words"


def render_dob(m: int, d: int, y: int, rng: random.Random) -> Tuple[str, str]:
    r = rng.random()
    mn = MONTHS[m - 1]
    if r < 0.30:
        return f"{mn} {ordinal_words(d)} {year_words(y)}", "words"
    if r < 0.50:
        suf = "st" if d in (1, 21) else "nd" if d in (2, 22) else "rd" if d in (3, 23) else "th"
        return f"{mn} {d}{suf} {y}", "month dd yyyy"
    if r < 0.65:
        return f"{mn} {d} {y}", "month d yyyy"
    if r < 0.80:
        return f"{m:02d} {d:02d} {y}", "mm dd yyyy"
    if r < 0.90:
        return f"{m}/{d}/{y}", "m/d/yyyy"
    return f"{spoken_digits(f'{m:02d}', rng)} {spoken_digits(f'{d:02d}', rng)} {year_words(y)}", "spoken_words"


def render_phone(p: str, rng: random.Random) -> Tuple[str, str]:
    r = rng.random()
    if r < 0.45:
        return grouped(p, rng, [3, 3, 4]), "grouped_digits"
    if r < 0.60:
        return grouped(p, rng, [3, 3, 4], sep="-"), "dashed"
    if r < 0.70:
        return p, "raw_digits"
    return spoken_digits(p, rng), "spoken_words"


def render_member_id(mid: str, rng: random.Random) -> Tuple[str, str]:
    r = rng.random()
    if r < 0.4:
        return mid, "raw_digits"
    if r < 0.7:
        return grouped(mid, rng), "grouped_digits"
    return spoken_digits(mid, rng), "spoken_words"


def render_npi(npi: str, rng: random.Random) -> Tuple[str, str]:
    r = rng.random()
    if r < 0.5:
        return npi, "raw_digits"
    if r < 0.8:
        return grouped(npi, rng, [3, 3, 4]), "grouped_digits"
    return spoken_digits(npi, rng), "spoken_words"


def render_address(rng: random.Random) -> Tuple[str, str]:
    house = rng.choice([rng.randint(1, 99), rng.randint(100, 999), rng.randint(1000, 9999)])
    city, state = rng.choice(CITIES)
    z = f"{rng.randint(10000, 99999)}"
    unit = ""
    if rng.random() < 0.3:
        unit = " " + rng.choice(["apartment", "unit", "suite", "apt"]) + " " + rng.choice([str(rng.randint(1, 40)), f"{rng.randint(1, 12)}{rng.choice('abcd')}"])
    r = rng.random()
    hn = str(house) if r < 0.7 else spoken_digits(str(house), rng)
    zz = z if rng.random() < 0.7 else spoken_digits(z, rng)
    core = f"{hn} {rng.choice(STREETS)} {rng.choice(STYPES)}{unit}"
    r2 = rng.random()
    if r2 < 0.5:
        return f"{core} {city} {state} {zz}", "full"
    if r2 < 0.8:
        return f"{core} {city} {zz}", "no_state"
    return f"{core} {city}", "no_zip"


# --------------------------------------------------------------------------- #
# transcript model: turns are lists of segments; a segment may carry an entity
# --------------------------------------------------------------------------- #


@dataclass
class Seg:
    text: str
    label: Optional[str] = None
    fmt: str = ""
    role: str = ""


@dataclass
class Call:
    rng: random.Random
    first: str
    last: str
    agent: str
    company: str
    noise: Dict
    turns: List[Tuple[str, List[Seg]]] = field(default_factory=list)

    def A(self, *segs) -> None:
        self.turns.append(("agent", [s if isinstance(s, Seg) else Seg(s) for s in segs]))

    def C(self, *segs) -> None:
        self.turns.append(("customer", [s if isinstance(s, Seg) else Seg(s) for s in segs]))

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"


def E(text: str, label: str, fmt: str = "", role: str = "") -> Seg:
    return Seg(text, label, fmt, role)


def noisify_plain(text: str, rates: Dict, rng: random.Random) -> str:
    words = text.split()
    out: List[str] = []
    for w in words:
        if len(words) > 3 and len(w) <= 4 and rng.random() < rates["drop"]:
            continue
        if w in ASR_CONFUSIONS and rng.random() < rates["sub"]:
            w = rng.choice(ASR_CONFUSIONS[w])
        out.append(w)
        if rng.random() < rates["filler"]:
            out.append(rng.choice(FILLERS))
    if out and rng.random() < rates["tag"]:
        out.insert(rng.randint(0, len(out)), rng.choice(TAGS))
    return " ".join(out)


def render_call(call: Call) -> Tuple[str, List[Dict]]:
    """Join turns into one transcript string, applying noise to plain segments only; return exact entity offsets."""
    rng, rates = call.rng, call.noise
    items: List[object] = []            # str (plain text) or Seg (entity)
    for spk, segs in call.turns:
        items.append(f"{spk}:")
        for seg in segs:
            if seg.label is None:
                t = noisify_plain(seg.text, rates, rng)
                if t:
                    items.append(t)
            else:
                items.append(seg)
    parts: List[str] = []
    ents: List[Dict] = []
    pos = 0
    for k, it in enumerate(items):
        if k:
            parts.append(" ")
            pos += 1
        if isinstance(it, Seg):
            info = ENTITY_TYPES[it.label]
            ents.append({"start": pos, "end": pos + len(it.text), "text": it.text, "label": it.label,
                         "gliner_label": info["gliner"], "pci": info["pci"], "pii": info["pii"], "format": it.fmt, "role": it.role})
            parts.append(it.text)
            pos += len(it.text)
        else:
            parts.append(it)
            pos += len(it)
    return "".join(parts), ents


# --------------------------------------------------------------------------- #
# scenario building blocks
# --------------------------------------------------------------------------- #


def opening(c: Call) -> None:
    rng = c.rng
    c.A(rng.choice([f"thank you for calling {c.company} this is", f"{c.company} member services", f"hi you've reached {c.company} my name is"]),
        E(c.agent, "PERSON_NAME", "first", "agent"), rng.choice(["how can i help you today", "how can i help", "what can i do for you"]))
    intro = rng.choice(["hi this is", "yeah hi my name is", "hello it's", "hi um"])
    c.C(intro, E(c.name, "PERSON_NAME", "full", "customer"), rng.choice(["", "i have a question about my account", "i need some help", "i'm calling about my plan"]))


def verify_member(c: Call, with_dob: bool = True) -> None:
    rng = c.rng
    mid = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(8, 10)))
    c.A(rng.choice(["sure can i get your member id", "let me verify the account what's your member id", "okay and your member id please"]))
    mt, mf = render_member_id(mid, rng)
    c.C(rng.choice(["it's", "sure", "yeah", "um"]), E(mt, "MEMBER_ID", mf, "customer"))
    if with_dob:
        c.A(rng.choice(["thanks and your date of birth", "and can you confirm your date of birth", "date of birth to verify"]))
        m, d, y = rng.randint(1, 12), rng.randint(1, 28), rng.randint(1940, 2004)
        text, fmt = render_dob(m, d, y, rng)
        c.C(rng.choice(["", "it's", "i was born", "sure"]), E(text, "DATE_OF_BIRTH", fmt, "customer"))
    c.A(rng.choice([f"thank you {c.first} i have your account up", "perfect you're verified", "okay got it"]))


def closing(c: Call) -> None:
    rng = c.rng
    c.A(rng.choice(["is there anything else i can help with", "anything else today"]))
    c.C(rng.choice(["no that's it thanks", "no thank you", "that's all", "nope thanks"]))
    c.A(rng.choice([f"thank you for calling {c.company} have a good day", "take care goodbye", f"thanks {c.first} bye now"]))


def card_payment(c: Call) -> None:
    rng, fp = c.rng, c.noise["entity_filler"]
    pan, brand = gen_pan(rng)
    cvv = "".join(str(rng.randint(0, 9)) for _ in range(4 if brand == "amex" else 3))
    m, y = rng.randint(1, 12), rng.randint(2026, 2031)
    amt = f"{rng.randint(40, 900)} dollars" if rng.random() < 0.6 else f"${rng.randint(40, 900)}.{rng.randint(0, 99):02d}"
    c.C(rng.choice([f"i'd like to pay my premium of {amt} with a card", "can i make a payment over the phone", f"i need to pay the {amt} balance today"]))
    c.A(rng.choice(["sure i can take that what's the card number", "absolutely go ahead with the card number when you're ready",
                    "okay i'll need the card number the expiration and the security code"]))
    ptext, pfmt = render_pan(pan, rng, fp)
    c.C(rng.choice(["okay it's", "sure", "the number is", "it's a " + brand + " the number is", ""]), E(ptext, "CARD_NUMBER", pfmt, "customer"))
    if rng.random() < 0.25:   # agent reads back the last four (NOT a PAN; unlabeled distractor)
        c.A(f"okay ending in {pan[-4:]} is that right")
        c.C(rng.choice(["yes", "that's right", "correct"]))
    elif rng.random() < 0.15:  # correction: a second full PAN
        c.C("wait sorry that's my old card let me give you the other one")
        pan2, _ = gen_pan(rng)
        p2, f2 = render_pan(pan2, rng, fp)
        c.C(rng.choice(["it's", "okay"]), E(p2, "CARD_NUMBER", f2, "customer"))
    c.A(rng.choice(["and the expiration date", "expiration", "what's the expiration on that"]))
    etext, efmt = render_expiry(m, y, rng)
    c.C(rng.choice(["", "it's", "um"]), E(etext, "CARD_EXPIRATION", efmt, "customer"))
    c.A(rng.choice(["and the three digit security code on the back", "and the cvv", "security code please"]) if brand != "amex"
        else "and the four digit code on the front")
    ctext, cfmt = render_cvv(cvv, rng)
    c.C(rng.choice(["", "it's", "the code is"]), E(ctext, "CARD_CVV", cfmt, "customer"))
    if rng.random() < 0.5:
        c.A(rng.choice(["and the billing zip code", "billing address for the card"]))
        if rng.random() < 0.5:
            c.C(f"{rng.randint(10000, 99999)}")
        else:
            atext, afmt = render_address(rng)
            c.C(rng.choice(["it's", "same as my home"]), E(atext, "ADDRESS", afmt, "customer"))
    c.A(rng.choice([f"okay the payment of {amt} went through your confirmation number is {rng.randint(10000000, 99999999)}",
                    "that's processed you'll get a receipt by email", "done the payment posted"]))
    c.C(rng.choice(["great thank you", "perfect", "thanks"]))


def ssn_verification(c: Call) -> None:
    rng, fp = c.rng, c.noise["entity_filler"]
    c.C(rng.choice(["i need to update my information", "i'm locked out and need to verify my identity", "i'm calling about a letter i got"]))
    c.A(rng.choice(["i'll need to verify your identity first can i have your social security number",
                    "for security can you give me your social", "okay i need your social security number date of birth and address"]))
    stext, sfmt = render_ssn(gen_ssn(rng), rng, fp)
    c.C(rng.choice(["", "it's", "sure", "okay"]), E(stext, "SSN", sfmt, "customer"))
    c.A(rng.choice(["thank you and your date of birth", "and date of birth"]))
    m, d, y = rng.randint(1, 12), rng.randint(1, 28), rng.randint(1940, 2004)
    dt, df = render_dob(m, d, y, rng)
    c.C(rng.choice(["", "it's"]), E(dt, "DATE_OF_BIRTH", df, "customer"))
    c.A(rng.choice(["and the address on file", "and your home address"]))
    at, af = render_address(rng)
    c.C(rng.choice(["", "it's"]), E(at, "ADDRESS", af, "customer"))
    if rng.random() < 0.5:
        c.A("and a good callback number")
        ph = "".join(str(rng.randint(0, 9)) for _ in range(10))
        pt, pf = render_phone(ph, rng)
        c.C(rng.choice(["", "my cell is"]), E(pt, "PHONE_NUMBER", pf, "customer"))
    c.A(rng.choice(["thank you that all matches", "perfect you're verified what can i help with"]))
    c.C(rng.choice(["i need to change my email on the account", "can you send me a new id card", "i want to add my spouse to the plan"]))
    c.A(rng.choice(["done", "i've taken care of that", "that's updated"]))


def provider_lookup(c: Call) -> None:
    rng = c.rng
    prov = f"dr {rng.choice(DOCTORS)}" if rng.random() < 0.5 else rng.choice(FACILITIES)
    c.C(rng.choice(["is", "i want to check if", "can you tell me whether"]), E(prov, "PROVIDER_NAME", "", "provider"),
        rng.choice(["in network", "covered under my plan", "still taking the plan"]))
    c.A(rng.choice(["let me check do you have the npi number", "do you have the provider's npi", "let me search the directory"]))
    if rng.random() < 0.6:
        nt, nf = render_npi(gen_npi(rng), rng)
        c.C(rng.choice(["yeah it's", "the npi is", "um"]), E(nt, "PROVIDER_NPI", nf, "provider"))
    else:
        c.C(rng.choice(["no i just have the name", "no sorry"]))
    at, af = render_address(rng)
    c.A(rng.choice(["okay i see them at", "i found them the address is", "yes they're in network at"]), E(at, "ADDRESS", af, "provider"))
    if rng.random() < 0.5:
        ph = "".join(str(rng.randint(0, 9)) for _ in range(10))
        pt, pf = render_phone(ph, rng)
        c.A("their phone number is", E(pt, "PHONE_NUMBER", pf, "provider"))
    c.C(rng.choice(["great thanks", "okay that's the one", "perfect"]))
    if rng.random() < 0.4:
        other = f"dr {rng.choice(DOCTORS)}"
        c.C("what about", E(other, "PROVIDER_NAME", "", "provider"))
        c.A(rng.choice(["they're out of network", "yes also in network"]))


def address_update(c: Call) -> None:
    rng = c.rng
    c.C(rng.choice(["i moved and need to update my address", "i need to change the address on my account"]))
    c.A(rng.choice(["sure what's the new address", "okay go ahead with the new address"]))
    at, af = render_address(rng)
    c.C(rng.choice(["it's", "", "the new one is"]), E(at, "ADDRESS", af, "customer"))
    if rng.random() < 0.5:
        c.A("and the old address was")
        ot, of_ = render_address(rng)
        c.C(E(ot, "ADDRESS", of_, "customer"))
    c.A(rng.choice(["and is your phone number still the same", "any change to the phone number"]))
    ph = "".join(str(rng.randint(0, 9)) for _ in range(10))
    pt, pf = render_phone(ph, rng)
    c.C(rng.choice(["no it's now", "yes it's", "actually it's"]), E(pt, "PHONE_NUMBER", pf, "customer"))
    c.A(rng.choice(["okay that's updated", "all set your next statement goes to the new address"]))


def claim_inquiry(c: Call) -> None:
    rng = c.rng
    prov = f"dr {rng.choice(DOCTORS)}" if rng.random() < 0.5 else rng.choice(FACILITIES)
    c.C(rng.choice(["i'm checking on a claim from", "my claim from", "i have a question about a bill from"]), E(prov, "PROVIDER_NAME", "", "provider"))
    c.A(rng.choice(["okay let me look that up what was the date of service", "do you have the claim number"]))
    c.C(rng.choice([f"it was {MONTHS[rng.randint(0, 11)]} {rng.randint(1, 28)}", f"claim number {rng.randint(100000000, 9999999999)}", "i don't have it"]))
    if rng.random() < 0.4:
        nt, nf = render_npi(gen_npi(rng), rng)
        c.A("i see the claim under npi", E(nt, "PROVIDER_NPI", nf, "provider"))
    c.A(rng.choice(["that claim is still processing", "that was paid last week", "it was denied for a coding issue i'll send it back"]))
    if rng.random() < 0.3:
        c.C("is the patient on that claim me or my daughter")
        c.A("it's for", E(f"{rng.choice(FIRST)} {c.last}", "PERSON_NAME", "full", "dependent"))
    c.C(rng.choice(["okay thanks", "alright", "got it"]))


def payment_and_provider(c: Call) -> None:
    provider_lookup(c)
    c.C(c.rng.choice(["oh and while i have you can i pay my premium", "also i need to make a payment"]))
    card_payment(c)


def general_no_pii(c: Call) -> None:
    rng = c.rng
    c.C(rng.choice(["what's your mailing address for claims", "when does open enrollment start", "how do i download the app"]))
    c.A(rng.choice(["open enrollment starts november first", "the app is in the app store under our name",
                    f"claims go to po box {rng.randint(1000, 99999)}"]))
    c.C(rng.choice(["okay thanks", "great"]))


SCENARIOS = {"card_payment": card_payment, "ssn_verification": ssn_verification, "provider_lookup": provider_lookup,
             "address_update": address_update, "claim_inquiry": claim_inquiry, "payment_and_provider": payment_and_provider,
             "general_no_pii": general_no_pii}


def wchoice(rng: random.Random, pairs: Sequence[Tuple]) -> object:
    r = rng.random() * sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if r <= acc:
            return v
    return pairs[-1][0]


def build_call(rng: random.Random, idx: int) -> Dict:
    scenario = wchoice(rng, SCENARIO_WEIGHTS)
    level = wchoice(rng, NOISE_LEVEL_WEIGHTS)
    c = Call(rng=rng, first=rng.choice(FIRST), last=rng.choice(LAST), agent=rng.choice(AGENTS), company=rng.choice(COMPANY), noise=NOISE[level])
    opening(c)
    if scenario in ("card_payment", "claim_inquiry", "address_update", "payment_and_provider") and rng.random() < 0.8:
        verify_member(c, with_dob=rng.random() < 0.7)
    SCENARIOS[scenario](c)
    closing(c)
    transcript, ents = render_call(c)
    for e in ents:
        assert transcript[e["start"]:e["end"]] == e["text"], (idx, e)
    counts = Counter(e["label"] for e in ents)
    pci_types = sorted({e["label"] for e in ents if e["pci"]})
    return {"id": f"ner_{idx:06d}", "transcript": transcript, "entities": ents, "contains_pci": bool(pci_types),
            "pci_types": pci_types, "contains_pii": any(e["pii"] for e in ents), "entity_counts": dict(counts),
            "scenario": scenario, "noise_level": level, "n_turns": len(c.turns), "n_words": len(transcript.split())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--split", default="0.8,0.1,0.1")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()
    fracs = tuple(float(x) for x in args.split.split(","))
    rng = random.Random(args.seed)
    t0 = time.time()
    calls = [build_call(rng, i) for i in range(args.n)]
    ids = list(range(len(calls)))
    rng.shuffle(ids)
    n_tr, n_va = int(round(len(ids) * fracs[0])), int(round(len(ids) * fracs[1]))
    split_of = {}
    for k, i in enumerate(ids):
        split_of[i] = "train" if k < n_tr else "validation" if k < n_tr + n_va else "test"
    os.makedirs(args.out, exist_ok=True)
    files = {s: open(os.path.join(args.out, f"{s}.jsonl"), "w", encoding="utf-8") for s in ("train", "validation", "test")}
    sample = open(os.path.join(args.out, "sample.jsonl"), "w", encoding="utf-8")
    label_counts, pci_calls, scen, noise, fmts = Counter(), 0, Counter(), Counter(), Counter()
    words = []
    try:
        for i, c in enumerate(calls):
            c["split"] = split_of[i]
            files[c["split"]].write(json.dumps(c, ensure_ascii=False) + "\n")
            if i < args.sample:
                sample.write(json.dumps(c, ensure_ascii=False) + "\n")
            label_counts.update(c["entity_counts"])
            pci_calls += c["contains_pci"]
            scen[c["scenario"]] += 1
            noise[c["noise_level"]] += 1
            words.append(c["n_words"])
            for e in c["entities"]:
                fmts[f"{e['label']}:{e['format']}"] += 1
    finally:
        for f in files.values():
            f.close()
        sample.close()
    words.sort()
    stats = {"generator": os.path.basename(__file__), "seed": args.seed, "n_calls": args.n,
             "entity_types": ENTITY_TYPES, "pci_types": PCI_TYPES,
             "splits": {s: sum(1 for i in split_of if split_of[i] == s) for s in ("train", "validation", "test")},
             "entity_counts": dict(label_counts), "entities_total": sum(label_counts.values()),
             "calls_with_pci": pci_calls, "scenarios": dict(scen), "noise_levels": dict(noise),
             "entity_formats": dict(sorted(fmts.items())),
             "words_per_call": {"min": words[0], "p50": words[len(words) // 2], "p90": words[int(len(words) * .9)], "max": words[-1]},
             "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"entity_types": ENTITY_TYPES, "gliner_labels": [v["gliner"] for v in ENTITY_TYPES.values()],
                   "gliner_to_label": {v["gliner"]: k for k, v in ENTITY_TYPES.items()}, "pci_types": PCI_TYPES}, f, indent=2)
    print(json.dumps({k: stats[k] for k in ("n_calls", "entities_total", "calls_with_pci", "splits", "seconds")}))
    print("entity counts:", dict(label_counts))
    print("scenarios:", dict(scen), "| words/call:", stats["words_per_call"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
