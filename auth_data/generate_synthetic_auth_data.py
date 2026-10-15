#!/usr/bin/env python3
"""
Synthetic call-transcript generator for authentication-chunk classification.

Generates N agent/customer conversations (speech-to-text style: lowercase, no
punctuation, light disfluencies). Each conversation authenticates the caller in
one of two ways:

  * ID path    (~80%): the customer utters a 9-12 digit account / member ID.
  * OTHER path (~20%): the customer says they do NOT have their ID number, and
                       the agent verifies name + address + date of birth instead.

Conversations are then chunked into windows that fit a bert-base-uncased token
budget and every chunk is labelled:

  ID_AUTH     chunk contains the spoken 9-12 digit identifier exchange
  OTHER_AUTH  chunk contains the name / address / DOB verification exchange
  NO_AUTH     everything else (greeting, issue, resolution, close, distractor
              numbers such as phone numbers, order numbers, amounts, dates)

An authentication exchange is never split across two chunks, so chunk labels
are unambiguous. Conversations are split into train / validation / test at the
conversation level (no leakage between splits), stratified by auth path.

Usage:
    python3 auth_data/generate_synthetic_auth_data.py                 # defaults
    python3 auth_data/generate_synthetic_auth_data.py --n 10000 --seed 13 \
        --max-tokens 200 --out data

    # exact token counts (either form):
    python3 auth_data/generate_synthetic_auth_data.py --tokenizer bert-base-uncased   # transformers
    python3 auth_data/generate_synthetic_auth_data.py --tokenizer /path/to/vocab.txt  # tokenizers

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Vocabularies (all fictional / generic)
# --------------------------------------------------------------------------- #

FIRST_NAMES = """
james mary robert patricia john jennifer michael linda david elizabeth william
barbara richard susan joseph jessica thomas sarah charles karen christopher lisa
daniel nancy matthew betty anthony margaret mark sandra donald ashley steven
kimberly paul emily andrew donna joshua michelle kenneth carol kevin amanda brian
dorothy george melissa timothy deborah ronald stephanie edward rebecca jason
sharon jeffrey laura ryan cynthia jacob kathleen gary amy nicholas angela eric
shirley jonathan anna stephen brenda larry pamela justin emma scott nicole brandon
helen benjamin samantha samuel katherine gregory christine alexander debra frank
rachel patrick carolyn raymond janet jack catherine dennis maria jerry heather
tyler diane aaron ruth jose julie adam olivia nathan joyce henry virginia douglas
victoria zachary kelly peter lauren kyle christina noah joan ethan evelyn jeremy
judith walter megan christian andrea keith cheryl roger hannah terry jacqueline
austin martha sean gloria gerald teresa carl ann dylan sara harold madison jordan
frances jesse kathryn bryan janice lawrence jean arthur abigail gabriel alice
bruce judy logan sophia billy grace albert denise willie amber alan doris eugene
marilyn russell danielle vincent beverly louis isabella bobby theresa philip
diana johnny natalie priya wei omar fatima diego mei amir rosa luis yuki kofi
"""
FIRST_NAMES = FIRST_NAMES.split()

LAST_NAMES = """
smith johnson williams brown jones garcia miller davis rodriguez martinez
hernandez lopez gonzalez wilson anderson thomas taylor moore jackson martin lee
perez thompson white harris sanchez clark ramirez lewis robinson walker young
allen king wright scott torres nguyen hill flores green adams nelson baker hall
rivera campbell mitchell carter roberts gomez phillips evans turner diaz parker
cruz edwards collins reyes stewart morris morales murphy cook rogers gutierrez
ortiz morgan cooper peterson bailey reed kelly howard ramos kim cox ward richardson
watson brooks chavez wood james bennett gray mendoza ruiz hughes price alvarez
castillo sanders patel myers long ross foster jimenez powell jenkins perry russell
sullivan bell coleman butler henderson barnes gonzales fisher vasquez simmons
romero jordan patterson alexander hamilton graham reynolds griffin wallace moreno
west cole hayes bryant herrera gibson ellis tran medina aguilar stevens murray
ford castro marshall owens harrison fernandez mcdonald woods washington kennedy
wells vargas henry chen freeman webb tucker guzman burns crawford olson simpson
porter hunter gordon mendez silva shaw snyder mason dixon munoz hunt hicks holmes
palmer wagner black robertson boyd rose stone salazar fox warren mills meyer rice
schmidt garza daniels ferguson nichols stephens soto weaver ryan gardner payne
grant dunn kelley spencer hawkins arnold pierce vazquez hansen peters santos hart
bradley knight elliott cunningham duncan armstrong hudson carroll lane riley
andrews alvarado ray delgado berry perkins hoffman johnston matthews pena
richards contreras willis carpenter lawrence sandoval guerrero george chapman
rios estrada ortega watkins greene nunez wheeler valdez harper burke larson
santiago maldonado morrison franklin carlson austin dominguez carr lawson jacobs
obrien lynch singh vega bishop montgomery oliver jensen harvey williamson gilbert
dean sims espinoza howell li wong reid hanson le mccoy garrett burton fuller wang
weber welch rojas lucas marquez fields park yang little banks padilla day walsh
bowman schultz luna fowler mejia okafor haddad novak lindqvist ivanova nakamura
"""
LAST_NAMES = LAST_NAMES.split()

STREET_NAMES = """
maple oak pine cedar elm walnut birch spruce willow chestnut hickory sycamore
magnolia laurel cypress poplar aspen juniper main park lake hill river ridge
valley forest meadow sunset highland washington lincoln jefferson madison
jackson franklin adams monroe harrison church school mill spring summit prospect
union center grove orchard garden vine lincoln liberty victory pleasant pearl
market water front bridge canal harbor bay beach dune ocean prairie mesa canyon
"""
STREET_NAMES = STREET_NAMES.split()
STREET_TYPES = ["street", "avenue", "road", "drive", "lane", "court", "boulevard",
                "way", "place", "circle", "terrace", "trail"]

CITIES = [
    ("springfield", "illinois"), ("riverside", "california"), ("franklin", "tennessee"),
    ("greenville", "south carolina"), ("fairview", "oregon"), ("madison", "wisconsin"),
    ("georgetown", "texas"), ("arlington", "virginia"), ("clinton", "iowa"),
    ("salem", "oregon"), ("bristol", "connecticut"), ("dayton", "ohio"),
    ("lexington", "kentucky"), ("ashland", "oregon"), ("burlington", "vermont"),
    ("milford", "massachusetts"), ("newport", "rhode island"), ("oxford", "mississippi"),
    ("jackson", "michigan"), ("manchester", "new hampshire"), ("auburn", "alabama"),
    ("dover", "delaware"), ("hudson", "new york"), ("kingston", "pennsylvania"),
    ("lancaster", "pennsylvania"), ("marion", "indiana"), ("mount vernon", "washington"),
    ("troy", "new york"), ("winchester", "virginia"), ("plymouth", "minnesota"),
    ("aurora", "colorado"), ("mesa", "arizona"), ("boulder", "colorado"),
    ("tempe", "arizona"), ("reno", "nevada"), ("tacoma", "washington"),
    ("eugene", "oregon"), ("tulsa", "oklahoma"), ("wichita", "kansas"),
    ("omaha", "nebraska"), ("boise", "idaho"), ("fargo", "north dakota"),
    ("savannah", "georgia"), ("tampa", "florida"), ("orlando", "florida"),
    ("charlotte", "north carolina"), ("raleigh", "north carolina"),
    ("baton rouge", "louisiana"), ("little rock", "arkansas"), ("albuquerque", "new mexico"),
]

COMPANIES = [
    "northwind mobile", "harbor bank", "acme energy", "blue ridge insurance",
    "summit cable", "lakeside credit union", "evergreen utilities", "cascade wireless",
    "meridian health plan", "pioneer gas and electric", "redwood financial",
    "coastal internet", "granite state telecom", "sunrise pharmacy benefits",
]

AGENT_NAMES = [
    "alex", "jordan", "taylor", "morgan", "casey", "riley", "sam", "jamie", "dana",
    "chris", "pat", "lee", "robin", "kim", "devon", "quinn", "avery", "skyler",
    "priya", "marcus", "elena", "tomas", "nina", "kwame", "sofia", "ravi",
]

FILLERS = ["um", "uh", "so", "okay", "yeah", "let me see", "hold on", "alright",
           "one second", "right", "sure", "hmm"]

ID_LABELS = ["account number", "member id", "customer id", "account id",
             "policy number", "member number", "customer number", "subscriber id"]

MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety"]
ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
            6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
            11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
            15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
            19: "nineteenth", 20: "twentieth", 30: "thirtieth"}

LABELS = ("ID_AUTH", "OTHER_AUTH", "NO_AUTH")

# --------------------------------------------------------------------------- #
# Number / date verbalisation helpers
# --------------------------------------------------------------------------- #


def two_digit_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    t, o = divmod(n, 10)
    return TENS[t] if o == 0 else f"{TENS[t]} {ONES[o]}"


def number_words(n: int) -> str:
    """Cardinal words for 0..9999 (enough for house numbers and years)."""
    if n < 100:
        return two_digit_words(n)
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{ONES[h]} hundred" + (f" {two_digit_words(r)}" if r else "")
    th, r = divmod(n, 1000)
    return f"{ONES[th]} thousand" + (f" {number_words(r)}" if r else "")


def ordinal_words(day: int) -> str:
    if day in ORDINALS:
        return ORDINALS[day]
    t, o = divmod(day, 10)
    return f"{TENS[t]} {ORDINALS[o]}"


def ordinal_suffix(day: int) -> str:
    if 10 <= day % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def year_words(y: int, rng: random.Random) -> str:
    if y < 2000:
        return f"nineteen {two_digit_words(y - 1900)}"
    if y == 2000:
        return "two thousand"
    if y < 2010:
        return rng.choice([f"two thousand {ONES[y - 2000]}", f"two thousand and {ONES[y - 2000]}"])
    return rng.choice([f"twenty {two_digit_words(y - 2000)}", f"two thousand {two_digit_words(y - 2000)}"])


def digits_to_words(digits: str, rng: random.Random) -> str:
    """Speak a digit string one digit at a time, with occasional 'double'/'triple'."""
    out: List[str] = []
    i = 0
    while i < len(digits):
        d = digits[i]
        run = 1
        while i + run < len(digits) and digits[i + run] == d:
            run += 1
        if run >= 3 and rng.random() < 0.5:
            out.append(f"triple {ONES[int(d)]}")
            i += 3
            continue
        if run >= 2 and rng.random() < 0.5:
            out.append(f"double {ONES[int(d)]}")
            i += 2
            continue
        if d == "0":
            out.append(rng.choice(["zero", "oh", "zero"]))
        else:
            out.append(ONES[int(d)])
        i += 1
    return " ".join(out)


def group_digits(digits: str, rng: random.Random) -> str:
    """Split a digit string into spoken groups, e.g. '472 913 860'."""
    groups: List[str] = []
    i = 0
    n = len(digits)
    while i < n:
        remaining = n - i
        if remaining <= 4:
            size = remaining
        else:
            size = rng.choice([3, 3, 4, 2])
            size = min(size, remaining)
        groups.append(digits[i:i + size])
        i += size
    return " ".join(groups)


def speak_id(digits: str, rng: random.Random) -> str:
    """Render a 9-12 digit identifier the way it might appear in an STT transcript."""
    style = rng.random()
    if style < 0.30:
        return digits                                  # 472913860
    if style < 0.55:
        return group_digits(digits, rng)               # 472 913 860
    if style < 0.80:
        return digits_to_words(digits, rng)            # four seven two ...
    # mixed: part grouped digits, part words, with a pause
    cut = rng.randint(3, len(digits) - 3)
    first, second = digits[:cut], digits[cut:]
    sep = rng.choice([" ", " um ", " uh ", " and then "])
    if rng.random() < 0.5:
        return group_digits(first, rng) + sep + digits_to_words(second, rng)
    return digits_to_words(first, rng) + sep + group_digits(second, rng)


def speak_dob(month: int, day: int, year: int, rng: random.Random) -> str:
    style = rng.random()
    mname = MONTHS[month - 1]
    if style < 0.30:
        return f"{mname} {ordinal_words(day)} {year_words(year, rng)}"      # march third nineteen eighty two
    if style < 0.50:
        return f"{mname} {day}{ordinal_suffix(day)} {year}"                 # march 3rd 1982
    if style < 0.65:
        return f"{mname} {day} {year}"                                      # march 3 1982
    if style < 0.80:
        return f"{month:02d} {day:02d} {year}"                              # 03 03 1982
    if style < 0.90:
        mm = digits_to_words(f"{month:02d}", rng)
        dd = digits_to_words(f"{day:02d}", rng)
        return f"{mm} {dd} {year_words(year, rng)}"                         # oh three oh three nineteen eighty two
    return f"{month}/{day}/{year}"                                          # 3/3/1982


def speak_house_number(n: int, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.6:
        return str(n)
    if r < 0.8:
        return digits_to_words(str(n), rng)
    # "fourteen twenty eight" style for 4-digit numbers
    s = str(n)
    if len(s) == 4 and s[2] != "0":
        return f"{two_digit_words(int(s[:2]))} {two_digit_words(int(s[2:]))}"
    return number_words(n)


def speak_zip(z: str, rng: random.Random) -> str:
    return z if rng.random() < 0.7 else digits_to_words(z, rng)


def spell_word(w: str) -> str:
    return " ".join(w)


# --------------------------------------------------------------------------- #
# Token counting (bert-base-uncased)
# --------------------------------------------------------------------------- #

_COMMON_ONE_PIECE = set("""
agent customer the a an and i you to it is my your that of for on in with me we
this can could would should have has had do does did not no yes okay ok so um uh
yeah sure right just like what when where how why which who there here one two
three four five six seven eight nine zero oh ten eleven twelve thirteen fourteen
fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty
seventy eighty ninety hundred thousand double triple number account name address
street avenue road drive lane court place circle date birth born thank thanks
please help today calling call phone bill payment card order service plan great
perfect alright let see second hold moment minute sorry about again also any
anything else all back been before being but by from get got going good had
into last make more much need new now off only other our out over same some
than them then they time up very was were will work year years month months
day days week weeks first second third next
""".split())


def estimate_bert_tokens(text: str) -> int:
    """
    Conservative estimate of bert-base-uncased WordPiece tokens (no CLS/SEP).
    Over-estimates on purpose so a budget set here is safe for the real tokenizer.
    """
    words = re.findall(r"[a-z]+|\d+|[^\sa-z\d]", text.lower())
    total = 0
    for w in words:
        if w in _COMMON_ONE_PIECE:
            total += 1
        elif w.isdigit():
            total += max(1, math.ceil(len(w) / 2))
        elif w.isalpha():
            total += 1 if len(w) <= 4 else 1 + math.ceil((len(w) - 4) / 3)
        else:
            total += 1
    return total


def make_token_counter(tokenizer: Optional[str]) -> Tuple[Callable[[str], int], str]:
    """
    Returns (count_fn, name). `tokenizer` may be a HF model name (needs
    `transformers`) or a path to a bert-base-uncased vocab.txt (needs
    `tokenizers`). Falls back to the built-in conservative estimator.
    """
    if tokenizer and os.path.isfile(tokenizer):
        try:
            from tokenizers import BertWordPieceTokenizer  # type: ignore
            tok = BertWordPieceTokenizer(tokenizer, lowercase=True)
            return (lambda s: len(tok.encode(s, add_special_tokens=False).ids)), f"tokenizers:{tokenizer}"
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"[warn] could not load vocab {tokenizer!r} ({exc}); using built-in estimator", file=sys.stderr)
    elif tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore
            tok = AutoTokenizer.from_pretrained(tokenizer)
            return (lambda s: len(tok.tokenize(s))), f"transformers:{tokenizer}"
        except Exception as exc:  # pragma: no cover - depends on environment
            print(f"[warn] could not load tokenizer {tokenizer!r} ({exc}); using built-in estimator", file=sys.stderr)
    return estimate_bert_tokens, "builtin-estimate"


# --------------------------------------------------------------------------- #
# Conversation model
# --------------------------------------------------------------------------- #


@dataclass
class Turn:
    speaker: str          # "agent" | "customer"
    text: str
    span: Optional[str] = None   # "ID_AUTH" | "OTHER_AUTH" | None

    def render(self) -> str:
        return f"{self.speaker}: {self.text}"


@dataclass
class Customer:
    first: str
    last: str
    house: int
    street: str
    street_type: str
    unit: Optional[str]
    city: str
    state: str
    zip: str
    dob: Tuple[int, int, int]      # (month, day, year)
    phone: str
    id_number: str


@dataclass
class Conversation:
    conv_id: str
    auth_type: str                  # "ID_AUTH" | "OTHER_AUTH"
    turns: List[Turn] = field(default_factory=list)


def maybe_filler(rng: random.Random, p: float = 0.35) -> str:
    return (rng.choice(FILLERS) + " ") if rng.random() < p else ""


def make_customer(rng: random.Random) -> Customer:
    city, state = rng.choice(CITIES)
    month = rng.randint(1, 12)
    day = rng.randint(1, MONTH_DAYS[month - 1])
    year = rng.randint(1940, 2004)
    id_len = rng.randint(9, 12)
    id_number = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(id_len - 1))
    unit = None
    if rng.random() < 0.3:
        unit = rng.choice(["apartment", "unit", "suite"]) + " " + rng.choice(
            [str(rng.randint(1, 40)), f"{rng.randint(1, 12)}{rng.choice('abcd')}", str(rng.randint(100, 420))])
    return Customer(
        first=rng.choice(FIRST_NAMES),
        last=rng.choice(LAST_NAMES),
        house=rng.choice([rng.randint(1, 99), rng.randint(100, 999), rng.randint(1000, 9999)]),
        street=rng.choice(STREET_NAMES),
        street_type=rng.choice(STREET_TYPES),
        unit=unit,
        city=city,
        state=state,
        zip=f"{rng.randint(10000, 99999)}",
        dob=(month, day, year),
        phone="".join(str(rng.randint(0, 9)) for _ in range(10)),
        id_number=id_number,
    )


# ----------------------------- opening ------------------------------------- #

REASONS = [
    "billing", "password", "outage", "cancel", "upgrade", "charge", "address_change",
    "shipping", "autopay", "payment_arrangement", "coverage", "device",
]


def opening_turns(rng: random.Random, cust: Customer, agent: str, company: str, reason: str) -> List[Turn]:
    greet = rng.choice([
        f"thank you for calling {company} this is {agent} how can i help you today",
        f"hi thanks for calling {company} my name is {agent} what can i do for you",
        f"{company} customer care this is {agent} speaking how may i help you",
        f"good {rng.choice(['morning', 'afternoon', 'evening'])} you've reached {company} this is {agent}",
        f"thanks for holding this is {agent} with {company} how can i help",
    ])
    intro = rng.choice(["hi", "hello", "hey", "yes hi", "hi there", "good morning", "hi yeah"])
    self_name = ""
    if rng.random() < 0.45:
        self_name = rng.choice([
            f" this is {cust.first}", f" my name is {cust.first} {cust.last}",
            f" {cust.first} {cust.last} here", f" this is {cust.first} {cust.last}",
        ])
    problem = {
        "billing": [
            "i'm calling about my bill it's higher than usual this month",
            "i have a question about some charges on my latest statement",
            f"my bill went up by like {rng.randint(10, 60)} dollars and i don't know why",
        ],
        "password": [
            "i'm locked out of my online account and i need to reset my password",
            "i can't log in to the app it keeps saying my password is wrong",
            "i need help resetting the password on my account",
        ],
        "outage": [
            "my service has been out since this morning",
            "the internet keeps dropping every few minutes",
            "i've got no signal at my house and it's been like that since yesterday",
        ],
        "cancel": [
            "i'd like to cancel my service",
            "i'm moving and i need to close my account",
            "i want to cancel one of the lines on my plan",
        ],
        "upgrade": [
            "i want to look at upgrading my plan",
            "i'm interested in the new plan i saw advertised",
            "can you tell me what it would cost to move to a bigger plan",
        ],
        "charge": [
            f"there's a charge for {rng.randint(5, 90)} dollars and {rng.randint(0, 99):02d} cents that i don't recognize",
            "i was charged twice this month and i only made one payment",
            "i see a late fee but i paid on time",
        ],
        "address_change": [
            "i recently moved and i need to update the address on my account",
            "i need to change my mailing address",
            "my statements are going to my old address",
        ],
        "shipping": [
            "i ordered a replacement device and it hasn't shown up",
            "i'm checking on the status of a shipment",
            f"the tracking says delivered but i never got it",
        ],
        "autopay": [
            "i want to set up automatic payments",
            "i need to turn off autopay for a couple months",
            "my autopay didn't go through and i want to know why",
        ],
        "payment_arrangement": [
            "i'm going to be a little late on my payment and wanted to set something up",
            "can i split this month's payment into two",
            "i need an extension on my bill",
        ],
        "coverage": [
            "i'm trying to find out if a procedure is covered under my plan",
            "i have a question about my deductible",
            "i got a letter saying a claim was denied",
        ],
        "device": [
            "my router is blinking red and nothing is working",
            "the new phone you sent me won't activate",
            "my modem keeps restarting on its own",
        ],
    }[reason]
    return [
        Turn("agent", greet),
        Turn("customer", f"{intro}{self_name} {maybe_filler(rng)}{rng.choice(problem)}".strip()),
    ]


# ----------------------------- authentication ----------------------------- #


def ask_for_id(rng: random.Random, cust: Customer, id_label: str) -> str:
    return rng.choice([
        f"i can definitely help with that first can i get your {id_label}",
        f"sure thing let me pull up your account do you have your {id_label} handy",
        f"okay i'm sorry to hear that to get started what's the {id_label} on the account",
        f"happy to help can you read me your {id_label} please",
        f"no problem before i can look into that i need to verify your account can i have your {id_label}",
        f"alright let's take a look can you give me the {id_label}",
        f"got it for security can you confirm your {id_label} for me",
    ])


def id_auth_turns(rng: random.Random, cust: Customer, id_label: str) -> List[Turn]:
    turns: List[Turn] = [Turn("agent", ask_for_id(rng, cust, id_label), span="ID_AUTH")]
    spoken = speak_id(cust.id_number, rng)
    lead = rng.choice([
        "sure it's", "yeah it's", "okay it's", "yes hold on um it's", "let me grab it okay it's",
        "yep one second it should be", "i think it's", "it's", "okay i have it here it's",
        "yes the number is", "sure that's",
    ])
    if rng.random() < 0.25:
        turns.append(Turn("customer", rng.choice([
            "yeah hold on let me find my card", "one second it's on the statement",
            "um i think so give me a second", "yes let me pull it up on my phone",
        ]), span="ID_AUTH"))
        turns.append(Turn("agent", rng.choice(["take your time", "no rush", "sure go ahead"]), span="ID_AUTH"))
    turns.append(Turn("customer", f"{lead} {spoken}", span="ID_AUTH"))

    r = rng.random()
    if r < 0.30:
        # agent reads it back
        readback = rng.choice([group_digits(cust.id_number, rng), digits_to_words(cust.id_number, rng), cust.id_number])
        turns.append(Turn("agent", rng.choice([
            f"okay let me read that back {readback} is that right",
            f"so that's {readback}",
            f"i have {readback} correct",
        ]), span="ID_AUTH"))
        turns.append(Turn("customer", rng.choice(["yes that's right", "correct", "yep", "that's it", "yes"]), span="ID_AUTH"))
    elif r < 0.45:
        # agent missed part of it
        tail = cust.id_number[-4:]
        turns.append(Turn("agent", rng.choice([
            "sorry you cut out a bit can you repeat the last few digits",
            "i missed the end of that could you say the last four again",
        ]), span="ID_AUTH"))
        turns.append(Turn("customer", rng.choice([
            f"{digits_to_words(tail, rng)}", f"sure {tail}", f"the last four are {group_digits(tail, rng)}",
        ]), span="ID_AUTH"))
    confirm = rng.choice([
        f"thank you i have your account pulled up am i speaking with {cust.first}",
        f"perfect i see the account here for {cust.first} {cust.last}",
        "great that pulled right up thank you",
        f"okay i found it thanks {cust.first}",
        "alright you're verified thank you for that",
        f"thank you {cust.first} i've got your account open",
    ])
    turns.append(Turn("agent", confirm, span="ID_AUTH"))
    if "am i speaking" in confirm:
        turns.append(Turn("customer", rng.choice(["yes that's me", "yes", "that's right", "yep that's me"]), span="ID_AUTH"))
    return turns


def no_id_statement(rng: random.Random, id_label: str) -> str:
    return rng.choice([
        f"um i don't actually have my {id_label} on me",
        f"i don't have my {id_label} with me right now",
        f"honestly i have no idea what my {id_label} is i don't have the card",
        f"i'm not at home so i don't have the {id_label} handy",
        f"no i don't have that number i can't find my statement",
        f"i don't know my {id_label} off the top of my head is there another way",
        f"sorry i don't have my {id_label} i'm calling from work",
        f"i never memorized that number and i don't have it in front of me",
        f"no i don't have my {id_label} can you look me up another way",
        f"i don't have my {id_label} i lost the card actually that's part of why i'm calling",
    ])


def other_auth_turns(rng: random.Random, cust: Customer, id_label: str) -> List[Turn]:
    turns: List[Turn] = [
        Turn("agent", ask_for_id(rng, cust, id_label), span="OTHER_AUTH"),
        Turn("customer", no_id_statement(rng, id_label), span="OTHER_AUTH"),
    ]
    turns.append(Turn("agent", rng.choice([
        "that's okay i can verify you another way can i get your full name",
        "no problem we can look you up with some other information what's your first and last name",
        "that's fine i just need to verify a few things can you give me your full name",
        "not a problem let's do it the other way what's the name on the account",
        "sure i can look it up by name and address what is your full name please",
    ]), span="OTHER_AUTH"))
    name = f"{cust.first} {cust.last}"
    name_utt = rng.choice([
        f"{name}", f"it's {name}", f"{name} that's {spell_word(cust.last)}",
        f"{cust.first} {cust.last} last name is {spell_word(cust.last)}", f"sure {name}",
    ])
    turns.append(Turn("customer", name_utt, span="OTHER_AUTH"))
    if rng.random() < 0.25:
        turns.append(Turn("agent", rng.choice([
            f"can you spell the last name for me", f"how do you spell {cust.last}",
        ]), span="OTHER_AUTH"))
        turns.append(Turn("customer", spell_word(cust.last), span="OTHER_AUTH"))

    turns.append(Turn("agent", rng.choice([
        f"thank you {cust.first} and what's the address on the account",
        "okay and can you confirm your street address",
        "got it and the service address please",
        "and what's the mailing address we have for you",
        "thanks and your address including zip code",
    ]), span="OTHER_AUTH"))
    addr = f"{speak_house_number(cust.house, rng)} {cust.street} {cust.street_type}"
    if cust.unit:
        addr += f" {cust.unit}"
    r = rng.random()
    if r < 0.5:
        addr += f" {cust.city} {cust.state} {speak_zip(cust.zip, rng)}"
    elif r < 0.8:
        addr += f" in {cust.city} {speak_zip(cust.zip, rng)}"
    else:
        addr += f" {cust.city}"
    addr_utt = rng.choice([f"{addr}", f"it's {addr}", f"{maybe_filler(rng, 0.6)}{addr}", f"that would be {addr}"])
    turns.append(Turn("customer", addr_utt.strip(), span="OTHER_AUTH"))
    if r >= 0.8 and rng.random() < 0.6:
        turns.append(Turn("agent", "and the zip code", span="OTHER_AUTH"))
        turns.append(Turn("customer", speak_zip(cust.zip, rng), span="OTHER_AUTH"))

    turns.append(Turn("agent", rng.choice([
        "perfect and lastly your date of birth",
        "okay and can you verify your date of birth for me",
        "thank you and what's your date of birth",
        "great last thing i need is your birth date",
        "and for security what is your date of birth",
    ]), span="OTHER_AUTH"))
    m, d, y = cust.dob
    dob_utt = rng.choice([
        speak_dob(m, d, y, rng), f"it's {speak_dob(m, d, y, rng)}",
        f"{maybe_filler(rng, 0.5)}{speak_dob(m, d, y, rng)}", f"i was born {speak_dob(m, d, y, rng)}",
    ]).strip()
    turns.append(Turn("customer", dob_utt, span="OTHER_AUTH"))
    turns.append(Turn("agent", rng.choice([
        f"thank you {cust.first} that all matches i've got your account pulled up",
        "perfect that matches what we have on file you're verified",
        "great thank you that's everything i needed you're all set",
        f"okay {cust.first} i found your account thanks for bearing with me",
        "alright that checks out i have your account open now",
    ]), span="OTHER_AUTH"))
    return turns


# ----------------------------- resolution body ----------------------------- #


def rand_amount(rng: random.Random) -> str:
    d, c = rng.randint(5, 300), rng.randint(0, 99)
    return rng.choice([
        f"{d} dollars and {c:02d} cents",
        f"${d}.{c:02d}",
        f"{number_words(d)} {two_digit_words(c)}" if c else f"{number_words(d)} dollars",
        f"{d} dollars",
        f"{d} {c:02d}",
    ])


def rand_order_number(rng: random.Random) -> str:
    # 6-8 digits so it is NOT a 9-12 digit identifier
    n = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(6, 8)))
    return rng.choice([n, group_digits(n, rng), digits_to_words(n, rng)])


def rand_confirmation(rng: random.Random) -> str:
    letters = "".join(rng.choice("abcdefghjkmnpqrstuvwxyz") for _ in range(rng.randint(2, 3)))
    digits = "".join(str(rng.randint(0, 9)) for _ in range(rng.randint(4, 6)))
    return rng.choice([f"{letters} {digits}", f"{spell_word(letters)} {group_digits(digits, rng)}",
                       f"{spell_word(letters)} {digits_to_words(digits, rng)}"])


def rand_phone(rng: random.Random, cust: Customer) -> str:
    p = cust.phone
    return rng.choice([
        f"{p[:3]} {p[3:6]} {p[6:]}", digits_to_words(p, rng),
        f"{digits_to_words(p[:3], rng)} {p[3:6]} {p[6:]}",
    ])


def rand_future_date(rng: random.Random) -> str:
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return rng.choice([f"{MONTHS[m - 1]} {ordinal_words(d)}", f"{MONTHS[m - 1]} {d}{ordinal_suffix(d)}",
                       f"the {ordinal_words(d)}", f"{m} {d}"])


def body_turns(rng: random.Random, cust: Customer, reason: str) -> List[Turn]:
    t: List[Turn] = []
    A = lambda s: t.append(Turn("agent", s))        # noqa: E731
    C = lambda s: t.append(Turn("customer", s))     # noqa: E731

    if reason == "billing":
        A(rng.choice(["okay let me take a look at your most recent statement give me just a moment",
                      "alright i'm looking at the bill now one second"]))
        C(rng.choice(["okay", "sure", "thanks"]))
        A(f"so i see the total this month is {rand_amount(rng)} and last month it was {rand_amount(rng)}")
        A(rng.choice([
            f"the difference looks like a promotional credit of {rand_amount(rng)} that expired on {rand_future_date(rng)}",
            f"it looks like there's a one time equipment fee of {rand_amount(rng)} on here",
            "there's a prorated charge because your plan changed mid cycle",
        ]))
        C(rng.choice(["nobody told me that was going to expire", "oh i didn't realize that", "is there anything you can do about that"]))
        A(rng.choice([
            f"i can apply a courtesy credit of {rand_amount(rng)} to the account today and i'll note it",
            "let me see what promotions you qualify for right now",
            "i can switch you to a comparable plan that would bring the bill back down",
        ]))
        C(rng.choice(["that would be great", "okay yeah let's do that", "i appreciate it"]))
    elif reason == "password":
        A(rng.choice(["okay i can send a reset link what's the best email or phone number for that",
                      "no problem i'll trigger a reset where should i send the code"]))
        C(rng.choice([f"you can text it to {rand_phone(rng, cust)}", "the email on file is fine", f"my cell is {rand_phone(rng, cust)}"]))
        A(rng.choice(["okay i just sent that it should arrive in a minute or two",
                      "alright that's sent let me know when you get it"]))
        C(rng.choice(["okay got it", "yep it just came through", f"i see it the code is {digits_to_words(''.join(str(rng.randint(0, 9)) for _ in range(6)), rng)}"]))
        A("perfect go ahead and set a new password and you should be able to log in right away")
        C(rng.choice(["okay that worked thank you", "alright i'm in now", "great that did it"]))
    elif reason == "outage":
        A(rng.choice(["i'm sorry about that let me check for outages in your area",
                      "okay let me run a diagnostic on your line"]))
        C(rng.choice(["okay", "thank you", "sure"]))
        A(rng.choice([
            f"i do see a known outage affecting about {rng.randint(40, 900)} customers the estimated restoration is {rng.randint(1, 6)} pm",
            "i'm not seeing an outage so let's try a quick reset on the equipment",
            "the signal to your modem looks weak i'd like to send a refresh to it",
        ]))
        C(rng.choice(["okay should i unplug it", "alright go ahead", "how long will that take"]))
        A(rng.choice(["yes unplug it for thirty seconds and plug it back in",
                      f"that will take about {rng.randint(2, 5)} minutes you'll lose connection briefly",
                      f"i'll also open a ticket the ticket number is {rand_order_number(rng)}"]))
        C(rng.choice(["okay it's back up now", "alright the lights are green", "okay thanks i'll keep an eye on it"]))
    elif reason == "cancel":
        A(rng.choice(["i'm sorry to hear you're leaving can i ask what's prompting the cancellation",
                      "okay i can help with that may i ask the reason"]))
        C(rng.choice(["it's just too expensive", "i'm moving out of state", "i found a better deal somewhere else", "i just don't use it anymore"]))
        A(rng.choice([
            f"i understand before i process that i can offer you {rng.randint(10, 30)} percent off for the next {rng.randint(6, 12)} months",
            "i understand we do have service in that area if you'd like to transfer instead",
            "okay i can process that the service will remain active through the end of the billing cycle",
        ]))
        C(rng.choice(["no thanks i just want to cancel", "hmm let me think about that", "no i've made up my mind"]))
        A(f"understood i've submitted the cancellation your confirmation number is {rand_confirmation(rng)}")
        C(rng.choice(["okay thanks", "alright can you email that to me", "got it"]))
    elif reason == "upgrade":
        A(rng.choice(["sure let me look at what's available for you",
                      f"okay you're currently on the plan at {rand_amount(rng)} a month"]))
        C(rng.choice(["okay", "what are the options", "yeah what do you have"]))
        A(rng.choice([
            f"the next tier up is {rand_amount(rng)} a month and includes unlimited data",
            f"for {rng.randint(10, 40)} dollars more you'd get the premium package",
            f"there's a promotion right now that waives the {rand_amount(rng)} activation fee",
        ]))
        C(rng.choice(["is there a contract", "when would that start", "okay let's do it"]))
        A(rng.choice([f"no contract and it would take effect on {rand_future_date(rng)}",
                      "it starts at the beginning of your next cycle",
                      f"i've made the change your new monthly total will be {rand_amount(rng)}"]))
        C(rng.choice(["great", "okay sounds good", "perfect thanks"]))
    elif reason == "charge":
        A(rng.choice(["let me look at that transaction for you", "okay i see the charge you're talking about"]))
        A(rng.choice([
            f"that was posted on {rand_future_date(rng)} and it's for {rand_amount(rng)}",
            f"it looks like it's a duplicate the second one for {rand_amount(rng)} should have been reversed",
            f"that's the late fee and i can see your payment posted on the {ordinal_words(rng.randint(1, 28))}",
        ]))
        C(rng.choice(["yeah i don't recognize that", "can you refund it", "so can you take that off"]))
        A(rng.choice([
            f"i've reversed it you'll see the credit of {rand_amount(rng)} in three to five business days",
            f"i'm issuing a refund the reference number is {rand_confirmation(rng)}",
            "i've waived that fee and it will show on your next statement",
        ]))
        C(rng.choice(["thank you so much", "okay great", "appreciate it"]))
    elif reason == "address_change":
        A(rng.choice(["sure what's the new address", "okay i can update that what's the new mailing address"]))
        # NOTE: this is a *new* address, not the on-file address, so it stays NO_AUTH.
        nh = rng.randint(10, 9999)
        ns = rng.choice(STREET_NAMES)
        nt = rng.choice(STREET_TYPES)
        nc, nst = rng.choice(CITIES)
        nz = str(rng.randint(10000, 99999))
        C(f"{maybe_filler(rng, 0.4)}it's {speak_house_number(nh, rng)} {ns} {nt} {nc} {nst} {speak_zip(nz, rng)}".strip())
        A(rng.choice([f"okay {nh} {ns} {nt} {nc} got it", "alright i have that updated", "and is that effective immediately"]))
        C(rng.choice(["yes", "yep that's right", "yes as of today"]))
        A(rng.choice(["done your next statement will go to the new address",
                      f"all set i've sent a confirmation to the email on file"]))
        C(rng.choice(["perfect thanks", "great", "okay thank you"]))
    elif reason == "shipping":
        A(rng.choice(["let me check the tracking on that order do you have the order number",
                      "okay let me find that shipment"]))
        C(rng.choice([f"yeah the order number is {rand_order_number(rng)}", "no i don't have it", f"it's {rand_order_number(rng)}"]))
        A(rng.choice([
            f"okay it shows it shipped on {rand_future_date(rng)} and the tracking number is {rand_order_number(rng)}",
            "i see it was marked delivered to the front door yesterday",
            f"it looks like it's delayed at the carrier the new estimate is {rand_future_date(rng)}",
        ]))
        C(rng.choice(["well i never got it", "can you send another one", "okay i'll check with my neighbor"]))
        A(rng.choice([f"i'll send a replacement at no charge the new order number is {rand_order_number(rng)}",
                      "i've opened a claim with the carrier and you'll hear back within forty eight hours"]))
        C(rng.choice(["thanks", "okay great", "appreciate that"]))
    elif reason == "autopay":
        A(rng.choice(["sure i can set that up which card or bank account would you like to use",
                      "okay let me pull up the autopay settings"]))
        C(rng.choice([f"the card ending in {rng.randint(1000, 9999)}", "the checking account on file", f"you can use the card ending {digits_to_words(str(rng.randint(1000, 9999)), rng)}"]))
        A(rng.choice([f"okay autopay is set and the first draft will be {rand_amount(rng)} on {rand_future_date(rng)}",
                      "i've paused autopay for the next two cycles",
                      f"i see the payment of {rand_amount(rng)} was returned by the bank"]))
        C(rng.choice(["okay", "great thank you", "oh okay i'll call the bank"]))
    elif reason == "payment_arrangement":
        A(rng.choice([f"i can help with that your current balance is {rand_amount(rng)}",
                      "sure let me see what arrangements are available"]))
        C(rng.choice(["can i pay half now and half in two weeks", f"i can pay on the {ordinal_words(rng.randint(1, 28))}", "what are my options"]))
        A(rng.choice([
            f"yes i can set {rand_amount(rng)} for today and the rest on {rand_future_date(rng)}",
            f"i can extend the due date to {rand_future_date(rng)} with no late fee",
        ]))
        C(rng.choice(["that works", "okay let's do that", "perfect"]))
        A(f"done the arrangement confirmation number is {rand_confirmation(rng)}")
        C(rng.choice(["thank you", "okay got it", "great"]))
    elif reason == "coverage":
        A(rng.choice(["okay let me check the benefits on your plan", "let me look at that claim"]))
        C(rng.choice(["okay", "thanks"]))
        A(rng.choice([
            f"that procedure is covered at {rng.choice([70, 80, 90])} percent after your deductible",
            f"your deductible is {rand_amount(rng)} and you've met {rand_amount(rng)} of it so far",
            f"the claim was denied because the provider billed it under the wrong code the claim number is {rand_order_number(rng)}",
        ]))
        C(rng.choice(["okay so what do i owe", "can you resubmit it", "okay that makes sense"]))
        A(rng.choice(["i'll send it back for reprocessing you should see an update in ten business days",
                      f"your estimated out of pocket would be around {rand_amount(rng)}"]))
        C(rng.choice(["thanks", "okay thank you", "great"]))
    else:  # device
        A(rng.choice(["okay let's troubleshoot that what color are the lights on the front",
                      "alright can you tell me what the screen is showing"]))
        C(rng.choice(["the power light is red and the rest are off", "it says activation failed", "it's just cycling on and off"]))
        A(rng.choice([f"okay i'm sending a signal to the device now this takes about {rng.randint(1, 4)} minutes",
                      "let's try a factory reset hold the button on the back for fifteen seconds",
                      f"i'll ship a replacement the order number is {rand_order_number(rng)}"]))
        C(rng.choice(["okay it's rebooting", "alright done", "okay"]))
        A(rng.choice(["give it a minute and let me know if the lights go green", "it should come up online shortly"]))
        C(rng.choice(["okay it's working now", "yeah it's green", "alright looks good"]))
    return t


def closing_turns(rng: random.Random, cust: Customer, agent: str, company: str) -> List[Turn]:
    t = [Turn("agent", rng.choice([
        "is there anything else i can help you with today",
        "anything else i can do for you",
        "was there anything else",
    ]))]
    if rng.random() < 0.3:
        t.append(Turn("customer", rng.choice([
            "actually yeah can you text me a copy of this", "oh one more thing what's my next due date",
            "no but can i get a confirmation email",
        ])))
        t.append(Turn("agent", rng.choice([
            f"sure i've sent that to the number ending in {cust.phone[-4:]}",
            f"your next due date is {rand_future_date(rng)}",
            "yes that's on its way to the email on file",
        ])))
    t.append(Turn("customer", rng.choice([
        "no that's it thanks", "nope that's everything thank you", "no that was it i appreciate your help",
        f"no thanks {agent} you've been great",
    ])))
    t.append(Turn("agent", rng.choice([
        f"thank you for calling {company} have a great day",
        "you're welcome thanks for being a customer have a good one",
        f"my pleasure {cust.first} take care",
        "thank you have a wonderful day goodbye",
    ])))
    return t


def build_conversation(rng: random.Random, idx: int, auth_type: str) -> Conversation:
    cust = make_customer(rng)
    agent = rng.choice(AGENT_NAMES)
    company = rng.choice(COMPANIES)
    reason = rng.choice(REASONS)
    id_label = rng.choice(ID_LABELS)
    turns: List[Turn] = []
    turns += opening_turns(rng, cust, agent, company, reason)
    if rng.random() < 0.3:
        turns.append(Turn("agent", rng.choice([
            "i'm sorry to hear that", "okay i understand", "i can definitely take a look at that for you",
            "oh no let's get that sorted out", "sure i can help with that",
        ])))
        turns.append(Turn("customer", rng.choice(["thank you", "okay", "great", "thanks"])))
    turns += id_auth_turns(rng, cust, id_label) if auth_type == "ID_AUTH" else other_auth_turns(rng, cust, id_label)
    turns += body_turns(rng, cust, reason)
    turns += closing_turns(rng, cust, agent, company)
    return Conversation(conv_id=f"conv_{idx:05d}", auth_type=auth_type, turns=turns)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #


def group_turns(turns: Sequence[Turn]) -> List[List[Turn]]:
    """Group consecutive turns that belong to the same auth span; others are singletons."""
    groups: List[List[Turn]] = []
    for t in turns:
        if t.span and groups and groups[-1][0].span == t.span:
            groups[-1].append(t)
        else:
            groups.append([t])
    return groups


def chunk_conversation(conv: Conversation, count_tokens: Callable[[str], int],
                       max_tokens: int, min_tokens: int, rng: random.Random) -> List[Dict]:
    """
    Greedy, turn-aligned chunking with a per-chunk budget drawn uniformly from
    [min_tokens, max_tokens]. Randomising the budget stops chunk length (or
    "the auth exchange always sits in the first chunk") from leaking the label.
    An auth span is an atomic unit and is never split, so each chunk's label is
    determined by whether it contains a span. Every unit is <= max_tokens, so
    every chunk is <= max_tokens.
    """
    units = group_turns(conv.turns)
    chunks: List[Dict] = []
    cur: List[Turn] = []
    cur_tokens = 0
    budget = rng.randint(min_tokens, max_tokens)

    def flush() -> None:
        nonlocal cur, cur_tokens, budget
        if not cur:
            return
        text = " ".join(t.render() for t in cur)
        spans = {t.span for t in cur if t.span}
        assert len(spans) <= 1, "a chunk must not contain two different auth spans"
        label = next(iter(spans)) if spans else "NO_AUTH"
        chunks.append({"text": text, "label": label, "n_turns": len(cur), "tokens": count_tokens(text)})
        cur, cur_tokens = [], 0
        budget = rng.randint(min_tokens, max_tokens)

    for unit in units:
        unit_text = " ".join(t.render() for t in unit)
        unit_tokens = count_tokens(unit_text)
        if unit_tokens > max_tokens:
            raise ValueError(
                f"{conv.conv_id}: atomic unit of {unit_tokens} tokens exceeds --max-tokens {max_tokens}; "
                f"raise the budget (units are at most a single auth exchange)")
        if cur and cur_tokens + 1 + unit_tokens > budget:
            flush()
        cur.extend(unit)
        cur_tokens += unit_tokens + (1 if cur_tokens else 0)
    flush()
    return chunks


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #


def stratified_split(convs: List[Conversation], rng: random.Random,
                     fracs: Tuple[float, float, float]) -> Dict[str, str]:
    """Return conv_id -> split, stratified by auth_type."""
    assignment: Dict[str, str] = {}
    by_type: Dict[str, List[Conversation]] = {}
    for c in convs:
        by_type.setdefault(c.auth_type, []).append(c)
    for _, group in sorted(by_type.items()):
        ids = [c.conv_id for c in group]
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * fracs[0]))
        n_val = int(round(n * fracs[1]))
        for i, cid in enumerate(ids):
            assignment[cid] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
    return assignment


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10_000, help="number of conversations (default 10000)")
    ap.add_argument("--seed", type=int, default=20261014, help="random seed")
    ap.add_argument("--no-id-frac", type=float, default=0.20,
                    help="fraction of conversations where the customer lacks an ID (default 0.20)")
    ap.add_argument("--max-tokens", type=int, default=200,
                    help="per-chunk token budget excluding [CLS]/[SEP] (default 200)")
    ap.add_argument("--min-tokens", type=int, default=48,
                    help="lower bound of the randomised per-chunk budget (default 48)")
    ap.add_argument("--tokenizer", default=None,
                    help="HF tokenizer name (needs transformers) or path to a bert-base-uncased "
                         "vocab.txt (needs tokenizers) for exact counts; default is a built-in estimate")
    ap.add_argument("--split", default="0.8,0.1,0.1", help="train,validation,test fractions")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)), help="output directory")
    args = ap.parse_args(argv)

    fracs = tuple(float(x) for x in args.split.split(","))
    if len(fracs) != 3 or abs(sum(fracs) - 1.0) > 1e-6:
        ap.error("--split must be three fractions summing to 1")

    if not (0 < args.min_tokens <= args.max_tokens):
        ap.error("--min-tokens must be in (0, --max-tokens]")
    rng = random.Random(args.seed)
    chunk_rng = random.Random(args.seed + 1)
    count_tokens, counter_name = make_token_counter(args.tokenizer)

    # Exactly round(n * no_id_frac) conversations take the OTHER_AUTH path.
    n_other = int(round(args.n * args.no_id_frac))
    auth_types = ["OTHER_AUTH"] * n_other + ["ID_AUTH"] * (args.n - n_other)
    rng.shuffle(auth_types)

    convs = [build_conversation(rng, i, at) for i, at in enumerate(auth_types)]
    assignment = stratified_split(convs, rng, fracs)  # type: ignore[arg-type]

    os.makedirs(args.out, exist_ok=True)
    files = {s: open(os.path.join(args.out, f"{s}.jsonl"), "w", encoding="utf-8")
             for s in ("train", "validation", "test")}
    stats: Dict = {
        "generator": os.path.basename(__file__),
        "seed": args.seed,
        "n_conversations": args.n,
        "no_id_frac_requested": args.no_id_frac,
        "max_tokens_per_chunk": args.max_tokens,
        "min_tokens_per_chunk": args.min_tokens,
        "token_counter": counter_name,
        "labels": list(LABELS),
        "conversations_by_auth_type": dict(Counter(c.auth_type for c in convs)),
        "splits": {},
    }
    per_split: Dict[str, Dict] = {s: {"conversations": 0, "chunks": 0, "labels": Counter(),
                                      "auth_type": Counter(), "max_tokens": 0, "sum_tokens": 0}
                                  for s in files}
    label_total: Counter = Counter()
    try:
        for conv in convs:
            split = assignment[conv.conv_id]
            chunks = chunk_conversation(conv, count_tokens, args.max_tokens, args.min_tokens, chunk_rng)
            ps = per_split[split]
            ps["conversations"] += 1
            ps["auth_type"][conv.auth_type] += 1
            for i, ch in enumerate(chunks):
                rec = {
                    "id": f"{conv.conv_id}_c{i:02d}",
                    "conversation_id": conv.conv_id,
                    "chunk_index": i,
                    "num_chunks": len(chunks),
                    "conversation_auth_type": conv.auth_type,
                    "label": ch["label"],
                    "text": ch["text"],
                    "n_turns": ch["n_turns"],
                    "bert_tokens": ch["tokens"],
                }
                files[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
                ps["chunks"] += 1
                ps["labels"][ch["label"]] += 1
                ps["max_tokens"] = max(ps["max_tokens"], ch["tokens"])
                ps["sum_tokens"] += ch["tokens"]
                label_total[ch["label"]] += 1
    finally:
        for f in files.values():
            f.close()

    for s, ps in per_split.items():
        stats["splits"][s] = {
            "conversations": ps["conversations"],
            "conversations_by_auth_type": dict(ps["auth_type"]),
            "chunks": ps["chunks"],
            "labels": dict(ps["labels"]),
            "max_chunk_tokens": ps["max_tokens"],
            "mean_chunk_tokens": round(ps["sum_tokens"] / max(1, ps["chunks"]), 1),
        }
    stats["chunks_total"] = sum(label_total.values())
    stats["labels_total"] = dict(label_total)
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
