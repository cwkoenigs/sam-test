#!/usr/bin/env python3
"""
Synthetic, noisy health-plan call-center transcripts for

  1. call classification (fine-tuned RoBERTa)
       * multi-class : `primary_label`  (the main reason for the call)
       * multi-label : `labels` / `label_vector` (every bucket substantively
         discussed on the call; 1-3 buckets per call)
  2. abstractive call summarisation (BART): `summary`

Buckets: BILLING, CLAIM_STATUS, COVERAGE, COMPLAINTS, PRIOR_AUTH, NEMT,
TECHNICAL_ISSUE  (NEMT = non-emergency medical transportation).

Transcripts use the same `agent:` / `customer:` turn format as auth_data but are
deliberately noisier: ASR confusions and homophones, dropped and repeated
words, fillers, [inaudible] / [crosstalk] tags, garbled numbers, truncated
turns, dropped speaker tags (diarisation errors), IVR fragments, hold music
and transfers. Each call carries a `noise_level` (low / medium / high).
Labels and summaries are clean; the summary is generated from the underlying
call facts, not from the noisy text.

Usage:
    uv run python class_data/generate_class_data.py            # 100k calls
    uv run python class_data/generate_class_data.py --n 2000 --seed 7
    # token counts come from the local roberta-base tokenizer (BART shares it);
    # pass --tokenizer none to skip counting.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

LABELS = ["BILLING", "CLAIM_STATUS", "COVERAGE", "COMPLAINTS", "PRIOR_AUTH", "NEMT", "TECHNICAL_ISSUE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
LABEL_DESCRIPTIONS = {
    "BILLING": "premium invoices, payments, autopay, refunds, balances owed to the plan or a provider",
    "CLAIM_STATUS": "status, denial, reprocessing or appeal of a submitted medical/pharmacy claim",
    "COVERAGE": "whether a service, drug or provider is covered; deductible, copay, network, referral questions",
    "COMPLAINTS": "dissatisfaction with the plan, a provider, a vendor or a prior interaction; grievances; supervisor requests",
    "PRIOR_AUTH": "prior authorization requests, status, denials, expedited review, peer-to-peer",
    "NEMT": "non-emergency medical transportation: scheduling, cancelling, late/no-show rides, mileage reimbursement",
    "TECHNICAL_ISSUE": "member portal / mobile app login, password, ID card download, errors, missing data online",
}

# realistic primary-topic mix (sums to 1)
PRIMARY_WEIGHTS = {
    "BILLING": 0.21, "CLAIM_STATUS": 0.18, "COVERAGE": 0.17, "PRIOR_AUTH": 0.13,
    "COMPLAINTS": 0.10, "NEMT": 0.11, "TECHNICAL_ISSUE": 0.10,
}
# which secondary topics tend to ride along with which primary
SECONDARY_AFFINITY = {
    "BILLING": {"CLAIM_STATUS": 3, "COVERAGE": 2, "TECHNICAL_ISSUE": 3, "COMPLAINTS": 2, "PRIOR_AUTH": 1, "NEMT": 1},
    "CLAIM_STATUS": {"BILLING": 3, "COVERAGE": 3, "COMPLAINTS": 2, "PRIOR_AUTH": 2, "TECHNICAL_ISSUE": 1, "NEMT": 1},
    "COVERAGE": {"PRIOR_AUTH": 3, "CLAIM_STATUS": 2, "BILLING": 2, "NEMT": 1, "TECHNICAL_ISSUE": 1, "COMPLAINTS": 1},
    "COMPLAINTS": {"CLAIM_STATUS": 3, "BILLING": 2, "NEMT": 2, "PRIOR_AUTH": 2, "TECHNICAL_ISSUE": 1, "COVERAGE": 1},
    "PRIOR_AUTH": {"COVERAGE": 3, "CLAIM_STATUS": 2, "COMPLAINTS": 2, "NEMT": 1, "BILLING": 1, "TECHNICAL_ISSUE": 1},
    "NEMT": {"COMPLAINTS": 3, "COVERAGE": 2, "BILLING": 1, "TECHNICAL_ISSUE": 1, "CLAIM_STATUS": 1, "PRIOR_AUTH": 1},
    "TECHNICAL_ISSUE": {"BILLING": 3, "CLAIM_STATUS": 2, "COVERAGE": 1, "COMPLAINTS": 2, "NEMT": 1, "PRIOR_AUTH": 1},
}
N_TOPICS_WEIGHTS = [(1, 0.55), (2, 0.35), (3, 0.10)]
NOISE_LEVEL_WEIGHTS = [("low", 0.30), ("medium", 0.45), ("high", 0.25)]

# --------------------------------------------------------------------------- #
# Vocabularies (fictional / generic)
# --------------------------------------------------------------------------- #

FIRST = ("james mary robert patricia john jennifer michael linda david elizabeth william barbara richard "
         "susan joseph jessica thomas sarah charles karen lisa daniel nancy matthew betty anthony margaret mark "
         "sandra donald ashley steven kimberly paul emily andrew donna joshua michelle kenneth carol kevin amanda "
         "brian dorothy george melissa deborah ronald stephanie edward rebecca jason sharon jeffrey laura ryan "
         "cynthia jacob kathleen gary amy nicholas angela eric shirley jonathan anna stephen brenda larry pamela "
         "justin emma scott nicole brandon helen benjamin samantha samuel katherine gregory christine priya wei "
         "omar fatima diego mei amir rosa luis yuki kofi aisha carlos lin ana marcus elena tomas nina ravi").split()
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
PLANS = ["gold hmo", "silver ppo", "bronze hdhp", "medicare advantage plan", "medicaid managed care plan",
         "select epo", "essential plan", "platinum ppo", "community health plan", "value hmo"]
COMPANY = ["evergreen health plan", "harbor health", "meridian health plan", "northstar care", "bluewater health",
           "summit health plan", "lakeside community health", "cascade health"]
PROVIDERS = ["dr patel", "dr nguyen", "dr okafor", "dr rivera", "dr chen", "dr johansson", "dr adeyemi", "dr kowalski",
             "lakeside orthopedics", "riverside imaging center", "northgate family medicine", "summit physical therapy",
             "harbor view cardiology", "pinecrest dermatology", "valley behavioral health", "oakwood dialysis center",
             "metro urgent care", "county general hospital", "westside womens health", "central pediatrics"]
SERVICES = ["an mri of the knee", "a ct scan", "physical therapy", "a sleep study", "a colonoscopy", "an echocardiogram",
            "a dermatologist visit", "a mental health visit", "a dental cleaning", "an eye exam", "chiropractic care",
            "lab work", "an allergy test", "a mammogram", "a specialist visit", "an ultrasound", "a hearing aid",
            "a cpap machine", "a wheelchair", "a blood pressure monitor", "insulin pumps", "a nutritionist visit"]
DRUGS = ["ozempic", "humira", "eliquis", "jardiance", "trulicity", "dupixent", "stelara", "xarelto", "entresto",
         "mounjaro", "symbicort", "lantus", "farxiga", "rinvoq", "skyrizi", "biktarvy"]
PA_PROCEDURES = ["an mri of the lower back", "a knee replacement", "a sleep study", "a ct scan of the abdomen",
                 "twelve physical therapy visits", "a cardiac stress test", "bariatric surgery", "an epidural injection",
                 "a pet scan", "home health nursing", "a power wheelchair", "cataract surgery", "a spinal fusion",
                 "genetic testing", "an inpatient rehab stay", "a continuous glucose monitor"]
APPT_TYPES = ["dialysis", "chemo", "a physical therapy session", "a cardiology follow up", "an eye appointment",
              "a dental appointment", "wound care", "a behavioral health appointment", "an infusion",
              "a primary care visit", "a specialist appointment", "radiation treatment"]
STREETS = "maple oak pine cedar elm walnut birch main park lake hill river ridge valley forest meadow sunset highland washington lincoln".split()
STYPES = ["street", "avenue", "road", "drive", "lane", "court", "boulevard", "way"]
CITIES = ["springfield", "riverside", "franklin", "greenville", "fairview", "madison", "georgetown", "arlington",
          "clinton", "salem", "bristol", "dayton", "lexington", "burlington", "milford", "dover", "hudson", "troy",
          "aurora", "mesa", "boulder", "reno", "tacoma", "eugene", "tulsa", "wichita", "omaha", "boise", "fargo"]
ERRORS = ["error code 403", "error 500", "something went wrong try again later", "session expired", "invalid credentials",
          "page not found", "we cannot verify your identity", "error code e 1 2 7", "service unavailable"]
BROWSERS = ["chrome", "safari", "the app on my iphone", "the android app", "firefox", "edge", "my ipad"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]
ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
       9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth", 15: "fifteenth",
       16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth", 21: "twenty first",
       22: "twenty second", 23: "twenty third", 24: "twenty fourth", 25: "twenty fifth", 26: "twenty sixth",
       27: "twenty seventh", 28: "twenty eighth"}
FILLERS = ["um", "uh", "like", "you know", "i mean", "so", "okay so", "hmm", "well", "yeah"]
TAGS = ["[inaudible]", "[crosstalk]", "[background noise]", "[unintelligible]", "[silence]", "[dog barking]",
        "[baby crying]", "[static]"]

# ASR confusions: token -> replacement options (applied with the per-level rate)
ASR_CONFUSIONS = {
    "claim": ["clam", "claims", "claim number"], "claims": ["clams", "claim"], "prior": ["pryor", "pry or"],
    "authorization": ["author ization", "authorisation", "authorize ation"], "auth": ["off", "oth"],
    "deductible": ["deductable", "the ductible", "deduct able"], "copay": ["co pay", "copays"],
    "coinsurance": ["co insurance", "coin surance"], "premium": ["premier", "premiums"], "portal": ["porter", "portal site"],
    "password": ["pass word", "passport"], "login": ["log in", "log on", "logan"], "log": ["long"],
    "appointment": ["a pointment", "appointments"], "dialysis": ["die alysis", "dialysis center"],
    "pharmacy": ["farm acy", "pharmacies"], "formulary": ["form a larry", "formula"], "referral": ["referal", "refer all"],
    "specialist": ["special list", "specialists"], "transportation": ["transport tation", "transport"],
    "wheelchair": ["wheel chair"], "grievance": ["grievence", "grief ants"], "supervisor": ["super visor", "supervise or"],
    "denied": ["the nied", "denide", "tonight"], "denial": ["the nile", "denile"], "benefits": ["bene fits", "benefit"],
    "medicaid": ["medic aid"], "medicare": ["medi care"], "explanation": ["explain nation"], "network": ["net work"],
    "ozempic": ["o zempic", "ozempik"], "humira": ["who mira", "humera"], "eliquis": ["ella quist", "eloquist"],
    "jardiance": ["jar dance", "guardians"], "mri": ["m r i", "emory"], "ct": ["c t", "city"], "id": ["i d", "eye d"],
    "to": ["too", "two"], "for": ["four"], "two": ["to", "too"], "there": ["their", "they're"], "our": ["are", "hour"],
    "your": ["you're", "you"], "know": ["no"], "right": ["write"], "whole": ["hole"], "week": ["weak"],
    "sent": ["cent", "scent"], "bill": ["build", "bills"], "paid": ["pay", "played"], "card": ["car", "cart"],
    "code": ["coat", "cold"], "ride": ["right", "rides"], "driver": ["drive her", "drivers"], "late": ["let", "later"],
    "plan": ["plant", "plane"], "cover": ["covered", "color"], "covered": ["cover", "color"], "in": ["and", "an"],
    "and": ["in", "an"], "a": ["uh", "the"], "the": ["a", "they"], "can": ["can't", "ken"], "can't": ["can", "cant"],
    "number": ["numbers", "no more"], "account": ["a count", "accounts"], "member": ["remember", "members"],
    "status": ["stat us", "statis"], "pending": ["spending", "pen ding"], "approved": ["a proved", "improved"],
    "appeal": ["a peel", "appeals"], "balance": ["balanced", "ballots"], "refund": ["re fund", "refunded"],
    "invoice": ["in voice", "invoices"], "autopay": ["auto pay", "otto pay"], "app": ["ap", "apt"], "reset": ["re set", "receipt"],
}
_CONF_KEYS = set(ASR_CONFUSIONS)

NOISE = {  # per-level rates
    "low":    dict(sub=0.015, drop=0.01, filler=0.025, repeat=0.008, tag=0.02, tagdrop=0.02, trunc=0.01, numgarble=0.10,
                   ivr=0.15, hold=0.10, dropcall=0.01),
    "medium": dict(sub=0.05, drop=0.03, filler=0.06, repeat=0.02, tag=0.06, tagdrop=0.06, trunc=0.03, numgarble=0.35,
                   ivr=0.35, hold=0.25, dropcall=0.03),
    "high":   dict(sub=0.11, drop=0.06, filler=0.10, repeat=0.04, tag=0.14, tagdrop=0.13, trunc=0.06, numgarble=0.65,
                   ivr=0.55, hold=0.40, dropcall=0.06),
}

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def wchoice(rng: random.Random, pairs: Sequence[Tuple]) -> object:
    r = rng.random() * sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if r <= acc:
            return v
    return pairs[-1][0]


def digits(rng: random.Random, n: int) -> str:
    return str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(n - 1))


def spoken_digits(s: str, rng: random.Random) -> str:
    return " ".join(rng.choice(["oh", "zero"]) if c == "0" else ONES[int(c)] for c in s if c.isdigit())


def grouped(s: str, rng: random.Random) -> str:
    out, i = [], 0
    while i < len(s):
        k = min(rng.choice([2, 3, 3, 4]), len(s) - i)
        out.append(s[i:i + k])
        i += k
    return " ".join(out)


def say_number(s: str, rng: random.Random) -> str:
    r = rng.random()
    return s if r < 0.45 else grouped(s, rng) if r < 0.75 else spoken_digits(s, rng)


def money(rng: random.Random, lo: int, hi: int) -> Tuple[str, float]:
    d, c = rng.randint(lo, hi), rng.choice([0, 0, rng.randint(1, 99)])
    val = d + c / 100
    forms = [f"{d} dollars", f"${d}", f"{d} bucks"] if c == 0 else [f"{d} dollars and {c:02d} cents", f"${d}.{c:02d}", f"{d} {c:02d}"]
    return rng.choice(forms), val


def a_date(rng: random.Random, past: bool = True) -> Tuple[str, str]:
    m, d = rng.randint(1, 12), rng.randint(1, 28)
    spoken = rng.choice([f"{MONTHS[m - 1]} {ORD[d]}", f"{MONTHS[m - 1]} {d}", f"the {ORD[d]} of {MONTHS[m - 1]}", f"{m} {d}",
                         f"{MONTHS[m - 1]} {d}{'st' if d == 1 else 'nd' if d == 2 else 'rd' if d == 3 else 'th'}"])
    clean = f"{MONTHS[m - 1].capitalize()} {d}"
    return spoken, clean


def a_time(rng: random.Random) -> str:
    h = rng.randint(6, 17)
    mm = rng.choice(["", " fifteen", " thirty", " forty five"])
    return f"{h if h <= 12 else h - 12}{mm} {'am' if h < 12 else 'pm'}"


def address(rng: random.Random) -> str:
    return f"{rng.randint(10, 9999)} {rng.choice(STREETS)} {rng.choice(STYPES)} {rng.choice(CITIES)}"


_MONEY_RES = [(re.compile(r"\b(\d+) dollars and (\d{2}) cents\b"), r"$\1.\2"),
              (re.compile(r"\b(\d+) (?:dollars|bucks)\b"), r"$\1"),
              (re.compile(r"(?<![\d$.])(\d{1,4}) (\d{2})(?!\d)"), r"$\1.\2")]


def clean_money(text: str) -> str:
    """Normalise spoken money forms ('794 dollars', '55 26', '55 bucks') to $794 / $55.26 in summaries."""
    for rx, rep_ in _MONEY_RES:
        text = rx.sub(rep_, text)
    return text


def cap(s: str) -> str:
    return s[0].upper() + s[1:] if s else s


# --------------------------------------------------------------------------- #
# call context
# --------------------------------------------------------------------------- #


@dataclass
class Ctx:
    rng: random.Random
    first: str
    last: str
    agent: str
    company: str
    plan: str
    member_id: str
    summary: List[str] = field(default_factory=list)      # clean sentences, in order
    outcomes: List[str] = field(default_factory=list)
    frustrated: bool = False

    @property
    def name(self) -> str:
        return f"{self.first} {self.last}"


Turn = Tuple[str, str]  # (speaker, text)


def A(t: List[Turn], s: str) -> None:
    t.append(("agent", s))


def C(t: List[Turn], s: str) -> None:
    t.append(("customer", s))


# --------------------------------------------------------------------------- #
# topic generators. each returns turns and appends clean summary sentences.
# role = "primary" (full exchange) or "secondary" (shorter add-on)
# --------------------------------------------------------------------------- #


def intro_secondary(rng: random.Random) -> str:
    return rng.choice(["oh and while i have you", "one more thing", "also i wanted to ask", "actually can i ask you something else",
                       "and another thing", "before i let you go", "sorry one other question", "oh wait also"])


def topic_billing(ctx: Ctx, role: str) -> List[Turn]:
    rng, t = ctx.rng, []
    sub = wchoice(rng, [("premium_increase", 2), ("duplicate", 2), ("missed", 2), ("refund", 1.5), ("autopay", 2),
                        ("provider_bill", 2), ("balance", 1.5)])
    amt, amt_v = money(rng, 40, 900)
    amt2, _ = money(rng, 40, 900)
    due_spoken, due_clean = a_date(rng)
    inv = digits(rng, 8)
    lead = intro_secondary(rng) + " " if role == "secondary" else ""
    if sub == "premium_increase":
        C(t, f"{lead}my premium went up to {amt} this month and nobody told me why")
        A(t, rng.choice(["let me pull up your billing i see the change took effect on the renewal",
                         "okay i'm looking at the invoice now one moment"]))
        A(t, rng.choice([f"it looks like the plan renewed and the new rate is {amt} the old rate was {amt2}",
                         f"the subsidy on the account changed so your portion went from {amt2} to {amt}"]))
        C(t, rng.choice(["that's a big jump", "nobody sent me anything about that", "can i get that lowered"]))
        A(t, rng.choice(["i can send you the renewal notice again and i can also have someone review your plan options",
                         "i can note the account and have enrollment call you about other plan options"]))
        ctx.summary.append(f"Member questioned a premium increase to {amt} per month; agent explained the renewal rate change and offered a plan options review.")
        ctx.outcomes.append("renewal notice re-sent")
    elif sub == "duplicate":
        C(t, f"{lead}i got charged twice for {amt} on {due_spoken}")
        A(t, "let me check the payment history")
        A(t, rng.choice([f"i do see two payments of {amt} posted the same day",
                         f"i see one payment posted and one pending for {amt} the pending one should drop off",
                         "i only see one payment on our side the second one may be a hold from your bank"]))
        C(t, rng.choice(["can you refund the second one", "i need that money back", "how long does that take"]))
        if rng.random() < 0.6:
            A(t, f"i've submitted a refund request for {amt} it takes seven to ten business days reference number {say_number(inv, rng)}")
            ctx.summary.append(f"Member reported a duplicate charge of {amt} on {due_clean}; agent reviewed payment history and submitted a refund request (reference {inv}).")
            ctx.outcomes.append("refund requested")
        else:
            A(t, "if it's a bank hold it releases in three to five days if not call us back and we'll refund it")
            ctx.summary.append(f"Member reported a duplicate charge of {amt} on {due_clean}; agent found only one posted payment and advised the second is a bank hold that will release in 3-5 days.")
            ctx.outcomes.append("bank hold explained")
    elif sub == "missed":
        C(t, f"{lead}i got a letter saying i missed a payment and my coverage might end")
        A(t, f"okay i see the {MONTHS[rng.randint(0, 11)]} premium of {amt} was not received and you're in the grace period")
        C(t, rng.choice(["i thought autopay was on", "i mailed a check", "can i pay it right now"]))
        k = rng.randint(0, 2)
        A(t, [f"i can take a payment today of {amt} and that brings you current",
              "i can set up a payment arrangement so you don't lose coverage",
              f"the grace period ends on {due_spoken} as long as it's paid by then you're fine"][k])
        C(t, rng.choice(["okay let's do it", "yes please", "alright"]))
        ctx.summary.append(f"Member received a missed-payment notice for a {amt} premium; agent confirmed grace period status and " +
                           ["took a payment to bring the account current.", "set up a payment arrangement.",
                            f"advised the grace period ends {due_clean}."][k])
        ctx.outcomes.append(["payment taken", "payment arrangement", "grace period explained"][k])
    elif sub == "refund":
        C(t, f"{lead}i overpaid and i'm still waiting on a refund of {amt}")
        A(t, f"i see the refund was approved on {due_spoken} it should arrive within thirty days")
        C(t, rng.choice(["it's been longer than that", "can you check the address it's going to", "okay"]))
        if rng.random() < 0.6:
            A(t, "i'll escalate it to the finance team and you should hear back in five business days")
            ctx.summary.append(f"Member followed up on an outstanding {amt} refund; agent confirmed approval date {due_clean} and escalated to finance.")
            ctx.outcomes.append("refund escalated")
        else:
            A(t, "the check was mailed last week give it a few more days")
            ctx.summary.append(f"Member followed up on an outstanding {amt} refund; agent confirmed it was approved {due_clean} and the check was mailed last week.")
            ctx.outcomes.append("refund check mailed")
    elif sub == "autopay":
        C(t, f"{lead}i want to set up automatic payments so i stop forgetting")
        A(t, "sure i can do that would you like to use a bank account or a card")
        C(t, rng.choice([f"the card ending in {rng.randint(1000, 9999)}", "my checking account", "whatever is easier"]))
        A(t, f"okay autopay is set for the {ORD[rng.randint(1, 28)]} of each month for {amt}")
        ctx.summary.append(f"Member requested autopay enrollment; agent set up monthly automatic payments of {amt}.")
        ctx.outcomes.append("autopay enrolled")
    elif sub == "provider_bill":
        prov = rng.choice(PROVIDERS)
        C(t, f"{lead}i got a bill from {prov} for {amt} but my explanation of benefits says i only owe {amt2}")
        A(t, "let me compare the claim to what they billed you")
        A(t, rng.choice(["you're right your responsibility is the lower amount the provider may have billed before the claim finished",
                         "it looks like the provider balance billed you which they are not allowed to do as an in network provider"]))
        C(t, rng.choice(["so what do i do", "should i pay it", "this is ridiculous"]))
        A(t, rng.choice([f"don't pay the {amt} i'll send the provider a notice and you only owe {amt2}",
                         "i'll open a provider billing dispute and you'll get a letter when it's resolved"]))
        ctx.summary.append(f"Member disputed a {amt} bill from {prov} that conflicted with the EOB amount of {amt2}; agent opened a provider billing dispute.")
        ctx.outcomes.append("billing dispute opened")
    else:  # balance
        C(t, f"{lead}what's my current balance and when is it due")
        A(t, f"your balance is {amt} and it's due on {due_spoken}")
        k = rng.randint(0, 2)
        C(t, ["okay can i pay half now", "can you email me the invoice", "thanks that's all i needed"][k])
        A(t, ["sure i'll take a partial payment now", "i've sent the invoice to the email on file", "you're welcome"][k])
        ctx.summary.append(f"Member asked for the current balance; agent confirmed {amt} due {due_clean}" +
                           [" and took a partial payment.", " and emailed the invoice.", "."][k])
        ctx.outcomes.append(["partial payment taken", "invoice emailed", "balance confirmed"][k])
    return t


def topic_claim_status(ctx: Ctx, role: str) -> List[Turn]:
    rng, t = ctx.rng, []
    claim = digits(rng, rng.choice([9, 10, 11]))
    prov = rng.choice(PROVIDERS)
    dos_spoken, dos_clean = a_date(rng)
    billed, _ = money(rng, 80, 4000)
    status = wchoice(rng, [("pending", 2), ("denied", 3), ("paid", 1.5), ("needs_info", 1.5), ("not_found", 1)])
    lead = intro_secondary(rng) + " " if role == "secondary" else ""
    C(t, f"{lead}" + rng.choice([f"i'm checking on a claim from {prov} date of service {dos_spoken}",
                                 f"i need the status of claim number {say_number(claim, rng)}",
                                 f"my doctor's office says you guys haven't paid them for my visit on {dos_spoken}"]))
    if rng.random() < 0.5:
        A(t, "do you have the claim number")
        C(t, rng.choice([f"yeah it's {say_number(claim, rng)}", "no i just have the date", f"um {say_number(claim, rng)} i think"]))
    A(t, "okay give me a second to pull that up")
    if status == "pending":
        A(t, f"that claim for {billed} is still processing it was received on {a_date(rng)[0]} and they have thirty days")
        C(t, rng.choice(["okay so when will it be done", "it's been a while", "alright"]))
        A(t, "it should finalize within the next two weeks you'll get an explanation of benefits in the mail")
        ctx.summary.append(f"Member checked the status of claim {claim} from {prov} (DOS {dos_clean}, billed {billed}); agent confirmed it is still processing within the 30-day window.")
        ctx.outcomes.append("claim pending")
    elif status == "denied":
        reason = rng.choice(["the provider was out of network", "no prior authorization was on file", "it was billed with the wrong code",
                             "the service wasn't medically necessary per the review", "it was a duplicate submission",
                             "the timely filing limit passed"])
        A(t, f"i see that claim was denied because {reason}")
        C(t, rng.choice(["that's wrong", "nobody told me i needed that", "so what are my options", "what does that mean"]))
        ctx.frustrated = ctx.frustrated or rng.random() < 0.4
        A(t, rng.choice(["you can file an appeal within sixty days i can start that for you",
                         "i can send it back for reprocessing the provider needs to submit a corrected claim",
                         "i'll open a reconsideration and you'll get a letter with the outcome"]))
        C(t, rng.choice(["yes start the appeal", "okay do that", "fine"]))
        A(t, f"done your reference number is {say_number(digits(rng, 8), rng)}")
        ctx.summary.append(f"Member inquired about claim {claim} from {prov} (DOS {dos_clean}), which was denied because {reason}; agent initiated an appeal/reprocessing request.")
        ctx.outcomes.append("appeal opened")
    elif status == "paid":
        paid, _ = money(rng, 50, 3000)
        A(t, f"that claim was paid on {a_date(rng)[0]} we paid {paid} and your responsibility is {money(rng, 0, 400)[0]}")
        if rng.random() < 0.5:
            C(t, rng.choice(["great thanks", "can you send me that"]))
            A(t, "i'll resend the explanation of benefits")
            ctx.summary.append(f"Member checked claim {claim} from {prov}; agent confirmed it was paid ({paid}) and re-sent the EOB.")
            ctx.outcomes.append("eob re-sent")
        else:
            C(t, "okay so why is the office still billing me")
            A(t, "the provider may not have posted the payment yet give them a week")
            ctx.summary.append(f"Member checked claim {claim} from {prov}; agent confirmed it was paid ({paid}) and advised the provider may not have posted the payment yet.")
            ctx.outcomes.append("claim paid, provider posting delay")
    elif status == "needs_info":
        A(t, "the claim is on hold because we need additional information from the provider medical records")
        C(t, rng.choice(["do i need to do anything", "they said they sent it", "okay"]))
        A(t, "i'll send a second request to the provider nothing needed from you")
        ctx.summary.append(f"Member asked about claim {claim} from {prov}, which is pended for medical records; agent re-requested records from the provider.")
        ctx.outcomes.append("records re-requested")
    else:
        A(t, f"i'm not finding a claim for {prov} on {dos_spoken} it may not have been submitted yet")
        C(t, rng.choice(["they told me they sent it", "so what do i do", "hm okay"]))
        A(t, "i'd ask the provider's billing office to confirm the submission and the member id they used")
        ctx.summary.append(f"Member asked about a claim from {prov} for DOS {dos_clean}; no claim was on file and agent advised contacting the provider's billing office.")
        ctx.outcomes.append("no claim on file")
    return t


def topic_coverage(ctx: Ctx, role: str) -> List[Turn]:
    rng, t = ctx.rng, []
    sub = wchoice(rng, [("service", 3), ("drug", 2.5), ("network", 2), ("deductible", 2), ("referral", 1.5)])
    lead = intro_secondary(rng) + " " if role == "secondary" else ""
    if sub == "service":
        svc = rng.choice(SERVICES)
        copay, _ = money(rng, 10, 150)
        C(t, f"{lead}is {svc} covered under my plan")
        A(t, f"let me check your benefits on the {ctx.plan}")
        cov = rng.random()
        if cov < 0.6:
            A(t, rng.choice([f"yes {svc} is covered with a {copay} copay", f"it's covered at eighty percent after your deductible",
                             f"yes it's covered but it does require a prior authorization"]))
            ctx.summary.append(f"Member asked whether {svc} is covered; agent confirmed coverage under the {ctx.plan} and explained the cost share.")
        elif rng.random() < 0.5:
            A(t, f"{svc} is not a covered benefit on this plan")
            ctx.summary.append(f"Member asked whether {svc} is covered; agent explained it is not a covered benefit under the {ctx.plan}.")
        else:
            A(t, "it's covered only if it's medically necessary and in network")
            ctx.summary.append(f"Member asked whether {svc} is covered; agent explained it is covered only when medically necessary and in network under the {ctx.plan}.")
        C(t, rng.choice(["okay good", "what if i go somewhere out of network", "do i need a referral for that", "hm okay"]))
        A(t, rng.choice(["out of network would be covered at a lower rate", "no referral needed for that", "i'd stay in network to keep your cost down"]))
        ctx.outcomes.append("benefits explained")
    elif sub == "drug":
        drug = rng.choice(DRUGS)
        C(t, f"{lead}my doctor wants to put me on {drug} is that on the formulary")
        A(t, "let me look at the drug list")
        tier = rng.randint(2, 5)
        k = rng.randint(0, 2)
        dcopay = money(rng, 25, 250)[0]
        A(t, [f"{drug} is on tier {tier} so the copay would be {dcopay} for a thirty day supply",
              f"{drug} is covered but it requires step therapy you'd need to try a preferred alternative first",
              f"{drug} is not on the formulary your doctor can request a formulary exception"][k])
        C(t, rng.choice(["that's expensive", "okay", "what's the alternative", "how do they do that"]))
        A(t, ["i can mail you the formulary", "there's a preferred alternative on tier two i can give you the name for your doctor",
              "the pharmacy team can walk your doctor through the exception form"][k])
        ctx.summary.append(f"Member asked about formulary coverage for {drug}; agent " +
                           [f"confirmed it is tier {tier} with a {dcopay} copay for a 30-day supply.",
                            "explained it requires step therapy with a preferred tier-2 alternative first.",
                            "explained it is not on the formulary and described the formulary exception process."][k])
        ctx.outcomes.append("formulary explained")
    elif sub == "network":
        prov = rng.choice(PROVIDERS)
        C(t, f"{lead}is {prov} in network i want to make sure before i go")
        A(t, "let me search the provider directory")
        A(t, rng.choice([f"yes {prov} is in network for your plan", f"{prov} is showing as out of network for the {ctx.plan}",
                         f"they're in network but only at the {rng.choice(CITIES)} location"]))
        C(t, rng.choice(["okay thanks", "the directory online said something different", "ugh okay"]))
        ctx.summary.append(f"Member asked whether {prov} is in network; agent checked the directory and advised network status.")
        ctx.outcomes.append("network status confirmed")
    elif sub == "deductible":
        ded, _ = money(rng, 500, 6000)
        met, _ = money(rng, 0, 3000)
        C(t, f"{lead}how much of my deductible have i met this year")
        A(t, f"your deductible is {ded} and you've met {met} so far your out of pocket max is {money(rng, 3000, 9000)[0]}")
        C(t, rng.choice(["okay so a visit would cost me what", "does that reset in january", "thanks"]))
        A(t, rng.choice(["yes it resets at the start of the plan year", "you'd pay the full allowed amount until it's met then coinsurance"]))
        ctx.summary.append(f"Member asked about deductible progress; agent confirmed {met} met of a {ded} deductible and explained cost sharing.")
        ctx.outcomes.append("deductible explained")
    else:
        svc = rng.choice(["a dermatologist", "a cardiologist", "an orthopedic specialist", "a therapist", "an allergist", "a podiatrist"])
        C(t, f"{lead}do i need a referral to see {svc}")
        A(t, rng.choice([f"on the {ctx.plan} you do need a referral from your primary care doctor for {svc}",
                         f"no the {ctx.plan} doesn't require referrals for specialists"]))
        C(t, rng.choice(["okay i'll call my doctor", "good", "that's annoying"]))
        ctx.summary.append(f"Member asked whether a referral is required to see {svc}; agent explained the referral rules for the {ctx.plan}.")
        ctx.outcomes.append("referral rules explained")
    return t


def topic_complaints(ctx: Ctx, role: str) -> List[Turn]:
    rng, t = ctx.rng, []
    ctx.frustrated = True
    sub = wchoice(rng, [("wait_times", 2), ("rude_rep", 2), ("repeat_calls", 2.5), ("provider", 1.5), ("vendor", 1.5), ("directory", 1)])
    ref = digits(rng, 8)
    lead = intro_secondary(rng) + " " if role == "secondary" else ""
    if sub == "wait_times":
        C(t, f"{lead}i've been on hold for {rng.randint(35, 90)} minutes this is unacceptable")
        A(t, "i'm really sorry about the wait i understand that's frustrating")
    elif sub == "rude_rep":
        C(t, f"{lead}the last person i talked to was rude and hung up on me i want to file a complaint")
        A(t, "i apologize for that experience i can file a grievance for you")
    elif sub == "repeat_calls":
        C(t, f"{lead}this is the {rng.choice(['third', 'fourth', 'fifth'])} time i've called about the same thing and nothing gets fixed")
        A(t, "i'm sorry you've had to call back so many times let me see the notes")
    elif sub == "provider":
        C(t, f"{lead}i want to complain about {rng.choice(PROVIDERS)} they made me wait two hours and then cancelled")
        A(t, "i'm sorry to hear that i can document a quality of care complaint against the provider")
    elif sub == "vendor":
        C(t, f"{lead}the ride company has been terrible they left me stranded")
        A(t, "i apologize i can file a formal complaint against the transportation vendor")
    else:
        C(t, f"{lead}your provider directory is wrong half the doctors listed don't take the plan anymore")
        A(t, "i apologize i'll report the directory inaccuracy")
    C(t, rng.choice(["i want to talk to a supervisor", "i want this documented", "what are you going to do about it",
                     "i'm about to switch plans", "just fix it"]))
    k = rng.randint(0, 2)
    A(t, [f"i've filed a grievance the case number is {say_number(ref, rng)} you'll get a letter within thirty days",
          "i can have a supervisor call you back within twenty four hours",
          f"i've documented the complaint reference {say_number(ref, rng)} and escalated it"][k])
    C(t, rng.choice(["fine", "okay thank you", "i'll believe it when i see it", "alright"]))
    desc = {"wait_times": "excessive hold times", "rude_rep": "a prior representative's conduct",
            "repeat_calls": "repeated unresolved calls", "provider": "a provider's service",
            "vendor": "the transportation vendor", "directory": "provider directory inaccuracies"}[sub]
    action = [f"filed a grievance (case {ref}) with a letter due within 30 days", "arranged a supervisor callback within 24 hours",
              f"documented and escalated the complaint (reference {ref})"][k]
    ctx.summary.append(f"Member complained about {desc}; agent apologized and {action}.")
    ctx.outcomes.append(["grievance filed", "supervisor callback", "complaint escalated"][k])
    return t


def topic_prior_auth(ctx: Ctx, role: str) -> List[Turn]:
    rng, t = ctx.rng, []
    item = rng.choice(PA_PROCEDURES) if rng.random() < 0.65 else rng.choice(DRUGS)
    prov = rng.choice(PROVIDERS)
    auth = digits(rng, 9)
    status = wchoice(rng, [("pending", 3), ("approved", 2), ("denied", 2.5), ("not_received", 1.5), ("expedite", 1)])
    lead = intro_secondary(rng) + " " if role == "secondary" else ""
    C(t, f"{lead}" + rng.choice([f"{prov} submitted a prior authorization for {item} and i need to know if it's approved",
                                 f"i'm checking on an authorization for {item}", f"what's the status of auth number {say_number(auth, rng)}"]))
    A(t, "let me look at the authorization")
    if status == "pending":
        A(t, f"it's in clinical review it was received on {a_date(rng)[0]} standard reviews take up to fourteen days")
        C(t, rng.choice(["my procedure is next week", "can you speed it up", "okay"]))
        A(t, rng.choice(["if the doctor says it's urgent they can request an expedited review which is seventy two hours",
                         "i'll add a note that the procedure date is approaching"]))
        ctx.summary.append(f"Member checked on a prior authorization from {prov} for {item}; agent confirmed it is in clinical review and explained expedited review options.")
        ctx.outcomes.append("auth pending")
    elif status == "approved":
        A(t, f"good news it's approved authorization number {say_number(auth, rng)} valid through {a_date(rng)[0]}")
        C(t, rng.choice(["great", "can you send that to the doctor", "perfect thank you"]))
        A(t, "the provider was notified and i can also mail you the approval letter")
        ctx.summary.append(f"Member checked on a prior authorization for {item}; agent confirmed approval (auth {auth}) and that the provider was notified.")
        ctx.outcomes.append("auth approved")
    elif status == "denied":
        reason = rng.choice(["the clinical criteria weren't met", "conservative treatment hasn't been tried first",
                             "the documentation didn't include the imaging results", "the service isn't covered for this diagnosis"])
        A(t, f"the authorization was denied because {reason}")
        C(t, rng.choice(["my doctor says i need it", "that's insane", "what can we do", "so now what"]))
        ctx.frustrated = ctx.frustrated or rng.random() < 0.5
        A(t, rng.choice(["your doctor can request a peer to peer review with our medical director",
                         "you can appeal within sixty days and i can start that now", "the provider can resubmit with the missing documentation"]))
        C(t, rng.choice(["okay start the appeal", "i'll tell my doctor", "fine"]))
        ctx.summary.append(f"Member inquired about a prior authorization for {item} from {prov}, which was denied because {reason}; agent explained appeal and peer-to-peer options.")
        ctx.outcomes.append("auth denied, appeal options given")
    elif status == "not_received":
        A(t, f"i don't see an authorization request from {prov} for {item} on file")
        C(t, rng.choice(["they said they faxed it", "ugh", "what do they need to send"]))
        A(t, "i'd have the office resubmit through the provider portal and include the clinical notes")
        ctx.summary.append(f"Member asked about a prior authorization for {item}; none was on file and agent advised the provider to resubmit.")
        ctx.outcomes.append("no auth on file")
    else:
        A(t, "i see the request is marked standard but i can convert it to expedited if the doctor indicates urgency")
        C(t, rng.choice(["yes please it's urgent", "the doctor said it's urgent", "okay"]))
        A(t, "done the expedited review will complete within seventy two hours")
        ctx.summary.append(f"Member requested expedited review of a prior authorization for {item}; agent converted the request to expedited.")
        ctx.outcomes.append("auth expedited")
    return t


def topic_nemt(ctx: Ctx, role: str) -> List[Turn]:
    rng, t = ctx.rng, []
    sub = wchoice(rng, [("schedule", 3), ("no_show", 2.5), ("late", 1.5), ("cancel", 1.5), ("wheelchair", 1), ("mileage", 1)])
    appt = rng.choice(APPT_TYPES)
    when_spoken, when_clean = a_date(rng, past=False)
    tm = a_time(rng)
    trip = digits(rng, 7)
    addr = address(rng)
    lead = intro_secondary(rng) + " " if role == "secondary" else ""
    if role == "primary" and rng.random() < 0.3:
        A(t, "let me get you over to the transportation line one moment")
        A(t, f"transportation this is {rng.choice(AGENTS)} how can i help")
    if sub == "schedule":
        C(t, f"{lead}i need a ride to {appt} on {when_spoken} at {tm}")
        A(t, "sure what's the pickup address")
        C(t, rng.choice([addr, f"it's {addr}", f"{addr} apartment {rng.randint(1, 30)}"]))
        A(t, rng.choice(["and the name and address of the facility", "where is the appointment"]))
        C(t, f"{rng.choice(PROVIDERS)} in {rng.choice(CITIES)}")
        A(t, f"okay the ride is booked pickup at {tm} trip number {say_number(trip, rng)} the driver can arrive up to fifteen minutes early")
        C(t, rng.choice(["is it a round trip", "okay thank you", "will they call me"]))
        A(t, rng.choice(["yes the return is a will call you call when you're done", "yes you'll get a text the night before"]))
        ctx.summary.append(f"Member requested non-emergency medical transportation to {appt} on {when_clean} at {tm}; agent booked the trip (trip {trip}) from {addr}.")
        ctx.outcomes.append("ride booked")
    elif sub == "no_show":
        C(t, f"{lead}my ride never showed up this morning and i missed my {appt}")
        A(t, "i'm so sorry let me look at the trip")
        A(t, rng.choice([f"i see trip {say_number(trip, rng)} was assigned but the driver marked it as a no show on their end",
                         "it looks like the vendor cancelled the trip without notifying you"]))
        ctx.frustrated = True
        C(t, rng.choice(["i was outside waiting for forty minutes", "that's a lie i was there", "this keeps happening"]))
        A(t, rng.choice(["i'm filing a complaint against the vendor and i can rebook the appointment ride for you",
                         "i'll report this and set you up with a different vendor going forward"]))
        C(t, rng.choice(["okay rebook it", "yes please", "fine"]))
        ctx.summary.append(f"Member reported a driver no-show for {appt} (trip {trip}); agent filed a vendor complaint and offered to rebook.")
        ctx.outcomes.append("vendor complaint and rebook")
    elif sub == "late":
        C(t, f"{lead}my driver is {rng.randint(20, 75)} minutes late and i'm going to miss my {appt}")
        A(t, "let me contact the vendor for an eta")
        A(t, rng.choice([f"the driver is about {rng.randint(5, 20)} minutes out", "they're sending a different driver eta twenty minutes"]))
        C(t, rng.choice(["okay", "this is the second time", "hurry"]))
        ctx.summary.append(f"Member reported a late driver for {appt}; agent contacted the vendor and provided an ETA.")
        ctx.outcomes.append("eta provided")
    elif sub == "cancel":
        C(t, f"{lead}i need to cancel my ride on {when_spoken} my appointment got moved")
        A(t, f"okay i've cancelled trip {say_number(trip, rng)} would you like to schedule the new date")
        C(t, rng.choice([f"yes it's now on {a_date(rng)[0]} at {a_time(rng)}", "not yet i'll call back", "yes"]))
        A(t, rng.choice(["booked you'll get a confirmation text", "okay call us at least two days before"]))
        ctx.summary.append(f"Member cancelled a scheduled ride (trip {trip}) for {when_clean} due to an appointment change; agent cancelled and offered rescheduling.")
        ctx.outcomes.append("ride cancelled")
    elif sub == "wheelchair":
        C(t, f"{lead}i need a wheelchair van for my {appt} the regular car doesn't work for me")
        A(t, "i can update your mobility level so the vendor sends a wheelchair accessible vehicle")
        C(t, rng.choice(["thank you", "last time they sent a sedan", "okay"]))
        A(t, "it's updated and your upcoming trips are flagged for a wheelchair van")
        ctx.summary.append(f"Member requested a wheelchair-accessible vehicle for {appt}; agent updated the mobility level and flagged upcoming trips.")
        ctx.outcomes.append("mobility level updated")
    else:
        miles = rng.randint(8, 80)
        C(t, f"{lead}how do i get reimbursed for driving myself to {appt} it's about {miles} miles each way")
        A(t, "you'd submit a mileage reimbursement form with the appointment verification signed by the provider")
        C(t, rng.choice(["can you mail me the form", "how much do they pay", "okay"]))
        A(t, rng.choice(["i'll mail the form today", "it's reimbursed per mile at the current rate within thirty days of receipt"]))
        ctx.summary.append(f"Member asked about mileage reimbursement for self-driving to {appt} ({miles} miles each way); agent explained the form process and mailed the form.")
        ctx.outcomes.append("mileage form sent")
    return t


def topic_technical(ctx: Ctx, role: str) -> List[Turn]:
    rng, t = ctx.rng, []
    sub = wchoice(rng, [("login", 3), ("password", 2), ("id_card", 2), ("two_factor", 1.5), ("missing_data", 1.5), ("payment_page", 1.5), ("app_crash", 1)])
    err = rng.choice(ERRORS)
    browser = rng.choice(BROWSERS)
    lead = intro_secondary(rng) + " " if role == "secondary" else ""
    if sub == "login":
        C(t, f"{lead}i can't log in to the member portal it keeps saying {err}")
        A(t, f"are you using the app or a browser")
        C(t, f"{browser}")
        A(t, rng.choice(["let's try clearing the cache and using an incognito window", "i see your account is locked after too many attempts i can unlock it",
                         "the portal had an outage this morning it should be back now"]))
        C(t, rng.choice(["okay let me try", "okay that worked", "still not working"]))
        A(t, rng.choice(["great you're in", "i'll open a ticket with the web team they'll email you", "try again in an hour"]))
        ctx.summary.append(f"Member could not log in to the member portal ({err}); agent troubleshot the login and unlocked the account or opened a web ticket.")
        ctx.outcomes.append("login troubleshot")
    elif sub == "password":
        C(t, f"{lead}i need to reset my portal password the reset email never comes")
        A(t, "let me check the email address on file")
        k = rng.randint(0, 2)
        A(t, ["the email on file has a typo i've corrected it try the reset again", "i've sent a manual reset link it expires in fifteen minutes",
              "check your spam folder it comes from a no reply address"][k])
        C(t, rng.choice(["got it", "okay i see it now", "thanks"]))
        ctx.summary.append("Member could not receive a portal password reset email; agent " +
                           ["corrected a typo in the email on file.", "sent a manual reset link.", "advised checking the spam folder."][k])
        ctx.outcomes.append(["email corrected", "manual reset link sent", "spam folder advised"][k])
    elif sub == "id_card":
        C(t, f"{lead}i'm trying to download my id card from the app and it just spins")
        if rng.random() < 0.6:
            A(t, "i can email you a digital copy right now")
            C(t, rng.choice(["email is fine", "i need it today for an appointment", "okay"]))
            A(t, "sent you should have it in a few minutes")
            ctx.summary.append("Member could not download the ID card in the mobile app; agent emailed a digital ID card.")
            ctx.outcomes.append("digital id card sent")
        else:
            A(t, "that feature is down we're aware of it i can mail a new card")
            C(t, rng.choice(["okay", "how long will that take", "i need it today for an appointment"]))
            A(t, "it goes out today and arrives in seven to ten days")
            ctx.summary.append("Member could not download the ID card in the mobile app; agent noted a known outage and mailed a replacement card.")
            ctx.outcomes.append("replacement card mailed")
    elif sub == "two_factor":
        C(t, f"{lead}the portal wants a verification code but the text never arrives")
        A(t, f"what's the last four of the phone number you're expecting it on")
        C(t, f"{rng.randint(1000, 9999)}")
        A(t, rng.choice(["we have a different number on file i'll update it", "i'll switch your verification to email instead"]))
        C(t, rng.choice(["okay try now", "thanks"]))
        A(t, "you should get the code now")
        ctx.summary.append("Member was not receiving two-factor verification codes for the portal; agent updated the contact method.")
        ctx.outcomes.append("2fa contact updated")
    elif sub == "missing_data":
        C(t, f"{lead}my claims from last month aren't showing up in the portal")
        A(t, rng.choice(["the portal updates a few days after the claim finalizes", "i see them on my side there's a sync issue i'll report it"]))
        C(t, rng.choice(["okay", "it's been three weeks", "can you just read them to me"]))
        A(t, rng.choice(["i've opened a ticket with the web team", "sure i can read you the status"]))
        ctx.summary.append("Member reported recent claims not appearing in the portal; agent explained the sync delay and opened a web ticket.")
        ctx.outcomes.append("web ticket opened")
    elif sub == "payment_page":
        C(t, f"{lead}i tried to pay my premium online and the payment page gave me {err}")
        A(t, "let me check whether the payment went through")
        if rng.random() < 0.6:
            A(t, "nothing posted so you weren't charged i can take the payment by phone")
            C(t, rng.choice(["okay let's do it by phone", "i don't want to be charged twice"]))
            ctx.summary.append(f"Member encountered {err} on the online payment page; agent verified nothing posted and took the payment by phone.")
            ctx.outcomes.append("payment by phone")
        else:
            A(t, "it actually went through you should see a confirmation email")
            C(t, rng.choice(["oh okay good", "i don't want to be charged twice"]))
            ctx.summary.append(f"Member encountered {err} on the online payment page; agent confirmed the payment had posted successfully.")
            ctx.outcomes.append("payment confirmed")
    else:
        C(t, f"{lead}the app crashes every time i open the find a doctor section")
        A(t, "what phone are you on and is the app up to date")
        C(t, f"{browser} i think it's updated")
        A(t, "i'll report the crash to the app team in the meantime the website has the same directory")
        ctx.summary.append("Member reported the mobile app crashing in the provider directory; agent logged the crash with the app team.")
        ctx.outcomes.append("crash reported")
    return t


TOPIC_FN: Dict[str, Callable[[Ctx, str], List[Turn]]] = {
    "BILLING": topic_billing, "CLAIM_STATUS": topic_claim_status, "COVERAGE": topic_coverage,
    "COMPLAINTS": topic_complaints, "PRIOR_AUTH": topic_prior_auth, "NEMT": topic_nemt, "TECHNICAL_ISSUE": topic_technical,
}

# --------------------------------------------------------------------------- #
# opening / verification / closing
# --------------------------------------------------------------------------- #


def opening(ctx: Ctx, rates: Dict) -> List[Turn]:
    rng, t = ctx.rng, []
    if rng.random() < rates["ivr"]:
        A(t, rng.choice(["thank you for calling this call may be recorded for quality and training purposes",
                         "para espanol oprima el numero dos", "please say or enter your member id",
                         "your call is important to us please stay on the line", "[hold music]"]))
        if rng.random() < 0.4:
            C(t, rng.choice(["representative", "agent agent", "operator", "[silence]", "representative representative"]))
    A(t, rng.choice([f"thank you for calling {ctx.company} member services this is {ctx.agent} how can i help you",
                     f"{ctx.company} this is {ctx.agent} may i have your name and member id",
                     f"hi this is {ctx.agent} with {ctx.company} who do i have the pleasure of speaking with",
                     f"member services {ctx.agent} speaking how can i help"]))
    C(t, rng.choice([f"hi this is {ctx.name}", f"yeah hi {ctx.first} {ctx.last}", f"hello my name is {ctx.name} and i have a question",
                     f"hi um {ctx.name} member id {say_number(ctx.member_id, rng)}", "hi i need some help with my account"]))
    if rng.random() < 0.7:
        A(t, rng.choice(["can i get your member id and date of birth to verify", "thanks and your member id please",
                         "let me verify the account can you confirm your date of birth and zip code"]))
        C(t, rng.choice([f"{say_number(ctx.member_id, rng)}", f"it's {say_number(ctx.member_id, rng)} and {MONTHS[rng.randint(0, 11)]} {rng.randint(1, 28)} {rng.randint(1940, 2002)}",
                         f"{MONTHS[rng.randint(0, 11)]} {ORD[rng.randint(1, 28)]} {rng.randint(1940, 2002)} zip {digits(rng, 5)}",
                         f"hold on {say_number(ctx.member_id, rng)}"]))
        A(t, rng.choice([f"thank you {ctx.first} i have your account up", "perfect you're verified", "okay got it what can i do for you today"]))
    return t


def closing(ctx: Ctx, rates: Dict) -> List[Turn]:
    rng, t = ctx.rng, []
    if rng.random() < rates["dropcall"]:
        A(t, rng.choice(["hello are you still there", "hello [silence]", "i think we lost them [call ended]"]))
        return t
    A(t, rng.choice(["is there anything else i can help with today", "anything else", "did that take care of everything"]))
    C(t, rng.choice(["no that's it thanks", "no thank you", "that's all", "nope thanks for your help", "i guess not"] +
                    (["no but i'm still not happy about this"] if ctx.frustrated else [])))
    A(t, rng.choice([f"thank you for calling {ctx.company} have a good day", "take care goodbye", f"thanks {ctx.first} bye now",
                     "you may get a short survey after this call thank you"]))
    return t


# --------------------------------------------------------------------------- #
# noise
# --------------------------------------------------------------------------- #


def noisify_text(text: str, rates: Dict, rng: random.Random) -> str:
    words = text.split()
    out: List[str] = []
    for w in words:
        if w.startswith("[") and w.endswith("]"):
            out.append(w)
            continue
        # ASR mostly loses short function words; longer content words and numbers survive
        if len(words) > 3 and len(w) <= 4 and not w.isdigit() and rng.random() < rates["drop"]:
            continue
        if w in _CONF_KEYS and rng.random() < rates["sub"]:
            w = rng.choice(ASR_CONFUSIONS[w])
        elif w.isdigit() and len(w) >= 4 and rng.random() < rates["numgarble"]:
            g = rng.random()
            w = spoken_digits(w, rng) if g < 0.4 else grouped(w, rng) if g < 0.7 else w[:-1] + " " + w[-1]
        out.append(w)
        if rng.random() < rates["repeat"]:
            out.append(w.split()[0])
        if rng.random() < rates["filler"]:
            out.append(rng.choice(FILLERS))
    if rng.random() < rates["tag"]:
        out.insert(rng.randint(0, len(out)), rng.choice(TAGS))
    if rng.random() < rates["trunc"] and len(out) > 6:
        out = out[: rng.randint(3, len(out) - 2)] + [rng.choice(["--", "...", "[cut off]"])]
    return " ".join(out)


def noisify_turns(turns: List[Turn], rates: Dict, rng: random.Random) -> List[Turn]:
    noisy: List[Turn] = []
    for spk, text in turns:
        text = noisify_text(text, rates, rng)
        if noisy and rng.random() < rates["tagdrop"]:
            # diarisation error: speaker tag dropped, utterance merges into the previous turn
            pspk, ptext = noisy[-1]
            noisy[-1] = (pspk, ptext + " " + text)
        else:
            noisy.append((spk, text))
    if rng.random() < rates["hold"] and len(noisy) > 6:
        pos = rng.randint(3, len(noisy) - 2)
        noisy.insert(pos, ("agent", rng.choice(["can you hold for just a moment [hold music]", "one moment please [silence]",
                                                "bear with me my system is slow today [typing]", "[hold music] thanks for holding"])))
    return noisy


# --------------------------------------------------------------------------- #
# call assembly
# --------------------------------------------------------------------------- #


def build_call(rng: random.Random, idx: int, primary: str) -> Dict:
    ctx = Ctx(rng=rng, first=rng.choice(FIRST), last=rng.choice(LAST), agent=rng.choice(AGENTS), company=rng.choice(COMPANY),
              plan=rng.choice(PLANS), member_id=digits(rng, rng.choice([9, 10, 11])))
    noise_level = wchoice(rng, NOISE_LEVEL_WEIGHTS)
    rates = NOISE[noise_level]
    n_topics = wchoice(rng, N_TOPICS_WEIGHTS)
    secondaries: List[str] = []
    pool = dict(SECONDARY_AFFINITY[primary])
    while len(secondaries) < n_topics - 1 and pool:
        s = wchoice(rng, list(pool.items()))
        secondaries.append(s)
        pool.pop(s)

    turns: List[Turn] = []
    turns += opening(ctx, rates)
    turns += TOPIC_FN[primary](ctx, "primary")
    for s in secondaries:
        turns += TOPIC_FN[s](ctx, "secondary")
    # frustration can surface as a brief complaint without becoming a labelled topic
    if ctx.frustrated and "COMPLAINTS" not in secondaries and primary != "COMPLAINTS" and rng.random() < 0.5:
        C(turns, rng.choice(["honestly this whole thing has been a hassle", "i just wish this was easier",
                             "you guys really need to fix your system", "sorry i'm just frustrated"]))
        A(turns, rng.choice(["i understand and i appreciate your patience", "i hear you i'm sorry about that"]))
    turns += closing(ctx, rates)
    noisy = noisify_turns(turns, rates, rng)

    labels = sorted({primary, *secondaries}, key=LABEL2ID.get)
    sentences = [f"Member {cap(ctx.first)} {cap(ctx.last)} called {ctx.company} member services."] + ctx.summary
    if len(ctx.outcomes) > 1:
        sentences.append(f"Outcomes: {'; '.join(ctx.outcomes)}.")
    summary = clean_money(" ".join(sentences))
    transcript = " ".join(f"{s}: {x}" for s, x in noisy)
    return {
        "id": f"call_{idx:06d}",
        "transcript": transcript,
        "primary_label": primary,
        "labels": labels,
        "label_vector": [1 if l in labels else 0 for l in LABELS],
        "summary": summary,
        "noise_level": noise_level,
        "n_turns": len(noisy),
        "n_words": len(transcript.split()),
    }


# --------------------------------------------------------------------------- #
# tokenizer / splits / main
# --------------------------------------------------------------------------- #


def make_counter(spec: Optional[str]) -> Tuple[Optional[Callable[[List[str]], List[int]]], str]:
    if not spec or spec == "none":
        return None, "none"
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(spec)
        return (lambda texts: [len(x) for x in tok(texts, add_special_tokens=True)["input_ids"]]), f"transformers:{spec}"
    except Exception as exc:  # pragma: no cover
        print(f"[warn] tokenizer {spec!r} unavailable ({exc}); token counts skipped", file=sys.stderr)
        return None, "none"


def stratified_split(items: List[Dict], rng: random.Random, fracs: Tuple[float, float, float]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    by: Dict[str, List[str]] = {}
    for it in items:
        by.setdefault(it["primary_label"], []).append(it["id"])
    for _, ids in sorted(by.items()):
        rng.shuffle(ids)
        n_tr, n_va = int(round(len(ids) * fracs[0])), int(round(len(ids) * fracs[1]))
        for i, cid in enumerate(ids):
            out[cid] = "train" if i < n_tr else "validation" if i < n_tr + n_va else "test"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--tokenizer", default="/Users/sam/models/roberta-base",
                    help="HF tokenizer path/name for token counts (BART uses the same BPE); 'none' to skip")
    ap.add_argument("--max-tokens", type=int, default=512, help="calls longer than this are regenerated")
    ap.add_argument("--split", default="0.8,0.1,0.1")
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--sample", type=int, default=200, help="rows copied into sample.jsonl for quick inspection")
    args = ap.parse_args()
    fracs = tuple(float(x) for x in args.split.split(","))
    assert len(fracs) == 3 and abs(sum(fracs) - 1) < 1e-6

    rng = random.Random(args.seed)
    t0 = time.time()
    primaries = [wchoice(rng, list(PRIMARY_WEIGHTS.items())) for _ in range(args.n)]
    calls = [build_call(rng, i, p) for i, p in enumerate(primaries)]
    print(f"generated {len(calls)} calls in {time.time() - t0:.0f}s", flush=True)

    counter, counter_name = make_counter(args.tokenizer)
    regenerated = 0
    if counter:
        t1 = time.time()
        B = 2000
        for i in range(0, len(calls), B):
            for c, n in zip(calls[i:i + B], counter([c["transcript"] for c in calls[i:i + B]])):
                c["roberta_tokens"] = n
        # regenerate over-length calls (same primary label) until they fit
        for i, c in enumerate(calls):
            tries = 0
            while c["roberta_tokens"] > args.max_tokens and tries < 20:
                c = build_call(rng, i, c["primary_label"])
                c["roberta_tokens"] = counter([c["transcript"]])[0]
                tries += 1
                regenerated += 1
            calls[i] = c
        print(f"token counts ({counter_name}) in {time.time() - t1:.0f}s; regenerated {regenerated} over-length calls", flush=True)

    split_of = stratified_split(calls, rng, fracs)  # type: ignore[arg-type]
    os.makedirs(args.out, exist_ok=True)
    files = {s: open(os.path.join(args.out, f"{s}.jsonl"), "w", encoding="utf-8") for s in ("train", "validation", "test")}
    sample_f = open(os.path.join(args.out, "sample.jsonl"), "w", encoding="utf-8")
    per: Dict[str, Dict] = {s: {"calls": 0, "primary": Counter(), "multilabel": Counter(), "n_labels": Counter(),
                                "noise": Counter(), "tokens": []} for s in files}
    try:
        for i, c in enumerate(calls):
            s = split_of[c["id"]]
            c["split"] = s
            files[s].write(json.dumps(c, ensure_ascii=False) + "\n")
            if i < args.sample:
                sample_f.write(json.dumps(c, ensure_ascii=False) + "\n")
            p = per[s]
            p["calls"] += 1
            p["primary"][c["primary_label"]] += 1
            for l in c["labels"]:
                p["multilabel"][l] += 1
            p["n_labels"][len(c["labels"])] += 1
            p["noise"][c["noise_level"]] += 1
            if "roberta_tokens" in c:
                p["tokens"].append(c["roberta_tokens"])
    finally:
        for f in files.values():
            f.close()
        sample_f.close()

    def tok_stats(v: List[int]) -> Dict:
        if not v:
            return {}
        v = sorted(v)
        return {"min": v[0], "p50": v[len(v) // 2], "p90": v[int(len(v) * 0.9)], "p99": v[int(len(v) * 0.99)], "max": v[-1],
                "mean": round(sum(v) / len(v), 1), "over_512": sum(x > 512 for x in v)}

    stats = {
        "generator": os.path.basename(__file__), "seed": args.seed, "n_calls": args.n,
        "labels": LABELS, "label_descriptions": LABEL_DESCRIPTIONS, "token_counter": counter_name,
        "max_tokens": args.max_tokens, "regenerated_over_length": regenerated,
        "splits": {s: {"calls": p["calls"], "primary_label": dict(p["primary"]), "multilabel_counts": dict(p["multilabel"]),
                       "labels_per_call": {str(k): v for k, v in sorted(p["n_labels"].items())}, "noise_level": dict(p["noise"]),
                       "roberta_tokens": tok_stats(p["tokens"])} for s, p in per.items()},
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(args.out, "labels.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "label2id": LABEL2ID, "descriptions": LABEL_DESCRIPTIONS}, f, indent=2)
    print(json.dumps({k: stats[k] for k in ("n_calls", "token_counter", "regenerated_over_length", "seconds")}))
    for s, p in stats["splits"].items():
        print(s, p["calls"], "primary", p["primary_label"], "labels/call", p["labels_per_call"], "tokens", p["roberta_tokens"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
