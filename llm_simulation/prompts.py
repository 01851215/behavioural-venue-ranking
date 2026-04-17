"""Prompt templates — v2 (Phase 1): stronger behavioural anchoring, archetype
feature emphasis in ranking task, manipulation check prompts."""
from __future__ import annotations
from typing import Dict


# ── Archetype: which behavioural feature to highlight in ranking prompts ───────
# Used to surface the dimension each archetype *should* react to.

_ARCHETYPE_EMPHASIS: Dict[str, str] = {
    # Coffee
    "Loyalist":           "Pay close attention to whether a venue has a strong loyal, regular customer base.",
    "Weekday Regular":    "Pay close attention to how consistent and reliable the venue's traffic patterns are.",
    "Casual Weekender":   "Pay close attention to whether a venue appeals broadly to many different types of people.",
    "Infrequent Visitor": "Pay close attention to the star rating and overall review count as signals of safety.",
    # Restaurant
    "Explorer":           "Pay close attention to how distinctive and varied the cuisine and dining experience is.",
    "Mixed / Average":    "Pay close attention to the overall rating and how easy the venue is to reach.",
    "Nightlife Seeker":   "Pay close attention to how buzzy and lively the venue is, especially in the evenings.",
    # Hotel
    "One-Time Tourist (Business)": "Pay close attention to how consistently professional and reliable the venue is.",
    "Leisure Traveler":            "Pay close attention to the atmosphere, experience quality, and destination appeal.",
    "One-Time Tourist":            "Pay close attention to overall ratings and review volume as proxies for quality.",
    "Budget Explorer":             "Pay close attention to value and variety — you want a good base without overpaying.",
}

# First-person frequency anchors injected into system prompts
_ARCHETYPE_STATS: Dict[str, str] = {
    "Loyalist":           "In the past year, you visited your regular café over 200 times.",
    "Weekday Regular":    "You grab coffee near work roughly 4 mornings a week — same 2-3 spots in rotation.",
    "Casual Weekender":   "You visit cafés maybe 6-8 times a month, almost always somewhere you haven't tried before.",
    "Infrequent Visitor": "You go to a coffee shop once every couple of weeks at most.",
    "Explorer":           "You eat at a new restaurant almost every week. Repeating a restaurant in a month is rare.",
    "Mixed / Average":    "You eat out 2-3 times a week, usually picking whatever is close and well-reviewed.",
    "Nightlife Seeker":   "Most of your restaurant visits happen after 9pm as part of a night out.",
    "One-Time Tourist (Business)": "You've stayed in hotels over 40 nights this year, almost all for work.",
    "Leisure Traveler":   "You take 3-4 leisure trips a year and really look forward to where you'll stay.",
    "One-Time Tourist":   "You travel a couple of times a year and just need somewhere comfortable to stay.",
    "Budget Explorer":    "You've stayed in 8 different cities this year, mixing hostels, guesthouses, and budget hotels.",
}


def build_system_prompt(persona) -> str:
    stat = _ARCHETYPE_STATS.get(persona.archetype, "")
    stat_line = f"\n\n**A concrete fact about you:** {stat}" if stat else ""
    return (
        f"You are {persona.name}, a {persona.age}-year-old {persona.occupation} "
        f"living in {persona.city}.{stat_line}\n\n"
        f"{persona.behavioral_profile}\n\n"
        "You make decisions that genuinely reflect your habits. You do NOT try to guess "
        "what an AI expects — you respond as this specific person, driven by your real "
        "priorities. When you disagree with what looks 'objectively best', you say so."
    )


# ── Task 1: Ranking Task ──────────────────────────────────────────────────────

def build_ranking_prompt(persona, candidate_df, domain: str) -> str:
    venue_lines = _format_venues(candidate_df, domain)
    emphasis = _ARCHETYPE_EMPHASIS.get(persona.archetype, "")
    task_intro = {
        "coffee": "You're choosing a coffee shop to visit.",
        "restaurant": "You're choosing a restaurant for dinner.",
        "hotel": "You're choosing a hotel to book.",
    }[domain]

    emphasis_line = f"\n\n**Your priority:** {emphasis}" if emphasis else ""

    prompt = (
        f"**Your situation:** {persona.task_context}{emphasis_line}\n\n"
        f"{task_intro} Here are {len(candidate_df)} options:\n\n"
        f"{venue_lines}\n\n"
        "Rank these venues from most to least appealing **for you specifically**, "
        "given your habits, priorities, and what you're looking for right now. "
        "Do not rank by what you think is objectively best — rank by what you personally "
        "would choose.\n\n"
        "Return a JSON object with this exact structure:\n"
        '{"ranking": [1, 3, 7, 2, 5, 8, 4, 6, 9, 10], "reasoning": "one sentence"}\n\n'
        "where the list contains the venue numbers (1–10) in order from your top pick "
        "to your least preferred. Include all 10 numbers exactly once."
    )
    return prompt


# ── Task 2: Pairwise Head-to-Head ─────────────────────────────────────────────

def build_pairwise_prompt(persona, birank_venue: dict, star_venue: dict, domain: str) -> str:
    d_label = {"coffee": "café", "restaurant": "restaurant", "hotel": "hotel"}[domain]
    a = _venue_card(birank_venue, domain, "A")
    b = _venue_card(star_venue, domain, "B")
    emphasis = _ARCHETYPE_EMPHASIS.get(persona.archetype, "")
    emphasis_line = f"\n**Your priority:** {emphasis}" if emphasis else ""

    prompt = (
        f"**Your situation:** {persona.task_context}{emphasis_line}\n\n"
        f"You must choose between exactly two {d_label}s:\n\n"
        f"{a}\n\n{b}\n\n"
        "Which would you choose, given your personal preferences and habits? "
        "There is no correct answer — choose the one that genuinely suits you.\n\n"
        'Return JSON: {"choice": "A" or "B", "reason": "one sentence"}'
    )
    return prompt


# ── Task 3: Revisit Prediction ────────────────────────────────────────────────

def build_revisit_prompt(persona, venue: dict, domain: str) -> str:
    d_label = {"coffee": "café", "restaurant": "restaurant", "hotel": "hotel"}[domain]
    card = _venue_card(venue, domain, "this venue")
    stat = _ARCHETYPE_STATS.get(persona.archetype, "")
    stat_line = f" (remember: {stat})" if stat else ""

    prompt = (
        f"You visited the following {d_label} last month{stat_line}:\n\n"
        f"{card}\n\n"
        "Given your typical patterns and what this venue offered, how likely are you "
        "to return? Answer as yourself, not as a general customer.\n\n"
        "Scale: 0 = definitely not returning, 10 = certain to return.\n\n"
        'Return JSON: {"revisit_score": <integer 0-10>, "reason": "one sentence"}'
    )
    return prompt


# ── Task 4: Manipulation check prompts (Phase 2) ──────────────────────────────

_MANIPULATION_QUESTIONS: Dict[str, list] = {
    "coffee": [
        "How many different coffee shops have you visited in the past 6 months?",
        "When did you last try a café you had never been to before?",
        "What matters most when choosing a café — habit/familiarity, convenience/location, discovery/novelty, or safety/reliability?",
    ],
    "restaurant": [
        "How often do you eat at the same restaurant more than twice in a single month?",
        "When choosing where to eat, how important is it that you haven't been to the restaurant before — 'very important', 'somewhat important', 'not important', or 'actively prefer familiar'?",
        "Would you travel 30 minutes for a restaurant you haven't tried, or pick something closer and familiar?",
    ],
    "hotel": [
        "Do you typically book the same hotel chain across different cities, or do you vary?",
        "What matters more when booking a hotel: atmosphere and experience, or reliability and consistency?",
        "How many different cities have you stayed in overnight in the past 12 months?",
    ],
}


def build_manipulation_check_prompt(persona) -> str:
    questions = _MANIPULATION_QUESTIONS.get(persona.domain, _MANIPULATION_QUESTIONS["coffee"])
    q_list = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
    return (
        "Answer these three short questions about yourself, as honestly as you can. "
        "Keep each answer to 1-2 sentences.\n\n"
        f"{q_list}\n\n"
        'Return JSON: {"answers": ["answer to Q1", "answer to Q2", "answer to Q3"]}'
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_venues(df, domain: str) -> str:
    lines = []
    for i, row in df.iterrows():
        num = i + 1
        lines.append(f"**Venue {num}:** {_venue_inline(row, domain)}")
    return "\n".join(lines)


def _venue_inline(row, domain: str) -> str:
    name = row.get("name", "Unknown")
    city = row.get("city", "")
    state = row.get("state", "")
    stars = row.get("stars", "?")
    tag = row.get("behavioral_tag", "")
    review_count = row.get("review_count", row.get("n_reviews", ""))

    location = f"{city}, {state}" if city and state else city or state or ""
    reviews_str = f" · {int(review_count):,} reviews" if review_count and str(review_count) != "nan" else ""

    if domain == "coffee":
        return f"{name} ({location}) — ⭐ {stars}{reviews_str} | {tag}"
    elif domain == "restaurant":
        cuisine = str(row.get("categories", row.get("cuisine_categories", ""))).split(",")[0].strip()
        cuisine_str = f" · {cuisine}" if cuisine and cuisine != "nan" else ""
        return f"{name} ({location}){cuisine_str} — ⭐ {stars}{reviews_str} | {tag}"
    else:
        subcat = row.get("subcategory", "Hotel")
        return f"{name} ({location}) [{subcat}] — ⭐ {stars}{reviews_str} | {tag}"


def _venue_card(venue: dict, domain: str, label: str) -> str:
    name = venue.get("name", "Unknown")
    city = venue.get("city", "")
    state = venue.get("state", "")
    stars = venue.get("stars", "?")
    tag = venue.get("behavioral_tag", "")
    review_count = venue.get("review_count", venue.get("n_reviews", ""))

    location = f"{city}, {state}" if city and state else city or state or ""
    rc = str(review_count)
    reviews_str = f" · {int(float(rc)):,} reviews" if rc not in ("", "nan", "None") else ""

    if domain == "hotel":
        subcat = venue.get("subcategory", "Hotel")
        return f"**Option {label}:** {name} ({location}) [{subcat}] — ⭐ {stars}{reviews_str}\n_{tag}_"
    elif domain == "restaurant":
        cuisine = str(venue.get("categories", venue.get("cuisine_categories", ""))).split(",")[0].strip()
        cuisine_str = f" · {cuisine}" if cuisine and cuisine != "nan" else ""
        return f"**Option {label}:** {name} ({location}){cuisine_str} — ⭐ {stars}{reviews_str}\n_{tag}_"
    else:
        return f"**Option {label}:** {name} ({location}) — ⭐ {stars}{reviews_str}\n_{tag}_"
