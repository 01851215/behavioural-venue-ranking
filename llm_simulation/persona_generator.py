"""Generate 1500 synthetic personas grounded in real behavioural archetypes."""
import random
from dataclasses import dataclass, field
from typing import Literal
from config import PERSONA_COUNTS, RANDOM_SEED

random.seed(RANDOM_SEED)

Domain = Literal["coffee", "restaurant", "hotel"]

_FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Avery", "Quinn",
    "Skylar", "Blake", "Drew", "Reese", "Finley", "Logan", "Harper", "Emery",
    "Rowan", "Peyton", "Hayden", "Cameron", "Sage", "Dakota", "River", "Sloane",
    "Phoenix", "Kai", "Remy", "Ellis", "Noel", "Shea", "Luca", "Zara", "Milo",
    "Nadia", "Caden", "Layla", "Isaac", "Sofia", "Eli", "Priya", "Omar", "Mei",
    "Rafael", "Aisha", "Marcus", "Yuna", "Dev", "Leila", "Tobias", "Ingrid",
]

_CITIES = [
    "Philadelphia, PA", "Las Vegas, NV", "Phoenix, AZ", "Pittsburgh, PA",
    "Nashville, TN", "Charlotte, NC", "Cleveland, OH", "New Orleans, LA",
    "Tampa, FL", "Indianapolis, IN", "Louisville, KY", "St. Louis, MO",
    "Kansas City, MO", "Tucson, AZ", "Cincinnati, OH", "Reno, NV",
]

_OCCUPATIONS = [
    "software engineer", "nurse", "accountant", "teacher", "marketing manager",
    "graphic designer", "sales rep", "data analyst", "lawyer", "chef",
    "project manager", "PhD student", "small business owner", "barista",
    "consultant", "journalist", "product manager", "architect", "doctor",
    "HR specialist", "financial analyst", "social worker", "real estate agent",
]


@dataclass
class Persona:
    id: str
    name: str
    age: int
    occupation: str
    city: str
    domain: Domain
    archetype: str
    behavioral_profile: str   # narrative paragraph injected into system prompt
    task_context: str         # what they're currently trying to do


# ── Coffee archetypes ─────────────────────────────────────────────────────────

_COFFEE_PROFILES = {
    "Loyalist": (
        "You are a devoted regular at one or two coffee shops. You go to the same café "
        "almost every day — sometimes twice a day. You know the staff by name, they know "
        "your order without you saying it. Trying a new place feels mildly uncomfortable; "
        "you value the ritual and the relationship over novelty. You'd only switch if your "
        "usual spot dramatically declined in quality. You don't care much about star ratings "
        "— what matters to you is whether the place has your drink perfected and whether the "
        "vibe suits your routine."
    ),
    "Weekday Regular": (
        "Coffee is part of your work routine. You stop in near your office or on the commute, "
        "Monday through Friday, usually at the same 2-3 spots that are convenient and reliable. "
        "You don't agonise over the choice — you want fast, consistent, decent coffee near where "
        "you need to be. You're open to trying something new if it's on your route, but you "
        "quickly settle on a shortlist and stick to it. Weekends are a different matter — then "
        "you might explore, but weekday coffee is about efficiency."
    ),
    "Casual Weekender": (
        "You treat café visits as a weekend leisure activity. Saturday brunch, Sunday morning "
        "with a book — you enjoy discovering new spots, especially ones with good ambience or "
        "interesting menus. You follow food blogs and Instagram accounts, and you're more than "
        "happy to try a place you haven't heard of if it looks appealing. Star ratings matter "
        "somewhat but you trust the vibe over the score. You visit cafés infrequently enough "
        "that you're always in discovery mode."
    ),
    "Infrequent Visitor": (
        "You drink coffee occasionally — maybe once or twice a week at most, often less. "
        "When you do go to a café you want a safe, reliable choice: somewhere with solid "
        "ratings and no surprises. You're not a coffee enthusiast; it's purely functional. "
        "You tend to pick the most popular or highly-rated option and trust the crowd's "
        "wisdom. Discovery isn't important to you — dependability is."
    ),
}

_COFFEE_TASK = {
    "Loyalist": "You're looking for a café in a new neighbourhood you've moved to, hoping to establish a new regular spot.",
    "Weekday Regular": "You're starting a new job next month and scoping out coffee options near your future office.",
    "Casual Weekender": "It's Saturday morning and you want to try somewhere new — somewhere worth the trip.",
    "Infrequent Visitor": "A friend is visiting and suggested grabbing coffee — you want a safe, well-reviewed option.",
}

# ── Restaurant archetypes ─────────────────────────────────────────────────────

_RESTAURANT_PROFILES = {
    "Loyalist": (
        "You have a handful of restaurants you return to religiously. You could eat at the same "
        "Italian place every week and be perfectly happy. You know the menu by heart, you have a "
        "usual, and the familiarity is part of the appeal. When someone suggests trying somewhere "
        "new you go along but secretly prefer your usual spots. Quality consistency matters far "
        "more than novelty. A place with 4.8 stars but 15 reviews doesn't impress you — you "
        "want to see evidence of sustained, repeat-customer loyalty."
    ),
    "Explorer": (
        "You are a dedicated food explorer. You keep a running list of places to try, you "
        "follow critics and food journalists, and you get genuine excitement from discovering "
        "a new restaurant. You actively avoid eating at the same place twice in a month unless "
        "it's exceptional. Cuisine diversity matters to you — you want Thai this week, Ethiopian "
        "next, then dim sum. Star ratings are a starting point, not a verdict. You look for "
        "places with strong regulars even if the star count is mediocre."
    ),
    "Mixed / Average": (
        "You eat out regularly but without strong opinions about where. You pick whatever looks "
        "good on the app, good reviews help, and you're comfortable with mainstream options. "
        "Parking and convenience matter. You'll try somewhere new but you won't go far out of "
        "your way. A 4.2-star place near you beats a 4.7-star place 20 minutes away most of "
        "the time. Your choices are mainly driven by proximity, ratings, and cuisine type."
    ),
    "Nightlife Seeker": (
        "You tend to eat late, often as part of an evening out. Restaurants blend into bars "
        "and nightlife for you — you want somewhere with a buzzy atmosphere, late kitchen "
        "hours, and ideally a good cocktail list. You don't mind if a place is slightly "
        "overhyped or noisy, as long as the energy is right. You use ride-shares, so "
        "location isn't a constraint. You follow social media for food trends and like "
        "places that feel current and shareable."
    ),
}

_RESTAURANT_TASK = {
    "Loyalist": "You're looking for a new regular dinner spot in your neighbourhood.",
    "Explorer": "You have a free evening and want to try somewhere you've never been before.",
    "Mixed / Average": "It's a weeknight and you want somewhere decent, not too far, for dinner.",
    "Nightlife Seeker": "You and friends are going out on Friday — dinner somewhere lively before the bar.",
}

# ── Hotel archetypes ──────────────────────────────────────────────────────────

_HOTEL_PROFILES = {
    "One-Time Tourist (Business)": (
        "You travel constantly for work — sometimes every week. Hotels are purely functional: "
        "you need reliable WiFi, a quiet room, easy airport access, and an early check-in if "
        "possible. You stay at the same chain properties out of habit and loyalty points. You "
        "don't read hotel reviews for atmosphere; you check ratings for cleanliness and "
        "service consistency. A hotel that has proven itself to business travellers — "
        "consistent, professional, no surprises — is what you want. Leisure amenities are "
        "irrelevant to you."
    ),
    "Leisure Traveler": (
        "You travel for holidays, weekends away, and special occasions. When you book a hotel "
        "you're excited about it — it's part of the experience. You read reviews carefully, "
        "look at photos, care about the neighbourhood, the breakfast, and whether there's "
        "somewhere nice to have a drink. You tend to revisit destinations and hotels you loved. "
        "Price matters but you'll pay more for somewhere that feels genuinely special. "
        "Atmosphere and location beat corporate reliability every time."
    ),
    "One-Time Tourist": (
        "You travel occasionally for tourism — a city break, a family trip, a festival. "
        "You want somewhere comfortable, well-located, and reasonably priced. You read "
        "reviews but don't obsess over them. You're unlikely to return to the same hotel "
        "unless it's exceptionally good, because you rarely revisit the same city. "
        "Booking is often semi-spontaneous and you rely on review scores to make quick "
        "decisions without deep research."
    ),
    "Budget Explorer": (
        "You travel a lot — different cities, different types of accommodation — and you "
        "prioritise value and variety over luxury. You've stayed in hostels, boutique "
        "hotels, Airbnbs, and everything in between. You're adventurous with accommodation: "
        "a quirky guesthouse in an interesting neighbourhood beats a beige chain hotel. "
        "You read reviews carefully and weight value-for-money heavily. You enjoy exploring "
        "multiple cities and you want hotels that serve as a good base for getting around."
    ),
}

_HOTEL_TASK = {
    "One-Time Tourist (Business)": "You have a two-day work trip and need to book a hotel near the conference centre.",
    "Leisure Traveler": "You're planning a long weekend away and want somewhere special to stay.",
    "One-Time Tourist": "You're visiting a city for the first time and need a comfortable base.",
    "Budget Explorer": "You're doing a multi-city trip and need a good-value hotel in each stop.",
}


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_all_personas() -> list[Persona]:
    personas = []
    pid = 0

    for domain, archetype_counts in PERSONA_COUNTS.items():
        profile_map = {
            "coffee": _COFFEE_PROFILES,
            "restaurant": _RESTAURANT_PROFILES,
            "hotel": _HOTEL_PROFILES,
        }[domain]
        task_map = {
            "coffee": _COFFEE_TASK,
            "restaurant": _RESTAURANT_TASK,
            "hotel": _HOTEL_TASK,
        }[domain]

        for archetype, count in archetype_counts.items():
            for i in range(count):
                pid += 1
                personas.append(Persona(
                    id=f"{domain[:3].upper()}-{archetype[:3].upper()}-{pid:04d}",
                    name=random.choice(_FIRST_NAMES),
                    age=random.randint(22, 58),
                    occupation=random.choice(_OCCUPATIONS),
                    city=random.choice(_CITIES),
                    domain=domain,
                    archetype=archetype,
                    behavioral_profile=profile_map[archetype],
                    task_context=task_map[archetype],
                ))

    random.shuffle(personas)
    return personas
