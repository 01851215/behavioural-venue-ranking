"""Generate 3,000 personas from the 5-age × 10-occupation × 3-domain cross-matrix.

Each cell produces 20 personas with:
  - Random name / exact age within band / specific job title
  - City drawn from occupation-appropriate pool
  - Behavioral profile grounded in research (demographic_profiles.py)
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List

from config import RANDOM_SEED
from demographic_profiles import (
    VALID_CELLS, AGE_RANGES,
    get_profile, get_cities, get_job_titles,
)

random.seed(RANDOM_SEED)

PERSONAS_PER_CELL = 20

_FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Taylor", "Casey", "Riley", "Avery", "Quinn",
    "Skylar", "Blake", "Reese", "Logan", "Harper", "Rowan", "Peyton", "Cameron",
    "Sage", "Dakota", "Phoenix", "Kai", "Remy", "Ellis", "Noel", "Luca", "Zara",
    "Milo", "Nadia", "Isaac", "Sofia", "Priya", "Omar", "Mei", "Rafael", "Aisha",
    "Marcus", "Yuna", "Dev", "Leila", "Tobias", "Ingrid", "Sam", "Jamie", "Drew",
    "Finley", "Hayden", "Shea", "River", "Sloane", "Caden", "Layla",
]

_LAST_NAMES = [
    "Chen", "Osei", "Mitchell", "Rivera", "Kim", "Patel", "Nguyen", "Schmidt",
    "Lopez", "Williams", "Torres", "Park", "Hansen", "Silva", "Rossi", "Müller",
    "Nakamura", "Andersen", "Kowalski", "Petrov", "Dubois", "O'Brien", "Johansson",
    "Garcia", "Johnson", "Brown", "Wilson", "Thompson", "White", "Harris",
    "Martin", "Jackson", "Lee", "Walker", "Hall", "Allen", "Young", "King",
]


@dataclass
class DemographicPersona:
    id: str
    name: str
    age: int
    occupation_label: str   # full job title (e.g. "Registered Nurse")
    occupation_cluster: str  # cluster name (e.g. "Healthcare")
    age_group: str
    city: str
    domain: str
    behavioral_profile: str
    task_context: str
    sort_col: str            # venue feature to rank by for this cell
    loyalty_score: int       # 1 (explorer) to 5 (loyalist)
    price_sensitivity: int   # 1 (low) to 5 (high)

    @property
    def archetype(self) -> str:
        """Composite label used by shared prompt functions."""
        return self.occupation_cluster

    @property
    def occupation(self) -> str:
        return self.occupation_label


def generate_study2_personas() -> List[DemographicPersona]:
    personas: List[DemographicPersona] = []
    pid = 0

    rng = random.Random(RANDOM_SEED)

    for (age_group, occupation) in VALID_CELLS:
        age_lo, age_hi = AGE_RANGES[age_group]
        cities = get_cities(age_group, occupation)
        job_titles = get_job_titles(age_group, occupation)

        for domain in ("coffee", "restaurant", "hotel"):
            dom_profile = get_profile(age_group, occupation, domain)

            for _ in range(PERSONAS_PER_CELL):
                pid += 1
                name = f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
                age = rng.randint(age_lo, age_hi)
                job = rng.choice(job_titles)
                city = rng.choice(cities)

                # Age group abbreviation for ID
                age_abbr = {
                    "Gen Z (18-25)": "GZ",
                    "Young Millennial (26-33)": "YM",
                    "Senior Millennial (34-40)": "SM",
                    "Gen X (41-56)": "GX",
                    "Boomer (57+)": "BO",
                }[age_group]
                occ_abbr = occupation[:3].upper().replace(" ", "")
                dom_abbr = domain[:3].upper()

                personas.append(DemographicPersona(
                    id=f"S2-{age_abbr}-{occ_abbr}-{dom_abbr}-{pid:04d}",
                    name=name,
                    age=age,
                    occupation_label=job,
                    occupation_cluster=occupation,
                    age_group=age_group,
                    city=city,
                    domain=domain,
                    behavioral_profile=dom_profile["profile"],
                    task_context=dom_profile["task"],
                    sort_col=dom_profile["sort_col"],
                    loyalty_score=dom_profile["loyalty_score"],
                    price_sensitivity=dom_profile["price_sens"],
                ))

    rng.shuffle(personas)
    return personas
