"""
EmergeRec Dataset Registry

Central registry of all available datasets.
Each dataset entry specifies: file path, split date, has_stars flag.
"""

from pathlib import Path

REPO = Path(__file__).parent.parent.parent
DATA = REPO / "data"

DATASETS = {
    "uk_foursquare": {
        "interactions": DATA / "results/uk_fsq_interactions.csv",
        "businesses":   DATA / "results/uk_fsq_businesses.csv",
        "split_date":   "2013-07-01",
        "has_stars":    False,
        "description":  "Foursquare WWW2019 GB check-ins · 288K events · 70K venues",
    },
    "tripadvisor_london": {
        "interactions": DATA / "results/london_interactions.csv",
        "businesses":   DATA / "results/london_businesses.csv",
        "split_date":   "2018-01-01",
        "has_stars":    True,
        "description":  "TripAdvisor London restaurants · 997K reviews · 1,877 venues",
    },
    "yelp_coffee": {
        "interactions": Path("/Users/chris/Desktop/Master Project/yelp_dataset/coffee_interactions.csv"),
        "businesses":   None,
        "split_date":   "2020-01-01",
        "has_stars":    False,
        "description":  "Yelp US coffee shops · 2.2M interactions · 8,509 venues",
    },
    "yelp_restaurant": {
        "interactions": DATA / "processed/restaurant_interactions.csv",
        "businesses":   None,
        "split_date":   "2020-01-01",
        "has_stars":    True,
        "description":  "Yelp US restaurants",
    },
    "yelp_hotel": {
        "interactions": DATA / "processed/hotel_interactions.csv",
        "businesses":   None,
        "split_date":   "2020-01-01",
        "has_stars":    True,
        "description":  "Yelp US hotels",
    },
    "tokyo_foursquare": {
        "interactions": DATA / "processed/tokyo_fsq_interactions.csv",
        "businesses":   DATA / "processed/tokyo_fsq_businesses.csv",
        "split_date":   "2013-07-01",
        "has_stars":    False,
        "description":  "Foursquare WWW2019 Tokyo/Japan check-ins",
    },
}


def list_datasets():
    """Print all available datasets and their status."""
    print(f"{'Dataset':<25} {'Available':>10} Description")
    print("-" * 70)
    for name, cfg in DATASETS.items():
        available = "✓" if cfg["interactions"].exists() else "✗ (missing)"
        print(f"{name:<25} {available:>10}  {cfg['description']}")


def get_dataset(name: str) -> dict:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list(DATASETS.keys())}")
    cfg = DATASETS[name]
    if not cfg["interactions"].exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {cfg['interactions']}.\n"
            f"Run the appropriate extraction pipeline first."
        )
    return cfg
