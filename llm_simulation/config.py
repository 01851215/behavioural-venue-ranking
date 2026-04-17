import os

# Models — updated to gpt-5.4 / gpt-5.4-mini per user specification
MODEL = "gpt-5.4"          # full model: pairwise + revisit tasks
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MAX_CONCURRENT = 5        # 30K TPM limit → ~35 req/min max; keep headroom
TEMPERATURE = 0.8         # some variation across personas
VENUES_PER_TASK = 10      # candidate venues shown per ranking task
CANDIDATE_POOL_SIZE = 40  # draw from top-N model-ranked venues

# 1500 total personas distributed proportionally across archetypes
PERSONA_COUNTS = {
    "coffee": {
        "Loyalist": 45,           # 9% of real users
        "Weekday Regular": 245,   # 49%
        "Casual Weekender": 160,  # 32%
        "Infrequent Visitor": 50, # 10%
    },
    "restaurant": {
        "Loyalist": 50,
        "Explorer": 150,
        "Mixed / Average": 200,
        "Nightlife Seeker": 100,
    },
    "hotel": {
        "One-Time Tourist (Business)": 249,  # 49.7%
        "Leisure Traveler": 182,             # 36.5%
        "One-Time Tourist": 42,              # 8.4%
        "Budget Explorer": 27,               # 5.4%
    },
}

BOOTSTRAP_SAMPLES = 1000
RANDOM_SEED = 42

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CACHE_PATH = os.path.join(RESULTS_DIR, "response_cache.db")
