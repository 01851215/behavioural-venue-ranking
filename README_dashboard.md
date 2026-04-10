# Coffee Shop Ranking Inspector Dashboard

## Quick Start

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Overview

This is a **research prototype** for inspecting coffee shop ranking outputs. It allows exploration of how different behavioral modes and ranking algorithms produce different venue recommendations.

### Important Notes

- **NOT a consumer app** - This is a research tool for analyzing ranking algorithms
- **No demographic inference** - "Modes" are user-selected behavioral preferences, not age/income/demographics
- **Purpose:** Inspect and understand how group-conditioned rankings work

## Features

### Sidebar Controls

1. **City Selector** - Choose from available cities in the dataset
2. **Area Filter** - Select a reference café and radius (0.5-10 km)
3. **Ranking Mode:**
   - Behaviour-based (BiRank)
   - Rating-based
   - Popularity-based
   - Revisit-rate-based
4. **Behaviour Mode** (user preference):
   - Regular / routine visits
   - Explorer / try new places
   - Morning quick stop
   - Weekend casual
5. **Top-K Selector** - Show top 10/20/50 results

### Main Panel

**Left Column:** Ranking table with:
- Rank, café name, distance, score, rating, revisit rate
- Sortable and downloadable as CSV

**Right Column:** Interactive map showing:
- Reference point (red marker)
- Radius circle
- Top-K cafés (green/orange star markers)
- Other nearby cafés (gray circles)
- Click markers for popups with details

### Venue Details

Select any café to see:
- Plain-language explanation of why it's ranked highly
- Key metrics (score, rating, revisit rate, total visits)
- **No algorithm jargon** - explanations in human-friendly terms

## Data Requirements

### Required Files (must be present):
- `business_coffee.csv` - Business info with coordinates
- `coffee_birank_venue_scores.csv` - Global BiRank scores
- `coffee_baselines.csv` - Rating, popularity, revisit baselines

### Optional Files (enhance functionality):
- `coffee_birank_venue_scores_by_group.csv` - Group-specific scores
- `coffee_venue_features.csv` - Additional venue features

If optional files are missing, the app uses composite scoring based on weighted baselines.

## Behavior Mode Mapping

When group-specific scores are available:
- **Regular / routine visits** → Loyalists
- **Explorer / try new places** → Weekday Regulars
- **Morning quick stop** → Weekday Regulars (proxy)
- **Weekend casual** → Casual Weekenders

When group scores are unavailable, uses weighted composites:
- **Routine:** 60% revisit + 30% BiRank + 10% rating
- **Explorer:** 40% popularity + 30% rating + 30% BiRank
- **Morning:** 50% rating + 50% revisit
- **Weekend:** 100% popularity

## Technical Details

- **Framework:** Streamlit + Folium
- **Distance calculation:** Haversine formula (great circle distance)
- **Map tiles:** OpenStreetMap
- **Normalization:** Min-max scaling for composite scores

## Limitations

1. **Morning/Weekend modes are proxies** - No time-of-day data in current dataset
2. **Group scores require clustering** - Falls back to composite if unavailable
3. **Distance is "as the crow flies"** - Not walking/driving distance

## Development

To modify the app:
1. Edit `app.py`
2. Save changes
3. Streamlit auto-reloads on file changes

To debug:
```bash
streamlit run app.py --logger.level=debug
```

## Example Use Cases

### Research Questions

1. **"Do routine users prefer different cafés than explorers?"**
   - Set mode to "Regular / routine" → Note top 10
   - Change to "Explorer / try new places" → Compare

2. **"How does BiRank differ from rating-based ranking?"**
   - Set mode to "Behaviour-based" → Note results
   - Change to "Rating-based" → See differences

3. **"What are the best cafés near Reading Terminal Market?"**
   - Select Philadelphia
   - Pick "Reading Terminal Market" as reference
   - Set radius to 2 km
   - Explore results

### Inspection Workflow

1. Select a city and area
2. Choose ranking mode (e.g., BiRank)
3. Choose behavior preference (e.g., routine)
4. Review top-K table
5. Check map visualization
6. Click a café for explanation
7. Compare with other modes
8. Download results as CSV

## Credits

**Data:** Yelp Open Dataset  
**Analysis:** Coffee shop behavioral clustering + BiRank graph ranking  
**Tool:** Research prototype for ranking inspection  

---

*For questions or issues, refer to the main project documentation.*
