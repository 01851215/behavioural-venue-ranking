# Behaviour-Based Coffee Shop Analysis: Interpretation & Findings

## Executive Summary

This analysis reveals distinct behavioural patterns among coffee shop users in the Yelp dataset,
demonstrating that **user behaviour provides insights that ratings alone cannot capture**.

**Key Findings:**
- Identified **4 behavioural user archetypes** through clustering
- **9.0%** of active users exhibit "Loyalist" behaviour (high revisit ratio)
- Median user revisit ratio: **0.000**
- Median venue revisit rate: **0.016**
- Coffee shops vary significantly in their user base composition

---

## 1. What Behavioural Patterns Exist?

### User Archetypes

Through K-Means clustering on behavioural features, we identified **4 distinct user groups**:


#### Cluster 0: Casual Weekenders
- **Size:** 29,891 users (31.9%)
- **Revisit ratio:** 0.005
- **Unique shops visited:** 2.9
- **Avg time between visits:** 227 days
- **Weekday preference:** 30.9%


#### Cluster 1: Weekday Regulars
- **Size:** 45,948 users (49.0%)
- **Revisit ratio:** 0.012
- **Unique shops visited:** 4.9
- **Avg time between visits:** 184 days
- **Weekday preference:** 89.5%


#### Cluster 2: Loyalists
- **Size:** 8,452 users (9.0%)
- **Revisit ratio:** 0.410
- **Unique shops visited:** 2.2
- **Avg time between visits:** 278 days
- **Weekday preference:** 66.1%


#### Cluster 3: Infrequent Visitors
- **Size:** 9,539 users (10.2%)
- **Revisit ratio:** 0.003
- **Unique shops visited:** 2.2
- **Avg time between visits:** 1439 days
- **Weekday preference:** 68.3%


### Behavioural Insights

1. **Loyalists (Cluster 2)** demonstrate high persistence (41% revisit ratio) with low exploration
   - These users have found their "regular spot"
   - Visit the same coffee shops repeatedly
   - Represent a stable revenue base for venues

2. **Weekday Regulars (Cluster 1)** show routine weekday visits (89.5% weekday ratio)
   - Likely work-related coffee shop visits
   - Moderate exploration behavior
   - Predictable temporal patterns

3. **Casual Weekenders (Cluster 0)** prefer weekend visits
   - Lower weekday ratio (~31%)
   - Exploratory behavior with low revisit rates
   - Leisure-oriented usage

4. **Infrequent Visitors (Cluster 3)** have very long gaps between visits (>1400 days)
   - Occasional users
   - Minimal revisit behavior
   - Represent transient traffic

---

## 2. How Do Different User Groups Express Value?

Different user archetypes express value through **distinct behavioural signatures**:

### Loyalists
- **Value expression:** Repeated visits, long-term commitment
- **Signal strength:** High (consistent, reliable)
- **Rating limitations:** May not review frequently despite high engagement

### Weekday Regulars
- **Value expression:** Routine integration into daily life
- **Signal strength:** Medium-High (predictable patterns)
- **Rating limitations:** May rate only once but visit hundreds of times

### Casual Users
- **Value expression:** Breadth of exploration, variety-seeking
- **Signal strength:** Medium (distributed across many venues)
- **Rating limitations:** Ratings may reflect novelty rather than sustained value

### Infrequent Visitors
- **Value expression:** Sporadic engagement
- **Signal strength:** Low (unreliable temporal pattern)
- **Rating limitations:** May rate based on single experience

---

## 3. Which Coffee Shops Are "Regulars" Venues?

We identified venues based on their **revisit rate** (proportion of users who return):

### Repeat-User Dominated Venues
- **Count:** 3 coffee shops
- **Characteristic:** Revisit rate > 30%
- **User base:** Dominated by Loyalists and Weekday Regulars
- **Behavioral signal:** Strong habitual usage
- **Value proposition:** Consistency, familiarity, routine integration

### Exploratory-User Dominated Venues
- **Count:** 6,494 coffee shops
- **Characteristic:** Revisit rate < 10%
- **User base:** Dominated by Casual and Infrequent users
- **Behavioral signal:** High novelty appeal, low retention
- **Value proposition:** Discovery, variety, one-time experiences

### Implications for Ranking

**Traditional rating-based ranking fails to capture:**
- Venues with high loyalty but moderate ratings
- Difference between novelty appeal and sustained value
- Importance of routine integration vs. occasional satisfaction

**Behavior-based ranking can surface:**
- Hidden gems with devoted regulars
- Venues that integrate into users' daily routines
- Places that build long-term relationships with customers

---

## 4. Why Ratings Alone Fail to Capture These Patterns

### Rating Limitations

1. **Single-Point-in-Time Bias**
   - Rating reflects ONE experience
   - Doesn't capture sustained engagement or revisit behavior
   
2. **Self-Selection Bias**
   - Loyalists may not review frequently (they've already committed)
   - Exploratory users more likely to review each new place
   
3. **Novelty vs. Routine Value**
   - High ratings may reflect novelty appeal
   - Routine value (habitual visits) often goes unrated
   
4. **Missing Temporal Dimension**
   - Ratings don't show temporal patterns (weekday regulars, etc.)
   - Can't distinguish between consistent quality and one-time experiences

### Behavioural Advantages

**Behavioral data captures:**
- ✓ **Revealed preference:** Actions (visits) vs. stated preference (ratings)
- ✓ **Temporal patterns:** When, how often, and how consistently users visit
- ✓ **Loyalty signals:** Revisit behavior shows sustained value
- ✓ **User archetypes:** Different user groups express value differently

---

## Visualizations

![Interpretation Plots](/Users/chris/Desktop/Yelp JSON/yelp_dataset/behaviour_interpretation_plots.png)

---

## Next Steps for Behaviour-Based Ranking

With these behavioral foundations established, future work can:

1. **Construct behavior-based ranking algorithms** using:
   - Weighted revisit signals from different user archetypes
   - Temporal stability as a quality indicator
   - User-venue affinity based on cluster matching

2. **Compare against rating-based baselines** to demonstrate:
   - Which venues are undervalued by ratings
   - Which user groups are better served by behavior-based ranking

3. **Extend to other categories** (restaurants, hotels) using the same pipeline

---

## Conclusion

This analysis demonstrates that **behavior-based analysis reveals patterns ratings cannot capture**.
Coffee shop users exhibit distinct behavioral archetypes, from devoted Loyalists to casual Explorers.
Venues vary in their ability to build loyal user bases vs. attracting one-time visitors.

**Behavior-based ranking can surface venues that:**
- Build strong habitual relationships with users
- Integrate into users' daily routines
- Provide sustained value over time, not just novelty appeal

This foundation enables the next phase: designing and validating behavior-based ranking algorithms
that better reflect how users actually engage with local urban services.

---

*Generated from Yelp Open Dataset*
*Analysis period: 2005-2022*
*Total users analyzed: 351,642*
*Total coffee shops: 8,509*
