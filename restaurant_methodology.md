# Restaurant Behavioral Ranking Model: Methodology

## Introduction

Choosing a restaurant isn't just about finding the place with the highest star rating. We often care about how far away it is, whether it fits our specific dining habits (are we feeling adventurous or sticking to our favorites?), and if it's too crowded right now. 

To solve this, we built a **Behavioral Restaurant Ranking Engine**. Unlike traditional systems that just show you the most popular places, this engine learns from how you and others interact with the city. It combines data from three different sources to score restaurants on three distinct criteria: **Behavior**, **Mobility**, and **Context**. 

Here is a plain-language explanation of how the algorithm works and the knowledge it relies on. 

---

## The Knowledge Base: Three Datasets

To build a complete picture of the dining landscape, we integrated three datasets:

1. **Yelp Dataset (The Behavior):** We analyzed over 15 million genuine interactions (reviews and check-ins) across 64,000 restaurants. This tells us *who* is going *where*, *how often*, and *what they think about it*. 
2. **Transitland Dataset (The Mobility):** We mapped hundreds of thousands of public transit stops across the country. By looking at bus and train departure frequencies, we can measure how easily a restaurant can be reached without a car.
3. **Busyness Data (The Context):** Using check-in timestamps as a proxy (similar to Google's Popular Times), we estimated the hourly busyness and peak capacity of every restaurant. 

---

## Step 1: Understanding the User (Archetypes)

Before we can recommend a restaurant, we need to understand the diner. The model profiles every user based on their past Yelp history and groups them into behavioral "Archetypes." 

We measure several factors:
- **Variety Seeking (Entropy):** Do you visit 10 different restaurants in a month, or do you visit your 2 favorite spots 5 times each? This separates the **"Explorers"** from the **"Loyalists"**.
- **Spatial Range:** How far are you willing to travel for food? This separates **"Distance-First"** diners (who stay in their neighborhood) from **"Preference-First"** diners (who will cross the city for a specific meal).
- **Rating Sensitivity:** Do you only leave 5-star or 1-star reviews? Are you easily satisfied or highly critical? This identifies the **"Critics"** versus the **"Casuals"**.

---

## Step 2: Understanding the Venue

We also analyze the restaurants themselves. A 4.5-star rating isn't enough information. We compute:
- **Loyalty Magnetism:** What percentage of a restaurant's traffic comes from repeat customers? 
- **Niche vs. Broad Appeal (Gini Coefficient):** Is this place kept alive by a small, dedicated group of regulars, or is it a tourist trap that everyone visits exactly once?
- **Transit Accessibility:** We draw an 800-meter walking radius around every restaurant and count the daily transit departures. Places near major subway hubs get higher mobility scores.

---

## Step 3: The S(R,U,C) Scoring Engine

When calculating the final ranking, we use a formula called **S(R,U,C)**. This stands for a **S**core based on the **R**estaurant, the **U**ser, and the **C**ontext. 

The algorithm grades every candidate restaurant on three sub-scores:

### 1. Behavioral Utility ($U_{beh}$)
This measures how much you will genuinely *like* the restaurant. It looks at the baseline popularity of the venue but adds a **Critic Penalty**: if you are flagged as a "Critic" archetype and the venue's average rating is lower than your personal historical average, the algorithm penalizes the restaurant. 

### 2. Mobility Convenience ($C_{mob}$)
This measures how painful it will be to get there. It calculates the direct distance between you and the restaurant. If you have a small "Spatial Range" archetype, distant restaurants are heavily penalized. However, restaurants gain a **Walking Bonus** if they are within 800 meters, and a **Transit Bonus** if they are near high-frequency bus/train stops.

### 3. Contextual Relevance ($R_{ctx}$)
This modifies the score based on the current situation. The engine applies a **Queue Penalty** based on the venue's peak busyness to avoid sending you to places that are overcrowded. It also looks at how well the restaurant's cuisine matches your historical category preferences.

---

## Step 4: Intelligent Balancing & Diversity

How do we decide which of those three scores matters most? 

### Dynamic Weighting
We use a mathematical technique called the **Entropy Weight Method (EWM)**. Instead of assigning a fixed weight (e.g., 40% Behavior, 40% Mobility, 20% Context), the algorithm looks at the candidate restaurants for *you specifically*. If all the restaurants near you are equally far away, Distance becomes a useless metric. The algorithm automatically lowers the weight of distance and increases the weight of Behavior to break the tie. 

### Diversity-Aware Ranking
Finally, we don't want to show you ten identical burger joints on the same street. We use an algorithm called **Maximal Marginal Relevance (MMR)**. As the engine builds your Top 10 list, it constantly checks the cuisines and locations of the restaurants it has already picked. If it just added a Mexican restaurant, it strictly penalizes other Mexican restaurants in the same immediate area to guarantee you get a diverse set of options. 

---

## Validation: Proving It Works

We didn't just assume this worked; we tested it against the future. We hid all interactions that happened after January 2020 and asked the algorithm to predict where users would eat in 2020 and beyond. 

The results were clear:
- The full multi-objective model was dramatically more accurate than simply recommending the most popular or highest-rated places.
- **Ablation Study:** By systematically turning off parts of the algorithm, we proved that **Mobility** is the single most important factor. If the algorithm doesn't understand distance and transit, prediction accuracy plummets. Recognizing a user's **Critic Penalty** also resulted in mathematically better, more tailored matches.
