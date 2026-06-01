# How We Ranked Coffee Shops by Real Behavior — A Plain-Language Guide

## The Big Idea

Most review platforms rank businesses by star ratings—the number you see next to a restaurant's name. But ratings can be misleading. A café with a perfect 5-star score from just three reviews tells you very little compared to a café that hundreds of people keep coming back to, week after week, for years.

We built a system that ranks coffee shops not by what people *say* (ratings), but by what people *do* (their actual visit patterns). The idea is simple: **actions speak louder than stars.**

This document explains—step by step, in everyday language—how we did it, what knowledge we drew on, and why it works.

---

## Step 1: Gathering the Raw Ingredients

**What we did:** We started with Yelp's publicly available dataset, which contains records of businesses, user reviews, check-ins, and tips. We extracted the key pieces of information we needed: *who* visited *where* and *when*.

**Why it matters:** Think of it like gathering receipts. Each review or check-in is a receipt that proves someone was at a particular coffee shop on a particular date. We don't care what they wrote—we only care that they showed up.

**Knowledge used:** This is standard *data extraction*—a routine first step in any analysis where you pull out just the columns you need from a large spreadsheet.

---

## Step 2: Filtering to Coffee Shops Only

**What we did:** The Yelp dataset covers everything from dentists to nightclubs. We filtered it down to only businesses labeled as "Coffee," "Café," "Espresso," or "Coffee & Tea."

**Why it matters:** Comparing a coffee shop to a car dealership makes no sense. By focusing on one category, we ensure we're comparing apples to apples—or in this case, lattes to lattes.

**Result:** We identified **8,509** coffee shops across the US and Canada.

---

## Step 3: Measuring How People Actually Behave

This is the heart of the method. We turned raw visit records into meaningful *behavioral signals*—numbers that describe patterns of habit, loyalty, and regularity. Here's what we measured and why:

### Visit Frequency & Revisit Rate

- **What:** How many times does a typical person come back to the same café?
- **Why:** A cafe where 30% of visitors return is doing something right. One where almost nobody returns might have gotten lucky with tourists.

### Burstiness: Steady Habit vs. One-Time Hype

- **What:** We measured how *regular* someone's visits are. Do they come every Tuesday like clockwork, or did they visit three times in one week and never again?
- **The Formula:** We used a well-known measure from physics called **Burstiness** (developed by researchers Goh and Barabási). It produces a score between −1 and +1:
  - **−1** = perfectly regular, like a commuter who visits the same café every morning
  - **0** = random, like someone rolling dice to decide where to go
  - **+1** = maximally bursty, like a tourist who visits several times during a vacation and never returns
- **Why:** This helps us separate genuine regulars from one-time visitors. A café packed with regulars (low burstiness) is likely a genuinely good daily spot.

### Exploration Diversity (Entropy)

- **What:** How many *different* cafés does a user spread their visits across?
- **Why:** A person who visits 50 different cafés is an explorer—their visits to any single café are less meaningful as evidence of quality. A person who visits only 3 cafés but keeps coming back to one of them is a loyalist—their preference is strong evidence.
- **The measure:** Shannon Entropy, borrowed from information theory. High entropy = high diversity (explorer). Low entropy = concentrated loyalty.

### Venue Stability

- **What:** How consistent is a café's traffic week-to-week?
- **Why:** A café with steady traffic all year is more reliably good than one that spikes during holiday season and goes quiet the rest of the year.
- **The measure:** Coefficient of Variation (standard deviation divided by the mean of weekly visits). Lower = more stable.

### Loyalty Concentration (Gini Coefficient)

- **What:** Is a café's traffic spread across many people, or does it depend on a few "super-fans"?
- **Why:** Broad loyalty (many people returning) is healthier than narrow dependence on a handful of people.
- **The measure:** The Gini coefficient, commonly used in economics to measure inequality. Low Gini = many people contribute equally. High Gini = a few people dominate.

---

## Step 4: Building a Network (Bipartite Graph)

**What we did:** We organized all the data into a network with two types of nodes: **Users** and **Venues**. Lines (called *edges*) connect each user to each café they visited, with thicker lines for more visits.

**Why it matters:** This structure captures relationships. Instead of looking at each café in isolation, we can now see the web of connections—who goes where, and how those patterns overlap.

**Key design choice:** We kept the network *bipartite*—meaning users only connect to venues, never to other users or venues to other venues. This preserves the raw data without introducing artificial connections or losing information.

---

## Step 5: The BiRank Algorithm — Mutual Reinforcement

**What we did:** We applied an algorithm called **BiRank** to our network. BiRank works on a simple but powerful idea:

> *A café is important if it's visited by important people.
> A person is important if they visit important cafés.*

This sounds circular, but the algorithm resolves it through *iteration*—it starts with rough guesses and refines them repeatedly until the scores stabilize (usually within 20–30 rounds).

**The innovation — Behavioral Priors:** Instead of starting BiRank with equal guesses for everyone (the default), we *injected* the behavioral features from Step 3 as starting points:

- **Users** who are steady, high-frequency visitors (low burstiness, many visits) start with higher importance
- **Venues** with high repeat-user rates and stable traffic start with higher importance

This anchors the mathematical algorithm in real-world behavioral evidence, preventing it from being swayed by noise.

**Knowledge used:** BiRank was originally developed for ranking in bipartite networks (like ranking web pages and the people who link to them). We adapted it for the user-venue context. The concept of "priors" comes from Bayesian statistics—the idea that you should start with your best guess based on what you already know, rather than starting from total ignorance.

---

## Step 6: Proving It Works — Prediction Test

Any ranking system needs to prove it's not just making things up. We tested ours with a simple, powerful idea: **can past behavior predict the future?**

### How the test works

1. We split the data in time: everything **before 2020** is the "training" set, and everything **from 2020 onward** is the "test" set.
2. We built our rankings using *only* the training data (the past).
3. We then checked: did the cafés we ranked highly actually get visited by those users in the future?

### What we measured

- **NDCG@k** (Normalized Discounted Cumulative Gain): A standard measure from information retrieval (the science behind search engines). It asks: "If I show you the top k recommendations, how many of them match what you actually ended up visiting?" It gives extra credit for getting the top spots right.
- **Hit Rate @k**: Simply—did at least one of our top-k suggestions match a place the user actually went?
- **Spearman Correlation**: Do the venues we rank highly also tend to be the ones that get more actual future visits?

### What we compared against

| Method | Description |
|--------|-------------|
| **BiRank (Behavioral)** | Our method—network ranking with behavioral priors |
| **Rating (Stars)** | Traditional star-rating average |
| **Popularity (Visits)** | Simply counting total visits |
| **Random** | Random guess (the "monkey with a dartboard" baseline) |

If our behavioral model can't beat a random guess, it has no value. If it also beats ratings and popularity, it demonstrates that behavior captures something that those simpler measures miss.

---

## Step 7: Making It Visible — The Dashboard

We built an interactive dashboard (using Streamlit and Folium maps) so that anyone can explore the results. The dashboard lets you:

- **Search for a city** — with fuzzy matching, so "philly" finds Philadelphia
- **Draw a search radius** on the map around any café
- **Compare ranking methods** side by side (behavioral vs. ratings vs. popularity)
- **See a behavioral profile** for each venue — a plain-language tag like "Steady / High Retention / Broad Loyalty" that explains *why* a venue ranks where it does
- **View validation results** — the prediction test scores, so you can verify the model's accuracy yourself

---

## Summary: What Makes This Different

| Traditional Ranking | Our Behavioral Ranking |
|---------------------|------------------------|
| Based on what people *say* (star ratings) | Based on what people *do* (visit patterns) |
| A single number (1–5 stars) | Multiple behavioral dimensions (loyalty, regularity, diversity) |
| Easily gamed by fake reviews | Hard to fake—requires real, repeated visits |
| Treats all users equally | Distinguishes regulars from one-time visitors |
| No way to verify it predicts anything | Validated by predicting actual future visits |

---

## The Knowledge Behind It

The methods we used draw from several established fields:

- **Network Science** (BiRank algorithm) — ranking nodes in networks, pioneered by researchers studying web link structures
- **Information Theory** (Shannon Entropy) — measuring diversity and uncertainty, developed by Claude Shannon in 1948
- **Econometrics** (Gini Coefficient) — measuring concentration and inequality, widely used in economics since the early 1900s
- **Physics** (Burstiness Index) — characterizing temporal patterns, developed by Goh & Barabási in their study of human dynamics
- **Bayesian Statistics** (Prior Injection) — the idea of starting with informed assumptions rather than blind guesses
- **Information Retrieval** (NDCG) — the standard evaluation metric used by companies like Google and Netflix to test recommendation quality

None of these ideas are new individually. Our contribution is **combining** them into a single pipeline that transforms raw visit data into validated, interpretable venue rankings.

---

## How to Read the Results

When you look at a venue's profile on the dashboard, here's what the key tags mean:

| Tag | What It Means |
|-----|---------------|
| **Steady** | Traffic is consistent week to week — not a flash-in-the-pan |
| **High Retention** | A large portion of visitors come back at least twice |
| **Broad Loyalty** | Many different people contribute to the traffic, not just a few super-fans |
| **Year-Round** | Equally popular across all seasons — not just a summer or holiday spot |
| **Niche Following** | A small but devoted group drives most of the traffic |
| **Variable Traffic** | Busy some weeks, quiet others — could be location or event dependent |

These tags are generated automatically from the numbers, so they're always grounded in real data—not opinion.
