# Redrob Equilibrium 🏆

> **Dynamic Behavioral Weighting & High-Velocity Talent Sourcing Workspace** > Built for the official *Redrob AI "India Runs" Hackathon*.

Redrob Equilibrium is a production-grade talent sourcing platform designed to bridge the gap between deep semantic text alignment and real-world candidate reliability. Instead of matching text keywords statically, Equilibrium uses a dual-engine hybrid matrix computation system to rank talent dynamically.

---

## 🧠 Core Engineering Features

* **Sub-Second Semantic Matching Engine:** Leverages a `SentenceTransformer` text-embedding architecture to vectorize complex job descriptions instantly.
* **Hybrid Allocation Matrix:** Features an interactive live sliding matrix that calculates candidate scores based on user-defined balance ratios between **Semantic Job Fit** and **Platform Behavioral Signals**.
* **Tiered Computational Optimization:** Implements pre-filtering text rules to cleanse the data pipeline, cutting down calculation workloads from minutes to milliseconds.

## 🛠️ Tech Stack & Architecture

* **Frontend Dashboard:** Streamlit UI Components
* **Machine Learning Backend:** PyTorch, Sentence-Transformers (`all-MiniLM-L6-v2`)
* **Data Processing & Analytics:** Pandas DataFrames, Pickle Caches
