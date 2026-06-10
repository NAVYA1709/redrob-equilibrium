# Redrob Equilibrium

### A Hybrid Dynamic Ranking Engine for Reliable Engineering Sourcing

Redrob Equilibrium is a deterministic candidate ranking workspace designed to eliminate recruiter ghosting. Instead of relying on fragile keyword matching or slow, non-deterministic generative LLMs, this architecture pairs deep-learning semantic profiling with real-time platform responsiveness signals to rank the best—and most active—backend and data engineering talent.

---

## 🚀 Key Features & Extraction Logic

Our core extraction pipeline parses unstructured Job Descriptions (JDs) and maps them directly to critical engineering pillars rather than shallow syntax:

* **Backend Architecture:** Evaluating server-side design, structural patterns, and codebase scalability.
* **Language Depth:** Assessing programmatic knowledge and execution mastery over surface-level keyword presence.
* **Data Layer Infrastructure:** Identifying SQL data warehouses, pipeline architectures, and storage integrity.

> **Example Pipeline Mapping:** > *"Build scalable server code in deep Python"* ➔ `Backend Architecture` | `Language Depth`

---

## 🧮 Mathematical Consensus Model

Unlike generative AI "black boxes," Redrob Equilibrium features **100% mathematical auditability**. Candidates are scored deterministically by blending text matching with real-world availability metrics using a dynamic recruiter-controlled weight slider ($W_{\text{semantic}}$).

$$\text{Final Score} = \left(\text{Semantic Fit} \times \frac{W_{\text{semantic}}}{100}\right) + \left(\text{Behavioral Signals} \times \frac{100 - W_{\text{semantic}}}{100}\right)$$

### Component Breakdown:
1. **Semantic Fit (Resume Match):** Dense vector mapping using the `all-MiniLM-L6-v2` transformer model to compute the exact **Cosine Similarity** between candidate profiles and the parsed JD requirements.
2. **Behavioral Signals (Platform Activity):** Real-time engagement analytics including response latency indexes, interview completion histories, and availability states.
3. **Weight Slider ($W_{\text{semantic}}$):** A variable percentage constraint that lets recruiters prioritize technical alignment versus immediate platform responsiveness dynamically.

---

## 🛠️ System Architecture

The core engine is built strictly on top of predictable matrix structures to eliminate hallucination risks completely:
* **Logic Layer:** Python processing pipeline driving fast geometric computation.
* **Model Pipeline:** Deep learning tensor operations powered by `PyTorch` and `SentenceTransformers`.
* **Persistence Layer:** Fast, file-based structured storage employing static vector collections (`.pkl`) and JSON Lines format (`.jsonl`).

---

## 💻 Local Setup & Execution Guide

Follow these steps to initialize the environment and run the ranking engine processing pipeline locally.

### 1. Prerequisites & Installation
Ensure you have Python 3.8+ installed, then clone the repository and install the backend mathematical packages:
```bash
# Install core deep learning and tensor processing tools
pip install torch sentence-transformers numpy
