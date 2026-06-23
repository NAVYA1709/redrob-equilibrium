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
# Install core deep learning, pandas and tensor processing tools
pip install torch sentence-transformers numpy pandas python-docx
```

### 2. Execution Command for Submission Reproduction
To reproduce the top-100 candidates ranking submission file from the candidate pool:
```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```
This command runs in less than 5 minutes on CPU. It does the following:
* Parses the Job Description from `data/job_description.docx` (falling back to baseline if unavailable).
* Performs strict pre-filtering to remove honeypots (impossibilities like `last_active_date < signup_date`, skill durations > YOE, or expert skills with 0 duration), unqualified titles, and 100% consulting backgrounds.
* Computes dense embeddings using `SentenceTransformer('all-MiniLM-L6-v2')` for the filtered candidate corpora.
* Scores candidates using the hybrid mathematical consensus blend combined with heuristic target bonuses (YOE 5-9, target skills like vector databases/embeddings/evaluation/MLOps, Pune/Noida locations, and short notice periods).
* Breaks score ties deterministically using `candidate_id` ascending.
* Generates a unique, natural-language reasoning string for each of the top 100 candidates.
* Outputs the final 100 rows into the specified output path (columns: `candidate_id`, `rank`, `score`, `reasoning`).

### 3. Submission Validation
Verify that the output format adheres strictly to the hackathon specifications:
```bash
python data/validate_submission.py data/submission.csv
```
