import json
import os
import pickle
from sentence_transformers import SentenceTransformer

print("🔄 Loading Local AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_custom_signals_score(signals):
    """
    STRICT FRACTIONAL NORMALIZATION ENGINE
    Guarantees a clean, high-variance decimal spread strictly bounded between 0 and 100.
    """
    if not signals or not isinstance(signals, dict):
        return 50.0
    
    # 1. GitHub Vitality Component (Max 25 Pts)
    # Scaled natively out of 10.0
    gh_raw = max(0.0, min(10.0, float(signals.get("github_activity_score", 5.0))))
    gh_component = (gh_raw / 10.0) * 25.0
    
    # 2. Interview Reliability Component (Max 35 Pts)
    # Already a percentage decimal (0.0 to 1.0)
    int_raw = max(0.0, min(1.0, float(signals.get("interview_completion_rate", 0.5))))
    int_component = int_raw * 35.0
    
    # 3. Final Conversion Intent Component (Max 25 Pts)
    # Already a percentage decimal (0.0 to 1.0)
    offer_raw = max(0.0, min(1.0, float(signals.get("offer_acceptance_rate", 0.5))))
    offer_component = offer_raw * 25.0
    
    # 4. Behavioral Responsiveness Component (Max 15 Pts)
    resp_hours = float(signals.get("avg_response_time_hours", 72.0))
    if resp_hours <= 12.0:
        speed_component = 15.0   # Fast responder elite tier
    elif resp_hours <= 24.0:
        speed_component = 12.0
    elif resp_hours <= 72.0:
        speed_component = 8.0
    else:
        speed_component = 3.0    # Slow response penalty
        
    # Strictly bound maximum sum: 25 + 35 + 25 + 15 = 100.0
    total_score = gh_component + int_component + offer_component + speed_component
    
    # Shave off a subtle fraction based on completeness to force granular variance
    completeness = max(0, min(100, int(signals.get("profile_completeness_score", 100))))
    variance_tweak = (completeness / 100.0) * 0.98
    
    return round(total_score * variance_tweak, 1)

def run_precomputation():
    candidates_path = "data/candidates.jsonl"
    brain_out_path = "data/candidate_embeddings.pkl"
    
    candidate_records = []
    text_corpus_batch = []
    tech_keywords = ["engineer", "developer", "data", "backend", "programmer", "software"]
    
    print("📥 Ingesting and filtering candidate pool...")
    with open(candidates_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            cleaned_line = line.strip().strip("'").strip('"').replace('\\n', '').strip()
            if not cleaned_line:
                continue
            try:
                candidate = json.loads(cleaned_line)
                profile = candidate.get("profile", {})
                current_title = str(profile.get("current_title", "")).lower()
                signals = candidate.get("redrob_signals", {})
                
                if not any(kw in current_title for kw in tech_keywords):
                    continue
                if not signals.get("open_to_work_flag", False) or signals.get("profile_completeness_score", 0) < 75:
                    continue
                
                skills_list = [s.get("name", "") if isinstance(s, dict) else str(s) for s in candidate.get("skills", [])]
                candidate_corpus = f"Headline: {profile.get('headline', '')}. Summary: {profile.get('summary', '')}. Tech Stack: {', '.join(skills_list)}"
                
                candidate_records.append({
                    "Candidate ID": candidate.get("candidate_id", "UNKNOWN"),
                    "Name": profile.get("anonymized_name", "Anonymous Candidate"),
                    "Current Title": profile.get("current_title", "N/A"),
                    "precomputed_signal_rating": calculate_custom_signals_score(signals)
                })
                text_corpus_batch.append(candidate_corpus)
            except Exception:
                continue

    print(f"🧠 Encoding {len(text_corpus_batch)} elite profiles...")
    embeddings = model.encode(text_corpus_batch, convert_to_tensor=True, show_progress_bar=True)
    
    with open(brain_out_path, 'wb') as pkl_file:
        pickle.dump({"records": candidate_records, "embeddings": embeddings}, pkl_file)
    print("✨ BRAIN FILE UPDATED SUCCESSFULLY WITH PERFECT BOUNDS!")

if __name__ == "__main__":
    run_precomputation()