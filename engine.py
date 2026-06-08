import os
import pickle
import pandas as pd
from docx import Document
from sentence_transformers import SentenceTransformer, util

print("🔄 Loading Local AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_txt_from_docx(docx_path):
    if not os.path.exists(docx_path) or os.path.getsize(docx_path) < 1024:
        return ""
    try:
        doc = Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception:
        return ""

def run_ranking_engine():
    jd_path = "data/job_description.docx"
    brain_path = "data/candidate_embeddings.pkl"
    
    print("📄 Parsing Job Description parameters...")
    job_description_text = extract_txt_from_docx(jd_path)
    if not job_description_text:
        job_description_text = "Backend Engineer with experience building data pipelines, software infrastructure, SQL warehouses, and Python services."
        print("⚠️ Note: Using standard fallback baseline.")
        
    if not os.path.exists(brain_path):
        print(f"❌ Error: Could not find '{brain_path}'. Run precompute_brain.py first!")
        return
        
    print("💾 Loading precomputed candidate matrix...")
    with open(brain_path, 'rb') as pkl_file:
        brain_bundle = pickle.load(pkl_file)
        
    candidate_records = brain_bundle["records"]
    candidate_embeddings = brain_bundle["embeddings"]
    
    print("⚡ Matching and Ranking Instantly...")
    jd_embedding = model.encode(job_description_text, convert_to_tensor=True)
    cosine_scores = util.cos_sim(jd_embedding, candidate_embeddings)[0]
    
    final_ranked_list = []
    for idx, record in enumerate(candidate_records):
        semantic_score_pct = round(cosine_scores[idx].item() * 100, 2)
        
        # Pull the precomputed strict score out of the record directly
        signals_score = record["precomputed_signal_rating"]
        
        final_hybrid_score = round((semantic_score_pct * 0.70) + (signals_score * 0.30), 2)
        
        final_ranked_list.append({
            "Candidate ID": record["Candidate ID"],
            "Name": record["Name"],
            "Current Title": record["Current Title"],
            "Semantic Match": f"{semantic_score_pct}%",
            "Platform Signal Rating": f"{signals_score}/100",
            "Final Score": final_hybrid_score
        })

    df_results = pd.DataFrame(final_ranked_list)
    df_results = df_results.sort_values(by="Final Score", ascending=False).reset_index(drop=True)
    
   # --- ✨ NATIVE DISPLAY FORMATTING FOR JUDGES ---
    # Force Pandas to calculate optimal spacing automatically
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.colheader_justify', 'left')
    
    print("\n🏆 ADVANCED HYBRID AI RANKING CO-PILOT (TOP 5 RECOMMENDATIONS):")
    print("=" * 110)
    print(df_results.head(5).to_string(index=False))
    print("=" * 110)
    
    df_results.to_csv("data/semantic_ranked_output.csv", index=False)
    print("\n✅ Final matrix saved cleanly to data/semantic_ranked_output.csv!")

if __name__ == "__main__":
    run_ranking_engine()
    