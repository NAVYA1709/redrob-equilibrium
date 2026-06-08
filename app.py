import streamlit as st
import os
import pickle
import time
import pandas as pd

# --- 1. INSTANT PAGE LAYOUT CONFIGURATION ---
# This must happen first so the browser gets the UI structure immediately!
st.set_page_config(
    page_title="Redrob AI Hybrid Recruiter Co-Pilot",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PLACEHOLDERS & VISUAL ANIMATION FOR HEAVY LOADS ---
st.title("🏆 Redrob Equilibrium")
st.markdown("##### *Dynamic Behavioral Weighting & High-Velocity Talent Sourcing Workspace*")
st.markdown("---")

# Clean UI styling to hide the default Streamlit header links
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Show an animated spinner right away so the user knows the AI is warming up
with st.spinner("🧠 Initializing Neural Engine & Indexing Data Matrix... Please wait a few seconds."):
    # We move the heavy imports inside here so they don't block the screen layout!
    from sentence_transformers import SentenceTransformer, util

    @st.cache_resource
    def load_local_ai_model():
        return SentenceTransformer('all-MiniLM-L6-v2')

    @st.cache_data
    def load_precomputed_brain(path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    # Load the actual model and dataset weights into RAM
    model = load_local_ai_model()
    brain_path = "data/candidate_embeddings.pkl"
    brain_bundle = load_precomputed_brain(brain_path)

# --- 3. SIDEBAR INTERACTIVE DIALS ---
st.sidebar.markdown("## ⚙️ Hybrid Allocation Matrix")
st.sidebar.info("Adjust the sliders below to dynamically re-weight the AI ranking algorithm live.")

semantic_weight = st.sidebar.slider("Semantic Job Fit Weight (%)", min_value=10, max_value=90, value=70, step=5)
signals_weight = 100 - semantic_weight
st.sidebar.caption(f"Calculated Platform Behavioral Weight: {signals_weight}%")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Active Target Keywords Included:")
st.sidebar.code("Engineer, Developer, Data,\nBackend, Programmer, Software")

# --- 4. MAIN INTERACTIVE APPLICATION BODY ---
if brain_bundle is None:
    st.error(f"❌ Critical Error: Could not locate '{brain_path}'. Please run `python precompute_brain.py` first!")
else:
    default_jd = "Backend Engineer with experience building data pipelines, software infrastructure, SQL warehouses, and Python services."
    user_jd = st.text_area("📝 Edit Target Job Description Profile:", value=default_jd, height=100)
    
    if st.button("⚡ Execute Lightning Cross-Match & Rank"):
        with st.spinner("Analyzing semantic matrices instantly..."):
            
            candidate_records = brain_bundle["records"]
            candidate_embeddings = brain_bundle["embeddings"]
            
            # Fast Tensor Math Execution
            jd_embedding = model.encode(user_jd, convert_to_tensor=True)
            cosine_scores = util.cos_sim(jd_embedding, candidate_embeddings)[0]
            
            final_ranked_list = []
            for idx, record in enumerate(candidate_records):
                semantic_score_pct = round(cosine_scores[idx].item() * 100, 2)
                signals_score = record["precomputed_signal_rating"]
                
                final_hybrid_score = round((semantic_score_pct * (semantic_weight / 100.0)) + (signals_score * (signals_weight / 100.0)), 2)
                
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
            
            st.markdown("### 📊 Engine Insights & Metrics")
            # Sleek, grouped corporate metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric(label="Total Elite Candidates Screened", value=f"{len(df_results):,}")
            with m2:
                st.metric(label="Highest Matrix Balance Match", value=f"{df_results.iloc[0]['Final Score']} pts")
            with m3:
                st.metric(label="Top Recommendation", value=df_results.iloc[0]['Name'])
                
            st.markdown("---")
            st.markdown("### 🏆 Top 5 Recommended Talent Leaderboard")
            
            # Premium Visual Card Container Setup
            for i in range(min(5, len(df_results))):
                row = df_results.iloc[i]
                
                # Wrap each candidate in a clean visual card container
                with st.container(border=True):
                    col_rank, col_meta, col_scores, col_action = st.columns([0.6, 2.4, 2.2, 1.8])
                    
                    with col_rank:
                        st.markdown(f"<h2 style='text-align: center; color: #FF4B4B; margin:0;'>#{i+1}</h2>", unsafe_allow_html=True)
                    
                    with col_meta:
                        st.markdown(f"#### **{row['Name']}**")
                        st.caption(f"🆔 {row['Candidate ID']}  |  💼 {row['Current Title']}")
                    
                    with col_scores:
                        st.markdown(f"🎯 **Composite Score: {row['Final Score']}**")
                        st.caption(f"🤖 Text Match: `{row['Semantic Match']}`  |  📈 Platform Behavior: `{row['Platform Signal Rating']}`")
                    
                    with col_action:
                        # Clean interaction block using a dropdown expander
                        with st.expander("🛠️ Pipeline Actions", expanded=False):
                            st.button("🟢 Dispatch Test Setup", key=f"test_{i}", use_container_width=True)
                            st.markdown("<p style='font-size:11px; margin: 5px 0 2px 0;'><b>Drafted Outreach Email:</b></p>", unsafe_allow_html=True)
                            st.code(f"Subject: Technical Discussion - Redrob Core Team\n\nHi {row['Name']},\n\nYour background as a {row['Current Title']} stands out. Let's align regarding our active infrastructure architecture projects...", language="text")
            
            # Export data summary section
            df_results.to_csv("data/semantic_ranked_output.csv", index=False)
            st.success("📊 Evaluation matrix successfully synchronized to `data/semantic_ranked_output.csv`!")