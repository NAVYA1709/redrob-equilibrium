import streamlit as st
import os
import pickle
import time
import pandas as pd
import altair as alt
import numpy as np

# --- 1. INSTANT PAGE LAYOUT CONFIGURATION ---
st.set_page_config(
    page_title="Redrob Equilibrium AI Co-Pilot",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PREMIUM UI STYLING & FONTS ---
# CSS overrides to create a high-fidelity glassmorphic dark-mode dashboard
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sleek Title Banner */
    .title-banner {
        padding: 2rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(168, 85, 247, 0.1) 50%, rgba(244, 63, 94, 0.05) 100%);
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.25);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(4px);
    }
    .title-banner h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #818CF8, #C084FC, #F472B6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .title-banner p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    /* Modern Dashboard Metric Card */
    .dashboard-metric {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
        transition: transform 0.2s, border-color 0.2s;
    }
    .dashboard-metric:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.3);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #818CF8;
        margin-bottom: 0.2rem;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    /* Styled custom chips for skills */
    .skill-chip {
        background: rgba(99, 102, 241, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    
    /* Score and Rank Badges */
    .score-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 0.4rem 0.9rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.25);
    }
    .rank-chip {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.1rem;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(99, 102, 241, 0.2);
    }
    
    /* Visual Container adjustments */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        margin-top: 5px;
    }
    
    /* Form inputs custom styling */
    textarea {
        background-color: #1e293b !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Custom header banner
st.markdown("""
    <div class="title-banner">
        <h1>🏆 Redrob Equilibrium</h1>
        <p>Dynamic Behavioral Weighting & High-Velocity AI Talent Sourcing Workspace</p>
    </div>
    """, unsafe_allow_html=True)

# --- 3. HEAVY MODEL & DATA LOADING ---
with st.spinner("🧠 Initializing Neural Engine & Indexing Data Matrix... Please wait a few seconds."):
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

    model = load_local_ai_model()
    brain_path = "data/candidate_embeddings.pkl"
    brain_bundle = load_precomputed_brain(brain_path)

# --- 4. SIDEBAR ALGORITHM MATRIX & FILTERS ---
st.sidebar.markdown("## ⚙️ Allocation & Filters")

st.sidebar.info("Fine-tune the algorithm weights and target filters to tailor matching criteria.")

st.sidebar.markdown("### ⚖️ Algorithm Weighting")
semantic_weight = st.sidebar.slider("Semantic Job Fit Weight (%)", min_value=10, max_value=90, value=70, step=5)
signals_weight = 100 - semantic_weight
st.sidebar.caption(f"Calculated Behavioral weight: **{signals_weight}%**")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Talent Filter Criteria")
min_experience = st.sidebar.slider("Minimum Experience (Years)", min_value=0, max_value=15, value=0, step=1)

# Populate a unique set of skills if brain is loaded
all_possible_skills = []
if brain_bundle is not None:
    # Gather all unique skills
    skills_set = set()
    for record in brain_bundle["records"]:
        if "Skills" in record:
            for skill in record["Skills"]:
                if skill.strip():
                    skills_set.add(skill.strip())
    all_possible_skills = sorted(list(skills_set))

selected_skills = st.sidebar.multiselect("Filter by Target Skills", options=all_possible_skills)
search_query = st.sidebar.text_input("🔍 Search Name or Title")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Active Keywords in Corpus")
st.sidebar.code("Engineer, Developer, Data,\nBackend, Programmer, Software")

# --- 5. APPLICATION LOGIC ---
if brain_bundle is None:
    st.error(f"❌ Critical Error: Could not locate '{brain_path}'. Please run `python precompute_brain.py` first!")
else:
    candidate_records = brain_bundle["records"]
    candidate_embeddings = brain_bundle["embeddings"]

    # Interactive tabs
    tab_matcher, tab_analytics, tab_sandbox = st.tabs([
        "🎯 Candidate Matcher", 
        "📊 Analytics Dashboard", 
        "⚙️ Weighting Sandbox"
    ])

    default_jd = "Backend Engineer with experience building data pipelines, software infrastructure, SQL warehouses, and Python services."

    # Use session state to persist results across tab switching
    if "df_ranked" not in st.session_state or st.session_state.df_ranked is None:
        # Precompute initial matching with default JD so the page doesn't look empty!
        jd_embedding = model.encode(default_jd, convert_to_tensor=True)
        cosine_scores = util.cos_sim(jd_embedding, candidate_embeddings)[0]
        
        initial_list = []
        for idx, record in enumerate(candidate_records):
            semantic_score_pct = round(cosine_scores[idx].item() * 100, 2)
            signals_score = record["precomputed_signal_rating"]
            final_hybrid_score = round((semantic_score_pct * (semantic_weight / 100.0)) + (signals_score * (signals_weight / 100.0)), 2)
            
            # Merge dictionary data
            item = record.copy()
            item["Semantic Score"] = semantic_score_pct
            item["Final Score"] = final_hybrid_score
            initial_list.append(item)
            
        st.session_state.df_ranked = pd.DataFrame(initial_list)
        st.session_state.current_jd = default_jd

    with tab_matcher:
        st.markdown("### 📝 Define Search Parameters")
        user_jd = st.text_area("Edit Target Job Description Profile:", value=st.session_state.current_jd, height=100)
        
        col_btn, col_opts = st.columns([1, 3])
        with col_btn:
            execute_match = st.button("⚡ Execute Match & Rank", use_container_width=True)
        with col_opts:
            show_count = st.selectbox("Show Top Recommendations", options=[5, 10, 15, 20], index=0)

        # Trigger re-ranking if button is clicked or if weights change dynamically
        if execute_match or st.session_state.df_ranked is None:
            with st.spinner("Analyzing semantic matrices instantly..."):
                st.session_state.current_jd = user_jd
                jd_embedding = model.encode(user_jd, convert_to_tensor=True)
                cosine_scores = util.cos_sim(jd_embedding, candidate_embeddings)[0]
                
                ranked_list = []
                for idx, record in enumerate(candidate_records):
                    semantic_score_pct = round(cosine_scores[idx].item() * 100, 2)
                    signals_score = record["precomputed_signal_rating"]
                    final_hybrid_score = round((semantic_score_pct * (semantic_weight / 100.0)) + (signals_score * (signals_weight / 100.0)), 2)
                    
                    item = record.copy()
                    item["Semantic Score"] = semantic_score_pct
                    item["Final Score"] = final_hybrid_score
                    ranked_list.append(item)
                    
                st.session_state.df_ranked = pd.DataFrame(ranked_list)

        # Apply sidebar filters to cached results
        df_filtered = st.session_state.df_ranked.copy()
        
        # 1. Experience Filter
        df_filtered = df_filtered[df_filtered["Experience"] >= min_experience]
        
        # 2. Skills Filter
        if selected_skills:
            df_filtered = df_filtered[df_filtered["Skills"].apply(lambda skills: any(s in skills for s in selected_skills))]
            
        # 3. Text Search Filter (Name or Title)
        if search_query:
            q = search_query.lower()
            df_filtered = df_filtered[
                df_filtered["Name"].str.lower().str.contains(q) | 
                df_filtered["Current Title"].str.lower().str.contains(q)
            ]

        # Final Sort
        df_filtered = df_filtered.sort_values(by="Final Score", ascending=False).reset_index(drop=True)

        st.markdown("---")
        
        # Dashboard Overview Row
        if not df_filtered.empty:
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                    <div class="dashboard-metric">
                        <div class="metric-val">{len(df_filtered):,}</div>
                        <div class="metric-lbl">Total Matching Candidates</div>
                    </div>
                    """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                    <div class="dashboard-metric">
                        <div class="metric-val">{df_filtered.iloc[0]['Final Score']} pts</div>
                        <div class="metric-lbl">Highest Equilibrium Score</div>
                    </div>
                    """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                    <div class="dashboard-metric">
                        <div class="metric-val">{df_filtered.iloc[0]['Name']}</div>
                        <div class="metric-lbl">Top Talent Recommendation</div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No candidates matched the sidebar filters. Try relaxing experience or skills filters!")

        st.markdown("### 🏆 Talent Leaderboard")
        
        # Display matching candidates
        display_limit = min(show_count, len(df_filtered))
        for i in range(display_limit):
            row = df_filtered.iloc[i]
            
            # Glassmorphic Card Outer Border Container
            with st.container(border=True):
                col_rank, col_info, col_metrics, col_badge = st.columns([0.8, 4.5, 2.5, 1.2])
                
                with col_rank:
                    st.markdown(f"""
                        <div style="text-align: center; margin-top: 10px;">
                            <span class="rank-chip">#{i+1}</span>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_info:
                    st.markdown(f"""
                        <h4 style="margin: 0; color: #F8FAFC;">{row['Name']}</h4>
                        <div style="font-size: 0.9rem; color: #818CF8; font-weight: 600; margin-top: 2px; margin-bottom: 6px;">💼 {row['Current Title']}</div>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 8px;">📍 {row['Location']} &nbsp;|&nbsp; 📈 <b>{row['Experience']}</b> YOE</div>
                    """, unsafe_allow_html=True)
                    
                    # Custom skills badges
                    skills_list = row.get("Skills", [])
                    skills_html = "".join([f'<span class="skill-chip">{s}</span>' for s in skills_list])
                    st.markdown(f"<div>{skills_html}</div>", unsafe_allow_html=True)
                
                with col_metrics:
                    st.markdown(f"""
                        <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 5px;">
                            <div>🤖 Text Similarity: <b style="color:#F8FAFC;">{row['Semantic Score']}%</b></div>
                            <div style="margin-top: 4px;">📈 Platform Activity: <b style="color:#F8FAFC;">{row['precomputed_signal_rating']}/100</b></div>
                            <div style="margin-top: 4px;">💻 Profile Score: <b style="color:#F8FAFC;">{row['Profile Completeness']}%</b></div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col_badge:
                    st.markdown(f"""
                        <div style="text-align: center; margin-top: 12px;">
                            <span class="score-badge">{row['Final Score']}</span>
                            <div style="font-size: 0.75rem; color: #94A3B8; margin-top: 6px;">Composite Score</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Expandable details section
                with st.expander("🔍 Candidate Profile Summary & Automated Pipelines"):
                    st.markdown(f"**Profile Headline:** *{row['Headline']}*")
                    st.markdown(f"**Professional Summary:** {row['Summary']}")
                    
                    st.markdown("<div style='margin: 1rem 0; border-top: 1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)
                    st.markdown("🎯 **Behavioral Platform Signals Details**")
                    
                    # Display subscore indicators
                    sb1, sb2, sb3, sb4 = st.columns(4)
                    with sb1:
                        st.metric(label="GitHub Activity Score", value=f"{row['Github Activity']}/100")
                    with sb2:
                        st.metric(label="Interview Completion", value=f"{int(row['Interview Completion'] * 100)}%")
                    with sb3:
                        st.metric(label="Offer Acceptance Rate", value=f"{int(row['Offer Acceptance'] * 100)}%")
                    with sb4:
                        # Quick responsiveness label
                        hours = row['Response Time']
                        tier = "Fast (Elite)" if hours <= 24 else ("Standard" if hours <= 72 else "Slow")
                        st.metric(label="Avg Response Time", value=f"{hours} hrs", delta=tier, delta_color="inverse" if tier == "Slow" else "normal")
                    
                    st.markdown("<div style='margin: 1rem 0; border-top: 1px solid rgba(255,255,255,0.05);'></div>", unsafe_allow_html=True)
                    
                    # Actions
                    ab1, ab2 = st.columns([1.2, 2.5])
                    with ab1:
                        st.button("🟢 Dispatch Test Setup", key=f"test_btn_{row['Candidate ID']}_{i}", use_container_width=True)
                        st.button("✉️ Schedule Virtual Interview", key=f"invite_btn_{row['Candidate ID']}_{i}", use_container_width=True)
                    with ab2:
                        st.markdown("<span style='font-size: 11px; font-weight:700; color:#94A3B8;'>AI-DRAFTED OUTREACH EMAIL</span>", unsafe_allow_html=True)
                        email_body = f"Subject: Opportunity with Redrob Engineering Team\n\nHi {row['Name']},\n\nI was reviewing top talent metrics on Redrob and your profile stood out. With {row['Experience']} YOE and skills in {', '.join(skills_list[:3])}, you look like a fantastic fit for our active projects.\n\nLet me know your availability for a call.\n\nBest,\nRecruiting Team"
                        st.code(email_body, language="text")

        # Save ranked output to CSV
        if not df_filtered.empty:
            df_filtered.to_csv("data/semantic_ranked_output.csv", index=False)

    with tab_analytics:
        st.markdown("### 📊 Sourcing Analytics & Distribution Insights")
        
        if df_filtered.empty:
            st.info("No data available. Please relax filters in the sidebar to populate candidates.")
        else:
            # 1. Row of KPI metrics
            ak1, ak2, ak3 = st.columns(3)
            with ak1:
                st.metric("Avg Composite Score", f"{round(df_filtered['Final Score'].mean(), 1)} pts")
            with ak2:
                st.metric("Avg Years of Experience", f"{round(df_filtered['Experience'].mean(), 1)} Yrs")
            with ak3:
                # Count distinct titles
                st.metric("Unique Job Roles", f"{df_filtered['Current Title'].nunique()}")
            
            st.markdown("---")
            
            # Prepare plotting dataframe
            df_plot = df_filtered.copy()
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Scatter plot: Semantic Match vs Precomputed Signal
                scatter_chart = alt.Chart(df_plot).mark_circle(size=70).encode(
                    x=alt.X('Semantic Score:Q', title='Semantic Job Description Fit (%)', scale=alt.Scale(zero=False)),
                    y=alt.Y('precomputed_signal_rating:Q', title='Platform Behavior Score (0-100)'),
                    color=alt.Color('Final Score:Q', scale=alt.Scale(scheme='purples'), title='Composite Score'),
                    tooltip=['Name', 'Current Title', 'Experience', 'Final Score']
                ).properties(
                    title="Semantic Matching vs. Platform Behavior Index",
                    height=350
                ).interactive()
                st.altair_chart(scatter_chart, use_container_width=True)
                
            with col_chart2:
                # Bar Chart: Years of Experience Distribution
                experience_dist = alt.Chart(df_plot).mark_bar(color='#a855f7').encode(
                    x=alt.X('Experience:Q', bin=alt.Bin(maxbins=15), title='Years of Experience'),
                    y=alt.Y('count():Q', title='Number of Candidates')
                ).properties(
                    title="Talent Experience Distribution Profile",
                    height=350
                )
                st.altair_chart(experience_dist, use_container_width=True)

            col_chart3, col_chart4 = st.columns(2)
            
            with col_chart3:
                # Horizontal Bar Chart: Country/Geographic Distribution
                df_plot['Country'] = df_plot['Location'].apply(lambda loc: loc.split(',')[-1].strip() if ',' in str(loc) else str(loc))
                country_counts = df_plot['Country'].value_counts().reset_index()
                country_counts.columns = ['Country', 'Count']
                
                geo_chart = alt.Chart(country_counts.head(10)).mark_bar(color='#EC4899').encode(
                    x=alt.X('Count:Q', title='Number of Candidates'),
                    y=alt.Y('Country:N', sort='-x', title='Location / Country')
                ).properties(
                    title="Talent Geographic Distribution (Top 10 Locations)",
                    height=350
                )
                st.altair_chart(geo_chart, use_container_width=True)
                
            with col_chart4:
                # Box plot of scores by role
                top_roles = df_plot['Current Title'].value_counts().head(8).index.tolist()
                df_roles = df_plot[df_plot['Current Title'].isin(top_roles)]
                
                box_chart = alt.Chart(df_roles).mark_boxplot().encode(
                    x=alt.X('Current Title:N', sort='-y', title='Job Title', axis=alt.Axis(labelAngle=-45)),
                    y=alt.Y('Final Score:Q', title='Composite Score Distribution')
                ).properties(
                    title="Composite Scores Across Top Job Roles",
                    height=350
                )
                st.altair_chart(box_chart, use_container_width=True)

    with tab_sandbox:
        st.markdown("### ⚙️ Algorithm Formula Sandbox")
        
        st.markdown("""
        The **Redrob Equilibrium Co-Pilot** utilizes a hybrid heuristic engine to balance **Semantic Job Description Matching** and **Real-World Platform Behavioral Signals**.
        
        #### 📐 Mathematical Formulation
        The composite evaluation score is calculated using the following formula:
        """)
        
        # Display LaTeX formula
        st.latex(r"""
        \text{Composite Score} = \left(\text{Semantic Match \%} \times \frac{W_{\text{semantic}}}{100}\right) + \left(\text{Platform Signals} \times \frac{100 - W_{\text{semantic}}}{100}\right)
        """)
        
        st.markdown("""
        * Where $W_{\text{semantic}}$ is the slider weight (e.g. $70\%$).
        * **Semantic Match** is computed via cosine similarity between the Job Description embedding and Candidate profile embeddings generated by the `all-MiniLM-L6-v2` Sentence Transformer.
        * **Platform Signals** rating is computed using a strict normalized calculation of the candidate's active profile completeness, interview completions, and responses:
        """)
        
        # Show how sub-components work
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("""
            ##### 💻 Platform Signals Contribution (Max 100 Points):
            1. **GitHub Vitality Score** (Max 25 pts) - Scaled linearly from active open-source contribution metrics.
            2. **Interview Reliability** (Max 35 pts) - Completion rate of scheduled evaluation assessments.
            3. **Conversion Intent** (Max 25 pts) - Historical offer acceptance rates.
            4. **Behavioral Responsiveness** (Max 15 pts) - Penalties/tier scores for average messaging response hours.
            """)
        with sc2:
            st.success("""
            💡 **Recruitment Velocity Optimization**:
            Traditional search models rank talent based solely on keywords. By balancing qualifications with active behavioral metrics, Redrob Equilibrium helps teams prioritize responsive candidates and reduce interview scheduling times.
            """)
            
        st.markdown("---")
        st.markdown("### 🎛️ Simulation: Extreme Weighting Impacts")
        
        # Display side-by-side simulation columns
        s_col1, s_col2 = st.columns(2)
        
        # Extract current semantic scores and signal ratings from session state
        df_sim = st.session_state.df_ranked[["Name", "Current Title", "Semantic Score", "precomputed_signal_rating"]].copy()
        df_sim = df_sim.rename(columns={"Semantic Score": "Semantic", "precomputed_signal_rating": "Signals"})
        
        with s_col1:
            st.subheader("🎯 Scenario A: Pure Skill Match (90% Semantic)")
            df_a = df_sim.copy()
            df_a["Composite"] = round((df_a["Semantic"] * 0.9) + (df_a["Signals"] * 0.1), 2)
            df_a = df_a.sort_values(by="Composite", ascending=False).head(3)
            
            for rank_idx, (idx, r) in enumerate(df_a.iterrows(), 1):
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.5); padding: 10px; border-radius: 8px; border-left: 4px solid #6366f1; margin-bottom:8px;">
                    <b>#{rank_idx} {r['Name']}</b><br/>
                    <span style="font-size:12px; color:#94a3b8;">💼 {r['Current Title']}</span><br/>
                    <span style="font-size:12px;">Score: <b>{r['Composite']}</b> | Match: {r['Semantic']}% | Behavioral: {r['Signals']}/100</span>
                </div>
                """, unsafe_allow_html=True)
                
        with s_col2:
            st.subheader("⚡ Scenario B: Pure Responsiveness (90% Behavior)")
            df_b = df_sim.copy()
            df_b["Composite"] = round((df_b["Semantic"] * 0.1) + (df_b["Signals"] * 0.9), 2)
            df_b = df_b.sort_values(by="Composite", ascending=False).head(3)
            
            for rank_idx, (idx, r) in enumerate(df_b.iterrows(), 1):
                st.markdown(f"""
                <div style="background: rgba(30,41,59,0.5); padding: 10px; border-radius: 8px; border-left: 4px solid #10b981; margin-bottom:8px;">
                    <b>#{rank_idx} {r['Name']}</b><br/>
                    <span style="font-size:12px; color:#94a3b8;">💼 {r['Current Title']}</span><br/>
                    <span style="font-size:12px;">Score: <b>{r['Composite']}</b> | Match: {r['Semantic']}% | Behavioral: {r['Signals']}/100</span>
                </div>
                """, unsafe_allow_html=True)