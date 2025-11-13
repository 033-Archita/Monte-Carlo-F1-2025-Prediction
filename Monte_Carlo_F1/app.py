import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import time
import ast # For parsing list strings from CSV

# --- 1. APP CONFIGURATION ---

st.set_page_config(
    page_title="F1 Championship Predictor (2025)",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS (F1 PRO UI) ---
def load_css():
    st.markdown("""
        <style>
        /* --- Base & Animations --- */
        .stApp {
            background-color: #1a1a1a; /* Dark background */
            color: #f0f0f0;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes glow {
            0% { text-shadow: 0 0 2px #e10600; }
            50% { text-shadow: 0 0 10px #e10600; }
            100% { text-shadow: 0 0 2px #e10600; }
        }
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(225, 6, 0, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(225, 6, 0, 0); }
            100% { box-shadow: 0 0 0 0 rgba(225, 6, 0, 0); }
        }
        h1 {
            animation: fadeIn 1s ease-in, glow 2s infinite alternate;
        }
        
        /* --- Buttons --- */
        .stButton > button {
            background-color: #e10600; 
            color: white; font-weight: bold;
            border-radius: 8px; border: none;
            padding: 10px 20px;
            font-size: 1.1rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #b00000;
            color: white; border: none;
            transform: scale(1.05);
        }
        /* Primary button pulse */
        .stButton > button[kind="primary"] {
            animation: pulse 2s infinite;
        }

        /* --- Headers & Metrics --- */
        h1, h2, h3 { color: #e10600; }
        .stMetric {
            background-color: #2a2a2a; border-radius: 8px; padding: 15px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.5);
            transition: transform 0.2s ease;
        }
        .stMetric:hover {
            transform: scale(1.03);
        }
        .stMetric > label { color: #aaa; }
        .stMetric > div[data-testid="stMetricValue"] {
            color: #f0f0f0 !important; 
            font-weight: bold;
            font-size: 1.8rem; 
        }
        
        /* --- Tabs --- */
        .stTabs [data-baseweb="tab-list"] { gap: 24px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #333; 
            color: #aaa; 
            padding: 10px 15px;
            transition: all 0.2s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-3px);
            background-color: #444;
        }
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            background-color: #444; 
            color: #f0f0f0; 
            font-weight: bold;
            border-bottom: 3px solid #e10600;
        }
        
        /* --- Other Elements --- */
        .stPlotlyChart {
            border-radius: 8px; overflow: hidden;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.7);
        }
        .stMarkdown { color: #d0d0d0; }
        </style>
    """, unsafe_allow_html=True)

# --- 3. DATA & CONFIG ---

# Static driver image URLs
driver_images = {
    'Lando Norris': 'images/LN.jpeg',
    'Oscar Piastri': 'images/OS.jpeg',
    'Max Verstappen': 'images/MV.jpeg',
    'George Russell': 'images/GR.jpeg',
    'Charles Leclerc': 'images/CL.jpeg',
    'Lewis Hamilton': 'images/LH.jpeg',
    'Kimi Antonelli': 'images/KA.jpeg',
    'Alex Albon': 'images/AA.jpeg',
    'Nico Hulkenberg': 'images/NH.jpeg',
    'Isack Hadjar': 'images/IH.jpeg',
    'Oliver Bearman': 'images/OB.jpeg',
    'Fernando Alonso': 'images/FA.jpeg',
    'Carlos Sainz': 'images/CS.jpeg',
    'Liam Lawson': 'images/LL.jpeg',
    'Lance Stroll': 'images/LS.jpeg',
    'Esteban Ocon': 'images/EO.jpeg',
    'Yuki Tsunoda': 'images/YT.jpeg',
    'Pierre Gasly': 'images/PG.jpeg',
    'Gabriel Bortoleto': 'images/GB.jpeg',
    'Franco Colapinto': 'images/FC.jpeg',
}

@st.cache_data
def load_all_data():
    """Loads performance and history data."""
    # 1. Load performance data
    try:
        df_perf = pd.read_csv('driver_performance.csv')
    except FileNotFoundError:
        st.error("`driver_performance.csv` not found. App cannot load performance data.")
        st.stop() 

    current_standings = df_perf.set_index('Driver').T.to_dict()
    drivers = df_perf['Driver'].tolist()
    driver_means = df_perf['Season_Avg_Points'].to_numpy()
    driver_stds = df_perf['Season_Std_Points'].to_numpy()
    driver_current_points = df_perf['Points'].to_numpy()

    # 2. Load season history
    try:
        df_history = pd.read_csv('season_history.csv')
        real_season_history = {}
        for _, row in df_history.iterrows():
            try:
                # --- FIX: Clean data and ensure consistent length ---
                # 1. Clean any non-numeric strings (like '1G' for Gasly)
                history_str = row['HistoryString'].replace('1G', '1')
                
                # 2. Safely parse the string
                history_tuple = ast.literal_eval(history_str)
                history_list = list(history_tuple)
                
                # 3. Pad the list if it's shorter than 26 events
                if len(history_list) < 26:
                    history_list.extend([0] * (26 - len(history_list)))
                
                # 4. Save the guaranteed-length list
                real_season_history[row['Driver']] = history_list
            except (ValueError, SyntaxError):
                real_season_history[row['Driver']] = [0] * 26 # Fallback
    except FileNotFoundError:
        st.error("`season_history.csv` not found. History comparison will fail.")
        real_season_history = {}

    return (current_standings, drivers, driver_means, driver_stds, 
            driver_current_points, real_season_history)

@st.cache_data
def get_randomized_mock_history(driver_data):
    """Generates a plausible, high-variance history using mu/sigma."""
    history = {}
    for driver, data in driver_data.items():
        total_points = data['Points']
        if total_points == 0:
            history[driver] = [0] * 26
            continue
            
        sim_scores = np.random.normal(data['Season_Avg_Points'] + 1, data['Season_Std_Points'], 26)
        sim_scores[sim_scores < 0] = 0
        current_sum = np.sum(sim_scores)
        
        if current_sum > 0:
            scale_factor = total_points / current_sum
            scaled_events = sim_scores * scale_factor
        else:
            scaled_events = np.zeros(26)
            scaled_events[0] = total_points
            
        final_events = scaled_events.round().astype(int)
        diff = total_points - np.sum(final_events)
        if diff != 0: final_events[0] += diff
        final_events[final_events < 0] = 0
        history[driver] = final_events.tolist()
    return history

# --- 4. SIMULATION ENGINE ---
def get_race_points(num_sims, num_drivers, means, stds, dnf_probs, points_map):
    """Vectorized function to simulate one race with DNF model."""
    # 1. Simulate performance scores
    scores = np.random.normal(means, stds, size=(num_sims, num_drivers))
    
    # 2. Simulate DNF rolls
    dnf_roll = np.random.rand(num_sims, num_drivers)
    
    # 3. Apply DNF penalty
    # If DNF roll is less than prob, set score to -999, else keep original score
    scores = np.where(dnf_roll < dnf_probs, -999.0, scores)
    
    # Add jitter to break ties
    scores += np.random.uniform(0, 0.001, size=scores.shape)
    
    # 4. Rank and get points
    ranks = np.argsort(scores, axis=1)[:, ::-1].argsort(axis=1) + 1
    return points_map[ranks]

@st.cache_data(ttl=600)
def run_simulation(num_simulations, selected_driver_name, current_standings, drivers, 
                   driver_means, driver_stds, driver_current_points):
    """Fully vectorized Monte Carlo simulation with DNF model."""
    num_drivers_local = len(drivers)
    
    # --- CALCULATE DNF PROBABILITIES ---
    # Scale std devs from 0 (most consistent) to 1 (least)
    min_std = np.min(driver_stds)
    max_std = np.max(driver_stds)
    scaled_stds = (driver_stds - min_std) / (max_std - min_std)
    
    # Define DNF model: Base 2% chance + up to 10% extra based on inconsistency
    BASE_DNF_PROB = 0.02
    MAX_ADDITIONAL_DNF_PROB = 0.10
    dnf_probs_per_driver = BASE_DNF_PROB + (scaled_stds * MAX_ADDITIONAL_DNF_PROB)
    
    # --- Points Systems ---
    gp_points_system = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    sprint_points_system = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
    gp_points_map = np.zeros(21, dtype=int)
    sprint_points_map = np.zeros(21, dtype=int)
    for i in range(1, 21):
        gp_points_map[i] = gp_points_system.get(i, 0)
        sprint_points_map[i] = sprint_points_system.get(i, 0)
    
    # --- Define common arguments for get_race_points ---
    args = (num_simulations, num_drivers_local, driver_means, driver_stds)
    
    # Pass dnf_probs and points_map as keywords
    gp1 = get_race_points(*args, dnf_probs=dnf_probs_per_driver, points_map=gp_points_map)
    gp2 = get_race_points(*args, dnf_probs=dnf_probs_per_driver, points_map=gp_points_map)
    gp3 = get_race_points(*args, dnf_probs=dnf_probs_per_driver, points_map=gp_points_map)
    sprint = get_race_points(*args, dnf_probs=dnf_probs_per_driver, points_map=sprint_points_map)
    
    # --- Aggregation ---
    total_simulated_points = gp1 + gp2 + gp3 + sprint
    final_points_array = (total_simulated_points + driver_current_points).astype(np.uint16)
    
    # 1. Summaries
    avg_final_points = final_points_array.mean(axis=0)
    winner_indices = np.argmax(final_points_array, axis=1)
    winner_counts = np.bincount(winner_indices, minlength=num_drivers_local)
    
    win_probs = {driver: count / num_simulations for driver, count in zip(drivers, winner_counts)}
    avg_points_dict = {driver: points for driver, points in zip(drivers, avg_final_points)}

    # 2. Risk Stats
    all_race_points = np.stack([gp1, gp2, gp3, sprint], axis=0)
    all_race_points_flat = all_race_points.reshape(-1, num_drivers_local)
    total_trials = all_race_points_flat.shape[0]

    driver_prob_stats = {}
    for i, driver in enumerate(drivers):
        driver_points = all_race_points_flat[:, i]
        p_dnf_or_out = np.sum(driver_points == 0) / total_trials
        avg = current_standings[driver]['Season_Avg_Points']
        p_below_avg = np.sum(driver_points < avg) / total_trials
        driver_prob_stats[driver] = {'P_DNF_Out': p_dnf_or_out, 'P_BelowAvg': p_below_avg}

    # 3. Data for single driver histogram
    driver_index = drivers.index(selected_driver_name)
    selected_driver_data = final_points_array[:, driver_index]
    
    # 4. Sampled data for violin plot
    sorted_avg_indices = np.argsort(avg_final_points)[::-1]
    top_10_driver_indices = sorted_avg_indices[:10]
    top_10_driver_names = [drivers[i] for i in top_10_driver_indices]

    sample_size = min(num_simulations, 50_000)
    sample_indices = np.random.choice(num_simulations, sample_size, replace=(num_simulations < 50_000))
    
    violin_data_dict = {drivers[i]: final_points_array[sample_indices, i] for i in top_10_driver_indices}
    df_violin_long = pd.DataFrame(violin_data_dict).melt(var_name='Driver', value_name='Simulated Final Points')

    return (win_probs, avg_points_dict, selected_driver_data, 
            df_violin_long, top_10_driver_names, sample_size, driver_prob_stats)

# --- 5. PLOTTING FUNCTIONS ---

def plot_win_probability(df_probs, selected_driver):
    """Plots the championship win probability bar chart."""
    driver_colors = {d: '#e10600' if d == selected_driver else '#0070c0' for d in df_probs['Driver']}
    fig = px.bar(
        df_probs, x='Driver', y='Probability', color='Driver', text_auto='.2%',
        labels={'Probability': 'Win Probability (%)', 'Driver': 'Driver'},
        color_discrete_map=driver_colors
    )
    fig.update_layout(
        yaxis_ticksuffix='%', xaxis={'categoryorder':'total descending'}, 
        paper_bgcolor='#1a1a1a', plot_bgcolor='#2a2a2a', font_color='#f0f0f0',
        showlegend=False, transition_duration=500
    )
    return fig

def plot_final_standings(df_compare):
    """Plots the stacked bar chart for final standings."""
    df_stacked = df_compare.copy()
    df_stacked['Simulated Points Gained'] = df_stacked['Avg. Simulated Final Points'] - df_stacked['Current Points']
    df_stacked = df_stacked.sort_values(by='Avg. Simulated Final Points', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_stacked['Driver'], y=df_stacked['Current Points'],
        name='Current Points', marker_color='#e10600',
        text=df_stacked['Current Points'], textposition='auto'
    ))
    fig.add_trace(go.Bar(
        x=df_stacked['Driver'], y=df_stacked['Simulated Points Gained'],
        name='Simulated Points Gained', marker_color='#0070c0',
        text=df_stacked['Simulated Points Gained'].round(0).astype(int), textposition='auto'
    ))
    fig.update_layout(
        barmode='stack',
        title='Current Points + Avg. Simulated Gained Points',
        xaxis_title='Driver', yaxis_title='Points', xaxis_tickangle=-45,
        paper_bgcolor='#1a1a1a', plot_bgcolor='#2a2a2a', font_color='#f0f0f0',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition_duration=500
    )
    return fig

def plot_violin_distribution(df_violin, top_10_names):
    """Plots the violin chart for point distribution."""
    fig = px.violin(
        df_violin, x='Driver', y='Simulated Final Points',
        color='Driver', box=True,
        labels={'Simulated Final Points': 'Final Points'}
    )
    fig.update_layout(
        xaxis={'categoryorder':'array', 'categoryarray': top_10_names},
        paper_bgcolor='#1a1a1a', plot_bgcolor='#2a2a2a', font_color='#f0f0f0',
        showlegend=False, height=600, transition_duration=500
    )
    return fig

def plot_performance_scatter(df_perf):
    """Plots the performance vs. consistency scatter plot."""
    mean_avg_points = df_perf['Mean Points'].mean()
    mean_std_dev = df_perf['Std Dev (Consistency)'].mean()

    fig = px.scatter(
        df_perf, x='Mean Points', y='Std Dev (Consistency)',
        hover_data=['Driver', 'Team', 'Mean Points', 'Std Dev (Consistency)'],
        title='Driver Performance Model',
        labels={'Mean Points': 'Avg. Points Per Event (μ)', 'Std Dev (Consistency)': 'Points Standard Deviation (σ)'},
        color='Team', size='Mean Points'
    )
    fig.add_vline(x=mean_avg_points, line_dash="dash", line_color="grey", annotation_text="Avg. Performance")
    fig.add_hline(y=mean_std_dev, line_dash="dash", line_color="grey", annotation_text="Avg. Consistency")
    
    # Quadrant Annotations
    fig.add_annotation(text="<b>Ideal</b><br>(High Avg, Low Variance)", x=df_perf['Mean Points'].max(), y=df_perf['Std Dev (Consistency)'].min(), showarrow=False, font=dict(color="#28a745"), xanchor='right', yanchor='bottom')
    fig.add_annotation(text="<b>Inconsistent Top Tier</b><br>(High Avg, High Variance)", x=df_perf['Mean Points'].max(), y=df_perf['Std Dev (Consistency)'].max(), showarrow=False, font=dict(color="#dc3545"), xanchor='right', yanchor='top')
    fig.add_annotation(text="<b>Consistent Midfield</b><br>(Low Avg, Low Variance)", x=df_perf['Mean Points'].min(), y=df_perf['Std Dev (Consistency)'].min(), showarrow=False, font=dict(color="#ffc107"), xanchor='left', yanchor='bottom')
    fig.add_annotation(text="<b>Unreliable</b><br>(Low Avg, High Variance)", x=df_perf['Mean Points'].min(), y=df_perf['Std Dev (Consistency)'].max(), showarrow=False, font=dict(color="#6c757d"), xanchor='left', yanchor='top')

    fig.update_traces(marker=dict(size=12, opacity=0.8, line=dict(width=1, color='Black')))
    fig.update_layout(paper_bgcolor='#1a1a1a', plot_bgcolor='#2a2a2a', font_color='#f0f0f0', height=600, transition_duration=500)
    return fig

def plot_driver_histogram(driver_points_data, selected_driver, current_points, avg_points, num_sims):
    """Plots the histogram for the selected driver."""
    fig = px.histogram(
        x=driver_points_data, nbins=70,
        title=f"Distribution of {selected_driver}'s Final Points (from {num_sims:,} simulations)",
        labels={'x': 'Final Championship Points', 'y': 'Frequency'}
    )
    fig.update_traces(marker=dict(color='#0070c0'))
    fig.add_vline(x=current_points, line_dash="dash", line_color="#e10600", annotation_text="Current Points", annotation_font_color="#e10600")
    fig.add_vline(x=avg_points, line_dash="dot", line_color="#f0f0f0", annotation_text="Average Final", annotation_font_color="#f0f0f0")
    fig.update_layout(paper_bgcolor='#1a1a1a', plot_bgcolor='#2a2a2a', font_color='#f0f0f0', showlegend=False, transition_duration=500)
    return fig

def plot_history_comparison(real_pts, randomized_pts, selected_driver, current_points):
    """Plots the actual vs. randomized season history."""
    df_history = pd.DataFrame({
        'Event': list(range(1, 27)) * 2,
        # --- FIX: real_pts is now a list, no need to cast ---
        'Points Scored': real_pts + randomized_pts,
        'Distribution Type': ['Actual Season History'] * 26 + ['Randomized Mock Season'] * 26
    })
    fig = px.bar(
        df_history, x='Event', y='Points Scored',
        color='Distribution Type', barmode='group',
        title=f"Comparison of {selected_driver}'s 26-Event History Models (Total Points: {current_points})",
        labels={'Points Scored': 'Points', 'Distribution Type': 'Model'},
        color_discrete_map={'Actual Season History': '#e10600', 'Randomized Mock Season': '#444444'}
    )
    fig.update_layout(
        paper_bgcolor='#1a1a1a', plot_bgcolor='#2a2a2a', font_color='#f0f0f0',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        transition_duration=500
    )
    return fig

# --- 6. STREAMLIT UI ---

def main():
    load_css()
    
    st.title("🏎️ 2025 F1 Championship Predictor")
    st.markdown("A high-speed **NumPy-vectorized** Monte Carlo simulation forecasting the final 3 GPs and 1 Sprint.")
    st.markdown("---")

    (current_standings, drivers, driver_means, driver_stds, 
     driver_current_points, real_season_history) = load_all_data()
    
    randomized_history_data = get_randomized_mock_history(current_standings)

    # --- Centralized Controls ---
    MAX_SIMULATIONS = 2_000_000
    sim_options = {
        "Quick (10k)": 10000, "Standard (100k)": 100000,
        "Deep (500k)": 500000, "Ultra (2M)": MAX_SIMULATIONS, "Custom...": -1 
    }
    
    col1, col2, col_spacer, col3 = st.columns([2.5, 2.5, 1, 1.5]) 
    with col1:
        sim_choice = st.selectbox("Select Simulation Quality:", options=list(sim_options.keys()), index=1)
        if sim_choice == "Custom...":
            num_sims = st.number_input("Enter Custom Simulation Count:", min_value=1000, max_value=MAX_SIMULATIONS, value=10000, step=1000)
        else:
            num_sims = sim_options[sim_choice]
    with col2:
        selected_driver = st.selectbox("Select a Driver for Deep Dive:", drivers, index=0)
    with col3:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True) 
        predict_button = st.button("Run Simulation", type="primary")

    st.markdown("---")

    # --- Main Page Logic ---
    if not predict_button:
        st.info("**Select your simulation quality and a driver, then click 'Run Simulation'.**")
        st.image("https://placehold.co/1200x600/1a1a1a/e10600?text=2025+F1+Championship&font=roboto", caption="The 2025 title fight is on.")
        
        # --- THIS IS THE MODIFIED SECTION ---
        st.markdown("## Data Sources")
        col_dl_1, col_dl_2 = st.columns(2)
        
        with col_dl_1:
            try:
                with open('driver_performance.csv', 'r') as f:
                    csv_data_perf = f.read()
                st.download_button(
                    label="Download Driver Performance Data (CSV)",
                    data=csv_data_perf, 
                    file_name="driver_performance.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
            except FileNotFoundError:
                st.warning("Could not find `driver_performance.csv` for download.")
        
        with col_dl_2:
            try:
                with open('season_history.csv', 'r') as f:
                    csv_data_hist = f.read()
                st.download_button(
                    label="Download Season History Data (CSV)",
                    data=csv_data_hist, 
                    file_name="season_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except FileNotFoundError:
                st.warning("Could not find `season_history.csv` for download.")
        # --- END OF MODIFIED SECTION ---
            
    else:
        progress_bar = st.progress(0, text="Booting up simulation engine...")
        start_time = time.time()
        
        progress_bar.progress(25, text=f"Running {num_sims:,} simulations (Vectorized)...")
        (win_probs, avg_points, driver_points_data, df_violin_long, 
         top_10_drivers_list, sample_size, driver_prob_stats
        ) = run_simulation(
            num_sims, selected_driver, current_standings, drivers, 
            driver_means, driver_stds, driver_current_points
        )
        
        progress_bar.progress(75, text="Aggregating results and preparing visualizations...")
        
        # --- Prepare DataFrames ---
        df_d_probs = pd.DataFrame(win_probs.items(), columns=['Driver', 'Probability'])
        top_prob_drivers = df_d_probs[df_d_probs['Probability'] > 0].sort_values(by='Probability', ascending=False)['Driver'].head(10).tolist()
        final_drivers_for_prob_chart = [selected_driver] + top_prob_drivers
        final_drivers_for_prob_chart = list(dict.fromkeys(final_drivers_for_prob_chart))
        df_d_probs_filtered = df_d_probs[df_d_probs['Driver'].isin(final_drivers_for_prob_chart)].sort_values(by='Probability', ascending=False)

        df_perf = pd.DataFrame({
            'Driver': drivers, 'Mean Points': driver_means, 
            'Std Dev (Consistency)': driver_stds,
            'Team': [current_standings[d]['Team'] for d in drivers]
        })

        df_compare = pd.DataFrame({
            'Driver': drivers, 'Current Points': driver_current_points,
            'Avg. Simulated Final Points': [avg_points[d] for d in drivers]
        })

        end_time = time.time()
        progress_bar.progress(100, text=f"Simulation Complete! (Took {end_time - start_time:.2f}s)")

        # --- *** TABS FOR PAGE NAVIGATION *** ---
        tab1, tab2, tab3, tab4 = st.tabs([
            "Overview & Standings", 
            "Distribution & Consistency", 
            f"Driver Deep Dive: {selected_driver}", 
            "Methodology"
        ])
        
        # --- TAB 1: Overview & Standings ---
        with tab1:
            st.header("Championship Win Probability")
            
            # --- Winner's Spotlight ---
            top_driver = df_d_probs_filtered.iloc[0]['Driver']
            top_prob = df_d_probs_filtered.iloc[0]['Probability']
            
            st.subheader(f"Projected Winner: {top_driver}")
            spot_col1, spot_col2 = st.columns([1, 3])
            with spot_col1:
                st.image(driver_images.get(top_driver, 'https://placehold.co/400x400/2a2a2a/white?text=No+Image'), use_container_width=True)
            with spot_col2:
                st.metric(label=f"{top_driver}'s Win Probability", value=f"{top_prob:.2%}")
                st.metric(label="Avg. Simulated Final Points", value=f"{avg_points[top_driver]:.0f}")
                st.metric(label="Current Points", value=f"{current_standings[top_driver]['Points']}")
            
            st.plotly_chart(plot_win_probability(df_d_probs_filtered, selected_driver), use_container_width=True)
            
            st.header("Projected Final Standings")
            st.plotly_chart(plot_final_standings(df_compare), use_container_width=True)

        # --- TAB 2: Distribution & Consistency ---
        with tab2:
            st.header("Distribution and Performance Analysis")
            
            t2_col1, t2_col2 = st.columns(2)
            with t2_col1:
                st.subheader("Distribution of Final Points")
                st.plotly_chart(plot_violin_distribution(df_violin_long, top_10_drivers_list), use_container_width=True)
            with t2_col2:
                st.subheader("Performance vs. Consistency")
                st.plotly_chart(plot_performance_scatter(df_perf), use_container_width=True)

        # --- TAB 3: Driver Deep Dive ---
        with tab3:
            st.header(f"Driver Deep Dive: {selected_driver}")
            
            prob_data = driver_prob_stats[selected_driver]
            st.subheader("Race Event Risk Analysis (Per Race Probability)")
            col_prob1, col_prob2, col_col3 = st.columns(3)
            col_prob1.metric("Current Points", f"{current_standings[selected_driver]['Points']}")
            col_prob2.metric("P(DNF/Outside Points)", f"{prob_data['P_DNF_Out']:.2%}", help="Probability of scoring 0 points in any of the remaining races.")
            col_col3.metric("P(Score < Avg)", f"{prob_data['P_BelowAvg']:.2%}", help=f"Probability of scoring less than the Season Average ({current_standings[selected_driver]['Season_Avg_Points']:.2f}).")
            
            st.markdown("---")
            
            col_d1, col_d2 = st.columns([2, 3])
            with col_d1:
                st.image(driver_images.get(selected_driver, 'https://placehold.co/600x600/2a2a2a/white?text=No+Image'), caption=f"{selected_driver} ({current_standings[selected_driver]['Team']})", use_container_width=True)
                st.metric(label="Championship Win Probability", value=f"{win_probs[selected_driver]:.2%}")
                st.metric(label="Avg. Simulated Final Points", value=f"{avg_points[selected_driver]:.0f}")

            with col_d2:
                st.subheader(f"Distribution of {selected_driver}'s Final Points")
                st.plotly_chart(plot_driver_histogram(
                    driver_points_data, selected_driver, 
                    current_standings[selected_driver]['Points'], 
                    avg_points[selected_driver], num_sims
                ), use_container_width=True)
                
                st.subheader("Season History Comparison")
                real_pts = real_season_history.get(selected_driver, [0] * 26)
                randomized_pts = randomized_history_data[selected_driver]
                st.plotly_chart(plot_history_comparison(
                    real_pts, randomized_pts, selected_driver, 
                    current_standings[selected_driver]['Points']
                ), use_container_width=True)

        # --- TAB 4: Methodology ---
        with tab4:
            st.header("Methodology")
            st.markdown(f"""
            This dashboard uses a **Two-Part Stochastic Monte Carlo simulation** to forecast the end of the 2025 season.

            ### 1. High-Speed Simulation
            The model runs `{num_sims:,}` simulations for every remaining event to generate a wide range of possible outcomes. 
            This high number of simulations provides a robust statistical forecast.

            ### 2. The Two-Part DNF Model
            This model is more realistic than a simple bell curve. For each driver in each of the `{num_sims*4:,}` simulated races, it performs two checks:
            
            * **Part 1 (Reliability/Risk):** It first "rolls the dice" based on a driver's **DNF Probability**. This probability is calculated from their season consistency (Std Dev). Less consistent drivers have a higher DNF chance.
            * **Part 2 (Performance):** If the driver *doesn't* DNF, the model then simulates their performance using a **Normal Distribution** (bell curve) based on their `Season_Avg_Points` (μ) and `Season_Std_Points` (σ). A DNF driver gets a score of -999 to guarantee last place.

            ### 3. Simulating the Remaining Events
            We simulate the **3 remaining Grand Prix** (full points) and **1 remaining Sprint** (top 8 points) by:
            1.  Running the **Two-Part DNF Model** for all drivers in all `{num_sims:,}` simulations for each of the 4 events.
            2.  Ranking the drivers and awarding the correct GP or Sprint points.
            3.  Adding these new points to their `Current Points` to get a final championship total.

            ### 4. Risk Metrics
            The **Race Event Risk Analysis** is calculated by aggregating the results of all `{4 * num_sims:,}` simulated single-race outcomes to determine the likelihood of:
            * **P(DNF/Outside Points):** The percentage of simulated races where the driver scored **zero** points.
            * **P(Score < Avg):** The percentage of simulated races where the driver scored less than their Season Average Points (μ).
            """)

if __name__ == "__main__":
    main()