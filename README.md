

# 🏎️ F1 Championship Predictor (2025)

A **Monte Carlo simulation dashboard** that forecasts the outcome of the **2025 Formula 1 World Championship** — powered by Streamlit and Plotly.
It simulates millions of race outcomes across the final rounds to project **championship probabilities, performance risk, and final standings**.



##  Features

### Advanced Simulation Engine

* Fully **vectorized Monte Carlo simulation** for ultra-fast performance (up to **2 million simulations**).
* Models **3 Grand Prix** and **1 Sprint** race using realistic points systems.
* Integrates a **two-part DNF model** for reliability and performance variance:

  * Simulates driver performance with a **Normal Distribution** (`μ`, `σ` from season data).
  * Adds stochastic **DNF (Did Not Finish)** probability based on driver consistency.

### Interactive Visual Analytics

* **Win Probability Bar Chart:** Who’s most likely to win the 2025 title?
* **Stacked Points Projection:** Compare current vs. simulated final points.
* **Violin & Scatter Plots:** Visualize driver consistency and point distributions.
* **Driver Deep Dive:** Explore individual performance, risk, and historical comparison.
* **Season History Viewer:** Compare real vs. randomized race performances.

### Modern F1-Inspired UI

* Custom **dark-mode F1-themed interface** with CSS animations (glow, pulse, transitions).
* Clean **Plotly interactive visualizations** integrated seamlessly into Streamlit.
* Smart progress bar and metric cards for an engaging simulation experience.

---

## Tech Stack

| Component             | Description                                         |
| --------------------- | --------------------------------------------------- |
| **Frontend/UI**       | Streamlit + Custom CSS                              |
| **Visualization**     | Plotly Express & Graph Objects                      |
| **Data Handling**     | Pandas, NumPy, AST                                  |
| **Simulation Engine** | Vectorized Monte Carlo (NumPy)                      |
| **Image Handling**    | Pillow (PIL)                                        |
| **Caching**           | Streamlit’s `@st.cache_data` for high-speed re-runs |

---

## Data Files

This app uses two CSV files for inputs:

| File                     | Purpose                                                                            |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `driver_performance.csv` | Contains per-driver stats: Team, Avg. Points (μ), Std. Dev (σ), and Current Points |
| `season_history.csv`     | Contains actual event-by-event scoring history per driver                          |

*(Both files can be downloaded directly from the app interface.)*



## ⚙️ How It Works

1. **Load Data** – Reads driver performance and historical results.
2. **Simulate Remaining Races** – Generates probabilistic race outcomes using:

   * Gaussian performance model (based on `μ`, `σ`)
   * DNF chance derived from driver consistency
3. **Aggregate Results** – Calculates final championship points and win probabilities.
4. **Visualize** – Displays results through interactive dashboards and statistical breakdowns.



## 🏁 Getting Started

### 🔧 Requirements

Make sure you have Python 3.9+ and install dependencies:

```bash
pip install streamlit pandas numpy plotly pillow
```

### ▶️ Run the App

```bash
streamlit run app.py
```

Then open the provided local URL (usually `http://localhost:8501`) in your browser.



## Example Usage

* Select **simulation quality** (Quick → Ultra, up to 2 million runs).
* Choose a **driver** (e.g., Lando Norris).
* Click **“Run Simulation”** to forecast final standings and probabilities.
* Explore detailed charts and comparisons across tabs.



## Outputs

* **Championship win probabilities**
* **Projected final standings**
* **Performance risk metrics**
* **Driver-specific distribution histograms**
* **Actual vs. simulated season history**


## License

MIT License © 2025 — Open source and free to use for analysis, learning, or custom simulation projects.



## Author

Developed by Archita Chandwani
Passionate about **data-driven motorsport analytics** and **predictive modeling**.


