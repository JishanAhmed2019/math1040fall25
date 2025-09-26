# dice_lln.py
# Headless-safe Streamlit app for LLN dice simulation.
import os
import sys
import traceback

# --- Try to set a non-interactive matplotlib backend BEFORE importing pyplot ---
try:
    import matplotlib
    matplotlib.use("Agg")   # Safe for headless/container environments
    import matplotlib.pyplot as plt
    mpl_available = True
except Exception:
    mpl_available = False
    # Print full traceback to server logs so you can inspect the cause.
    traceback.print_exc()
    plt = None

import streamlit as st
import numpy as np
import pandas as pd

# If Matplotlib isn't available, we will use Altair as a fallback (Altair is in requirements.txt).
if not mpl_available:
    try:
        import altair as alt
        alt_available = True
    except Exception:
        alt_available = False
        traceback.print_exc()

st.set_page_config(page_title="Law of Large Numbers: Dice Simulation", layout="wide")

# Custom CSS for better visual appeal
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stats-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .highlight-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .explanation-box {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🎲 Law of Large Numbers</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Watch probability estimates converge with more data</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("⚙️ Controls")
    
    with st.container():
        sides = st.slider("🎲 Die Sides", min_value=2, max_value=20, value=6, step=1)
        
        event_type = st.selectbox(
            "🎯 Event to Track",
            [
                "Roll equals specific value",
                "Roll is even",
                "Roll is odd",
                "Roll ≥ threshold",
            ],
            index=0,
        )

        if event_type == "Roll equals specific value":
            target = st.slider("Target Value", min_value=1, max_value=int(sides), value=int(sides))
            def is_success(x): return x == target
            theoretical_p = 1.0 / sides

        elif event_type == "Roll is even":
            def is_success(x): return (x % 2) == 0
            theoretical_p = np.floor(sides / 2) / sides

        elif event_type == "Roll is odd":
            def is_success(x): return (x % 2) == 1
            theoretical_p = np.ceil(sides / 2) / sides

        elif event_type == "Roll ≥ threshold":
            threshold = st.slider("Threshold (≥)", min_value=1, max_value=int(sides), value=max(1, int(np.ceil(sides * 0.8))))
            def is_success(x): return x >= threshold
            theoretical_p = (sides - threshold + 1) / sides

        n_rolls = st.slider("📊 Total Rolls", min_value=100, max_value=5000, value=1000, step=100)
        n_runs = st.slider("🔄 Independent Runs", min_value=1, max_value=3, value=1, step=1)
        show_points = st.checkbox("🔍 Show First 30 Points", value=True)
        seed = st.number_input("🎲 Random Seed", value=42, min_value=0, step=1)

# Set random number generator
if seed != 0:
    rng = np.random.default_rng(int(seed))
else:
    rng = np.random.default_rng()

def simulate_run(n, sides, is_success_func, rng):
    """
    Simulate n rolls of an n-sided die and return the running probability estimate.
    Vectorized for efficiency.
    """
    rolls = rng.integers(1, sides + 1, size=n)
    successes = np.array([is_success_func(x) for x in rolls], dtype=float)
    running_p = np.cumsum(successes) / np.arange(1, n + 1)
    return running_p, rolls

# Run simulations
x = np.arange(1, n_rolls + 1)
all_estimates = []
runs_data = []

for run in range(n_runs):
    running_p, rolls = simulate_run(n_rolls, sides, is_success, rng)
    all_estimates.append(running_p[-1])
    runs_data.append(pd.DataFrame({
        "roll": x,
        "running_p": running_p,
        "run": f"Run {run+1}"
    }))

# Plotting in col2
with col2:
    st.header("📈 Convergence Plot")
    
    if mpl_available and plt is not None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        for run_idx, df in enumerate(runs_data):
            running_p = df["running_p"].to_numpy()
            alpha = 1.0 if n_runs == 1 else (0.9 if run_idx == 0 else 0.55)
            label = f"Run {run_idx+1}" if n_runs <= 3 else None
            ax.plot(x, running_p, linewidth=2, alpha=alpha, label=label, color=f'C{run_idx}')

            if show_points:
                k = min(30, n_rolls)
                ax.plot(x[:k], running_p[:k], marker="o", linestyle="None", 
                       markersize=6, color=f'C{run_idx}', alpha=0.7)

        # Highlight n=1 point
        if len(runs_data) > 0:
            ax.plot(1, runs_data[0]["running_p"].iloc[0], marker="s", 
                   markersize=10, color="red", label="First Roll", zorder=5)

        ax.axhline(theoretical_p, color='black', linestyle="--", linewidth=3, 
                  label=f"Theoretical p = {theoretical_p:.3f}")

        ax.set_xlim(1, n_rolls)
        y_max = max(1.0, theoretical_p * 2) if theoretical_p > 0 else 1.0
        ax.set_ylim(0, min(1.0, y_max))
        ax.set_xlabel("Number of Rolls (n)", fontsize=12)
        ax.set_ylabel("Running Estimate p̂ₙ", fontsize=12)
        ax.set_title("Law of Large Numbers: Probability Convergence", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if n_runs > 1 or show_points:
            ax.legend(loc='upper right', fontsize=10)

        st.pyplot(fig, clear_figure=True)

    elif 'alt_available' in globals() and alt_available:
        big_df = pd.concat(runs_data, ignore_index=True)
        base = alt.Chart(big_df).mark_line(strokeWidth=2).encode(
            x=alt.X("roll:Q", title="Number of Rolls (n)"),
            y=alt.Y("running_p:Q", title="Running Estimate p̂ₙ"),
            color="run:N",
            tooltip=["run", "roll", alt.Tooltip("running_p:Q", format=".4f")]
        )

        chart = base
        if show_points:
            points = alt.Chart(big_df[big_df["roll"] <= 30]).mark_circle(size=60).encode(
                x="roll:Q", y="running_p:Q", color="run:N"
            )
            chart = base + points

        rule = alt.Chart(pd.DataFrame({"y": [theoretical_p]})).mark_rule(
            strokeDash=[5,5], strokeWidth=3
        ).encode(y="y:Q")
        
        st.altair_chart((chart + rule).interactive(), use_container_width=True)

# Stats panel
st.markdown('<div class="stats-container">', unsafe_allow_html=True)
st.header("📊 Results Summary")

if n_runs == 1:
    last_estimate = all_estimates[0]
    error = abs(last_estimate - theoretical_p)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Final Estimate", f"{last_estimate:.4f}")
    with col_b:
        st.metric("Theoretical p", f"{theoretical_p:.4f}")
    with col_c:
        st.metric("Error", f"{error:.4f}")
else:
    ests = np.array(all_estimates)
    mean_est = ests.mean()
    std_est = ests.std(ddof=1) if len(ests) > 1 else 0.0
    mean_error = abs(mean_est - theoretical_p)
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Mean Estimate", f"{mean_est:.4f}")
    with col_b:
        st.metric("Std Dev", f"{std_est:.4f}")
    with col_c:
        st.metric("Mean Error", f"{mean_error:.4f}")

st.markdown('</div>', unsafe_allow_html=True)

# Explanation boxes
st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
st.subheader("💡 Key Insights")

col_x, col_y = st.columns(2)

with col_x:
    st.markdown("**Early Rolls (n=1 to 30):**")
    st.markdown("- Estimates are very noisy")
    st.markdown("- Can be 0 or 1 initially")
    st.markdown("- Large swings in probability")
    
with col_y:
    st.markdown("**Later Rolls (n > 100):**")
    st.markdown("- Estimates stabilize")
    st.markdown("- Converge toward theoretical value")
    st.markdown("- Less variability")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
st.subheader("🎯 Law of Large Numbers Explained")
st.markdown("""
As the number of trials increases:
- **Sample mean** → **Population mean**
- **Empirical probability** → **True probability**
- **Random fluctuations** become **less significant**

This is why casinos always win in the long run!
""")
st.markdown('</div>', unsafe_allow_html=True)

# Signature
st.markdown(
    """
    <div style="text-align: right; font-size: 0.9em; margin-top: 2em; color: gray;">
        Prof. Jishan Ahmed<br>
        Data Science Assistant Professor<br>
        Weber State University
    </div>
    """,
    unsafe_allow_html=True
)