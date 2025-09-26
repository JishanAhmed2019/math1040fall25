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

st.set_page_config(page_title="Law of Large Numbers: Dice Simulation", layout="centered")

st.title("🎲 Dice Roll Simulation: Running Probability")
st.write(
    "Simulate rolling an n-sided die and watch the running estimate "
    r"$\hat{p}_n$ (probability of an event) converge to the theoretical value."
)

with st.sidebar:
    st.header("Controls")
    sides = st.number_input("Number of die sides", min_value=2, max_value=100, value=6, step=1)
    event_type = st.selectbox(
        "Event to track (success)",
        [
            "Roll equals a specific value",
            "Roll is even",
            "Roll is odd",
            "Roll ≥ threshold",
            "Custom set of winning faces",
        ],
        index=0,
    )

    if event_type == "Roll equals a specific value":
        target = st.number_input("Target value", min_value=1, max_value=int(sides), value=int(sides))
        def is_success(x): return x == target
        theoretical_p = 1.0 / sides

    elif event_type == "Roll is even":
        def is_success(x): return (x % 2) == 0
        theoretical_p = np.floor(sides / 2) / sides

    elif event_type == "Roll is odd":
        def is_success(x): return (x % 2) == 1
        theoretical_p = np.ceil(sides / 2) / sides

    elif event_type == "Roll ≥ threshold":
        threshold = st.number_input("Threshold (≥)", min_value=1, max_value=int(sides), value=max(1, int(np.ceil(sides * 0.8))))
        def is_success(x): return x >= threshold
        theoretical_p = (sides - threshold + 1) / sides

    elif event_type == "Custom set of winning faces":
        faces_str = st.text_input("Enter winning faces (comma or space separated)", value=str(sides))
        def parse_faces(s):
            toks = [t.strip() for t in s.replace(",", " ").split() if t.strip()]
            vals = []
            for t in toks:
                try:
                    v = int(t)
                    if 1 <= v <= int(sides):
                        vals.append(v)
                except ValueError:
                    pass
            return sorted(set(vals))
        faces = parse_faces(faces_str)
        st.caption(f"Parsed winning faces: {faces if faces else '∅'}")
        def is_success(x): return x in set(faces)
        theoretical_p = len(set(faces)) / sides if faces else 0.0

    n_rolls = st.slider("Number of rolls (n)", min_value=1, max_value=10_000, value=2_000, step=1)
    n_runs = st.slider("Overlay independent runs", min_value=1, max_value=5, value=1, step=1)
    show_points = st.checkbox("Show markers for first 30 rolls", value=True)
    seed = st.number_input("Random seed (optional)", value=0, min_value=0, step=1, help="Use 0 for a different seed every run.")
    st.markdown("---")
    st.caption("Tip: Watch how wildly the estimate jumps at the start — this is why we need many trials!")

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
runs_data = []  # for altair fallback

for run in range(n_runs):
    running_p, rolls = simulate_run(n_rolls, sides, is_success, rng)
    all_estimates.append(running_p[-1])
    runs_data.append(pd.DataFrame({
        "roll": x,
        "running_p": running_p,
        "run": f"Run {run+1}"
    }))

# Plotting: prefer matplotlib (faster control), fallback to Altair if matplotlib import failed
if mpl_available and plt is not None:
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150)

    for run_idx, df in enumerate(runs_data):
        running_p = df["running_p"].to_numpy()
        alpha = 1.0 if n_runs == 1 else (0.9 if run_idx == 0 else 0.55)
        label = f"Run {run_idx+1}" if n_runs <= 3 else None
        ax.plot(x, running_p, linewidth=1.5, alpha=alpha, label=label)

        if show_points:
            k = min(30, n_rolls)
            ax.plot(x[:k], running_p[:k], marker="o", linestyle="None", markersize=4,
                    color=f'C{run_idx % 10}')

    # Highlight n=1 point explicitly
    if len(runs_data) > 0:
        ax.plot(1, runs_data[-1]["running_p"].iloc[0], marker="s", markersize=8, color="red", label="n=1", zorder=5)

    ax.axhline(theoretical_p, color='black', linestyle="--", linewidth=2, label="Theoretical p")
    ax.text(x[-1] * 0.95, theoretical_p, "  theoretical p", va="center", ha="right", fontsize=9, color='black')

    ax.set_xlim(1, n_rolls)
    y_max = max(1.0, theoretical_p * 2) if theoretical_p > 0 else 1.0
    ax.set_ylim(0, min(1.0, y_max))
    ax.set_xlabel("Number of rolls (n)")
    ax.set_ylabel(r"Running estimate $\hat{p}_n$")
    ax.set_title("Convergence of Empirical Probability (Law of Large Numbers)", fontsize=10)
    ax.grid(True, alpha=0.3)
    if n_runs > 1 or show_points:
        ax.legend(loc='upper right', fontsize=8)

    st.pyplot(fig, clear_figure=True)

elif 'alt_available' in globals() and alt_available:
    # Build a combined DataFrame for Altair
    big_df = pd.concat(runs_data, ignore_index=True)
    # Altair line chart with optional points
    base = alt.Chart(big_df).mark_line().encode(
        x=alt.X("roll:Q", title="Number of rolls (n)"),
        y=alt.Y("running_p:Q", title=r"Running estimate $\hat{p}_n$"),
        color="run:N",
        tooltip=["run", "roll", alt.Tooltip("running_p:Q", format=".4f")]
    )

    chart = base
    if show_points:
        points = alt.Chart(big_df[big_df["roll"] <= 30]).mark_circle(size=30).encode(
            x="roll:Q", y="running_p:Q", color="run:N"
        )
        chart = base + points

    # Add theoretical p as a rule
    rule = alt.Chart(pd.DataFrame({"y": [theoretical_p]})).mark_rule(strokeDash=[5,5]).encode(y="y:Q")
    st.altair_chart((chart + rule).interactive(), use_container_width=True)

else:
    st.error("Unable to render plots: neither matplotlib nor Altair is available. Check server logs.")
    st.stop()

# Stats panel
st.markdown("### Latest Stats")
if n_runs == 1:
    last_estimate = all_estimates[0]
    st.write(
        f"**Final running estimate**: {last_estimate:.4f}  |  "
        f"**Theoretical p**: {theoretical_p:.4f}  |  "
        f"**Absolute error**: {abs(last_estimate - theoretical_p):.4f}"
    )
else:
    ests = np.array(all_estimates)
    st.write(
        f"**Across {n_runs} independent runs**, mean final estimate = {ests.mean():.4f}, "
        f"std = {ests.std(ddof=1) if len(ests) > 1 else 0.0:.4f}. "
        f"Theoretical p = {theoretical_p:.4f}."
    )

st.markdown("---")
st.caption(
    "💡 **Why show the first 30 points?** At n=1, your estimate is either 0 or 1 — very noisy! "
    "As you roll more, the estimate stabilizes toward the true probability. This is the Law of Large Numbers in action: "
    "more data → less noise → better estimate."
)

# Signature
st.markdown(
    """
    <div style="text-align: right; font-size: 0.9em; margin-top: 2em; color: gray;">
        Jishan Ahmed, PhD<br>
        Data Science Assistant Professor<br>
        Department of Mathematics<br>
        Weber State University
    </div>
    """,
    unsafe_allow_html=True
)
