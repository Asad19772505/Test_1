import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # Ensure 3D plots work

# --- Streamlit UI ---
st.set_page_config(page_title="What-If NPV Simulator", layout="centered")
st.title("📊 What-If Analysis: NPV Based on Revenue Growth & Cost Growth")

with st.expander("ℹ️ About this tool"):
    st.write("""
    This tool performs a what-if analysis on Net Present Value (NPV) by simulating different combinations of:
    - Revenue growth rates
    - Cost growth rates

    **Key outputs:**
    - NPV heatmap showing how different growth scenarios affect project profitability
    - Detailed scenario table with all combinations
    """)

# --- User Inputs ---
col1, col2 = st.columns(2)
with col1:
    initial_investment = st.number_input("💰 Initial Investment ($)", value=50000)
    revenue_base = st.number_input("📈 Base Annual Revenue ($)", value=20000)
    cost_base = st.number_input("📉 Base Annual Cost ($)", value=10000)

with col2:
    years = st.slider("⏳ Project Duration (Years)", min_value=1, max_value=20, value=5)
    discount_rate = st.slider("🏦 Discount Rate (WACC %)", 0.0, 20.0, 10.0, step=0.5) / 100

st.subheader("Growth Rate Scenarios")
revenue_growth_range = st.slider("📈 Revenue Growth Range (%)", -50, 100, (-10, 50))
cost_growth_range = st.slider("📈 Cost Growth Range (%)", -20, 100, (0, 50))

# --- Simulation Grid ---
@st.cache_data
def run_simulation(initial_investment, revenue_base, cost_base, years, discount_rate,
                  revenue_growth_range, cost_growth_range, num_points=20):
    revenue_growths = np.linspace(revenue_growth_range[0], revenue_growth_range[1], num_points) / 100
    cost_growths = np.linspace(cost_growth_range[0], cost_growth_range[1], num_points) / 100

    results = []
    for r_growth in revenue_growths:
        for c_growth in cost_growths:
            cash_flows = []
            for year in range(1, years + 1):
                revenue = revenue_base * ((1 + r_growth) ** year)
                cost = cost_base * ((1 + c_growth) ** year)
                cash_flows.append(revenue - cost)

            npv = npf.npv(discount_rate, [-initial_investment] + cash_flows)
            try:
                irr = npf.irr([-initial_investment] + cash_flows)
                irr = irr if not np.isnan(irr) else None
            except:
                irr = None

            cumulative_cf = np.cumsum(cash_flows)
            payback = next((i+1 for i, val in enumerate(cumulative_cf) if val >= initial_investment), years)

            results.append({
                "Revenue Growth %": round(r_growth * 100, 1),
                "Cost Growth %": round(c_growth * 100, 1),
                "NPV ($)": round(npv, 2),
                "IRR (%)": round(irr * 100, 1) if irr is not None else "N/A",
                "Payback (Years)": payback
            })

    return pd.DataFrame(results)

df = run_simulation(initial_investment, revenue_base, cost_base, years,
                    discount_rate, revenue_growth_range, cost_growth_range)

# --- Visualization ---
st.subheader("📊 Visualization")
plot_type = st.radio("Select visualization type:", ["Heatmap", "3D Surface"])

pivot_df = df.pivot(index="Revenue Growth %", columns="Cost Growth %", values="NPV ($)")

if plot_type == "Heatmap":
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("RdYlGn")

    zero_crossings = np.abs(pivot_df.values) < (0.05 * np.max(np.abs(pivot_df.values)))

    heatmap = ax.imshow(pivot_df.values, cmap=cmap, origin="lower", aspect="auto")
    ax.contour(zero_crossings, levels=[0.5], colors='black', linestyles='dashed')

    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels([f"{x}%" for x in pivot_df.columns])
    ax.set_yticklabels([f"{y}%" for y in pivot_df.index])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel("Cost Growth %")
    ax.set_ylabel("Revenue Growth %")
    ax.set_title(f"NPV Heatmap (Discount Rate: {discount_rate*100:.1f}%)")

    cbar = plt.colorbar(heatmap)
    cbar.set_label("NPV ($)")

    st.pyplot(fig)

else:  # 3D Surface plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(pivot_df.columns, pivot_df.index)
    Z = pivot_df.values
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('Cost Growth %')
    ax.set_ylabel('Revenue Growth %')
    ax.set_zlabel('NPV ($)')
    ax.set_title(f"NPV Surface (Discount Rate: {discount_rate*100:.1f}%)")

    fig.colorbar(surf, shrink=0.5, aspect=5)
    st.pyplot(fig)

# --- Display Results ---
st.subheader("📋 Detailed Scenario Table")
st.dataframe(df.style.format({
    "NPV ($)": "{:,.0f}",
    "IRR (%)": "{:,.1f}"
}).background_gradient(cmap="RdYlGn", subset=["NPV ($)"]))

# --- Key Metrics ---
st.subheader("🔑 Key Metrics")
max_npv = df.loc[df["NPV ($)"].idxmax()]
min_npv = df.loc[df["NPV ($)"].idxmin()]

col1, col2 = st.columns(2)
with col1:
    st.metric("Best Scenario (Max NPV)",
              f"${max_npv['NPV ($)']:,.0f}",
              f"Rev: {max_npv['Revenue Growth %']}%, Cost: {max_npv['Cost Growth %']}%")

with col2:
    st.metric("Worst Scenario (Min NPV)",
              f"${min_npv['NPV ($)']:,.0f}",
              f"Rev: {min_npv['Revenue Growth %']}%, Cost: {min_npv['Cost Growth %']}%")

# --- Download Results ---
st.download_button(
    label="💾 Download Results as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='npv_simulation_results.csv',
    mime='text/csv'
)
