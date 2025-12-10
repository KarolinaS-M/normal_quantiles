# app.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import streamlit as st


# -------------------- Page config --------------------
st.set_page_config(
    page_title="Normal quantiles demo",
    layout="wide"
)

st.title("Standard normal quantiles and shaded areas")

st.markdown(
    """
This app illustrates the relationship between:

- a **quantile order** $q$ (left side) and the corresponding quantile value $x = \\Phi^{-1}(q)$,
- a **quantile value** $x$ (right side) and the corresponding order $q = \\Phi(x)$.

For each input, the standard normal density is plotted with the area to the left of the point shaded.
"""
)


# -------------------- Helper function for plotting --------------------
def plot_shaded_normal(x_q: float, color: str, area_label: str, title: str):
    """
    Plot the standard normal density and shade the area to the left of x_q.
    """
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)

    fig, ax = plt.subplots(figsize=(5, 3))

    # Density curve
    ax.plot(x, y, label="Standard normal density")

    # Shaded area to the left of x_q
    x_fill = x[x <= x_q]
    y_fill = y[x <= x_q]
    ax.fill_between(x_fill, 0, y_fill, alpha=0.3, color=color, label=area_label)

    # Vertical line at x_q
    ax.axvline(x=x_q, linestyle="--", color=color)

    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    return fig


# -------------------- Layout: two columns --------------------
left_col, right_col = st.columns(2)


# ==================== LEFT SIDE: order -> quantile ====================
with left_col:
    st.subheader("From quantile order to value")

    st.markdown(
        "Enter a **quantile order** $q$ (left-tail probability). "
        "The app will compute the corresponding value $x = \\Phi^{-1}(q)$."
    )

    # Restrict q a bit away from 0 and 1 to avoid ±∞
    q = st.slider(
        "Quantile order q (left-tail probability)",
        min_value=0.001,
        max_value=0.999,
        value=0.25,
        step=0.001
    )

    x_q = norm.ppf(q)

    st.markdown(
        f"""
**Result**

- Quantile order: $q = {q:.4f}$
- Quantile value: $x = \\Phi^{{-1}}(q) \\approx {x_q:.4f}$  
- Interpretation: approximately **{q*100:.1f}%** of observations lie **below** this value.
"""
    )

    fig_left = plot_shaded_normal(
        x_q=x_q,
        color="tab:blue",
        area_label=f"Area for X ≤ {x_q:.4f}",
        title=f"Standard normal density (q = {q:.4f})"
    )
    st.pyplot(fig_left)


# ==================== RIGHT SIDE: value -> order ====================
with right_col:
    st.subheader("From quantile value to order")

    st.markdown(
        "Enter a **quantile value** $x$. "
        "The app will compute the corresponding order $q = \\Phi(x)$."
    )

    x_input = st.number_input(
        "Quantile value x",
        min_value=-4.0,
        max_value=4.0,
        value=-0.6745,
        step=0.0001,
        format="%.4f"
    )

    q_from_x = norm.cdf(x_input)

    st.markdown(
        f"""
**Result**

- Quantile value: $x = {x_input:.4f}$
- Quantile order: $q = \\Phi(x) \\approx {q_from_x:.4f}$  
- Interpretation: approximately **{q_from_x*100:.1f}%** of observations lie **below** this value.
"""
    )

    fig_right = plot_shaded_normal(
        x_q=x_input,
        color="green",
        area_label=f"Area for X ≤ {x_input:.4f}",
        title=f"Standard normal density (x = {x_input:.4f})"
    )
    st.pyplot(fig_right)