import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, export_text

st.set_page_config(page_title="Rule-Extraction Playground", layout="wide")
st.title("Rule-Extraction Playground")
st.markdown("Utforska globala surrogatmodeller och regelextraktion interaktivt!")

st.sidebar.header("Modellinställningar")
n_samples = st.sidebar.slider("Antal datapunkter", 100, 1000, 300, 50)
noise_level = st.sidebar.slider("Brusnivå (Noise)", 0.0, 0.3, 0.1, 0.05)
max_depth = st.sidebar.slider("Max trädhöjd (Max depth)", 1, 6, 3)

X, y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2, random_state=42, flip_y=noise_level)

clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
clf.fit(X, y)
accuracy = clf.score(X, y)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Toy dataset (Visualisering)")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k", alpha=0.8, s=40)
    ax.set_xlabel("Variabel x1")
    ax.set_ylabel("Variabel x2")
    st.pyplot(fig)

with col2:
    st.subheader("Decision-tree surrogate (Global)")
    st.metric(label="Modellens noggrannhet", value=f"{accuracy * 100:.1f}%")
    tree_rules = export_text(clf, feature_names=["x1", "x2"])
    st.text("Extraherade regler:")
    st.code(tree_rules, language="text")
