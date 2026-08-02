# TrustLens AI — Phase 2 Fraud Intelligence Specification

## Overview
**TrustLens AI Phase 2** evolves the platform from a single-text review classifier into a multi-dimensional **Fraud Intelligence Platform**. By combining NLP, graph theory, behavioral risk profiling, and time-series anomaly detection, TrustLens AI detects sophisticated review fraud rings, coordinated bot campaigns, and burst anomalies.

---

## 1. Reviewer Network Analysis (Graph Construction)

### Graph Architecture
The reviewer network is modeled as a **Bipartite Heterogeneous Graph** $G = (V_U, V_P, E)$:
- **Nodes $V_U$**: Reviewer / User Entities (`user_id`).
- **Nodes $V_P$**: Target Product Entities (`product_id`).
- **Edges $E$**: Directed/Weighted edges representing review submissions. Edge attributes contain `rating`, `timestamp`, and text preview.

### Interactive Visualization (PyVis)
The graph is rendered interactively using `PyVis`:
- **Nodes**: Users rendered as cyan nodes (`#38bdf8`), products as amber diamonds (`#f59e0b`).
- **Edges**: Color-coded based on rating extremity (red `#ef4444` for extreme 1.0 or 5.0 ratings, cyan for neutral ratings).
- **Physics Engine**: ForceAtlas2-based layout allows users to zoom, drag, inspect nodes, and analyze topological clusters directly in the web browser.

---

## 2. Reviewer Risk Scoring Engine

Every reviewer is assigned a normalized **Risk Score ($0 - 100$)** based on 5 weighted behavioral pillars:

$$\text{Risk Score} = \min\Big(100, S_{\text{vol}} + S_{\text{rating}} + S_{\text{conc}} + S_{\text{dup}} + S_{\text{vel}}\Big)$$

| Pillar | Weight | Description & Methodology |
| :--- | :---: | :--- |
| **1. Review Volume ($S_{\text{vol}}$)** | 20 pts | Scaled proportional to user's total review volume relative to baseline thresholds ($S_{\text{vol}} = \min(20, N_{\text{reviews}} \times 3.5)$). |
| **2. Rating Extremity ($S_{\text{rating}}$)** | 25 pts | Evaluates mean rating deviation $|\mu_{\text{rating}} - 3.0| / 2.0$ and zero variance ($\sigma^2 = 0$) across reviews. |
| **3. Product Concentration ($S_{\text{conc}}$)** | 15 pts | Measures ratio of total reviews to unique products targeted ($N_{\text{reviews}} / N_{\text{products}}$). |
| **4. Text Duplication ($S_{\text{dup}}$)** | 25 pts | Computes pairwise TF-IDF cosine similarity matrix $S_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{v}_j}{\|\mathbf{v}_i\| \|\mathbf{v}_j\|}$ across user's reviews and global corpus. Cosine similarity $> 0.85$ triggers full 25 pt penalty. |
| **5. Temporal Velocity ($S_{\text{vel}}$)** | 15 pts | Evaluates minimum time delta $\Delta t_{\min}$ between consecutive posts by the same user. Intervals $< 2$ minutes indicate automated bot scripts. |

### Risk Level Thresholds
- **Critical**: $80 \le \text{Score} \le 100$ (Red badge)
- **High**: $60 \le \text{Score} < 80$ (Orange badge)
- **Medium**: $35 \le \text{Score} < 60$ (Blue badge)
- **Low**: $0 \le \text{Score} < 35$ (Green badge)

---

## 3. Coordinated Reviewer Detection (Louvain Algorithm)

### Community Detection
Coordinated review syndicates often act in groups to target specific product sets. To detect these groups:
1. A **User-Co-Review Projection Graph** $G_U$ is constructed where an edge exists between two users if they have reviewed the same products.
2. The **Louvain Community Detection Algorithm** optimizes network modularity $Q$:

$$Q = \frac{1}{2m} \sum_{i,j} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

3. Clusters are evaluated for **Suspicion Score**:

$$\text{Suspicion} = \min\left(100, \left(\frac{|\mu_{\text{cluster\_rating}} - 3.0|}{2.0} \times 50\right) + \min\left(50, \frac{N_{\text{users}}}{N_{\text{products}}} \times 15\right)\right)$$

Clusters with suspicion scores $\ge 60\%$ are flagged as coordinated fraud rings.

---

## 4. Temporal Fraud Burst Detection

### Time-Series Aggregation
Timestamps are aggregated into **Hourly**, **Daily**, and **Weekly** frequency bins to track review velocity.

### ML Anomaly Models
1. **Isolation Forest (`sklearn.ensemble.IsolationForest`)**:
   - Feature Matrix $X \in \mathbb{R}^{n \times 3}$: `[review_count, unique_users, unique_products]`.
   - Isolates anomalous time windows by recursively partitioning feature space.
2. **DBSCAN (`sklearn.cluster.DBSCAN`)**:
   - Density-based spatial clustering identifies dense time bursts ($\text{eps}=2.5, \text{min\_samples}=2$).
3. **Statistical Velocity Thresholding**:
   - Flags time windows where volume $V > \mu_{\text{volume}} + 1.5 \sigma_{\text{volume}}$.

Anomalous time windows are flagged with an **Anomaly Confidence Score** and highlighted in the interactive Plotly timeline and dashboard.

---

## 5. Future Roadmap: Integration with Explainable AI & Synthetic LLM Detection

The Phase 2 Fraud Intelligence Platform is architected for seamless integration with upcoming Phase 3 modules:

### A. Explainable AI (XAI) Integration (SHAP & LIME)
- **SHAP (SHapley Additive exPlanations)**: Will calculate token-level feature attribution values for individual TF-IDF + XGBoost text predictions.
- **LIME (Local Interpretable Model-agnostic Explanations)**: Will generate natural language visual explanations highlighting specific suspicious phrases (e.g. *"super fast shipping buy now"*) alongside the graph risk scores.

### B. AI-Generated Synthetic Review Detection
- Integration of perplexity scoring and token distribution entropy detectors to spot synthetic LLM-generated reviews (ChatGPT/Llama clones).
- Combined score: $\text{Final Fraud Score} = 0.4 \times \text{Text XGBoost} + 0.3 \times \text{Reviewer Risk Score} + 0.3 \times \text{LLM Synthetic Probability}$.
