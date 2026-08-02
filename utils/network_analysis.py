"""Reviewer Network Analysis & Community Detection Module.

Uses NetworkX for bipartite graph representation of User-Product interactions,
Louvain algorithm for coordinated community detection, and PyVis for interactive visualization.
"""

from typing import Any
import networkx as nx
import pandas as pd
from pyvis.network import Network


class ReviewNetworkAnalyzer:
    """Analyzes reviewer-product graph structure and detects coordinated communities."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.graph = nx.Graph()
        self.communities: list[set] = []
        self.cluster_summaries: list[dict[str, Any]] = []
        self._build_graph()
        self._detect_communities()

    def _build_graph(self) -> None:
        """Build bipartite NetworkX graph from reviews dataframe."""
        self.graph.clear()

        for _, row in self.df.iterrows():
            user_id = str(row["user_id"])
            prod_id = str(row["product_id"])
            rating = float(row.get("rating", 3.0))

            # Add node types
            if not self.graph.has_node(user_id):
                self.graph.add_node(user_id, node_type="user", label=user_id)
            if not self.graph.has_node(prod_id):
                self.graph.add_node(prod_id, node_type="product", label=prod_id)

            # Add edge or update weight/rating attributes
            if self.graph.has_edge(user_id, prod_id):
                self.graph[user_id][prod_id]["weight"] += 1
            else:
                self.graph.add_edge(user_id, prod_id, weight=1, rating=rating)

    def _detect_communities(self) -> None:
        """Detect reviewer communities using Louvain Algorithm on projected user graph."""
        users = [n for n, d in self.graph.nodes(data=True) if d.get("node_type") == "user"]
        if not users:
            self.communities = []
            return

        # Project bipartite graph onto user nodes to identify co-reviewing networks
        user_graph = nx.bipartite.weighted_projected_graph(self.graph, users)

        if len(user_graph.nodes) > 0:
            try:
                # Use NetworkX built-in Louvain community detection
                self.communities = list(nx.community.louvain_communities(user_graph, random_state=42))
            except Exception:
                # Fallback to connected components if Louvain fails or graph is trivial
                self.communities = list(nx.connected_components(user_graph))
        else:
            self.communities = []

        self._build_cluster_summaries()

    def _build_cluster_summaries(self) -> None:
        """Calculate summary statistics for each detected community cluster."""
        self.cluster_summaries = []

        for idx, comm in enumerate(self.communities, start=1):
            users_in_comm = list(comm)
            comm_reviews = self.df[self.df["user_id"].isin(users_in_comm)]

            if comm_reviews.empty:
                continue

            targeted_products = comm_reviews["product_id"].unique().tolist()
            avg_rating = round(float(comm_reviews["rating"].mean()), 2)
            review_count = len(comm_reviews)
            user_count = len(users_in_comm)
            prod_count = len(targeted_products)

            # Suspicion calculation: extreme rating bias + high density of users per product
            rating_bias = abs(avg_rating - 3.0) / 2.0  # 0 to 1 scale
            product_concentration = user_count / max(prod_count, 1)
            raw_suspicion = (rating_bias * 50) + min(50, product_concentration * 15)
            suspicion_score = min(100.0, round(raw_suspicion, 1))

            self.cluster_summaries.append(
                {
                    "cluster_id": f"CLUSTER_{idx:02d}",
                    "user_count": user_count,
                    "product_count": prod_count,
                    "avg_rating": avg_rating,
                    "suspicion_score": suspicion_score,
                    "flagged": suspicion_score >= 60.0,
                    "members": users_in_comm,
                    "targeted_products": targeted_products,
                }
            )

        # Sort clusters by suspicion score descending
        self.cluster_summaries.sort(key=lambda c: c["suspicion_score"], reverse=True)

    def generate_pyvis_html(self) -> str:
        """Generate interactive HTML visualizer using PyVis with cybersecurity styling."""
        net = Network(height="550px", width="100%", bgcolor="#0b1120", font_color="#e2e8f0", notebook=False)

        # PyVis options for smooth dark UI physics
        net.options.physics.enabled = True
        net.options.physics.solver = "forceAtlas2Based"

        # Color maps for community clusters
        community_colors = [
            "#ef4444", "#06b6d4", "#a855f7", "#3b82f6",
            "#10b981", "#f59e0b", "#ec4899", "#8b5cf6"
        ]

        # Map user -> community index
        user_to_cluster: dict[str, int] = {}
        for comm_idx, comm in enumerate(self.communities):
            for u in comm:
                user_to_cluster[u] = comm_idx

        # Add nodes with styled aesthetics
        for node, data in self.graph.nodes(data=True):
            node_type = data.get("node_type", "user")

            if node_type == "user":
                comm_idx = user_to_cluster.get(node, 0)
                color = community_colors[comm_idx % len(community_colors)]
                net.add_node(
                    node,
                    label=f"User: {node}",
                    color=color,
                    shape="dot",
                    size=18,
                    title=f"User ID: {node}<br>Cluster: #{comm_idx+1}"
                )
            else:
                net.add_node(
                    node,
                    label=f"Prod: {node}",
                    color="#f59e0b",
                    shape="diamond",
                    size=24,
                    title=f"Product ID: {node}"
                )

        # Add edges
        for u, v, data in self.graph.edges(data=True):
            rating = data.get("rating", 3.0)
            edge_color = "#ef4444" if rating == 1.0 or rating == 5.0 else "#38bdf8"
            net.add_edge(u, v, color=edge_color, width=1.5, title=f"Rating: {rating} ★")

        return net.generate_html()

    def get_summary(self) -> dict[str, Any]:
        """Return structured json payload for API consumers."""
        total_users = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "user")
        total_prods = sum(1 for _, d in self.graph.nodes(data=True) if d.get("node_type") == "product")

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_users": total_users,
            "total_products": total_prods,
            "community_count": len(self.communities),
            "clusters": self.cluster_summaries,
        }
