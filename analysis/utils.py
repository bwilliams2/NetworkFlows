from functools import reduce
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict
from ortools.sat.python import cp_model
import networkx as nx
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import plotly.graph_objects as go


@dataclass
class ProcessStep:
    """Container class for network flow process steps"""

    id: int
    send_from_cnt: int
    to_processing_cnt: int
    for_process: int
    week: int
    amount: int
    treatment: str
    product: str
    is_used: any = None
    is_used_by_node: dict[int, any] = None
    contrib_amounts: list[any] = None

    def dict(self):
        return dict(
            id=self.id,
            amount=self.amount,
            week=self.week,
            send_from_cnt=self.send_from_cnt,
            to_processing_cnt=self.to_processing_cnt,
            for_process=self.for_process,
            product=self.product,
            treatment=self.treatment,
        )


class NetworkFlowModel:
    PROCESS_FLOW = ("Sourcing", "Conditioning", "Treatment", "Forwarding", "Delivery")

    def __init__(self, input_data: pd.DataFrame):
        """Generate NetworkModel from input data

        Args:
            input_data (pd.DataFrame): DataFrame describing historic process steps
        """
        self._input_data = input_data
        self._create_encoded_data()
        self._create_process_steps()
        self._steps_by_process = None

    def _create_encoded_data(self):
        """If necessary, create scaled data from input data."""
        encoded_data = self._input_data.copy()
        # Check for decimal values and convert to integers
        if encoded_data["Amount"].dtype == "float64":
            # Assume that decimal values have two significant digits and make integer values
            self._scale = 10**2
            encoded_data["Amount"] = (
                (encoded_data["Amount"] * self._scale).round().astype(int)
            )
        else:
            self._scale = 1

        self._encoded_data = encoded_data

    def _create_process_steps(self):
        """Create ProcessStep objects from input data"""
        process_steps = []
        data = self._encoded_data

        for index, row in data.iterrows():
            process_steps.append(
                ProcessStep(
                    id=index,
                    send_from_cnt=row["send_from_cnt"],
                    to_processing_cnt=row["to_processing_cnt"],
                    for_process=row["for_process"],
                    week=row["Week"],
                    amount=row["Amount"],
                    product=row["product"],
                    treatment=row["treatment"],
                    is_used_by_node={},
                    contrib_amounts={},
                )
            )
        self._process_steps = process_steps

    def _create_upstream_constraints(
        self,
        model: cp_model.CpModel,
        current_nodes: list[ProcessStep],
        upstream_nodes: list[ProcessStep],
    ):
        """Create constraints for upstream process steps

        Args:
            model (cp_model.CpModel): Model to add constraints to
            current_nodes (list[ProcessStep]): List of current process steps
            upstream_nodes (list[ProcessStep]): List of upstream process steps
        """
        for n, current_node in enumerate(current_nodes):
            node_is_used = current_node.is_used
            contrib_amounts = []
            for m, up_node in enumerate(upstream_nodes):
                # Boolean for if upstream node is used by current node
                up_node_uses_node = model.NewBoolVar(
                    f"up_node_{up_node.id}_uses_node_{current_node.id}"
                )

                # Amount of upstream node that is used by current node
                up_node_contrib = model.NewIntVar(
                    0,
                    up_node.amount,
                    f"up_node_{up_node.id}_contribution_to_node_{current_node.id}",
                )

                # Upstream node must be before or equal to current node week
                up_node_allowed = model.NewBoolVar(
                    f"up_node_{up_node.id}_allowed_for_node_{current_node.id}"
                )

                # Check that upstream process occurs before current process
                model.Add(up_node.week <= current_node.week).OnlyEnforceIf(
                    up_node_allowed
                )
                model.Add(up_node.week > current_node.week).OnlyEnforceIf(
                    up_node_allowed.Not()
                )

                # If upstream process is after current process, it cannot be used
                model.AddImplication(up_node_allowed.Not(), up_node_uses_node.Not())

                # Set contribution constraints for upstream process into current process
                model.Add(up_node_contrib >= 0).OnlyEnforceIf(
                    [up_node_uses_node, node_is_used]
                )
                model.Add(up_node_contrib == 0).OnlyEnforceIf(
                    [up_node_uses_node.Not(), node_is_used]
                )

                # Collect contribution amounts and save use booleans to ProcessStep for up_node
                contrib_amounts.append(up_node_contrib)
                up_node.is_used_by_node[current_node.id] = up_node_uses_node
                up_node.contrib_amounts[current_node.id] = up_node_contrib

            # Require process amount to equal all upstream contributions
            model.Add(current_node.amount == sum(contrib_amounts)).OnlyEnforceIf(
                node_is_used
            )

        for node in upstream_nodes:
            contrib_amounts = list(node.contrib_amounts.values())
            used_by_values = list(node.is_used_by_node.values())
            # Node is_used if any current nodes use it
            model.Add(sum(used_by_values) > 0).OnlyEnforceIf(node.is_used)
            model.Add(sum(used_by_values) == 0).OnlyEnforceIf(node.is_used.Not())
            # Upstream nodes cannot contribute more than their amount
            model.Add(node.amount >= sum(contrib_amounts))

    def solve(self):
        """Setup and solve network flow model"""

        processes = defaultdict(list)
        model = cp_model.CpModel()

        # Sort steps into process type and initialize is_used
        for step in self._process_steps:
            if step.for_process == self.PROCESS_FLOW[-1]:
                step.is_used = 1
            else:
                step.is_used = model.NewBoolVar(f"step_{step.id}_is_used")
            processes[step.for_process].append(step)

        self._steps_by_process = processes

        # For each demand, find upstream nodes until sourcing is reached
        for n in range(len(self.PROCESS_FLOW) - 1, 0, -1):
            current_nodes = processes[self.PROCESS_FLOW[n]]
            upstream_nodes = processes[self.PROCESS_FLOW[n - 1]]
            self._create_upstream_constraints(model, current_nodes, upstream_nodes)

        # Minimize number of node uses to generate simplified network flows
        num_used = []
        for step in self._process_steps:
            num_used += list(step.is_used_by_node.values())
        model.Minimize(sum(num_used))

        solver = cp_model.CpSolver()
        solver_status = solver.Solve(model)
        return NetworkFlowSolution(self, solver)


class NetworkFlowSolution:
    def __init__(
        self,
        model: NetworkFlowModel,
        solver: cp_model.CpSolver,
    ):
        """Provides data and network visualization for a NetworkFlowModel solution.

        Args:
            model (NetworkFlowModel): _description_
            solver (cp_model.CpSolver): _description_
        """
        self._solver = solver
        self._model = model
        self._network = None
        self._demands = None
        self._create_network()
        self._create_demands()

    def _create_network(self):
        """Generate NetworkX graph network from model solution"""
        model = self._model
        sol = self._solver

        G = nx.DiGraph()
        scale = model._scale
        steps_by_process = model._steps_by_process

        input_data = model._input_data
        max_amount = input_data["Amount"].max()
        min_amount = input_data["Amount"].min()
        min_weight = 1.0
        max_weight = 5.0

        # Scale weight based on amount compared to maximum
        def scale_weight(amount):
            return (amount - min_amount) / (max_amount - min_amount) * (
                max_weight - min_weight
            ) + min_weight

        x_pos = {
            "Delivery": 1,
            "Forwarding": 0.75,
            "Treatment": 0.5,
            "Conditioning": 0.25,
            "Sourcing": 0,
        }
        edges = []
        for i, (process, steps) in enumerate(steps_by_process.items()):
            separation = 1 / len(steps)
            node_ids = []
            for n, step in enumerate(steps):
                step_properties = step.dict()
                step_properties["amount"] = step_properties["amount"] / scale
                G.add_node(
                    step.id,
                    label=step.id,
                    is_used=sol.Value(step.is_used),
                    **step_properties,
                    pos=(x_pos[process], n * separation),
                )
                node_ids.append(step.id)
                for upstream_id, amount in step.contrib_amounts.items():
                    amount = sol.Value(amount) / scale
                    if amount > 0:
                        edges.append(
                            (
                                step.id,
                                upstream_id,
                                {"amount": amount, "weight": scale_weight(amount)},
                            )
                        )

        G.add_edges_from(edges)
        self._network = G

    def _create_demands(self):
        """Generate demand data from model solution"""
        model = self._model
        G = self._network
        rows = []

        for delivery_node in model._steps_by_process["Delivery"]:
            for source_node in model._steps_by_process["Sourcing"]:
                for path in nx.all_simple_paths(G, source_node.id, delivery_node.id):
                    if len(path) > 5:
                        pass
                    row = {}
                    for n, X in enumerate(path):
                        node = G.nodes[X]
                        prefix = f"Process{n+1}_"
                        if n < len(path) - 1:
                            process_amount = G.edges[(X, path[n + 1])]["amount"]
                        else:
                            process_amount = node["amount"]
                        row[prefix + "id"] = X
                        row[prefix[:-1]] = node["for_process"]
                        row[prefix + "Cnt"] = node["to_processing_cnt"]
                        row[prefix + "Amount"] = process_amount
                        row[prefix + "Week"] = node["week"]

                    rows.append(row)
        demands = pd.DataFrame(rows)
        id_columns = [col for col in demands.columns[::-1] if col.endswith("id")]
        demands = demands.sort_values(id_columns)

        # Current process amounts represent cumulative node steps rather than demand specific amounts
        # In each row, fill all amounts to the left of the minimum amount with the minimum amount
        amount_cols = [col for col in demands.columns if col.endswith("Amount")]
        for n, row in demands.iterrows():
            minimum_amount = row[amount_cols].astype(float).min()
            minimum_arg = row[amount_cols].astype(float).argmin()
            demands.loc[n, amount_cols[:minimum_arg]] = minimum_amount

        # Remove entries on duplicates from branched paths
        # Branch paths have duplicate process step columns so shifting by one will identify
        # duplicate processes.
        shifted_mask = demands == demands.shift(1)
        for n in range(len(id_columns), 1, -1):
            columns = [col for col in demands.columns if col.startswith(f"Process{n}")]
            demands.loc[shifted_mask[columns].all(axis=1), columns] = np.NaN

        self._demands = demands

    @property
    def demands(self):
        return self._demands.copy()

    def visualize(self):
        """Visualize network flow solution using Matplotlib"""
        G = self._network
        colors = sns.color_palette()
        fig, ax = plt.subplots(figsize=(10, 10))

        legend_elements = []
        process_nodes = {}
        node_positions = {k: node["pos"] for k, node in G.nodes.items()}
        for n, (process, steps) in enumerate(self._model._steps_by_process.items()):
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=colors[n],
                    lw=0,
                    label=process,
                    markersize=10,
                )
            )
            nodes = [step.id for step in steps]
            nx.draw_networkx_nodes(
                G, node_positions, nodelist=nodes, node_size=600, node_color=[colors[n]]
            )
            nx.draw_networkx_labels(
                G,
                node_positions,
                dict(zip(nodes, nodes)),
                font_size=14,
                font_color="whitesmoke",
            )
        weight = [edge["weight"] for edge in G.edges.values()]
        nx.draw_networkx_edges(G, node_positions, width=weight)
        ax.legend(handles=legend_elements)

    def ivisualize(self, ret: bool = False):
        """Visualize the network graph using Plotly

        Args:
            ret (bool, optional): If True, Plotly Figure is returned. Defaults to False.

        Returns:
            Figure: Plotly figure
        """
        G = self._network
        fig = go.Figure()

        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            plot_bgcolor="white",
        )

        # Add edges to the plotly figure
        for k, v in G.edges.items():
            start_node = G.nodes[k[0]]
            end_node = G.nodes[k[1]]
            x = [start_node["pos"][0], end_node["pos"][0]]
            y = [start_node["pos"][1], end_node["pos"][1]]
            fig.add_trace(
                go.Scatter(
                    x=[*x, None],
                    y=[*y, None],
                    mode="lines",
                    line=dict(color="black", width=v["weight"]),
                )
            )
            xn = (x[0] + x[1]) / 2.0
            yn = (y[0] + y[1]) / 2.0
            fig.add_trace(
                go.Scatter(
                    x=[xn],
                    y=[yn],
                    mode="markers",
                    text=[f"Amount: {v['amount']}"],
                    hovertemplate="%{text}",
                    marker=dict(
                        opacity=0,
                        size=20,
                        color="black",
                    ),
                )
            )

        colors = [
            "#1f77b4",  # muted blue
            "#ff7f0e",  # safety orange
            "#2ca02c",  # cooked asparagus green
            "#d62728",  # brick red
            "#9467bd",  # muted purple
        ]

        for n, (process, steps) in enumerate(self._model._steps_by_process.items()):
            x = []
            y = []
            text = []
            for i, step in enumerate(steps):
                X = step.id
                node = G.nodes[X]
                text = f"Id: {X}<br>Process: {node['for_process']}<br>Location: {node['to_processing_cnt']}<br>Amount: {node['amount']}<br>Week: {node['week']}"
                fig.add_trace(
                    go.Scatter(
                        x=[node["pos"][0]],
                        y=[node["pos"][1]],
                        mode="markers",
                        marker=dict(size=25, color=colors[n]),
                        text=[text],
                        hovertemplate="%{text}",
                    )
                )

        # fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        fig.update_layout(
            width=900,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(
                tickmode="array",
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=[
                    "Sourcing",
                    "Conditioning",
                    "Treatment",
                    "Forwarding",
                    "Delivery",
                ],
                tickfont=dict(size=20),
            ),
            yaxis=dict(visible=False),
        )

        # Display the network graph
        if ret:
            return fig
        else:
            fig.show()
