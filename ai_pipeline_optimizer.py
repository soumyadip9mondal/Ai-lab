# ============================================================
# AI & SYSTEMS ENGINEERING CHALLENGE
# PREMIUM ANIMATED AI PIPELINE OPTIMIZER
#
# AND / OR GRAPH + AO* STYLE VISUALIZATION
# Google Colab Ready
# ============================================================

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML, display
from dataclasses import dataclass, field


# ============================================================
# 1. NODE MODEL
# ============================================================

@dataclass
class Node:

    name: str

    cost: float = 0

    heuristic: float = 0

    node_type: str = "OR"

    children: list = field(
        default_factory=list
    )


# ============================================================
# 2. GRAPH CLASS
# ============================================================

class AndOrGraph:

    def __init__(self):

        self.nodes = {}

    def add_node(
        self,
        name,
        cost=0,
        heuristic=0,
        node_type="OR"
    ):

        self.nodes[name] = Node(
            name=name,
            cost=cost,
            heuristic=heuristic,
            node_type=node_type
        )

    def connect(
        self,
        parent,
        children
    ):

        self.nodes[parent].children = children


# ============================================================
# 3. CREATE GRAPH
# ============================================================

graph = AndOrGraph()


# ---------------- GOAL ----------------

graph.add_node(
    "Process Pipeline",
    0,
    12,
    "OR"
)


# ---------------- STRATEGIES ----------------

graph.add_node(
    "Deep Learning",
    0,
    30,
    "AND"
)

graph.add_node(
    "Ensemble Methods",
    0,
    40,
    "AND"
)


# ---------------- DEEP LEARNING ----------------

graph.add_node(
    "Feature Extraction",
    0,
    6,
    "AND"
)

graph.add_node(
    "Conv Extractor",
    4,
    2,
    "AND"
)

graph.add_node(
    "Vision Transformer",
    6,
    3,
    "OR"
)


graph.add_node(
    "Architecture Optimization",
    0,
    4,
    "AND"
)

graph.add_node(
    "Grid Search",
    5,
    1,
    "AND"
)

graph.add_node(
    "Bayesian Tuning",
    3,
    0,
    "OR"
)


# ---------------- ENSEMBLE ----------------

graph.add_node(
    "Ensemble Construction",
    0,
    7,
    "AND"
)

graph.add_node(
    "Gradient Boosting",
    4,
    3,
    "AND"
)

graph.add_node(
    "Random Forest",
    3,
    2,
    "OR"
)

graph.add_node(
    "Model Evaluation",
    5,
    0,
    "OR"
)


# ============================================================
# 4. CONNECT GRAPH
# ============================================================

graph.connect(
    "Process Pipeline",
    [
        "Deep Learning",
        "Ensemble Methods"
    ]
)

graph.connect(
    "Deep Learning",
    [
        "Feature Extraction",
        "Architecture Optimization"
    ]
)

graph.connect(
    "Feature Extraction",
    [
        "Conv Extractor"
    ]
)

graph.connect(
    "Conv Extractor",
    [
        "Vision Transformer"
    ]
)

graph.connect(
    "Architecture Optimization",
    [
        "Grid Search"
    ]
)

graph.connect(
    "Grid Search",
    [
        "Bayesian Tuning"
    ]
)

graph.connect(
    "Ensemble Methods",
    [
        "Ensemble Construction",
        "Model Evaluation"
    ]
)

graph.connect(
    "Ensemble Construction",
    [
        "Gradient Boosting"
    ]
)

graph.connect(
    "Gradient Boosting",
    [
        "Random Forest"
    ]
)

graph.connect(
    "Model Evaluation",
    []
)


# ============================================================
# 5. SOLVER
# ============================================================

memo = {}

evaluation_order = []


def solve(node_name):

    if node_name in memo:

        return memo[node_name]


    node = graph.nodes[node_name]


    # --------------------------------------------------------
    # TERMINAL
    # --------------------------------------------------------

    if not node.children:

        result = {

            "cost": node.cost,

            "path": [node_name]
        }

        memo[node_name] = result

        evaluation_order.append(
            node_name
        )

        return result


    # --------------------------------------------------------
    # OR NODE
    # --------------------------------------------------------

    if node.node_type == "OR":

        candidates = []


        for child in node.children:

            result = solve(child)

            candidates.append(result)


        best = min(
            candidates,
            key=lambda x: x["cost"]
        )


        result = {

            "cost":
                node.cost +
                best["cost"],

            "path":
                [node_name] +
                best["path"]
        }


    # --------------------------------------------------------
    # AND NODE
    # --------------------------------------------------------

    else:

        total = node.cost

        path = [node_name]


        for child in node.children:

            result_child = solve(child)

            total += result_child["cost"]

            path.extend(
                result_child["path"]
            )


        result = {

            "cost": total,

            "path": path
        }


    memo[node_name] = result

    evaluation_order.append(
        node_name
    )

    return result


# ============================================================
# 6. SOLVE
# ============================================================

final_result = solve(
    "Process Pipeline"
)

deep_result = solve(
    "Deep Learning"
)

ensemble_result = solve(
    "Ensemble Methods"
)


# ============================================================
# 7. PRINT RESULT
# ============================================================

print("=" * 70)

print(
    "AI PIPELINE OPTIMIZATION RESULT"
)

print("=" * 70)

print()

print(
    f"Deep Learning Cost : "
    f"{deep_result['cost']}"
)

print(
    f"Ensemble Cost      : "
    f"{ensemble_result['cost']}"
)

print()

print(
    "Optimal Strategy   : "
    "Ensemble Methods"
)

print(
    f"Minimum Cost       : "
    f"{final_result['cost']}"
)

print()

print(
    "Optimal Sequence:"
)

for i, node in enumerate(
    final_result["path"],
    1
):

    print(
        f"{i}. {node}"
    )


# ============================================================
# 8. PREMIUM COLOR PALETTE
# ============================================================

BG = "#08111F"

PANEL = "#101D30"

PANEL2 = "#14253D"

WHITE = "#F4F7FB"

MUTED = "#8FA3BF"

BLUE = "#38BDF8"

PURPLE = "#8B5CF6"

GREEN = "#22C55E"

ORANGE = "#F59E0B"

RED = "#EF4444"

BORDER = "#243854"


# ============================================================
# 9. NODE POSITIONS
# ============================================================

positions = {

    "Process Pipeline":
        (0, 6.0),

    "Deep Learning":
        (-3.8, 4.7),

    "Ensemble Methods":
        (3.8, 4.7),


    "Feature Extraction":
        (-5.4, 3.2),

    "Architecture Optimization":
        (-2.0, 3.2),


    "Ensemble Construction":
        (2.0, 3.2),

    "Model Evaluation":
        (5.4, 3.2),


    "Conv Extractor":
        (-5.4, 1.65),

    "Grid Search":
        (-2.0, 1.65),

    "Gradient Boosting":
        (2.0, 1.65),


    "Vision Transformer":
        (-5.4, 0.1),

    "Bayesian Tuning":
        (-2.0, 0.1),

    "Random Forest":
        (2.0, 0.1)
}


# ============================================================
# 10. EDGES
# ============================================================

edges = [

    ("Process Pipeline", "Deep Learning"),

    ("Process Pipeline", "Ensemble Methods"),

    ("Deep Learning", "Feature Extraction"),

    ("Deep Learning", "Architecture Optimization"),

    ("Feature Extraction", "Conv Extractor"),

    ("Conv Extractor", "Vision Transformer"),

    ("Architecture Optimization", "Grid Search"),

    ("Grid Search", "Bayesian Tuning"),

    ("Ensemble Methods", "Ensemble Construction"),

    ("Ensemble Methods", "Model Evaluation"),

    ("Ensemble Construction", "Gradient Boosting"),

    ("Gradient Boosting", "Random Forest")
]


# ============================================================
# 11. FIGURE
# ============================================================

fig = plt.figure(
    figsize=(18, 10),

    facecolor=BG
)


# Main graph area

ax = fig.add_axes(
    [0.03, 0.08, 0.70, 0.84]
)

ax.set_facecolor(BG)


# Right dashboard panel

panel = fig.add_axes(
    [0.755, 0.08, 0.225, 0.84]
)

panel.set_facecolor(PANEL)

panel.set_xticks([])

panel.set_yticks([])


# ============================================================
# 12. DRAW DASHBOARD
# ============================================================

def draw_dashboard(
    frame
):

    ax.clear()

    panel.clear()


    ax.set_facecolor(BG)

    panel.set_facecolor(PANEL)


    # ========================================================
    # HEADER
    # ========================================================

    fig.text(

        0.04,

        0.965,

        "AI PIPELINE OPTIMIZER",

        fontsize=22,

        fontweight="bold",

        color=WHITE
    )


    fig.text(

        0.04,

        0.935,

        "Animated AND / OR Search • Minimum-Cost Model Training",

        fontsize=10,

        color=MUTED
    )


    # ========================================================
    # CURRENT STATUS
    # ========================================================

    if frame < len(evaluation_order):

        current = evaluation_order[frame]

        status = (
            "●  SEARCHING"
        )

        status_color = ORANGE

    else:

        current = None

        status = (
            "●  OPTIMIZATION COMPLETE"
        )

        status_color = GREEN


    fig.text(

        0.67,

        0.952,

        status,

        fontsize=10,

        color=status_color,

        fontweight="bold",

        ha="right"
    )


    # ========================================================
    # EDGES
    # ========================================================

    for parent, child in edges:

        x1, y1 = positions[parent]

        x2, y2 = positions[child]


        # Determine if edge is on final path

        final_path = final_result["path"]


        is_final_edge = (

            parent in final_path

            and child in final_path

        )


        if is_final_edge and frame >= len(
            evaluation_order
        ):

            line_color = GREEN

            width = 4

        else:

            line_color = BORDER

            width = 1.5


        ax.plot(

            [x1, x2],

            [y1, y2],

            color=line_color,

            linewidth=width,

            zorder=1
        )


    # ========================================================
    # EVALUATED NODES
    # ========================================================

    evaluated = set(

        evaluation_order[
            :frame + 1
        ]
    )


    final_path = set(
        final_result["path"]
    )


    # ========================================================
    # DRAW NODES
    # ========================================================

    for name, node in graph.nodes.items():

        x, y = positions[name]


        # -----------------------------------------------
        # NODE STATE
        # -----------------------------------------------

        if (

            frame >= len(evaluation_order)

            and

            name in final_path
        ):

            face = "#123A27"

            edge = GREEN

            text_color = WHITE

            width = 2.8


        elif name == current:

            face = "#3A2B0A"

            edge = ORANGE

            text_color = WHITE

            width = 3


        elif name in evaluated:

            face = "#122D45"

            edge = BLUE

            text_color = WHITE

            width = 2


        else:

            face = PANEL

            edge = BORDER

            text_color = MUTED

            width = 1.2


        # -----------------------------------------------
        # NODE BOX
        # -----------------------------------------------

        box = plt.Rectangle(

            (x - 0.95, y - 0.42),

            1.9,

            0.84,

            facecolor=face,

            edgecolor=edge,

            linewidth=width,

            joinstyle="round",

            zorder=3
        )


        ax.add_patch(box)


        # -----------------------------------------------
        # NODE NAME
        # -----------------------------------------------

        short_name = name


        if len(short_name) > 22:

            words = short_name.split()

            if len(words) > 1:

                middle = len(words) // 2

                short_name = (

                    " ".join(
                        words[:middle]
                    )

                    + "\n"

                    +

                    " ".join(
                        words[middle:]
                    )
                )


        # -----------------------------------------------
        # COST INFO
        # -----------------------------------------------

        if name in memo:

            g = memo[name]["cost"]

            f = g + node.heuristic

            info = (

                f"C {node.cost:.0f}   "

                f"h {node.heuristic:.0f}   "

                f"f {f:.0f}"
            )

        else:

            info = (

                f"C {node.cost:.0f}   "

                f"h {node.heuristic:.0f}"
            )


        ax.text(

            x,

            y + 0.12,

            short_name,

            ha="center",

            va="center",

            fontsize=8.5,

            color=text_color,

            fontweight="bold",

            zorder=4
        )


        ax.text(

            x,

            y - 0.20,

            info,

            ha="center",

            va="center",

            fontsize=6.5,

            color=MUTED,

            zorder=4
        )


    # ========================================================
    # AND / OR LABELS
    # ========================================================

    labels = [

        (
            -3.8,
            4.08,
            "AND",
            PURPLE
        ),

        (
            3.8,
            4.08,
            "AND",
            PURPLE
        ),

        (
            0,
            5.30,
            "OR",
            BLUE
        )
    ]


    for x, y, text, color in labels:

        ax.text(

            x,

            y,

            text,

            ha="center",

            fontsize=8,

            color=color,

            fontweight="bold"
        )


    # ========================================================
    # GRAPH LABEL
    # ========================================================

    ax.text(

        -6.6,

        6.55,

        "MODEL TRAINING SEARCH GRAPH",

        fontsize=10,

        color=MUTED,

        fontweight="bold"
    )


    # ========================================================
    # RIGHT PANEL
    # ========================================================

    panel.axis("off")


    # --------------------------------------------------------
    # PANEL TITLE
    # --------------------------------------------------------

    panel.text(

        0.08,

        0.95,

        "LIVE OPTIMIZATION",

        fontsize=14,

        color=WHITE,

        fontweight="bold",

        transform=panel.transAxes
    )


    panel.text(

        0.08,

        0.915,

        "AO* / AND-OR SEARCH",

        fontsize=8,

        color=MUTED,

        transform=panel.transAxes
    )


    # --------------------------------------------------------
    # COST CARD
    # --------------------------------------------------------

    panel.text(

        0.08,

        0.85,

        "CURRENT COST",

        fontsize=8,

        color=MUTED,

        transform=panel.transAxes
    )


    if current and current in memo:

        current_cost = memo[current]["cost"]

    else:

        current_cost = 0


    panel.text(

        0.08,

        0.805,

        f"{current_cost:.0f}",

        fontsize=28,

        color=ORANGE,

        fontweight="bold",

        transform=panel.transAxes
    )


    # --------------------------------------------------------
    # STRATEGY COMPARISON
    # --------------------------------------------------------

    panel.text(

        0.08,

        0.70,

        "STRATEGY COMPARISON",

        fontsize=9,

        color=WHITE,

        fontweight="bold",

        transform=panel.transAxes
    )


    # Deep Learning

    panel.text(

        0.08,

        0.655,

        "Deep Learning",

        fontsize=9,

        color=MUTED,

        transform=panel.transAxes
    )


    panel.text(

        0.86,

        0.655,

        f"{deep_result['cost']:.0f}",

        fontsize=11,

        color=WHITE,

        fontweight="bold",

        ha="right",

        transform=panel.transAxes
    )


    panel.add_patch(

        plt.Rectangle(

            (0.08, 0.625),

            0.78,

            0.012,

            transform=panel.transAxes,

            facecolor="#263B55",

            edgecolor="none"
        )
    )


    panel.add_patch(

        plt.Rectangle(

            (0.08, 0.625),

            0.78 *

            deep_result["cost"] / 20,

            0.012,

            transform=panel.transAxes,

            facecolor=PURPLE,

            edgecolor="none"
        )
    )


    # Ensemble

    panel.text(

        0.08,

        0.575,

        "Ensemble Methods",

        fontsize=9,

        color=MUTED,

        transform=panel.transAxes
    )


    panel.text(

        0.86,

        0.575,

        f"{ensemble_result['cost']:.0f}",

        fontsize=11,

        color=WHITE,

        fontweight="bold",

        ha="right",

        transform=panel.transAxes
    )


    panel.add_patch(

        plt.Rectangle(

            (0.08, 0.545),

            0.78,

            0.012,

            transform=panel.transAxes,

            facecolor="#263B55",

            edgecolor="none"
        )
    )


    panel.add_patch(

        plt.Rectangle(

            (0.08, 0.545),

            0.78 *

            ensemble_result["cost"] / 20,

            0.012,

            transform=panel.transAxes,

            facecolor=GREEN,

            edgecolor="none"
        )
    )


    # --------------------------------------------------------
    # FORMULA
    # --------------------------------------------------------

    panel.text(

        0.08,

        0.46,

        "SEARCH FUNCTION",

        fontsize=9,

        color=WHITE,

        fontweight="bold",

        transform=panel.transAxes
    )


    panel.text(

        0.08,

        0.415,

        "f(n) = g(n) + h(n)",

        fontsize=13,

        color=BLUE,

        fontweight="bold",

        transform=panel.transAxes
    )


    panel.text(

        0.08,

        0.37,

        "g(n)  Actual cost",

        fontsize=8,

        color=MUTED,

        transform=panel.transAxes
    )


    panel.text(

        0.08,

        0.335,

        "h(n)  Heuristic",

        fontsize=8,

        color=MUTED,

        transform=panel.transAxes
    )


    panel.text(

        0.08,

        0.30,

        "f(n)  Estimated cost",

        fontsize=8,

        color=MUTED,

        transform=panel.transAxes
    )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    progress = min(

        frame /

        max(
            1,
            len(evaluation_order)
        ),

        1
    )


    panel.text(

        0.08,

        0.22,

        "SEARCH PROGRESS",

        fontsize=9,

        color=WHITE,

        fontweight="bold",

        transform=panel.transAxes
    )


    panel.add_patch(

        plt.Rectangle(

            (0.08, 0.18),

            0.78,

            0.025,

            transform=panel.transAxes,

            facecolor="#263B55",

            edgecolor="none"
        )
    )


    panel.add_patch(

        plt.Rectangle(

            (0.08, 0.18),

            0.78 * progress,

            0.025,

            transform=panel.transAxes,

            facecolor=BLUE,

            edgecolor="none"
        )
    )


    panel.text(

        0.08,

        0.135,

        f"Nodes evaluated: {min(frame + 1, len(evaluation_order))}",

        fontsize=8,

        color=MUTED,

        transform=panel.transAxes
    )


    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    if frame >= len(evaluation_order):

        panel.text(

            0.08,

            0.075,

            "✓ OPTIMAL SOLUTION",

            fontsize=10,

            color=GREEN,

            fontweight="bold",

            transform=panel.transAxes
        )


        panel.text(

            0.08,

            0.04,

            "Ensemble Methods • Cost 12",

            fontsize=9,

            color=WHITE,

            transform=panel.transAxes
        )


    # ========================================================
    # AXIS SETTINGS
    # ========================================================

    ax.set_xlim(
        -6.8,
        6.8
    )

    ax.set_ylim(
        -0.7,
        6.8
    )

    ax.axis("off")


# ============================================================
# 13. ANIMATION
# ============================================================

total_frames = (

    len(evaluation_order) + 8
)


animation = FuncAnimation(

    fig,

    draw_dashboard,

    frames=total_frames,

    interval=900,

    repeat=False
)


# ============================================================
# 14. DISPLAY
# ============================================================

plt.close(fig)

display(

    HTML(
        animation.to_jshtml()
    )
)
