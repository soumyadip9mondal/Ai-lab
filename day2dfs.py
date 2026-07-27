import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Create Graph
G = nx.Graph()

edges = [
    ("A", "B"), ("A", "C"),
    ("B", "D"), ("B", "E"),
    ("C", "F"), ("C", "G"),
    ("E", "H"),
]

G.add_edges_from(edges)

# DFS
start = "A"
visited = []
stack = [start]
order = []

while stack:
    node = stack.pop()

    if node not in visited:
        visited.append(node)
        order.append(node)
        stack.extend(sorted(set(G.neighbors(node)) - set(visited), reverse=True))

print("DFS Traversal:", order)

# Fixed layout
pos = nx.spring_layout(G, seed=42)

# Create figure
fig, ax = plt.subplots(figsize=(6, 5))

# Animation function
def update(frame):
    ax.clear()

    current_visited = order[:frame + 1]

    node_colours = [
        "red" if node in current_visited else "white"
        for node in G.nodes()
    ]

    nx.draw(
        G,
        pos=pos,
        ax=ax,
        with_labels=True,
        node_color=node_colours,
        edgecolors="black",
        node_size=1000,
        font_size=12
    )

    ax.set_title(f"Depth First Search\nVisited: {current_visited}")

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(order),
    interval=1000,   # 1 second per node
    repeat=False
)

plt.close(fig)

# Display animation in Google Colab
HTML(ani.to_jshtml())