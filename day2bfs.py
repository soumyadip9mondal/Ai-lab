import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
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

# BFS
start = "A"
visited = []
queue = deque([start])
order = []

while queue:
    node = queue.popleft()

    if node not in visited:
        visited.append(node)
        order.append(node)
        queue.extend(sorted(set(G.neighbors(node)) - set(visited)))

print("BFS Traversal:", order)

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
        font_size=14
    )

    ax.set_title(f"Breadth First Search\nVisited: {current_visited}")

# Create animation
ani = FuncAnimation(
    fig,
    update,
    frames=len(order),
    interval=1000,   
    repeat=False
)

plt.close(fig)


HTML(ani.to_jshtml())