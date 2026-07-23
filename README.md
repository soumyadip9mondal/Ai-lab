# AI Lab Daily Tasks

Welcome to the **AI Lab** repository! This project contains daily tasks, experiments, and implementations of various Artificial Intelligence algorithms and concepts.

---

## 🛠️ Prerequisites & Setup

To run the scripts in this repository, you'll need Python installed along with some specific libraries.

**Dependencies:**
- `networkx` (for graph data structures and algorithms)
- `matplotlib` (for plotting and rendering visualizations)
- `IPython` (for displaying animations in notebook environments like Jupyter/Colab)

You can install these dependencies using `pip`:
```bash
pip install networkx matplotlib ipython
```

---

## 📅 Daily Task Breakdown

### Day 2: Graph Traversals & Visualizations

The objective for Day 2 is to understand and implement fundamental graph traversal algorithms. We are building step-by-step visualizers to see exactly how these algorithms explore nodes.

* **Breadth-First Search (BFS)**
  * **File:** `day2.py`
  * **Description:** This script builds a sample graph using `networkx` and implements a classic BFS algorithm using a queue (`collections.deque`).
  * **Visualization:** It utilizes `matplotlib.animation.FuncAnimation` to generate an animation that highlights nodes in red as they are visited during the BFS traversal. It also exports the animation to HTML so it can be played back dynamically.

* **Depth-First Search (DFS)** *(Work in Progress)*
  * **File:** `day2dfs.py`
  * **Description:** Implementation for traversing the graph using a depth-first approach (currently being set up).

---

## 🚀 How to Run

You can run the Python scripts directly from your terminal:

```bash
python day2.py
```
*(Note: Visualizations may require a GUI backend or running within a Jupyter Notebook / Google Colab environment to view the generated JavaScript HTML animation).*
