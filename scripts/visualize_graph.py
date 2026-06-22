"""
Generate results/graph.png - the required visualization of the complete
LangGraph graph (Feature 4 deliverable).

Run: python scripts/visualize_graph.py

Requires pygraphviz (and the system graphviz package) for draw_png().
If pygraphviz isn't available in your environment, install graphviz first:
    Ubuntu/Debian: sudo apt-get install graphviz graphviz-dev
    macOS:         brew install graphviz
    then:          pip install pygraphviz
"""
import sys

sys.path.insert(0, ".")

from src.graph import build_graph  # noqa: E402

OUT_PATH = "results/graph.png"

if __name__ == "__main__":
    app = build_graph()
    png_bytes = app.get_graph().draw_png()
    with open(OUT_PATH, "wb") as f:
        f.write(png_bytes)
    print(f"Saved graph visualization to {OUT_PATH}")
