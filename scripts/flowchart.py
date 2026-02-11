from graphviz import Digraph
import os

dot = Digraph(
    name="Legal_NLP_Pipeline",
    format="svg",
    graph_attr={
        "rankdir": "LR", 
        "fontsize": "12", 
        "splines": "ortho",
        "nodesep": "0.5", # Vertical space between nodes
        "ranksep": "0.8"  # Horizontal space between columns
    },
)

# Colors
HUMAN, LLM, MODEL, INFRA, EVAL = "#E3F2FD", "#F3E5F5", "#E8F5E9", "#EEEEEE", "#FFFDE7"

# Helper to apply styles
def add_styled_node(graph, id, label, color):
    graph.node(id, label, style="filled,rounded", shape="box", fillcolor=color)

# ---------------------
# Row 1: Training Prep
# ---------------------
with dot.subgraph(name="cluster_0") as c:
    c.attr(label="Phase 1: Training Prep", style="dotted")
    add_styled_node(c, "H", "Human-coded laws\n(JSON per law)", HUMAN)
    add_styled_node(c, "LLM1", "GPT-5\nText extraction", LLM)
    add_styled_node(c, "T1", "Reconstructed data", HUMAN)
    add_styled_node(c, "SYN", "GPT-5\nSynthetic generation", LLM)
    add_styled_node(c, "T2", "Final training dataset", HUMAN)
    c.edge("H", "LLM1")
    c.edge("LLM1", "T1")
    c.edge("T1", "SYN")
    c.edge("SYN", "T2")

# ---------------------
# Row 2: Azure ML
# ---------------------
with dot.subgraph(name="cluster_1") as c:
    c.attr(label="Phase 2: Fine-tuning Model", style="dotted")
    add_styled_node(c, "FT", "Training data rows", HUMAN)
    add_styled_node(c, "AZ1", "Azure ML", INFRA)
    add_styled_node(c, "ENC", "LEGAL-BERT\n(Frozen)", MODEL)
    add_styled_node(c, "LORA", "LoRA adapters", MODEL)
    add_styled_node(c, "HEAD", "Multihead classifier", MODEL)
    c.edge("FT", "AZ1")
    c.edge("AZ1", "ENC")
    c.edge("ENC", "LORA")
    c.edge("LORA", "HEAD")

# ---------------------
# Row 3: Test Inference
# ---------------------
with dot.subgraph(name="cluster_2") as c:
    c.attr(label="Phase 3: Test & Evaluation", style="dashed", color="gray")
    
    # Existing nodes
    add_styled_node(c, "TEST", "Human-labeled\ntest laws", HUMAN)
    add_styled_node(c, "S1", "GPT-5-mini\nStrict Splitting", LLM)
    add_styled_node(c, "I1", "Inference", MODEL)
    add_styled_node(c, "E1", "Evaluation (Strict)", EVAL)
    
    add_styled_node(c, "S2", "GPT-5-mini\nRelaxed Splitting", LLM)
    add_styled_node(c, "I2", "Inference", MODEL)
    add_styled_node(c, "E2", "Evaluation (Relaxed)", EVAL)

    # New Comparison Node
    add_styled_node(c, "COMP", "Comparison\n(F1, Precision, Recall)", EVAL)
    
    # Top path
    c.edge("TEST", "S1")
    c.edge("S1", "I1")
    c.edge("I1", "E1")
    
    # Bottom path
    c.edge("TEST", "S2")
    c.edge("S2", "I2")
    c.edge("I2", "E2")
    
    # Funneling both into Comparison
    c.edge("E1", "COMP")
    c.edge("E2", "COMP")

# ---------------------
# CRITICAL: Vertical Stacking
# ---------------------
# invisible edges to force Row 1 above Row 2 above Row 3
dot.edge("H", "FT", style="invis")
dot.edge("FT", "TEST", style="invis")

# ensuring horizontal alignment
with dot.subgraph() as s:
    s.attr(rank='same')
    s.node("H")
    s.node("FT")
    s.node("TEST")

output_dir = "../outputs"
os.makedirs(output_dir, exist_ok=True)

# full path without extension
output_path = os.path.join(output_dir, "pipeline_diagram")

# rendering
dot.render(
    filename=output_path,
    format="svg",
    cleanup=True
)

print(f"Success! File is at: {output_path}.svg")

