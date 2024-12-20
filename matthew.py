import streamlit as st
import io
import sys

# Page configuration
st.set_page_config(page_title="MATTHEW: Axiomatic AI Reasoning", page_icon="🌌")

# Title and General Description
st.title("**MATTHEW: Axiomatic AI with Reasoning**")
st.markdown("""
Welcome to the interactive presentation of **MATTHEW**, an axiomatic AI prototype
designed to generate theories and innovative proposals from multidisciplinary axioms
(physics, biology, mathematics, philosophy).

The architecture of MATTHEW, called **Algebraic Axiom Architecture (AAA)**, combines
algebraic principles, linear transformations, axiomatic graphs, and multi-logic reasoning
(deductive, inductive, and abductive) to create coherent, original hypotheses with potential practical applicability.
""")

# AAA Architecture
st.header("Mathematical Foundations of the Algebraic Axiom Architecture (AAA)")

st.markdown("""
The AAA of MATTHEW is based on representing axioms as vectors in a space:
""")
st.latex(r"\mathbb{R}^n")
st.markdown("""
and then applying linear and nonlinear transformations as well as operations incorporating different types of reasoning.
""")

st.markdown("""
1. **Vector Representation of Axioms:**
""")
st.latex(r"\mathbf{a} \in \mathbb{R}^n")

st.markdown("""
2. **Linear and Nonlinear Transformations:**
""")
st.latex(r"\mathbf{y} = W\mathbf{a} + b \quad \text{and} \quad \mathbf{z} = \sigma(W\mathbf{a} + b)")

st.markdown("""
3. **Multi-logic Reasoning:**
   - Deductive, Inductive, and Abductive:
""")
st.latex(r"\mathbf{h} = \mathbf{y} + \epsilon")

st.markdown("""
These combined operations enable MATTHEW to go beyond given information, generating new theories.
""")

st.subheader("Snippet: Linear and Nonlinear Transformation")
transform_code = r"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dimensions
n = 128  # dimension of the axiom (input)
m = 256  # dimension of the transformed space (hidden)

W = torch.randn(m, n)
b = torch.randn(m, 1)

# Example: axiom vector
a = torch.randn(n, 1)

# Linear transformation
y = W @ a + b  # shape: (m, 1)

# Apply nonlinearity (ReLU)
z = F.relu(y)
"""
st.code(transform_code, language='python')

# Datasets and Axioms
st.header("Datasets and Axiom Collection")

with st.expander("Knowledge Sources"):
    st.write("**General Pretraining:**")
    st.markdown("- [Wikipedia Summary Dataset](https://github.com/tscheepers/Wikipedia-Summary-Dataset)")
    st.markdown("- Reddit Conversations Dataset")

    st.write("**Reasoning and Mathematics:**")
    st.markdown("- [GSM8K](https://github.com/openai/grade-school-math)")
    st.markdown("- [RuleTaker](https://rule-reasoning.apps.allenai.org/about)")
    st.markdown("- [CLUTRR](https://github.com/facebookresearch/CLUTRR)")

with st.expander("Example of Curated Axioms in JSON"):
    sample_axiom = [
       {
         "axiom_id": "phy_001",
         "axiom_text": "Energy is neither created nor destroyed, only transformed.",
         "domain": "physics",
         "tags": ["conservation", "thermodynamics"],
         "compatibilities": ["phy_002", "che_003"],
         "incompatibilities": ["phi_007"]
       }
    ]
    st.json(sample_axiom)

    st.markdown("**Snippet: Generate Dataset from Anthropic API and Web Scraping**")
    code_snippet = r"""
import requests
import json
from bs4 import BeautifulSoup
import arxiv

# Anthropic API Configuration
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/axioms"
ANTHROPIC_API_KEY = "00000-00000-00000-00000-00000"
HEADERS = {"Authorization": f"Bearer {ANTHROPIC_API_KEY}", "Content-Type": "application/json"}

# Function to fetch axioms from Anthropic API
def get_axioms_from_anthropic(prompt, max_tokens=300):
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    response = requests.post(ANTHROPIC_API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json().get("axioms", [])

# Function to fetch axioms via scraping
def get_axioms_from_web(url, css_selector):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return [element.get_text(strip=True) for element in soup.select(css_selector)]

# Function to fetch axioms from arXiv
def get_axioms_from_arxiv(query, max_results=10):
    search = arxiv.Search(query=query, max_results=max_results)
    return [f"Axiom derived from: {result.title}" for result in search.results()]

# Main function to generate the axiom dataset
def generate_axiom_dataset():
    # Fetch axioms from various sources
    anthropic_axioms = get_axioms_from_anthropic("List biological axioms about unknown viruses.")
    web_axioms = get_axioms_from_web("https://www.example.com/biology/unknown-viruses", "p.axiom")
    arxiv_axioms = get_axioms_from_arxiv("unknown viruses biology", max_results=5)

    # Combine axioms
    all_axioms = anthropic_axioms + web_axioms + arxiv_axioms

    # Create structured dataset
    dataset = [
        {
            "axiom_id": f"bio_generated_{i}",
            "axiom_text": axiom,
            "domain": "biology",
            "tags": ["unknown", "virus"],
            "compatibilities": [],
            "incompatibilities": []
        }
        for i, axiom in enumerate(all_axioms)
    ]

    # Save to a JSON file
    with open("generated_axioms.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

# Execute the main function
generate_axiom_dataset()
"""
    st.code(code_snippet, language='python')

# Interactive Experimentation (Drug Discovery)
st.header("Interactive Experimentation: New Drug for an Unknown Virus")

st.markdown("""
In this experiment, we will use axioms related to an unknown virus and attempt to generate a hypothesis for a new drug. The process is as follows:
1. Input two axioms about the virus.
2. These axioms are converted to vectors (simulated).
3. They are combined, and deductive, inductive, or abductive reasoning is applied.
4. The result is a proposal for an innovative drug or therapy.
""")

# Text input for axioms
axiom_text_1 = st.text_input("Axiom 1", "Virus X binds to protein receptors in host cells.")
axiom_text_2 = st.text_input("Axiom 2", "Certain enzymatic inhibitors block viral proteases of Virus X.")

# Function to simulate text to vector conversion
def text_to_vector(text):
    return sum(ord(c) for c in text)

# Convert axioms to simulated values
ax1_value = text_to_vector(axiom_text_1)
ax2_value = text_to_vector(axiom_text_2)

# Combine encoded values
combined_value = ax1_value + ax2_value

# Select reasoning type
st.markdown("**Select reasoning type:**")
reasoning_type = st.selectbox("Reasoning Type", ["Deductive", "Inductive", "Abductive"])

# Generate drug proposal
if st.button("Generate Drug Proposal"):
    if reasoning_type == "Deductive":
        hypothesis = "A drug based on a stable protein inhibitor that blocks receptor-viral binding."
    elif reasoning_type == "Inductive":
        hypothesis = "A formulation of encapsulated enzymatic inhibitors that reduce viral proteolysis, preventing replication."
    else:
        hypothesis = "A nanoparticle that releases specific inhibitors upon detecting viral proteases, reducing replication in the host."

    st.success(f"**Generated Drug Proposal:** {hypothesis}")

# Multilayer Perceptrons

st.markdown("""
<div style="text-align: center; font-style: italic;">
A multilayer perceptron (MLP) is like a team of problem-solvers working together. Each "neuron" in the network receives inputs, processes them using a weighted sum, applies an activation function to introduce non-linearity (like ReLU or sigmoid), and passes the output to the next layer. The hidden layers act as feature extractors, capturing complex patterns, while the final layer combines these patterns to make predictions. Through backpropagation, the network learns by adjusting weights to minimize errors, improving its performance with each iteration!
</div>
""", unsafe_allow_html=True)

# MATTHEW

st.header("Validation of Theories and Future Perspectives")

with st.expander("Validation"):
    st.markdown("""
    - **Logical Coherence:** Compare the hypothesis with the base axioms.
    - **Originality:** Verify that the proposal is non-trivial (comparing embeddings with knowledge bases).
    - **Applicability:** Test in biological simulations or computational models.
    """)

with st.expander("Future Impact"):
    st.markdown("""
    In the future, MATTHEW could:
    - Propose new lines of scientific research.
    """)
   
st.markdown("---")
st.markdown("""**Thank you for your attention!** If you have additional comments or need more assistance, feel free to email me at [dylan2406010@hybridge.education](mailto:dylan2406010@hybridge.education).""")
st.markdown("""[GitHub Repository](https://github.com/dylansuttonchavez/matthew)""")
