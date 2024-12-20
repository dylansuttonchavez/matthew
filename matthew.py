import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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

# Model definitions
class AxiomEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AxiomEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class ReasoningEngine(nn.Module):
    def __init__(self, hidden_dim):
        super(ReasoningEngine, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, noise=0.0):
        x = self.linear(x)
        return x + noise * torch.randn_like(x)

# Initialize models
def get_models():
    encoder = AxiomEncoder(128, 256)
    engine = ReasoningEngine(256)
    return encoder, engine

axiom_encoder, reasoning_engine = get_models()

# Text input for axioms
axiom_text_1 = st.text_input("Axiom 1", "Virus X binds to protein receptors in host cells.")
axiom_text_2 = st.text_input("Axiom 2", "Certain enzymatic inhibitors block viral proteases of Virus X.")

# Function to simulate text to vector conversion
def text_to_vector(text):
    torch.manual_seed(sum(ord(c) for c in text))
    return torch.rand((1, 128))

# Convert axioms to vectors
ax1_vec = text_to_vector(axiom_text_1)
ax2_vec = text_to_vector(axiom_text_2)

# Encode vectors
enc_ax1 = axiom_encoder(ax1_vec)
enc_ax2 = axiom_encoder(ax2_vec)

# Combine encoded vectors
combined = enc_ax1 + enc_ax2

# Select reasoning type
st.markdown("**Select reasoning type:**")
reasoning_type = st.selectbox("Reasoning Type", ["Deductive (0.0 noise)", "Inductive (0.05 noise)", "Abductive (0.1 noise)"])
noise_map = {
    "Deductive (0.0 noise)": 0.0,
    "Inductive (0.05 noise)": 0.05,
    "Abductive (0.1 noise)": 0.1,
}
noise_level = noise_map[reasoning_type]

# Generate drug proposal
if st.button("Generate Drug Proposal"):
    output_vec = reasoning_engine(combined, noise=noise_level)
    # Generate hypothesis based on reasoning type
    if noise_level == 0.0:
        hypothesis = "A drug based on a stable protein inhibitor that blocks receptor-viral binding."
    elif noise_level == 0.05:
        hypothesis = "A formulation of encapsulated enzymatic inhibitors that reduce viral proteolysis, preventing replication."
    else:
        hypothesis = "A nanoparticle that releases specific inhibitors upon detecting viral proteases, reducing replication in the host."

    st.success(f"**Generated Drug Proposal:** {hypothesis}")

# Debugging and interactive code execution
st.header("Dynamic Code Execution")
st.markdown("""
Enter Python code to experiment with the combined vector or any project variable.
""")

user_code = st.text_area("Enter your Python code here:", value="""print('Shape of combined vector:', combined.shape)
print('Values of combined vector:', combined)""")

if st.button("Execute Code"):
    buffer = io.StringIO()
    sys.stdout = buffer
    local_vars = {"combined": combined, "torch": torch}

    try:
        exec(user_code, {}, local_vars)
    except Exception as e:
        st.error(f"Error executing code: {e}")
    finally:
        sys.stdout = sys.__stdout__

    st.text(buffer.getvalue())

st.markdown("---")
st.markdown("""**Thank you for your attention!** If you have additional comments or need more assistance, feel free to email me at [dylan2406010@hybridge.education](mailto:dylan2406010@hybridge.education).""")
st.markdown("""[GitHub Repository](https://github.com/dylansuttonchavez/matthew)""")
