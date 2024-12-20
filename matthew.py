import streamlit as st
import io
import sys

st.set_page_config(page_title="MATTHEW: Axiomatic AI Reasoning", page_icon="🌌")

st.title("**MATTHEW: Axiomatic AI with Reasoning**")
st.markdown("""
Welcome to the interactive presentation of **MATTHEW**, an axiomatic AI prototype
designed to generate theories and innovative proposals from multidisciplinary axioms
(physics, biology, mathematics, philosophy).

The architecture of MATTHEW, called **Algebraic Axiom Architecture (AAA)**, combines
algebraic principles, linear transformations, axiomatic graphs, and multi-logic reasoning
(deductive, inductive, and abductive) to create coherent, original hypotheses with potential practical applicability.
""")

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
import anthropic
import arxiv
import json

# Anthropic API Configuration
ANTHROPIC_API_KEY = "00000-00000-00000-00000-00000"

# Function to fetch axioms from Anthropic API
def get_axioms_from_anthropic(prompt, max_tokens=300):
    client = anthropic.Client(ANTHROPIC_API_KEY)
    response = client.completion(
        prompt=prompt,
        stop_sequences=["\n"],
        max_tokens_to_sample=max_tokens
    )
    return response["completion"].split("\n")

# Function to fetch axioms from arXiv
def get_axioms_from_arxiv(query, max_results=10):
    search = arxiv.Search(query=query, max_results=max_results)
    return [f"Axiom derived from: {result.title}" for result in search.results()]

# Main function to generate the axiom dataset
def generate_axiom_dataset():
    # Fetch axioms from Anthropic API
    anthropic_axioms = get_axioms_from_anthropic("List biological axioms about unknown viruses.")

    # Fetch axioms from arXiv
    arxiv_axioms = get_axioms_from_arxiv("unknown viruses biology", max_results=5)

    # Combine axioms
    all_axioms = anthropic_axioms + arxiv_axioms

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

st.header("Interactive Experimentation: New Drug for an Unknown Virus")

st.markdown("""
In this experiment, we will use axioms related to an unknown virus and attempt to generate a hypothesis for a new drug. The process is as follows:
1. Input two axioms about the virus.
2. These axioms are converted to vectors (simulated).
3. They are combined, and deductive, inductive, or abductive reasoning is applied.
4. The result is a proposal for an innovative drug or therapy.
""")

axiom_text_1 = st.text_input("Axiom 1", "Virus X binds to protein receptors in host cells.")
axiom_text_2 = st.text_input("Axiom 2", "Certain enzymatic inhibitors block viral proteases of Virus X.")

def text_to_vector(text):
    return sum(ord(c) for c in text)

ax1_value = text_to_vector(axiom_text_1)
ax2_value = text_to_vector(axiom_text_2)

combined_value = ax1_value + ax2_value

st.markdown("**Select reasoning type:**")
reasoning_type = st.selectbox("Reasoning Type", ["Deductive", "Inductive", "Abductive"])

if st.button("Generate Drug Proposal"):
    if reasoning_type == "Deductive":
        hypothesis = "A drug based on a stable protein inhibitor that blocks receptor-viral binding."
    elif reasoning_type == "Inductive":
        hypothesis = "A formulation of encapsulated enzymatic inhibitors that reduce viral proteolysis, preventing replication."
    else:
        hypothesis = "A nanoparticle that releases specific inhibitors upon detecting viral proteases, reducing replication in the host."

    st.success(f"**Generated Drug Proposal:** {hypothesis}")

st.header("**Multilayer Perceptron**")
st.markdown("""
The multilayer perceptron (MLP) is a type of artificial neural network composed of multiple layers of 
nodes organized in a hierarchical structure. It includes an input layer, one or more hidden layers, 
and an output layer. Each node, or neuron, in a layer is connected to nodes in the subsequent layer 
through weights that represent the significance of each connection. During the learning process, known 
as supervised training, the MLP employs the backpropagation algorithm to adjust these weights by 
minimizing the error between the predicted and desired outputs. Non-linear activation functions within 
the neurons enable the MLP to model complex relationships and capture intricate patterns in data, making 
it well-suited for tasks such as classification, regression, and pattern recognition.
""")

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
st.markdown("""[GitHub Repository](https://github.com/dylansuttonchavez/matthew).""")
