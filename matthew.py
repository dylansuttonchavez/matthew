import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import sys

# Configuración de página (favicon y título)
st.set_page_config(page_title="MATTHEW: IA Axiomática con Razonamiento", page_icon="🌌")

# Título y Descripción General de MATTHEW

st.title("**MATTHEW: IA Axiomática con Razonamiento**")
st.markdown("""
Bienvenido a la presentación interactiva de **MATTHEW**, un prototipo de IA axiomática 
diseñada para generar teorías y propuestas innovadoras a partir de axiomas multidisciplinares 
(física, biología, matemáticas, filosofía).

La arquitectura de MATTHEW, denominada **Algebraic Axiom Architecture (AAA)**, combina 
principios algebraicos, transformaciones lineales, grafos axiomáticos y razonamiento 
multilógico (deductivo, inductivo y abductivo) para crear hipótesis coherentes, 
originales y con potencial aplicabilidad práctica.
""")

# Arquitectura AAA

st.header("Fundamentos Matemáticos de la Algebraic Axiom Architecture (AAA)")

st.markdown("""
La AAA de MATTHEW se basa en representar axiomas como vectores en un espacio:
""")
st.latex(r"\mathbb{R}^n")
st.markdown("""
y luego aplicar transformaciones lineales, no lineales y operaciones que incorporan distintos tipos de razonamiento.
""")

st.markdown("""
1. **Representación Vectorial de Axiomas:**
""")
st.latex(r"\mathbf{a} \in \mathbb{R}^n")

st.markdown("""
2. **Transformaciones Lineales y No Lineales:**
""")
st.latex(r"\mathbf{y} = W\mathbf{a} + b \quad \text{y} \quad \mathbf{z} = \sigma(W\mathbf{a} + b)")

st.markdown("""
3. **Razonamiento Multilógico:**
   - Deductivo, Inductivo y Abductivo (introduciendo ruido \(\epsilon\)):
""")
st.latex(r"\mathbf{h} = \mathbf{y} + \epsilon")

st.markdown("""
Estas operaciones combinadas permiten a MATTHEW ir más allá de la información dada, generando nuevas teorías.
""")

st.subheader("Snippet de Código: Transformación Lineal y No Lineal")
transform_code = r"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dimensiones
n = 128  # dimensión del axioma (input)
m = 256  # dimensión del espacio transformado (hidden)

W = torch.randn(m, n)
b = torch.randn(m, 1)

# Ejemplo: vector de axioma
a = torch.randn(n, 1)

# Transformación lineal
y = W @ a + b  # shape: (m, 1)

# Aplicar no linealidad (ReLU)
z = F.relu(y)
"""
st.code(transform_code, language='python')

# Datasets y Axiomas

st.header("Datasets y Conjunto de Axiomas")

with st.expander("Fuentes de Conocimiento"):
    st.write("**Preentrenamiento General:**")
    st.markdown("- [Wikipedia Summary Dataset](https://github.com/tscheepers/Wikipedia-Summary-Dataset)")
    st.markdown("- Reddit Conversations Dataset")

    st.write("**Razonamiento y Matemáticas:**")
    st.markdown("- [GSM8K](https://github.com/openai/grade-school-math)")
    st.markdown("- [RuleTaker](https://allenai.org/data/rule-reasoning-dataset)")
    st.markdown("- [CLUTRR](https://github.com/facebookresearch/CLUTRR)")

with st.expander("Ejemplo de Axiomas Curados en JSON"):
    sample_axiom = [
       {
         "axiom_id": "phy_001",
         "axiom_text": "La energía no se crea ni se destruye, solo se transforma.",
         "domain": "física",
         "tags": ["conservación", "termodinámica"],
         "compatibilities": ["phy_002", "che_003"],
         "incompatibilities": ["phi_007"]
       }
    ]
    st.json(sample_axiom)

    st.markdown("**Snippet de Código: Generar Dataset desde la API de Anthropic y Web Scraping**")
    code_snippet = r"""
import requests
import json
from bs4 import BeautifulSoup
import arxiv

# Configuración de la API de Anthropic
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/axioms"
ANTHROPIC_API_KEY = "00000-00000-00000-00000-00000"
HEADERS = {"Authorization": f"Bearer {ANTHROPIC_API_KEY}", "Content-Type": "application/json"}

# Función para obtener axiomas de la API de Anthropic
def get_axioms_from_anthropic(prompt, max_tokens=300):
    payload = {"prompt": prompt, "max_tokens": max_tokens}
    response = requests.post(ANTHROPIC_API_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json().get("axioms", [])

# Función para obtener axiomas de scraping
def get_axioms_from_web(url, css_selector):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")
    return [element.get_text(strip=True) for element in soup.select(css_selector)]

# Función para obtener axiomas de arXiv
def get_axioms_from_arxiv(query, max_results=10):
    search = arxiv.Search(query=query, max_results=max_results)
    return [f"Axiom derived from: {result.title}" for result in search.results()]

# Función principal para generar el dataset de axiomas
def generate_axiom_dataset():
    # Obtener axiomas de diferentes fuentes
    anthropic_axioms = get_axioms_from_anthropic("Lista axiomas biológicos sobre virus desconocidos.")
    web_axioms = get_axioms_from_web("https://www.example.com/biologia/virus-desconocidos", "p.axioma")
    arxiv_axioms = get_axioms_from_arxiv("unknown viruses biology", max_results=5)

    # Combinar axiomas
    all_axioms = anthropic_axioms + web_axioms + arxiv_axioms

    # Crear dataset estructurado
    dataset = [
        {
            "axiom_id": f"bio_generated_{i}",
            "axiom_text": axiom,
            "domain": "biología",
            "tags": ["desconocido", "virus"],
            "compatibilities": [],
            "incompatibilities": []
        }
        for i, axiom in enumerate(all_axioms)
    ]

    # Guardar en un archivo JSON
    with open("generated_axioms.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

# Ejecutar la función principal
generate_axiom_dataset()
"""
    st.code(code_snippet, language='python')

# Experimentación Interactiva (Medicamento)

st.header("Experimentación Interactiva: Nuevo Medicamento para un Virus Desconocido")

st.markdown("""
En este experimento, partiremos de axiomas relacionados con un virus desconocido y 
trataremos de generar la hipótesis de un nuevo medicamento. El proceso es:
1. Ingrese dos axiomas sobre el virus.
2. Estos axiomas se convierten en vectores (simulado).
3. Se combinan y se aplica razonamiento deductivo, inductivo o abductivo.
4. El resultado es una propuesta de medicamento o terapia innovadora.
""")

# Creación de instancias para la demo interactiva
class AxiomEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AxiomEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class ReasoningEngine(nn.Module):
    def __init__(self, hidden_dim):
        super(ReasoningEngine, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, noise=0.0):
        x = self.linear(x)
        x = x + noise * torch.randn_like(x)

axiom_encoder = AxiomEncoder(128, 256)
reasoning_engine = ReasoningEngine(256)

axiom_text_1 = st.text_input("Axioma 1", "El virus X se adhiere a receptores proteínicos en las células del huésped.")
axiom_text_2 = st.text_input("Axioma 2", "Ciertos inhibidores enzimáticos bloquean las proteasas virales del virus X.")

def text_to_vector(text):
    torch.manual_seed(sum([ord(c) for c in text]))
    return torch.rand((1, 128))

ax1_vec = text_to_vector(axiom_text_1)
ax2_vec = text_to_vector(axiom_text_2)

enc_ax1 = axiom_encoder(ax1_vec)
enc_ax2 = axiom_encoder(ax2_vec)

combined = enc_ax1 + enc_ax2

st.markdown("**Seleccione el tipo de razonamiento:**")
reasoning_type = st.selectbox("Tipo de razonamiento", ["Deductivo (0.0 ruido)", "Inductivo (0.05 ruido)", "Abductivo (0.1 ruido)"])
noise_map = {
    "Deductivo (0.0 ruido)": 0.0,
    "Inductivo (0.05 ruido)": 0.05,
    "Abductivo (0.1 ruido)": 0.1,
}
noise_level = noise_map[reasoning_type]

if st.button("Generar Propuesta de Medicamento"):
    output_vec = reasoning_engine(combined, noise=noise_level)
    # Hipótesis simbólica
    if noise_level == 0.0:
        hypothesis = "Un medicamento basado en un inhibidor proteínico estable que bloquee la unión receptor-viral."
    elif noise_level == 0.05:
        hypothesis = "Una formulación de inhibidores enzimáticos encapsulados que reducen la proteólisis del virus, evitando su replicación."
    else:
        hypothesis = "Una nanopartícula que libere inhibidores específicos al detectar proteasas virales, reduciendo la replicación del virus en el huésped."

    st.success(f"**Propuesta de Medicamento Generada:** {hypothesis}")

# MATTHEW

st.header("Validación de Teorías y Perspectivas Futuras")

with st.expander("Validación"):
    st.markdown("""
    - **Coherencia Lógica:** Comparar la hipótesis con los axiomas base.
    - **Originalidad:** Verificar que la propuesta no sea trivial (comparando embeddings con bases de conocimiento).
    - **Aplicabilidad:** Probar en simulaciones biológicas o modelos computacionales.
    """)

with st.expander("Impacto Futuro"):
    st.markdown("""
    En el futuro, MATTHEW podría:
    - Generar tratamientos innovadores para nuevas cepas virales.
    - Proponer líneas de investigación farmacológica.
    - Evolucionar hacia un asistente AGI para medicina, biología y otras ciencias.
    """)

# Ejecución Dinámica de Código

st.header("Ejecución Dinámica de Código")

st.markdown("""
En esta sección, puedes experimentar con la arquitectura AAA ejecutando código Python relacionado con las variables del proyecto. Por ejemplo:

**Ejemplo de Código:**  
```python
# Visualizar la salida del vector combinado
print("Forma del vector combinado:", combined.shape)

# Analizar los valores del vector
print("Valores del vector combinado:", combined)
```

Introduce tu código a continuación:
""")

user_code = st.text_area("Ingrese su código Python aquí:", value="""print("Forma del vector combinado:", combined.shape)\nprint("Valores del vector combinado:", combined)""")

if st.button("Ejecutar Código"):
    if 'combined' not in locals():
        st.error("La variable `combined` no está definida. Por favor, genera los axiomas antes de ejecutar el código.")
    else:
        buffer = io.StringIO()
        sys.stdout = buffer
        local_vars = {"combined": combined, "torch": torch}

        try:
            exec(user_code, {}, local_vars)
        except Exception as e:
            st.error(f"Error al ejecutar el código: {e}")
        finally:
            sys.stdout = sys.__stdout__

        st.text(buffer.getvalue())

st.markdown("---")
st.markdown("""**¡Gracias por tu atención!** Si tienes comentarios adicionales o necesitas más ayuda, no dudes en escribirme a mi correo: [dylan2406010@hybridge.education](mailto:dylan2406010@hybridge.education).""")
st.markdown("""[Repositorio](https://github.com/dylansuttonchavez/matthew)""")
