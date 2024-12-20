# MATTHEW: Axiomatic AI with Reasoning

**MATTHEW** is a prototype of Axiomatic Artificial Intelligence designed to generate theories and innovative proposals based on multidisciplinary axioms (physics, biology, mathematics, philosophy). Its architecture, named **Algebraic Axiom Architecture (AAA)**, integrates algebraic principles, linear transformations, axiom graphs, and multilogical reasoning (deductive, inductive, and abductive).

## Mathematical Foundations of the Algebraic Axiom Architecture (AAA)

The AAA is based on representing axioms as vectors in a space:

$$
\mathbb{R}^n
$$

Each axiom is expressed as a vector:

$$
\mathbf{a} \in \mathbb{R}^n
$$

Linear and nonlinear transformations are applied to these representations, incorporating different forms of reasoning (deductive, inductive, abductive). These operations enable MATTHEW to generate new theories from known axioms, establishing novel connections and exploring solutions to complex problems.

### Key Principles:

1. **Vectorial Representation of Axioms:**  
   Each axiom is expressed as a vector:
   $$
   \mathbf{a} \in \mathbb{R}^n
   $$

2. **Linear and Nonlinear Transformations:**  
   Linear and nonlinear functions are employed to combine and transform axioms. For example:
   $$
   \mathbf{y} = W\mathbf{a} + b \quad \text{y} \quad \mathbf{z} = \sigma(W\mathbf{a} + b)
   $$
   where \( W \) and \( b \) are parameters of the transformation, and \( \sigma \) represents a nonlinear activation function.

3. **Multilogical Reasoning:**  
   Integrates different types of reasoning (deductive, inductive, abductive) through the introduction of controlled noise:
   $$
   \mathbf{h} = \mathbf{y} + \epsilon
   $$
   where \( \epsilon \) represents noise, facilitating the generation of more creative and nuanced hypotheses.

These operations allow MATTHEW to generate new theories by combining and transforming existing axioms, thereby establishing novel connections and exploring solutions to complex problems.

## Datasets and Set of Axioms

**Knowledge Sources:**
- General Pretraining (e.g., Wikipedia, Reddit)
- Reasoning and Mathematics (e.g., GSM8K, RuleTaker, CLUTRR)

These sources provide structured axioms and knowledge, which are curated, filtered, and combined into a final set of axioms that feed into MATTHEW.

## Interactive Experimentation: New Medication for an Unknown Virus

An illustrative example is the generation of medication proposals based on axioms about an unknown virus:

1. **Axioms about the Virus and Its Interaction with the Host:**  
   Define the foundational statements regarding the virus's behavior and interactions.

2. **Vectorial Representation of These Axioms:**  
   Convert each axiom into a vector in \( \mathbb{R}^n \).

3. **Combination and Application of Multilogical Reasoning:**  
   Combine the vectors and apply reasoning processes to generate new hypotheses:
   $$
   \mathbf{combined} = \mathbf{enc\_ax1} + \mathbf{enc\_ax2}
   $$
   $$
   \mathbf{output\_vec} = \text{ReasoningEngine}(\mathbf{combined}, \text{noise})
   $$

4. **Obtaining an Innovative Pharmacological Hypothesis:**  
   Derive a novel medication proposal, such as a new enzymatic inhibitor or therapeutic nanoparticles.

The methodology demonstrates the system's ability to explore non-trivial solutions based on axiomatized information.

## Theory Validation and Future Perspectives

**Validation:**
- **Logical Coherence:** Ensure that proposals do not contradict the base axioms.
- **Originality:** Ensure that theories are not mere repetitions of available knowledge.
- **Applicability:** Test hypotheses in simulated environments or computational models.

**Future Perspectives:**
- Expand MATTHEW to propose treatments for new viral strains.
- Generate novel pharmacological research lines.
- Evolve into an AGI assistant applicable across multiple scientific domains.

---

**Contact:** [dylan2406010@hybridge.education](mailto:dylan2406010@hybridge.education)  
**Project Repository:** [https://github.com/dylansuttonchavez/matthew](https://github.com/dylansuttonchavez/matthew)
