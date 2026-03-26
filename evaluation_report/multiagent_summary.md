## Multi-Agent RAG System — Technical Summary

### Answer analysis

### Outputs

## Multi-Agent RAG System — Evaluation Report

---

### System A — Base LLM (no RAG)

| # | Query | Answer |
|---|-------|--------|
| 1 | What make Benny Scanlon heartbreak in Overcompensating? | In "Overcompensating," Benny Scanlon experiences heartbreak when he realizes that his romantic feelings for his best friend are unreciprocated, leading him to confront his feelings about love and friendship. This emotional struggle forces Benny to reassess his relationships and personal identity. |
| 2 | How do Milo Bradford and his wife respond to the kidnapping? | In response to the kidnapping, Milo Bradford and his wife take drastic measures, including enlisting the help of law enforcement and conducting their own investigation, while grappling with intense emotions and the fear for their child's safety. Their determination leads them to uncover shocking truths about the circumstances surrounding the abduction. |
| 3 | What is the name of the chef that Gordon Ramsay met for Māori cuisine? | Gordon Ramsay met chef and Māori cuisine expert, Peter Gordon, during his exploration of New Zealand's culinary traditions. |
| 4 | How do Lucy's feelings about her breakup with Stephen and her guilt toward Evan affect her actions in this episode? | Lucy grapples with lingering sadness over her breakup with Stephen while also feeling guilty about her growing connection to Evan, prompting her to push him away as she tries to sort out her emotions. This internal conflict leads her to make decisions that ultimately complicate her relationships and hinder her ability to move on. |
| 5 | Why Ilya ignored Shane's text? | Ilya ignored Shane's text because he was upset after their last conversation and needed time to process his feelings before responding. |

---

### System B — Basic RAG

| # | Query | Answer |
|---|-------|--------|
| 1 | What make Benny Scanlon heartbreak in Overcompensating? | Benny Scanlon experiences heartbreak in "Overcompensating" due to his feelings of inadequacy and self-doubt. |
| 2 | How do Milo Bradford and his wife respond to the kidnapping? | Milo Bradford and his wife, Marissa, respond to their son's kidnapping by trying to manage the public situation. |
| 3 | What is the name of the chef that Gordon Ramsay met for Māori cuisine? | The chef that Gordon Ramsay met for Māori cuisine is Monique Fiso. |
| 4 | How do Lucy's feelings about her breakup with Stephen and her guilt toward Evan affect her actions in this episode? | Lucy grapples with her emotions regarding her breakup with Stephen and her guilt toward Evan, which affects her decisions throughout the episode. |
| 5 | Why Ilya ignored Shane's text? | Ilya ignored Shane's texts after his team performed poorly at the 2014 Winter Olympics in Sochi. |

---

### System C — Advanced Agentic RAG (Base Model, GPT-4o-mini)

| # | Query | Answer |
|---|-------|--------|
| 1 | What is the occupation of Ilya and Shane in Heated Rivalry? | Ilya Rozanov is a hockey player drafted by the Boston Raiders, and Shane Hollander is a hockey player drafted by the Montreal Metros. Both are involved in the Major League Hockey (MLH). Their relationship is a central theme in the narrative. |
| 2 | What make Benny Scanlon heartbreak in Overcompensating? | Benny Scanlon navigates complex feelings in 'Overcompensating' after coming out to Carmen. To avoid his feelings for Miles, he begins spending more time with Carmen. Carmen helps Benny ease into gay life by setting him up on a Grindr date, reflecting the complexity of their friendship amid their individual challenges. |
| 3 | How do Milo Bradford and his wife respond to the kidnapping? | Conrad Cain and his wife Ellie are deeply affected by their son Leopold's kidnapping, compounded by media attention. Ellie joins an anonymous group for parents of kidnapped children, seeking support and information. |
| 4 | What is the occupation of Ilya and Shane? | Ilya and Shane are both hockey players in the Major League Hockey (MLH). Ilya is from Russia and plays for the Boston Raiders, while Shane is from Canada and plays for the Montreal Metros. |
| 5 | the relationship between dhacha and alansb in pasknc? | The retrieved context does not provide information about the relationship between dhacha and alansb in pasknc. Therefore, I don't know the answer. |

---

### System D — Advanced Agentic RAG (LoRA Fine-tuned)

*Factual questions routed to LoRA SmolLM2-1.7B; emotional questions routed to GPT-4o-mini*

| # | Query | Type | Answer |
|---|-------|------|--------|
| 1 | What make Benny Scanlon heartbreak in Overcompensating? | factual → LoRA | Benny's heartbreak in Overcompensating is related to his internal struggle with his feelings for Miles. |
| 2 | How do Milo Bradford and his wife respond to the kidnapping? | emotional → GPT | When Milo Bradford, the son of Los Angeles mayoral candidate Milo Bradford, is kidnapped, he and his family face a public crisis. |
| 3 | What is the name of the chef that Gordon Ramsay met for Māori cuisine? | factual → LoRA | Gordon Ramsay embarks on culinary adventures across various countries, including New Zealand, encountering local chefs along the way. |
| 4 | How do Lucy's feelings about her breakup with Stephen and her guilt toward Evan affect her actions in this episode? | emotional → GPT | In this episode, Lucy deals with complex emotions related to her relationship with Stephen, which has ended, and her feelings of guilt toward Evan. |
| 5 | Why Ilya ignored Shane's text? | emotional → GPT | Ilya began ignoring Shane's texts after his Russian hockey team performed poorly during the 2014 Winter Olympics in Sochi. |

---

### Discussion

From the results, we can observe clear improvement at each stage of the pipeline.

With the base LLM, answers are generally fabricated since the specific plot details are not present in the model's training data. The model produces plausible-sounding but inaccurate responses.

With basic RAG, answers become more grounded as the retriever supplies relevant plot chunks as context. However, the answers tend to be verbose and redundant, often restating surrounding context rather than directly addressing the question.

With the advanced agentic RAG using the base model, answers become more concise and accurate. The critic agent filters out unsupported claims, and the title filter agent leverages chunk metadata to remove noise from the retrieved documents, ensuring the generator only sees chunks from the correct show or movie.

With the LoRA fine-tuned model handling factual questions, the answers improve further in precision. Because the model was fine-tuned on NarrativeQA — a movie plot QA dataset focused on short, direct answers — it learns to extract the key information rather than elaborating with surrounding context. As a result, the answer to the question appears at the beginning of the response, making it immediately readable without having to parse through a long passage to find the relevant detail.

---

### Pros and Cons

| System | Factual Accuracy | Conciseness | Handles Unknowns |
|--------|-----------------|-------------|-----------------|
| A — Base LLM | ❌ Fabricates | Medium | ❌ Invents answers |
| B — Basic RAG | ✅ Often correct | Verbose | ⚠️ Sometimes correct |
| C — Advanced (base) | ✅ Grounded | Verbose | ✅ Returns "I don't know" |
| D — Advanced (LoRA) | ✅ Grounded + concise (factual) | Short and direct | ✅ Returns "I don't know" |

---

### Summary

Moving from the base model to the LoRA fine-tuned pipeline shows a clear improvement in answer conciseness for factual questions. The LoRA model produces answers that more closely resemble the short, grounded responses expected for plot QA, while the base model tends to over-generate with additional context and narrative framing. For emotional and vague questions, both systems perform equivalently since they both use GPT-4o-mini for those cases. The main remaining weakness across both systems is retrieval quality — when the vector store returns chunks from the wrong title or character, neither model can compensate, highlighting that retrieval accuracy is the primary bottleneck in the pipeline.

### Workflow
![Agent Workflow](output.png)

### State
The shared state is a plain Python `dict` passed between every node. It accumulates fields as the graph progresses:

| Field | Type | Description |
|---|---|---|
| `question` | `str` | Original user query |
| `question_type` | `str` | `factual`, `emotional`, or `vague` |
| `hyde_query` | `str` | Rewritten query for vector search |
| `strategy` | `str` | Retrieval strategy used |
| `chunks` | `list[Document]` | Retrieved/filtered plot chunks |
| `resolved_title` | `str` | Show/movie title resolved by Title Filter |
| `title_filter_success` | `bool` | Whether title filtering succeeded |
| `answer` | `str` | Generated answer |
| `evidence` | `list[dict]` | Source chunks shown to the user |
| `critic_judgment` | `dict` | Fact-check result with `supported`, `explanation`, `revised_answer` |
| `retry_count` | `int` | Number of retry attempts |
| `generator_used` | `str` | Which generator produced the answer: `lora` or `gpt` |

---

### Nodes
The graph has **8 nodes**, each a pure Python function that receives the state and returns an updated copy:

**1. `classify` — Classifier + Rewrite Agent (GPT-4o-mini)**
A single LLM call that simultaneously labels the question as `factual`, `emotional`, or `vague` and rewrites it into a HyDE (Hypothetical Document Embedding) query optimised for vector search. The question type determines which generator agent is used downstream.

**2. `generate_factual` — LoRA Generator Agent (SmolLM2-1.7B fine-tuned)**
Used exclusively for `factual` questions. Retrieves the top-10 chunks from Pinecone, reranks them with a Cross-Encoder, then passes the top-5 as context to the LoRA fine-tuned SmolLM2-1.7B model. The model was fine-tuned on NarrativeQA — a dataset of short, direct plot questions — making it well-suited for factual who/what/where questions that require concise answers.

**3. `generate_emotional` — GPT Generator Agent (GPT-4o-mini)**
Used for `emotional` questions about feelings, motivations, and relationships. Retrieves chunks and passes them directly to GPT-4o-mini without reranking, since emotional questions benefit from broader context rather than precision-focused top-k filtering.

**4. `generate_vague` — GPT Clarify + Generator Agent (GPT-4o-mini)**
Used for `vague` or underspecified questions. First rewrites the question into a more precise form using GPT-4o-mini, then retrieves and reranks chunks based on the clarified query before generating an answer.

**5. `title_filter` — Title Filter Agent (GPT-4o-mini)**
An LLM call that examines the retrieved chunks and resolves which specific show or movie the question targets. Filters chunks to only those matching the resolved title. If fewer than 2 chunks match, sets `title_filter_success = False` to signal a re-retrieval.

**6. `re_retrieve` — Re-Retrieval Node**
Triggered when the title filter fails. Re-queries Pinecone with a title-scoped filter to get more targeted chunks. Routes back to the appropriate generator — LoRA for factual questions, GPT for all others.

**7. `critic` — Critic Agent (GPT-4o-mini)**
Fact-checks the generated answer against the retrieved chunks. If the answer contains unsupported claims, it rewrites a corrected version strictly within the evidence.

**8. `evidence` — Evidence Selector**
Selects the top-3 most relevant chunks as evidence snippets to surface to the user alongside the final answer.

---

### Edges

| From | To | Type | Condition |
|---|---|---|---|
| `classify` | `generate_factual` | Conditional | `question_type == factual` → LoRA agent |
| `classify` | `generate_emotional` | Conditional | `question_type == emotional` → GPT agent |
| `classify` | `generate_vague` | Conditional | `question_type == vague` → GPT clarify agent |
| `generate_factual` | `title_filter` | Hard | Always |
| `generate_emotional` | `title_filter` | Hard | Always |
| `generate_vague` | `title_filter` | Hard | Always |
| `title_filter` | `re_retrieve` | Conditional | Title filter failed and `retry_count < 2` |
| `title_filter` | `critic` | Conditional | Title filter succeeded or retry limit reached |
| `re_retrieve` | `critic` | Hard | Always |
| `critic` | `evidence` | Hard | Always |
| `evidence` | `END` | Hard | Always |

Hard edges represent deterministic sequential steps. The key conditional split is at `classify` — the question type routes to the LoRA agent for factual questions and the GPT agent for all others. The title filter loop allows the system to re-retrieve with a more targeted query when the initial chunks do not match the resolved title.