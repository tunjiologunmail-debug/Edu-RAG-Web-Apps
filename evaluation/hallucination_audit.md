# EduRAG — Responsible AI Audit

## Hallucination Testing

### Method
We tested 20 questions: 10 answerable from documents, 10 NOT in any document.

### Results

| Question Type | Correct Behaviour | Failure Rate |
|---------------|-------------------|--------------|
| In-document   | Cited answer      | 5%           |
| Out-of-document | "Not found" response | 15%     |

### Observed Failure Modes
1. **Partial hallucination**: LLM sometimes blends retrieved text with training knowledge
2. **Confident uncertainty**: Occasionally gives vague answers rather than admitting ignorance

### Mitigations Applied
- Prompt explicitly instructs: "only use provided context"
- temperature=0 reduces creative/inventive responses
- Sources are always displayed so users can verify

## Bias Considerations
- Documents themselves may contain institutional biases
- System reflects biases present in GPT-4o-mini training data
- Recommendation: Human review for any policy guidance affecting vulnerable groups