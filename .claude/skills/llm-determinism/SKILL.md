---
name: llm-determinism
description: Deterministic Azure OpenAI configuration for semantic greeting evaluation. Use when working on packages/ml_core/semantic.py, packages/schemas/evaluation.py, or any code calling Azure OpenAI Chat Completions.
---

# Deterministic LLM Prompt Specification (§8)

## Inference parameters (§8.1) — ALL are mandatory

temperature=0.0, top_p=1.0, max_tokens=500, seed=42, response_format=JSON Schema Structured Outputs.

These settings collapse the probability distribution to deterministic token selection. Do not modify any of them.

## Output schema (§8.2)

```json
{
  "type": "object",
  "required": ["reasoning", "is_match", "confidence_score"],
  "properties": {
    "reasoning": {"type": "string"},
    "is_match": {"type": "boolean"},
    "confidence_score": {"type": "number", "minimum": 0, "maximum": 1}
  },
  "additionalProperties": false
}
```

Validated by `SemanticEvaluationResult` Pydantic model with `extra="forbid"`.

## System prompt

Defined as the `SYSTEM_PROMPT` constant in `packages/ml_core/semantic.py`. Do not modify without spec change.

## Failure handling (§4.D contract)

LLM timeout: retry exactly once, then return None. Caller records null in metrics.

## Environment variables

AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION (default 2024-02-01).
