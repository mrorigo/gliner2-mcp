## Model Description

GLiNER2 extends the original GLiNER architecture to support multi-task information extraction with a schema-driven interface. This base model provides efficient CPU-based inference while maintaining high accuracy across diverse extraction tasks.

**Key Features:**
- Multi-task capability: NER, classification, and structured extraction
- Schema-driven interface with field types and constraints
- CPU-first design for fast inference without GPU requirements
- 100% local processing with zero external dependencies

## Installation

```bash
pip install gliner2
```

## Usage

### Entity Extraction

```python
from gliner2 import GLiNER2
# Load the model
extractor = GLiNER2.from_pretrained("fastino/gliner2-base-v1")
# Extract entities
text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino yesterday."
result = extractor.extract_entities(text, ["company", "person", "product", "location"])
print(result)
# Output: {'entities': {'company': ['Apple'], 'person': ['Tim Cook'], 'product': ['iPhone 15'], 'location': ['Cupertino']}}
```

### Text Classification

```python
# Single-label classification
result = extractor.classify_text(
    "This laptop has amazing performance but terrible battery life!",
    {"sentiment": ["positive", "negative", "neutral"]}
)
print(result)
# Output: {'sentiment': 'negative'}
# Multi-label classification
result = extractor.classify_text(
    "Great camera quality, decent performance, but poor battery life.",
    {
        "aspects": {
            "labels": ["camera", "performance", "battery", "display", "price"],
            "multi_label": True,
            "cls_threshold": 0.4
        }
    }
)
print(result)
# Output: {'aspects': ['camera', 'performance', 'battery']}
```

### Structured Data Extraction

```python
text = "iPhone 15 Pro Max with 256GB storage, A17 Pro chip, priced at $1199."
result = extractor.extract_json(
    text,
    {
        "product": [
            "name::str::Full product name and model",
            "storage::str::Storage capacity",
            "processor::str::Chip or processor information",
            "price::str::Product price with currency"
        ]
    }
)
print(result)
# Output: {
#     'product': [{
#         'name': 'iPhone 15 Pro Max',
#         'storage': '256GB',
#         'processor': 'A17 Pro chip',
#         'price': '$1199'
#     }]
# }
```

### Multi-Task Schema Composition

```python
# Combine all extraction types
schema = (extractor.create_schema()
    .entities({
        "person": "Names of people or individuals",
        "company": "Organization or business names",
        "product": "Products or services mentioned"
    })
    .classification("sentiment", ["positive", "negative", "neutral"])
    .structure("product_info")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("features", dtype="list")
)
text = "Apple CEO Tim Cook unveiled the iPhone 15 Pro for $999."
results = extractor.extract(text, schema)
print(results)
# Output: {
#     'entities': {'person': ['Tim Cook'], 'company': ['Apple'], 'product': ['iPhone 15 Pro']},
#     'sentiment': 'positive',
#     'product_info': [{'name': 'iPhone 15 Pro', 'price': '$999', 'features': [...]}]
# }
```

## Model Details

- **Model Type:** Bidirectional Transformer Encoder (BERT-based)
- **Parameters:** 205M
- **Input:** Text sequences
- **Output:** Entities, classifications, and structured data
- **Architecture:** Based on GLiNER with multi-task extensions
- **Training Data:** Multi-domain datasets for NER, classification, and structured extraction

## Performance

This model is optimized for:
- Fast CPU inference (no GPU required)
- Low latency applications
- Resource-constrained environments
- Multi-task extraction scenarios

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{zaratiana2025gliner2efficientmultitaskinformation,
      title={GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface},
      author={Urchade Zaratiana and Gil Pasternak and Oliver Boyd and George Hurn-Maloney and Ash Lewis},
      year={2025},
      eprint={2507.18546},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.18546},
}
```

## License

This project is licensed under the Apache License 2.0.

## Links

- **Repository:** https://github.com/fastino-ai/GLiNER2
- **Paper:** https://arxiv.org/abs/2507.18546
- **Organization:** [Fastino AI](https://fastino.ai)
