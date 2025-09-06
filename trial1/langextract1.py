# Minimal LangExtract example: extract structured entities from text

# pip install langextract

import os
import json
import textwrap
import langextract as lx

# Prefer unified key; fall back to GEMINI_API_KEY if already set


if not os.getenv("LANGEXTRACT_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["LANGEXTRACT_API_KEY"] = os.environ["GEMINI_API_KEY"]

# 1) Define the extraction task (prompt + few-shot example)
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

examples = [
    lx.data.ExampleData(
        text=(
            "ROMEO. But soft! What light through yonder window breaks? "
            "It is the east, and Juliet is the sun."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"},
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"},
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"},
            ),
        ],
    )
]

# 2) Provide input text (replace with any string or file content)
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

# 3) Run extraction (Gemini flash is fast and cost‑effective; switch to pro for deeper reasoning)
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash-lite",
)

# 4) Print results
print(f"Extracted {len(result.extractions)} entities:\n")
for e in result.extractions:
    print(f"• {e.extraction_class}: '{e.extraction_text}'")
    if e.attributes:
        for k, v in e.attributes.items():
            print(f"  - {k}: {v}")

# 5) (Optional) Persist and visualize
from pathlib import Path

# Save JSONL
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate HTML (may be an IPython display object in notebooks)
html_content = lx.visualize("extraction_results.jsonl")

# Normalize to a plain string
content = html_content.data if hasattr(html_content, "data") else str(html_content)

# Ensure a UTF-8 charset tag (optional but recommended for browsers)
if "<meta charset=" not in content:
    if "<head>" in content:
        content = content.replace("<head>", '<head><meta charset="utf-8">', 1)
    else:
        content = '<meta charset="utf-8">\n' + content

# Write as UTF-8 on Windows to avoid cp1252 encoding errors
Path("visualization.html").write_text(content, encoding="utf-8")

print("Saved visualization.html with interactive highlights.")
