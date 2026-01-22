# mini script for re-doing test prep for law that failed
import json
from pathlib import Path
import argparse
from test_prep_library import process_test_law

# -----------------------------
# CLI argument parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Re-run Stage 1 for a single law PDF and append provisions.")
parser.add_argument("law_id", type=str, help="Law ID to process, e.g., '(Guyana 1959)_01'")
parser.add_argument(
    "--output",
    type=str,
    default="../data/test_provisions.jsonl",
    help="Path to the output JSONL file (default: '../data/test_provisions.jsonl')"
)
args = parser.parse_args()

law_id = args.law_id
output_file = Path(args.output).resolve()

# -----------------------------
# Paths and model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
pdf_dir = BASE_DIR / "data" / "laws_pdf"
model = "gpt-5-mini"

pdf_path = pdf_dir / f"{law_id}.pdf"

# -----------------------------
# System instructions
# -----------------------------
system_instructions = """
You are a legal text processing assistant. Your job is to:
1. Split legal texts into distinct provisions or articles.
2. Ensure each provision is self-contained.
3. Preserve numbering, headings, and structure.
4. Output results in JSON format

# MULTILINGUAL TEXT RULE
If the law is not in English:
1. Provide a faithful, literal English translation.
2. Do NOT include the original-language text.
3. Do not summarize, paraphrase, shorten, or reinterpret.
4. Preserve legal modality (may, must, shall, shall not).

# CHAPEAU AND SUB-CLAUSE SPLITTING RULE
When an article contains a chapeau followed by numbered (1, 2, 3) or lettered (a, b, c) sub-clauses:
- Treat each lowest-level sub-clause as a separate provision.
- Each extracted provision MUST explicitly restate the governing actor and duty from the chapeau.
- Do NOT output sub-clauses without the actor.
- Preserve the article number and title verbatim.
- If a provision spans multiple paragraphs, always include the chapeau actor in each paragraph.

# OUTPUT FORMAT (STRICT)
Return ONLY a JSON object matching the following schema:
{
  "provisions": [
    {
      "text": "<EXTRACTED_VERBATIM_TEXT>"
    }
  ]
}

# STRICT OUTPUT FORMAT INSTRUCTIONS
You MUST respond with ONLY valid JSON that conforms exactly to the provided schema.
No explanations. No reasoning. No natural language.
Do NOT include markdown, code fences, or commentary.
The entire response must be a single JSON object and nothing else.
"""

# -----------------------------
# Ensure PDF exists
# -----------------------------
if not pdf_path.exists():
    print(f"PDF not found for law {law_id} at {pdf_path}")
    exit(1)

# -----------------------------
# Run Stage 1 for this law
# -----------------------------
provisions = process_test_law(
    pdf_path=str(pdf_path),
    law_id=law_id,
    model=model,
    system_instructions=system_instructions,
    max_chars=100000,
    overlap_chars=1000
)

# -----------------------------
# Append to output file
# -----------------------------
if provisions:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a", encoding="utf-8") as f:
        for prov in provisions:
            f.write(json.dumps(prov, ensure_ascii=False) + "\n")

print(f"Processed {law_id}, extracted {len(provisions)} provisions, appended to {output_file}")