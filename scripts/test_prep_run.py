# this is the run script for prepping the test data (initialization and stage 1)

import logging, os, json, requests
from test_prep_library import process_multiple_test_laws
from training_prep_library import iter_chunks
from pathlib import Path
import pandas as pd

# -----------------------------
# Logging
# -----------------------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / "law_test_prep.log"

# log config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler() # printing to console too
    ]
)

logger = logging.getLogger(__name__)

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent # going back up one level to the project folder

data_path = BASE_DIR / "data"
data_path.mkdir(parents=True, exist_ok=True)

pdf_dir = BASE_DIR / "data" / "laws_pdf"
pdf_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Test set
# -----------------------------

# setting path for test list
csv_path = data_path / "test.csv"

df = pd.read_csv(csv_path)

# creating law_list
law_list = (
    df["path"]
      .apply(lambda p: Path(p).stem)  # drop folder + .json
      .dropna()
      .unique()
      .tolist()
)


# -----------------------------
# Codebook
# -----------------------------

# Grabbing the codebook from the huggingface repo
codebook_url = "https://huggingface.co/spaces/raulpzs/expression_laws/resolve/main/data3/codebook.json"
filename = "cb_expression_rights.json"

response = requests.get(codebook_url)
response.raise_for_status()

# downloading the codebook JSON
with open(filename, "wb") as f:
    f.write(response.content)

# loading the codebook JSON once
with open(filename, "r") as f:
    codebook = json.load(f)

# -----------------------------
# Model
# -----------------------------

model = "gpt-5-mini"

# -----------------------------
# System instructions
# -----------------------------

system_instructions = """
You are a legal text processing assistant. Your job is to extract ONLY legally operative provisions regulating self-expression.

# TASK
1. Split legal texts into distinct provisions or articles.
2. Ensure each provision is self-contained.
3. Preserve numbering, headings, and structure.
4. Output results in JSON format

# TOPICAL SCOPE (STRICT)
Extract ONLY provisions that regulate, restrict, permit, or require conduct related to:
- speech, expression, opinion
- information dissemination
- media, press, journalism
- communication (including digital or online)
- misinformation or disinformation

If a provision does NOT concern self-expression or information-related conduct, DO NOT extract it.

# NORMATIVE RULE REQUIREMENT (CRITICAL)
Extract ONLY provisions that contain an explicit, enforceable legal rule.

A valid provision MUST:
- identify a specific actor, AND
- impose, prohibit, permit, or condition conduct using clear legal modality.

Legal modality includes, but is not limited to:
must, shall, may, may not, shall not,
is prohibited, is forbidden, is unlawful,
is required, is obliged, is mandatory,
is permitted only if, is allowed subject to,
is punishable by, is subject to sanctions.

DO NOT extract:
- definitions or interpretive clauses
- statements of purpose or objectives
- declaratory rights without conditions, duties, or restrictions

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
{{
  "provisions": [
    {{
      "text": "<EXTRACTED_VERBATIM_TEXT>"
    }}
  ]
}}

# STRICT OUTPUT FORMAT INSTRUCTIONS
You MUST respond with ONLY valid JSON that conforms exactly to the provided schema.
No explanations. No reasoning. No natural language.
Do NOT include markdown, code fences, or commentary.
The entire response must be a single JSON object and nothing else.
"""

# --- RUN STAGE 1 ---

print("Starting Stage 1 extraction...")

provisions = process_multiple_test_laws(
    law_list=law_list,
    model=model,
    system_instructions = system_instructions,
    pdf_dir=pdf_dir,
    max_chars=100000,
    overlap_chars=1000
)

# --- SAVE OUTPUT ---

output_file = data_path / "test_provisions.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for prov in provisions:
        f.write(json.dumps(prov, ensure_ascii=False) + "\n")

logger.info("Stage 1: %d laws processed, %d provisions extracted", len(law_list), len(provisions))
logger.info("Saved provisions to: %s", output_file)
