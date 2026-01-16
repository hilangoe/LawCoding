# this script contains all the functions to prep the training data for fine-tuning (LoRA) BERT models
# this code will take the raw law texts (PDFs) and reconstruct the snippets/provisions relevant to the human-coded data

import json, os, time, jsonschema, logging
from openai import AzureOpenAI, OpenAI

# setting the logger from the run script
logger = logging.getLogger(__name__)

# defining the base_url
base_url = "https://hans-ai-foundry.openai.azure.com/openai/v1/"

# models deployed: gpt-5, gpt-5-mini, gpt-40-mini, text-embedding-ada-002

api_key = os.getenv("AZURE_OPENAI_API_KEY_2")

# new
client = OpenAI(
    api_key=api_key,
    base_url=base_url,          # base url for our resource
#    api_version="2024-12-01-preview" # this is for chatgpt5, might need to switch depending on model
)

# pulling in this function that loads and cleans PDF
from extract_text import _ensure_text

# next we need to read the JSON, but holding off waiting on format
# this needs to output requested_keys and key_data
def load_human_data(human_json_path):
    """
    Loads the human-coded JSON for a law and extracts:
    - requested_keys: list of provision keys to extract with the LLM
    - key_data: mapping of key -> deontic code (-1 or 1)

    Assumes JSON has been cleaned:
      - Code is either -1, 1, or "N/A"
      - Only -1 and 1 are included in requested_keys
    """

    with open(human_json_path, "r") as f:
        data = json.load(f)

    provisions = data.get("Provisions", [])

    requested_keys = []
    key_data = {}

    for p in provisions:
        key = p.get("Provision")
        deontic = p.get("Code")
    
        if key is None or deontic is None or deontic == "N/A":
            continue

        # ensure deontic is numeric
        try:
            deontic = int(deontic)
        except (ValueError, TypeError):
            logger.warning(
                "Skipping provision %s with invalid deontic value: %r",
                key,
                deontic
            )
            continue

        if deontic not in (-1, 1):
            logger.warning(
                "Skipping provision %s with out-of-range deontic value: %d",
                key,
                deontic
            )
            continue

    
        if key not in key_data:  # avoids duplicates
            requested_keys.append(key)
            key_data[key] = deontic

    return requested_keys, key_data

# schema goes here
schema = {
  "type": "object",
  "properties": {
    "LawID": { "type": "string" },
    "Provisions": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "Provision": { "type": "string" },
          "Text": { "type": "string" }
        },
        "required": ["Provision", "Text"],
        "additionalProperties": False
      }
    }
  },
  "required": ["LawID", "Provisions"],
  "additionalProperties": False
}

# extracting key definitions here
def extract_key_definitions(requested_keys, codebook):
    """
    Extract Key and ADICO for actor-level keys inside each codebook entry.
    """
    requested = set(requested_keys)
    blocks = []

    for entry in codebook:
        actors = entry.get("Actors", {}) or {}
        for actor_info in actors.values():
            key = actor_info.get("Key")
            if key in requested:
                blocks.append(
                    f"Key: {key}\nADICO: {actor_info.get('ADICO', '')}\n"
                )

    return "\n".join(blocks)

# system instruction function
def get_system_instructions_train(law_id, requested_keys, codebook) -> str:
    key_definitions = extract_key_definitions(requested_keys, codebook)

    system_instructions = f"""
You are a legal expert trained in freedom-of-expression laws.
Your task is to read a full legal text and extract provisions relevant to specific codebook definitions.

# TASK
1. Read and understand the law carefully.
2. For each key listed below, identify ALL relevant provisions of the law.
3. Extract only the verbatim text of each relevant provision.
4. Return the output as a JSON object matching the schema.

# MULTILINGUAL TEXT RULE
If the law is not in English:
  1. Provide a faithful, literal English translation.
  2. Do NOT include the original-language text.
  3. Do not summarize, paraphrase, shorten, or reinterpret.
  4. Preserve legal modality (may, must, shall, shall not).

# TOKEN/KEY LIMIT RULES (MANDATORY)
- For each each key:
    * Produce exactly ONE provision object.
    * If multiple parts of the law are relevant, concatenate them into a single Text string.
    * Do NOT return multiple objects for the same key.
    * Do not exceed **500 tokens** of extracted text.
    * If the law contains more relevant text than this limit, include only the **most central, authoritative, or defining segments**.
    * Never concatenate entire chapters or long sections—extract only the essential normative part.
- Never output more than **1500 tokens total** for the entire JSON.

# OUTPUT FORMAT (STRICT)
Return ONLY a JSON object, no commentary or markdown, in the following format:
{{
  "LawID": "{law_id}",
  "Provisions": [
    {{
      "Provision": "<KEY>",
      "Text": "<EXTRACTED_VERBATIM_TEXT>"
    }}
  ]
}}

# STRICT OUTPUT FORMAT INSTRUCTIONS
You MUST respond with ONLY valid JSON that conforms exactly to the provided schema.
No explanations. No reasoning. No natural language.
Do NOT include markdown, code fences, or commentary.
The entire response must be a single JSON object and nothing else.

# KEY DEFINITIONS FOR THIS LAW
The following definitions apply only to the keys you must extract:

{key_definitions}
"""

    return system_instructions


# need function for chunking because of long laws, now with overlap
def iter_chunks(law_text, max_chars=100000, overlap_chars=1000):
    """
    Memory-safe chunk generator.
    Yields one chunk at a time instead of building a full list.
    """
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    start = 0
    text_len = len(law_text)

    while start < text_len:
        end = min(start + max_chars, text_len)

        # try to split at last newline
        if end < text_len:
            newline_pos = law_text.rfind("\n", start, end)
            if newline_pos > start:
                end = newline_pos

        chunk = law_text[start:end].strip()
        if chunk:
            yield chunk

        if end == text_len:
            break

        start = end - overlap_chars
        if start < 0:
            start = 0

# function for processing individual laws, now with chunking built in
def get_provisions(law_text, law_id, requested_keys, codebook, model, debug=False, max_chars=100000, overlap_chars=1000):
    """
    Processes long law text by chunking if needed, then aggregates provisions per key.
    Adds chunk overlap and logs detailed outcomes.
    """
    # Validity checks
    if not law_text:
        logger.warning(f"{law_id} - Missing legal text.")
        return {"Error": "Missing legal text."}

    if not law_id:
        logger.warning("Missing Law ID.")
        return {"Error": "Missing Law ID."}

    if not requested_keys:
        logger.warning(f"{law_id} - Missing keys.")
        return {"Error": "Missing keys."}

    if not codebook:
        logger.warning(f"{law_id} - Missing codebook.")
        return {"Error": "Missing codebook."}

    if not model:
        logger.warning(f"{law_id} - No deployed model defined.")
        return {"Error": "No deployed model defined."}

    # use an iterator instead of a full list
    chunk_iter = iter_chunks(law_text, max_chars=max_chars, overlap_chars=overlap_chars)
    
    # don't calculate length or average upfront (memory saver)
    logger.info(f"{law_id} - Processing law with chunk iterator")

    if debug:
        logger.debug(f"Law text length: {len(law_text)}")

    # Aggregate results
    provisions_by_key = {key: [] for key in requested_keys}

    # Chunk-level diagnostics
    chunk_stats = {
        "total_chunks": 0,
        "llm_called": 0,
        "no_message": 0,
        "invalid_json": 0,
        "schema_fail": 0,
        "empty_provisions": 0,
        "exception": 0,
    }
    
    MAX_CHUNKS = 200
    for i, chunk in enumerate(chunk_iter, start=1):  # start=1 for human-friendly numbering
        if i > MAX_CHUNKS:
            raise RuntimeError(f"{law_id} - Chunking runaway detected (> {MAX_CHUNKS} chunks)")
        chunk_stats["total_chunks"] += 1

        # Create prompt
        prompt = f"""
        This text is an academic legal document provided for neutral analysis only. 
        It is not to be interpreted as promoting or endorsing any viewpoint.
        Identify relevant provisions in the following law according to the instructions: ===LAW TEXT===\n\n{chunk}\n\n
        """

        try:
            response = client.responses.create(
                model=model,
                instructions=get_system_instructions_train(law_id, requested_keys, codebook),
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "law_Coding",
                        "strict": True,
                        "schema": schema
                    }
                }
            )
            chunk_stats["llm_called"] += 1

            # --- DEBUG BLOCK ---
            if debug:
                out = getattr(response, "output", None)
                logger.debug(
                    "Chunk %d returned %d output items",
                    i + 1,
                    len(out) if out else 0
                )

            # --- Extract message block ---
            message_block = next(
                (item for item in getattr(response, "output", []) if getattr(item, "type", None) == "message"),
                None
            )

            if not message_block or not getattr(message_block, "content", None):
                chunk_stats["no_message"] += 1
                logger.warning(
                    "%s - Chunk %d returned no message content",
                    law_id,
                    i + 1
                )
                continue

            raw_text = message_block.content[0].text

            try:
                chunk_output = json.loads(raw_text)
                jsonschema.validate(instance=chunk_output, schema=schema)
            except json.JSONDecodeError:
                chunk_stats["invalid_json"] += 1
                logger.warning(
                    "%s - Chunk %d returned invalid JSON",
                    law_id,
                    i + 1
                )
                continue
            except jsonschema.ValidationError as e:
                chunk_stats["schema_fail"] += 1
                logger.warning(
                    "%s - Chunk %d JSON schema validation failed",
                    law_id,
                    i + 1)
                continue

            provisions = chunk_output.get("Provisions", [])
            if not provisions:
                chunk_stats["empty_provisions"] += 1
                logger.info(
                    "%s - Chunk %d returned 0 provisions",
                    law_id,
                    i + 1)

            # Aggregate by key
            for item in provisions:
                key = item.get("Provision")
                text = item.get("Text", "").strip()
                if key in provisions_by_key and text:
                    provisions_by_key[key].append(text)

        except Exception as e:
            chunk_stats["exception"] += 1
            logger.error(
                "%s - Chunk %d LLM call failed",
                law_id,
                i + 1,
                exc_info=e
            )
            continue

    # Build final aggregated output
    aggregated_provisions = [
        {"Provision": key, "Text": "\n\n".join(texts)}
        for key, texts in provisions_by_key.items() if texts
    ]

    logger.info("%s - Chunk processing summary: %s", law_id, chunk_stats)
    
    if debug:
        logger.debug("--- Chunk processing summary ---")
        for k, v in chunk_stats.items():
            logger.debug("%s: %d", k, v)

    if not aggregated_provisions:
        return {"LawID": law_id, "Provisions": [], "Warning": "No provisions extracted from any chunk."}

    return {"LawID": law_id, "Provisions": aggregated_provisions}

# merging texts with human-coded data
def merge_text_data(provisions, key_data):
    """
    Merge LLM-extracted provisions with human-coded deontic values.

    Parameters
    ----------
    provisions : dict
        Output from get_provisions(), of the form:
        {
            "LawID": 123,
            "Provisions": [
                {"Provision": "C_DISINFO_GEN", "Text": "..."},
                {"Provision": "P_DISINFO_GEN", "Text": "..."},
                ...
            ]
        }

    key_data : dict
        From human-coded JSON:
        {
            "C_DISINFO_GEN": 1,
            "P_DISINFO_GEN": -1,
            ...
        }

    Returns
    -------
    list of dict
        Each dict has: law_id, key, text, deontic
    """
    # validate data input
    if not provisions or "Provisions" not in provisions:
        return []

    law_id = provisions.get("LawID")

    # initializing rows
    merged = []

    # merge inputs on key

    for item in provisions["Provisions"]:
        key = item.get("Provision")
        text = item.get("Text", "").strip()

        if not key:
            continue

        deontic = key_data.get(key)

        # only include keys that have human-coded deontic (might be unnecessary)
        if deontic is None:
            continue

        merged.append({
            "law_id": law_id, # adding law_id so we can distinguish between real and synthetic data
            "key": key,
            "text": text,
            "deontic": deontic
        })

    # return dictionary
    return merged

# note: the paths will be generated in the run script from the training sample input list
def process_law(law_id, pdf_path, human_json_path, model, codebook, debug=False, max_chars=100000, overlap_chars=1000):
    """
    Orchestrates processing for one law:
    - Read PDF
    - Load human-coded data
    - Extract keys
    - Run LLM extraction
    - Merge results
    """

    # extract PDF text
    law_text = _ensure_text(pdf_path)
    if not law_text:
        logger.error("%s - No extracted text from PDF", law_id)
        return {
            "LawID": law_id,
            "Success": False,
            "Results": [],
            "Error": f"No extracted text for {law_id}."
        }

    # load human-coded data
    try:
        requested_keys, key_data = load_human_data(human_json_path)
        logger.info(
            "%s - Loaded %d requested keys from human data",
            law_id,
            len(requested_keys)
        )

        if debug:
            logger.debug(
                "%s - Requested keys: %s",
                law_id,
                requested_keys
            )
    except Exception as e:
        return {
            "LawID": law_id,
            "Success": False,
            "Results": [],
            "Error": f"Failed to load human-coded data for {law_id}: {e}"
        }

    if not requested_keys:
        return {
            "LawID": law_id,
            "Success": False,
            "Results": [],
            "Error": f"No requested keys found in human data for {law_id}."
        }

    # run LLM extraction
    try:
        provisions = get_provisions(
            law_text=law_text,
            law_id=law_id,
            requested_keys=requested_keys,
            codebook=codebook,
            model=model,
            debug=debug,
            max_chars=50000
        )
        num_provisions = len(provisions.get("Provisions", [])) if provisions else 0
        logger.info(
            "%s - LLM returned %d provisions",
            law_id,
            num_provisions
        )

        for item in provisions.get("Provisions", []):
            key = item.get("Provision")
            if key not in key_data:
                logger.warning(
                    "%s - LLM returned key not in human JSON: %s",
                    law_id,
                    key
                )

    except Exception as e:
        return {
            "LawID": law_id,
            "Success": False,
            "Results": [],
            "Error": f"LLM extraction failed for {law_id}: {e}"
        }

    if not provisions or "Provisions" not in provisions:
        return {
            "LawID": law_id,
            "Success": False,
            "Results": [],
            "Error": f"LLM returned no provisions for {law_id}."
        }

    # merge data
    try:
        merged_rows = merge_text_data(provisions, key_data)
        logger.info(
            "%s - Merged %d rows",
            law_id,
            len(merged_rows)
        )
    except Exception as e:
        return {
            "LawID": law_id,
            "Success": False,
            "Results": [],
            "Error": f"Merging failed for {law_id}: {e}"
        }

    # success path
    return {
        "LawID": law_id,
        "Success": True,
        "Results": merged_rows,
        "Error": None
    }

# processing multiple laws and stitching together
def process_multiple_laws(law_list, model, codebook, pdf_dir, json_dir, debug=False, max_chars=100000, overlap_chars=1000):
    """
    Processes multiple laws end-to-end.

    Parameters
    ----------
    law_list : list[str]
        List of unique law IDs (e.g., ["ARG_2018_FOIA", "NOR_2022_DISINFO"]).

    model : str
        Deployed Azure model name to use for LLM extraction.

    codebook : list[dict]
        Parsed codebook JSON already loaded into Python.

    pdf_dir : str
        Folder where PDFs are stored (files must match law_id + ".pdf")

    json_dir : str
        Folder where human-coded JSON files are stored 
        (files must match law_id + ".json")

    Returns
    -------
    dict
        {
            "Successes": [...],   # rows ready for training
            "Errors": [...],      # error objects from process_law()
        }
    """

    all_rows = [] # all training rows across laws
    errors = [] # all error records

    for law_id in law_list:
        logger.info("--- Processing law: %s ---", law_id)
        # identifying all possible matches        
        all_files = os.listdir(pdf_dir)
        logger.info(
            "%s - Looking for PDF files in %s",
            law_id,
            pdf_dir
        )
        
        if debug:
            logger.debug(
                "%s - First 10 files in directory: %s",
                law_id,
                all_files[:10]
            )
        
        pdf_candidates = [
            os.path.join(pdf_dir, f)
            for f in all_files
            if law_id.lower() in f.lower() and f.lower().endswith(".pdf")
        ]
        if debug:
            logger.debug(
                "%s - PDF candidates found: %s",
                law_id,
                pdf_candidates
            )
        
        if not pdf_candidates:
            logger.error(
                "%s - No PDF files found in %s",
                law_id,
                pdf_dir
            )
            errors.append({
                "LawID": law_id,
                "Success": False,
                "Results": [],
                "Error": f"No PDF files found for {law_id}"
            })
            continue
        
        # prefer English translation if present, using next generator to loop through candidates looking for ENG
        pdf_path = next((p for p in pdf_candidates if "ENG" in os.path.basename(p).upper()), pdf_candidates[0]) # first item set as default fallback
        
        # JSON: building file paths
        json_path = os.path.join(json_dir, f"{law_id}.json")

        if debug:
            logger.debug(
                "%s - JSON exists at %s: %s",
                law_id,
                json_path,
                os.path.isfile(json_path)
            )


        # existence check
        if not os.path.isfile(json_path):
            logger.error(
                "%s - Missing JSON file at %s",
                law_id,
                json_path
            )
            errors.append({
                "LawID": law_id,
                "Success": False,
                "Results": [],
                "Error": f"Missing JSON file at {json_path}"
            })
            continue
        
        # run full processing for one law
        result = process_law(
            law_id=law_id,
            pdf_path=pdf_path,
            human_json_path=json_path,
            model=model,
            codebook=codebook,
            debug=debug,
            max_chars=50000
        )

        # collecting results
        if result["Success"]:
            all_rows.extend(result["Results"])
        else:
            errors.append(result)

    # --- PIPELINE SUMMARY LOG ---
    logger.info("=== PIPELINE SUMMARY ===")
    logger.info("Total laws processed: %d", len(law_list))
    logger.info("Successful rows: %d", len(all_rows))
    logger.info("Errors: %d", len(errors))
    
    for e in errors:
        logger.warning(
            "Error in %s: %s",
            e.get("LawID"),
            e.get("Error")
        )

    return {
        "Successes": all_rows,
        "Errors": errors
    }

# prepping data for fine-tuning
def create_training_rows(law_outputs, codebook, deontic_labels=None):
    """
    Build training rows from raw_results['Successes'] directly.

    Inputs:
        law_outputs: list of provision dicts
        codebook: list of dicts with full ontology
        deontic_labels: optional list of all deontic categories. If None, use [-1, 1]

    Returns:
        rows: list of dicts (training samples)
        key_to_id: mapping from key label -> numeric ID
        deontic_to_id: mapping from deontic label -> numeric ID
    """
    # extract all ontology keys from codebook
    ontology_keys = set()
    for entry in codebook:
        actors = entry.get("Actors", {}) or {}
        for actor_info in actors.values():
            key = actor_info.get("Key")
            if key:
                ontology_keys.add(key)

    key_to_id = {k: i for i, k in enumerate(sorted(ontology_keys))}

    if deontic_labels is None:
        deontic_labels = [-1, 1]

    deontic_to_id = {d: i for i, d in enumerate(deontic_labels)}

    rows = []
    for prov in law_outputs:
        rows.append({
            "law_id": prov["law_id"], # adding law_id here so we can distinguish between real and synthetic
            "text": prov["text"],
            "key_label": key_to_id.get(prov["key"]),
            "deontic_label": deontic_to_id.get(prov["deontic"]),
            "key": prov["key"],
            "deontic": prov["deontic"]
        })

    return rows, key_to_id, deontic_to_id

def build_metadata(codebook, metadata_path):
    """
    Build metadata needed for model training.
    """
    # Extract unique ontology keys
    ontology_keys = set()
    for entry in codebook:
        actors = entry.get("Actors", {}) or {}
        for actor_info in actors.values():
            key = actor_info.get("Key")
            if key:
                ontology_keys.add(key)

    metadata = {
        "num_keys": len(ontology_keys),
        "num_deontic": 2   # always two labels: -1 and 1
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)