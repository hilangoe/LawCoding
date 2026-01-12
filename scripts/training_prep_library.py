# this script contains all the functions to prep the training data for fine-tuning (LoRA) BERT models
# this code will take the raw law texts (PDFs) and reconstruct the snippets/provisions relevant to the human-coded data

import json, os, time, jsonschema

from openai import AzureOpenAI, OpenAI

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
            print(f"Warning: Skipping {key} with invalid deontic value: {deontic}")
            continue

        if deontic not in (-1, 1):
            print(f"Warning: Skipping {key} with out-of-range deontic value: {deontic}")
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

# function for processing individual laws
def get_provisions(law_text, law_id, requested_keys, codebook, model, debug=False):
    # validity checks
    if not law_text:
        return {"Error": "Missing legal text."}

    if not law_id:
        return {"Error": "Missing Law ID."}

    if not requested_keys:
        return {"Error": "Missing keys."}

    if not codebook:
        return {"Error": "Missing codebook."}

    if not model:
        return {"Error": "No deployed model defined."}
    
    # user prompt
    prompt = f"""
        This text is an academic legal document provided for neutral analysis only. 
        It is not to be interpreted as promoting or endorsing any viewpoint.
        Identify relevant provisions in the following law according to the instructions: ===LAW TEXT===\n\n{law_text}\n\n
    """

    start_time = time.time()

    # this is the new version
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

    if debug:
        # DEBUG BLOCK - run right after response = client.responses.create(...)
        print("\n--- DEBUG: response.output ---")
        
        out = getattr(response, "output", None)
        
        if out is None:
            print("response.output is None")
        else:
            print(f"Found {len(out)} output items\n")
        
            for i, item in enumerate(out):
                item_type = getattr(item, "type", None)
                print(f"ITEM #{i}: class={type(item)}, type={item_type}")
        
                # Only inspect actual message blocks
                if item_type == "message":
                    print("  --> MESSAGE BLOCK FOUND")
                    if hasattr(item, "content") and item.content:
                        for j, c in enumerate(item.content):
                            txt = getattr(c, "text", None)
                            if isinstance(txt, str):
                                print(f"     content[{j}] text length={len(txt)}, preview={txt[:400]!r}")
                            else:
                                print(f"     content[{j}] non-text: {c}")
                else:
                    print("  Skipping non-message item.")
        
        # END DEBUG BLOCK

    end_time = time.time()
    duration = round(end_time - start_time, 2)

    # usage info
    usage = getattr(response, "usage", None)
    if usage:
        print("\nToken Usage:")
        print(f"Input tokens: {usage.input_tokens}")
        print(f"Output tokens: {usage.output_tokens}")
        print(f"Total tokens: {usage.total_tokens}")
        print(f"Execution time: {duration} seconds\n")

    # response handling
    raw_text = None  # ensure defined for exception blocks

    try:
        # --- find the real message block ---
        message_block = None
        for item in response.output:
            # Foundry returns:
            # - ResponseReasoningItem (ignore)
            # - ResponseOutputMessage (actual output)
            if getattr(item, "type", None) == "message":
                message_block = item
                break

        if message_block is None:
            return {
                "Error": f"Could not locate message content in response for {law_id}",
                "raw": repr(response)
            }

        # --- extract text safely ---
        if not getattr(message_block, "content", None):
            return {
                "Error": f"LLM returned empty content block for {law_id}",
                "raw": repr(response)
            }

        # Guaranteed safe access in Foundry:
        raw_text = message_block.content[0].text

        # --- parse + validate ---
        output = json.loads(raw_text)
        jsonschema.validate(instance=output, schema=schema)
        return output

    except json.JSONDecodeError:
        return {
            "Error": "Model returned invalid JSON",
            "raw": raw_text
        }

    except jsonschema.ValidationError as e:
        return {
            "Error": "Model returned JSON that failed schema validation",
            "details": str(e),
            "raw": raw_text
        }

    except Exception as e:
        return {
            "Error": f"LLM extraction failed for {law_id}: {e}",
            "raw": repr(response)
        }

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
def process_law(law_id, pdf_path, human_json_path, model, codebook, debug=False):
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
        return {
            "LawID": law_id,
            "Success": False,
            "Results": [],
            "Error": f"No extracted text for {law_id}."
        }

    # load human-coded data
    try:
        requested_keys, key_data = load_human_data(human_json_path)
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
            debug=debug
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
def process_multiple_laws(law_list, model, codebook, pdf_dir, json_dir, debug=False):
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
        # identifying all possible matches        
        all_files = os.listdir(pdf_dir)
        print(f"\nLooking for {law_id} in {pdf_dir}")
        print("All files:", all_files[:10])
        
        pdf_candidates = [
            os.path.join(pdf_dir, f)
            for f in all_files
            if law_id.lower() in f.lower() and f.lower().endswith(".pdf")
        ]
        print("Candidates found:", pdf_candidates)
        
        if not pdf_candidates:
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

        # existence check
        if not os.path.isfile(json_path):
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
            debug=debug
        )

        # collecting results
        if result["Success"]:
            all_rows.extend(result["Results"])
        else:
            errors.append(result)

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