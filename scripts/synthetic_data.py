import json, os, time, jsonschema

from training_prep_library, import extract_key_definitions

from openai import AzureOpenAI, OpenAI

from collections, import Counter

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

# schema, the LawID will be "synth"
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
          "Text": { "type": "string" },
            "Deontic": {"type": "integer"}
        },
        "required": ["Provision", "Text", "Deontic"],
        "additionalProperties": False
      }
    }
  },
  "required": ["LawID", "Provisions"],
  "additionalProperties": False
}

def get_system_instructions_synth(key, n, codebook) -> str:
    # only grabbing one definition here, since we're making one set of synthetic observations per key
    key_definition = extract_key_definitions(key, codebook)

    system_instructions = f"""You are a legal expert trained in freedom-of-expression laws.
Your task is to read a generic ADICO statement and generate synthetic law provisions with an accompanying deontic value.

# TASK
1. Read and understand the ADICO definition carefully.
2. Write {n} different synthetic law provisions that vary in content and in deontic value.
3. Record the deontic value of each synthetic provision.
4. Return the output as a JSON object matching the schema with LawID = "synth" for all.
5. The "Provisions" array must contain exactly {n} items.
6. "Provision" must always equal "{key}" exactly.

# DEONTIC VALUES
Each synthetic provision must either indicate whether the ADICO statement is present (1) or negated (-1).
Deontic must be an integer, not a string.

# OUTPUT FORMAT (STRICT)
Return ONLY a JSON object, no commentary or markdown, in the following format:
{{
  "LawID": "synth",
  "Provisions": [
    {{
      "Provision": "{key}",
      "Text": "synthethic text",
      "Deontic": 1
    }}
  ]
}}

# STRICT OUTPUT FORMAT INSTRUCTIONS
You MUST respond with ONLY valid JSON that conforms exactly to the provided schema.
No explanations. No reasoning. No natural language.
Do NOT include markdown, code fences, or commentary.
Do NOT escape JSON.
Do NOT wrap the JSON in quotes.
The entire response must be a single JSON object and nothing else.

# KEY DEFINITION
The following definition represents the ADICO statement:

{key_definition}
"""

    return system_instructions

def generate_synthetic_provisions(key, n, codebook, model):

    # validation section

    # user prompt
    prompt = f"""
        Generate synthetic law provisions according to the instructions
    """
    
    start_time = time.time()
    
    response = client.responses.create(
        model=model,
        instructions=get_system_instructions_synth(key, n, codebook),
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
                "Error": f"Could not locate message content for {key}",
                "raw": repr(response)
            }

        # --- extract text safely ---
        if not getattr(message_block, "content", None):
            return {
                "Error": f"LLM returned empty content block for {key}",
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
            "Error": f"LLM extraction failed for {key}: {e}",
            "raw": repr(response)
        }

# function for identifying keys with too few samples
def sparse_keys(rows, threshold=20):
    """
    Identify underrepresented keys in training rows.

    Inputs:
        rows: list of dicts (output of create_training_rows)
        threshold: int or None
            - keys with count < threshold are returned
            - if None, all keys are returned

    Returns:
        dict: {key: count}
    """

    key_counts = Counter(row["key"] for row in rows)

    if threshold is None:
        return dict(key_counts)

    return {
        key: count
        for key, count in key_counts.items()
        if count < threshold
    }

def generate_all_synthetic_rows(keys, n_map, codebook, model, threshold=20):
    """
    Generate synthetic provisions for multiple keys and return them in
    the 'law_output' format compatible with create_training_rows.
    """
    all_rows_synth = []

    for key in keys:
        current = n_map.get(key, 0)
        n = threshold - current # number of samples needed per key to meet threshold
        if n <= 0:
            continue
        
        result = generate_synthetic_provisions(key, n, codebook, model)

        # Skip if there was an error
        if "Provisions" not in result:
            print(f"Error generating synthetic data for key {key}: {result}")
            continue

        # Warn but still process if the list is empty
        if not result["Provisions"]:
            print(f"Warning: no synthetic provisions generated for key {key}")

        # Convert to structure expected by create_training_rows
        result_converted = {
            "Success": True, # this is needed to match the output from the provision extraction
            "LawID": result.get("LawID", "synth"),
            "Results": [
                {
                    "key": p["Provision"],
                    "text": p["Text"],
                    "deontic": p["Deontic"]
                }
                for p in result["Provisions"]
            ]
        }

        all_rows_synth.append(result_converted)

    return all_rows_synth