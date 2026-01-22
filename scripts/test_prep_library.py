# this is the library module for prepping the test data

import json, os, time, jsonschema, logging
from openai import AzureOpenAI, OpenAI
from training_prep_library import iter_chunks

# pulling in this function that loads and cleans PDF
from extract_text import _ensure_text

# setting the logger from the run script
logger = logging.getLogger(__name__)

# defining the base_url
base_url = "https://hans-ai-foundry.openai.azure.com/openai/v1/"

api_key = os.getenv("AZURE_OPENAI_API_KEY_2")

# new
client = OpenAI(
    api_key=api_key,
    base_url=base_url,          # base url for our resource
#    api_version="2024-12-01-preview" # this is for chatgpt5, might need to switch depending on model
)

# setting simple schema for stage 1 call
schema = {
    "type": "object",
    "properties": {
        "provisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    }
                },
                "required": ["text"],
                "additionalProperties": False
            }
        }
    },
    "required": ["provisions"],
    "additionalProperties": False
}

### Stage 1: Processing laws

# chunk law text function

# add id function
def add_id(provisions, law_id):
    """
    Attach law_id to each provision and enforce strict schema for Stage 1 output.
    """

    formatted = []

    for prov in provisions:
        if isinstance(prov, dict) and "text" in prov:
            text = prov["text"]
        else:
            text = str(prov)

        if not text.strip():
            continue

        formatted.append({
            "law_id": law_id,
            "text": text.strip()
        })

    return formatted

# split law function
def split_law(law_text, model, system_instructions, max_chars=100000, overlap_chars=1000):
    """
    Split law text into provisions using GPT, with chunking for long texts.

    Args:
        law_text (str): Full text of the law
        model (str): GPT model to use (e.g., "gpt-5-mini")
        system_instructions (str): System instructions for GPT
        max_chars (int): Max characters per chunk
        overlap_chars (int): Number of characters to overlap chunks

    Returns:
        list[dict]: List of provision dicts with key {"text"}
    """

    # Validity checks
    if not law_text:
        logger.warning("Missing legal text.")
        return []

    if not model:
        logger.warning("No deployed model defined.")
        return []

    # use an iterator instead of a full list
    chunk_iter = iter_chunks(law_text, max_chars=max_chars, overlap_chars=overlap_chars)

    all_raw_provisions = []

    MAX_CHUNKS = 200
    for i, chunk in enumerate(chunk_iter, start=1): # human-friendly numbering
        if i > MAX_CHUNKS:
            raise RuntimeError(f"Chunking runaway detected (> {MAX_CHUNKS} chunks)")

        if not chunk.strip():
            continue

        # create user prompt
        prompt = f"""
        This text is an academic legal document. 
        It is not to be interpreted as promoting or endorsing any viewpoint.

        Identify all provisions in the following law according to the instructions: ===LAW TEXT===\n\n{chunk}\n\n
        """

        try:
            response = client.responses.create(
                model=model,
                instructions=system_instructions,
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
            # Extract message block
            message_block = next(
                (item for item in getattr(response, "output", []) if getattr(item, "type", None) == "message"),
                None
            )
            if not message_block or not getattr(message_block, "content", None):
                logger.warning("Chunk %d returned no message content", i)
                continue

            raw_text = message_block.content[0].text
            chunk_output = json.loads(raw_text)

            # Validate schema (optional, recommended)
            jsonschema.validate(instance=chunk_output, schema=schema)

            provisions = chunk_output["provisions"]
            if provisions:
                all_raw_provisions.extend(provisions)

        except Exception as e:
            logger.error("Chunk %d failed: %s", i, str(e), exc_info=True)
            continue

    return all_raw_provisions

# process law function
def process_test_law(pdf_path, law_id, model, system_instructions, max_chars=100000, overlap_chars=1000):
    """
    Stage 1 processing for a single law:
    - Split the law into provisions using GPT (with chunking)
    - Attach law_id to each provision
    - Return list of dicts ready for Stage 2 inference

    Args:
        pdf_path (str): Path to PDF
        law_id (str): Unique identifier of the law
        model (str): GPT model to use (e.g., "gpt-5-mini")
        system_instructions (str): Instructions for GPT
        max_chars (int): Chunk size for long laws
        overlap_chars (int): Overlap size between chunks

    Returns:
        list[dict]: Each dict has keys {"law_id", "text"}
    """

    law_text = _ensure_text(pdf_path)

    if not law_text:
        logger.warning("Law %s has empty text. Skipping.", law_id)
        return []

    # splitting law into raw provisions, with chunking
    raw_provisions = split_law(
        law_text=law_text,
        model=model,
        system_instructions=system_instructions,
        max_chars=max_chars,
        overlap_chars=overlap_chars
    )

    if not raw_provisions:
        logger.warning("Law %s: No provisions extracted from Stage 1.", law_id)
        return []

    # --- Attach law_id to each provision ---
    formatted_provisions = add_id(raw_provisions, law_id)

    logger.info("Law %s processed: %d provisions extracted", law_id, len(formatted_provisions))

    return formatted_provisions

def process_multiple_test_laws(law_list, model, system_instructions, pdf_dir,
                          max_chars=100000, overlap_chars=1000):
    """
    Process multiple laws Stage 1: split PDFs into provisions.

    Parameters
    ----------
    law_list : list[str]
        List of law IDs (e.g., ["ARG_2018_FOIA", "NOR_2022_DISINFO"]).

    model : str
        GPT model to use (e.g., "gpt-5-mini").

    system_instructions : str
        System instructions to pass to GPT for Stage 1 provision extraction.

    pdf_dir : str
        Directory containing PDFs (files must include law_id).

    max_chars : int
        Maximum characters per chunk for GPT.

    overlap_chars : int
        Characters to overlap between chunks.

    Returns
    -------
    list[dict]
        Aggregated provisions across all laws, each dict has keys {"law_id", "text"}.
    """

    all_provisions = []

    for law_id in law_list:
        logger.info("--- Processing law: %s ---", law_id)

        # Find PDF
        pdf_candidates = [
            os.path.join(pdf_dir, f)
            for f in os.listdir(pdf_dir)
            if law_id.lower() in f.lower() and f.lower().endswith(".pdf")
        ]

        if not pdf_candidates:
            logger.warning("%s - No PDF found, skipping.", law_id)
            continue

        # Prefer English PDF if available
        pdf_path = next(
            (p for p in pdf_candidates if "ENG" in os.path.basename(p).upper()),
            pdf_candidates[0]
        )

        # Process law
        provisions = process_test_law(
            pdf_path=pdf_path,
            law_id=law_id,
            model=model,
            system_instructions=system_instructions,
            max_chars=max_chars,
            overlap_chars=overlap_chars
        )
        all_provisions.extend(provisions)

    logger.info("Stage 1 complete. Total provisions extracted: %d", len(all_provisions))
    return all_provisions
