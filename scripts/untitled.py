# this is the new version
base_url = "https://hans-ai-foundry.openai.azure.com/openai/v1/"

# this is the old version
base_url = "https://hans-ai-foundry.openai.azure.com/"

# this is the new version
client = OpenAI(
    api_key=api_key,
    base_url=base_url,          # base url for our resource
#    api_version="2024-12-01-preview" # this is for chatgpt5, might need to switch depending on model
)


# this is the old version
client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=base_url,          # base url for our resource
    api_version="2024-12-01-preview" # this is for chatgpt5, might need to switch depending on model
)


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


# this is the old version
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": get_system_instructions_train(law_id, requested_keys, codebook)},
            {"role": "user", "content": prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "law_coding",
                "schema": schema,
                "strict": True,
            },
        },
    )