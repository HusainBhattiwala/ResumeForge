PROMPT = """
You are a JSON extraction engine. Convert the following resume text into precisely the JSON schema specified below.
- Do not compose any extra fields or commentary.
- Do not make up values for any fields.
- Use "Present" if an end date is ongoing.
- Make sure dates are in YYYY-MM-DD.
- For "Extracted Keywords", extract relevant technical skills, technologies, tools, programming languages, frameworks, certifications, and industry-specific terms from the entire resume text.
- Do not format the response in Markdown or any other format. Just output raw JSON.

Schema:
```json
{0}
```

Resume:
```text
{1}
```

NOTE: Please output only a valid JSON matching the EXACT schema.
"""
