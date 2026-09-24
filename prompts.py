def build_prompt(sample):

    mention = sample["mention"]
    left = sample["context_left"]
    right = sample["context_right"]

    prompt = f"""You are helping an Entity Linking system.

Your task:
Given a text with a marked entity mention, generate ONE short factual sentence
that helps identify which specific entity the mention refers to.

STRICT RULES:
- One sentence only
- Maximum 25 words
- Must include the exact mention
- Focus on disambiguating facts: type, origin, field, role
- Use world knowledge
- Describe the entity using general world knowledge, NOT information from the text above
- Focus on: type, origin, field, nationality, role — facts that uniquely identify the entity
- Do NOT invent facts
- Do NOT output XML, JSON, markdown, or tags
- Do NOT output more than one sentence

BAD EXAMPLE (only repeats the name):
Lionel Messi is Lionel Messi.

BAD EXAMPLE (uses XML tags):
<answer id="1">Lionel Messi is a footballer.</answer>

GOOD EXAMPLE (adds disambiguating facts):
Mention: Lionel Messi
Output: Lionel Messi is an Argentine football player who won multiple Ballon d'Or awards.

GOOD EXAMPLE (identifies role and country):
Mention: Secretary of State for India
Output: Secretary of State for India was a British government position overseeing colonial India.

GOOD EXAMPLE (short and specific):
Mention: New Look
Output: New Look is a British fashion retail company.

-------------------------------------

Text:
{left} {{ {mention} }} {right}

Mention: {mention}
Output:"""

    return prompt