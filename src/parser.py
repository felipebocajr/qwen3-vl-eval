"""Answer extraction from model responses."""

import re



# ESSE CODIGO PRECISA SER MUDADO PARA PEGAR APENAS A RESPOSTA (extracted_answer) E NÃO O TEXTO TODO. DEVE SER FEITO UM TESTE DE SIMILARIDADE COMPARANDO A RESPOSTA DO MODELO COM A RESPOSTA REAL.


def extract_answer(text: str, num_choices: int = 4) -> tuple[str | None, bool]:
    """
    Extract a multiple-choice letter answer from raw model text.

    Returns:
        (extracted_letter, succeeded)
    """
    text = text.strip()
    if not text:
        return None, False

    max_letter = chr(ord("A") + num_choices - 1)

    # Priority 1: first character is a valid option letter
    first = text[0].upper()
    if "A" <= first <= max_letter:
        return first, True

    # Priority 2: pattern like (B), [B], B), B., B:
    pattern = r"[\(\[]?\s*([A-" + max_letter + r"])\s*[\)\].:]"
    m = re.search(pattern, text)
    if m:
        return m.group(1).upper(), True

    # Priority 3: "answer is B", "option B", "correct answer: B"
    pattern = r"(?:answer|option|choice)\s*(?:is|:|\.|\s)\s*([A-" + max_letter + "])"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return m.group(1).upper(), True

    # Priority 4: any standalone valid capital letter
    pattern = r"(?<!\w)([A-" + max_letter + r"])(?!\w)"
    matches = re.findall(pattern, text.upper())
    if matches:
        return matches[-1], True

    return None, False
