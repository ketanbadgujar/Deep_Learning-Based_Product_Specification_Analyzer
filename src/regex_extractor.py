import re


def extract_dimensions(text):
    patterns = [
        r"(DN\d+\s?(?:to|-)\s?DN\d+)",

        r"\b\d+(\.\d+)?\s?[xX×]\s?\d+(\.\d+)?\s?(mm|cm|inch|in|m)\b",

        r"\b\d+(\.\d+)?\s?(mm|cm|inch|in|m|meter|meters|feet|ft)\b",

        r"\b(Diameter|Length|Width|Height)\s*[:\-]?\s*\d+(\.\d+)?\s?(mm|cm|inch|in|m|meter|feet|ft)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)

    return None
