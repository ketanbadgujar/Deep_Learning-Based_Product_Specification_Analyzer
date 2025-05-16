import re

def extract_value_from_text(label, text):
    label = label.lower()
    text = text.strip()

    if "standard" in label:
        match = re.search(r"\b(ASME|ANSI|JIS|BS|API|EN|ISO|DIN|JIC)(\s?[-]?[A-Z0-9./]*)?", text, re.IGNORECASE)
        return match.group(0) if match else None

    elif "class rating" in label or "pressure" in label:
        match = re.search(r"(?i)(#\d{2,4}|\d{2,4}#|class\s*#?\d{2,4})", text)
        return match.group(0) if match else None

    elif "material" in label:
        materials = [
            r"Sheet Gasket, Flexicarb RGS 3",
            r"Sheet Gasket",
            r"Stainless Steel",
            r"Carbon Steel",
            r"Cast Iron",
            r"Bronze",
            r"Alloy",
            r"Steel",
            r"CS",
            r"SS\d{3}",
            r"SS",
            r"MS",
        ]
        for pattern in materials:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)

    return None
