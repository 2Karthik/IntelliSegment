def make_text_prompts(labels, template="a photo of a {label}"):
    \"\"\"Create CLIP text prompts from class labels.

    These prompts are CLIP inputs, so create them before semantic verification.
    \"\"\"
    return [template.format(label=label.replace("_", " ")) for label in labels]


def make_sam_box_prompt(box_xyxy):
    \"\"\"Return a float32-compatible XYXY box prompt for SAM2.\"\"\"
    if len(box_xyxy) != 4:
        raise ValueError("A box must contain [x1, y1, x2, y2].")
    x1, y1, x2, y2 = map(float, box_xyxy)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid XYXY box: {box_xyxy}")
    return [x1, y1, x2, y2]
