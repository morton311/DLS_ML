import os
import json
from collections import OrderedDict

CONFIG_DIR = "configs"
OUTPUT_MD = "config_summary.md"
EXCLUDE_FIELDS = {"predictions", "pred_lim"}

def flatten_json(y, prefix=''):
    """Flatten nested JSON into a single dict with compound keys, excluding specified fields."""
    out = OrderedDict()
    for k, v in y.items():
        if k in EXCLUDE_FIELDS:
            continue
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_json(v, key))
        else:
            out[key] = v
    return out

def get_column_structure(configs):
    """
    Returns an ordered dict representing the column structure:
    {top_level_key: [subfield1, subfield2, ...] or None if not a dict}
    """
    structure = OrderedDict()
    for _, flat in configs:
        for key in flat:
            parts = key.split('.')
            if len(parts) == 1:
                structure.setdefault(parts[0], None)
            else:
                structure.setdefault(parts[0], set()).add(parts[1])
    # Convert sets to sorted lists
    for k, v in structure.items():
        if isinstance(v, set):
            structure[k] = sorted(v)
    return structure

def main():
    files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
    files.sort()
    configs = []
    all_flat = []
    for fname in files:
        with open(os.path.join(CONFIG_DIR, fname), "r") as f:
            data = json.load(f)
            flat = flatten_json(data)
            configs.append((fname, flat))
            all_flat.append(flat)
    column_structure = get_column_structure(configs)

    # Build header rows
    header1 = ["Config Name"]
    header2 = [""]
    col_keys = []
    for top, subs in column_structure.items():
        if subs is None:
            header1.append(f"**{top}**")
            header2.append("")
            col_keys.append(top)
        else:
            header1.extend([f"**{top}**"] * len(subs))
            header2.extend([f"**{sub}**" for sub in subs])
            col_keys.extend([f"{top}.{sub}" for sub in subs])

    # Write markdown table
    with open(OUTPUT_MD, "w") as out:
        out.write("# Configuration Summary\n\n")
        # Write header rows
        out.write("| " + " | ".join(header1) + " |\n")
        out.write("|" + "|".join("---" for _ in header1) + "|\n")
        out.write("| " + " | ".join(header2) + " |\n")
        out.write("|" + "|".join("---" for _ in header2) + "|\n")
        # Write config rows
        for fname, flat in configs:
            row = [f"`{fname}`"]
            for key in col_keys:
                val = flat.get(key, "")
                row.append(f"`{val}`" if val != "" else " ")
            out.write("| " + " | ".join(row) + " |\n")

if __name__ == "__main__":
    main()