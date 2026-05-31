import os
import re

# -----------------------------
# CONFIGURATION
# -----------------------------

replacements = {
    "ḍ": "d",
    "ḥ": "h",
    "ṇ": "n",
    "ṛ": "r",
    "ṣ": "sh",
    "Ś": "Sh",
    "ū": "u",
    "ṇ": "n",
    "ñ": "n",
    "ṅ": "n",
    "ā": "a",
    "ī": "i",
    "‘": "'",   
    "’": "'",
    "ṁ": "m",
    "ṭ": "t",
    "Krshna": "Krishna",
}

root_dir = "."

# -----------------------------
# PROCESS FILES
# -----------------------------

for folder, subfolders, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".txt"):
            file_path = os.path.join(folder, file)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            modified_content = content

            # Case-insensitive replacement
            for old_word, new_word in replacements.items():
                pattern = re.compile(re.escape(old_word), re.IGNORECASE)
                modified_content = pattern.sub(new_word, modified_content)

            if modified_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)

                print(f"Updated: {file_path}")