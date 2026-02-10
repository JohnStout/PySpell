"""
This script adds the 'if __name__ == "__main__":' guard to recurseConvert.py
and indents all the main execution code by 4 spaces.

This is required for Windows multiprocessing (ProcessPoolExecutor) to work.
"""

# Read the file
with open('recurseConvert.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the PARAMETER SPACE header line
for i, line in enumerate(lines):
    if '# -------PARAMETER SPACE--------------------' in line:
        insert_idx = i + 1
        break

print(f"Found PARAMETER SPACE header at line {insert_idx}")
print(f"Will indent {len(lines) - insert_idx} lines")

# Build new content
new_lines = lines[:insert_idx]  # Keep imports and header
new_lines.append("# IMPORTANT: Windows multiprocessing requires this guard\r\n")
new_lines.append("if __name__ == '__main__':\r\n")

# Indent all remaining lines by 4 spaces
for line in lines[insert_idx:]:
    if line.strip():  # Has content
        new_lines.append('    ' + line)
    else:  # Empty line
        new_lines.append(line)

# Write back
with open('recurseConvert.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Done! Guard added and code indented.")
