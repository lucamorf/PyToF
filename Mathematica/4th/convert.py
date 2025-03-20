import os
import glob
import re
import numpy as np

def transform_expression(expr):
    # Replace exponentiation for s-variables: e.g., s2**2 -> s2h2
    expr = re.sub(r"(s\d+)\*\*(\d+)", r"\1h\2", expr)
    # Replace exponentiation for z: e.g., z**2 -> zh2
    expr = re.sub(r"z\*\*(\d+)", r"zh\1", expr)
    # Replace multiplication between symbols: e.g., s2h2*s4 -> s2h2_s4, z*s4 -> z_s4
    expr = re.sub(r"([sz]\d*(?:h\d+)?)(\*)([sz]\d*(?:h\d+)?)", r"\1_\3", expr)
    return expr

def generate_definitions_by_type(original_expr, transformed_expr):
    power_defs_s = {}
    power_defs_z = {}
    mult_defs_s = {}
    mult_defs_z = {}

    # Power definitions for s-variables
    for match in re.finditer(r"(s\d+)\*\*(\d+)", original_expr):
        token = f"{match.group(1)}h{match.group(2)}"
        power_defs_s[token] = f"{token} = {match.group(1)}**{match.group(2)}"

    # Power definitions for z
    for match in re.finditer(r"z\*\*(\d+)", original_expr):
        token = f"zh{match.group(1)}"
        power_defs_z[token] = f"{token} = z**{match.group(1)}"

    # Multiplication definitions (separating s and z cases)
    for match in re.finditer(r"\b([sz]\d*(?:h\d+)?)_([sz]\d*(?:h\d+)?)\b", transformed_expr):
        token = f"{match.group(1)}_{match.group(2)}"
        if "z" in token:
            mult_defs_z[token] = f"{token} = {match.group(1)}*{match.group(2)}"
        else:
            mult_defs_s[token] = f"{token} = {match.group(1)}*{match.group(2)}"

    return power_defs_s, power_defs_z, mult_defs_s, mult_defs_z

# Files to be generated for definitions; these will be skipped in processing
def_files = {
    "power_definitions_s.txt", "power_definitions_z.txt",
    "multiplication_definitions_s.txt", "multiplication_definitions_z.txt",
    "new_s0.txt", "R_ratio.txt"
}

all_power_defs_s = {}
all_power_defs_z = {}
all_mult_defs_s = {}
all_mult_defs_z = {}

# Process all .txt files in the current directory, ignoring definition files
txt_files = [fname for fname in glob.glob("*.txt") if os.path.basename(fname) not in def_files]

for fname in txt_files:
    try:
        # Load file contents as a string (joining tokens if necessary)
        input_expr_array = np.loadtxt(fname, dtype=str, encoding='utf-8')
    except Exception as e:
        print(f"Error reading {fname}: {e}")
        continue
    if isinstance(input_expr_array, np.ndarray):
        input_expr = " ".join(input_expr_array.tolist())
    else:
        input_expr = input_expr_array

    # Transform the expression
    trans_expr = transform_expression(input_expr)
    
    # Generate definitions for this file
    power_s, power_z, mult_s, mult_z = generate_definitions_by_type(input_expr, trans_expr)
    all_power_defs_s.update(power_s)
    all_power_defs_z.update(power_z)
    all_mult_defs_s.update(mult_s)
    all_mult_defs_z.update(mult_z)
    
    # Overwrite the file with the transformed expression
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(trans_expr)
    
    # Print input and output expressions on a single line
    print()
    print("Input Expression:") 
    print(input_expr)
    print()
    print("Transformed Expression:") 
    print(trans_expr)

# Convert definitions to single-line strings
power_defs_s_str = "; ".join(sorted(all_power_defs_s.values()))
power_defs_z_str = "; ".join(sorted(all_power_defs_z.values()))
mult_defs_s_str = "; ".join(sorted(all_mult_defs_s.values()))
mult_defs_z_str = "; ".join(sorted(all_mult_defs_z.values()))

# Write definitions to their respective files
with open("power_definitions_s.txt", "w", encoding="utf-8") as f:
    f.write(power_defs_s_str)
with open("power_definitions_z.txt", "w", encoding="utf-8") as f:
    f.write(power_defs_z_str)
with open("multiplication_definitions_s.txt", "w", encoding="utf-8") as f:
    f.write(mult_defs_s_str)
with open("multiplication_definitions_z.txt", "w", encoding="utf-8") as f:
    f.write(mult_defs_z_str)

# Print final definitions on a single line
print()
print("Power Definitions (s):") 
print(power_defs_s_str)
print()
print("Power Definitions (z):") 
print(power_defs_z_str)
print()
print("Multiplication Definitions (s):") 
print(mult_defs_s_str)
print()
print("Multiplication Definitions (z):") 
print(mult_defs_z_str)