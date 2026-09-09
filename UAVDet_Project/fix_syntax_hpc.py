with open('run_uavdet_hpc.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'print(f"' in line and 'mAP' not in line and 'E0' not in line and 'S0' not in line:
        indent = len(line) - len(line.lstrip())
        new_lines.append(' ' * indent + 'print("======================================================================")\n')
        new_lines.append(' ' * indent + 'print(f"Starting Training: {self.exp_name} ({self.epochs} epochs) on {self.dev}")\n')
        new_lines.append(' ' * indent + 'print("======================================================================")\n')
    else:
        new_lines.append(line)

with open('run_uavdet_hpc.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(">>> SUCCESS: Fixed line 698 syntax error!")
