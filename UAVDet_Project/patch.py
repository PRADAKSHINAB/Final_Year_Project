with open('run_uavdet_hpc.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'add_param_group' in line:
        indent = len(line) - len(line.lstrip())
        new_lines.append(' ' * indent + 'pass  # fixed duplicate param group\n')
    else:
        new_lines.append(line)

with open('run_uavdet_hpc.py', 'w') as f:
    f.writelines(new_lines)

print(">>> SUCCESS: Fixed add_param_group bug in run_uavdet_hpc.py!")
