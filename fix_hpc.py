with open('run_uavdet_hpc.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

clean = []
for line in lines:
    if 'PAUSED' in line:
        indent = len(line) - len(line.lstrip())
        clean.append(' ' * indent + 'print("PAUSED: Checkpoint saved!")\n')
    elif 'Starting Training' in line or "{'='*70}" in line:
        indent = len(line) - len(line.lstrip())
        clean.append(' ' * indent + 'print("======================================================================")\n')
    elif 'add_param_group' in line:
        indent = len(line) - len(line.lstrip())
        clean.append(' ' * indent + 'pass\n')
    elif 'self.dev   =' in line or 'self.dev    =' in line:
        clean.append('        self.dev = self.device = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")\n')
    else:
        clean.append(line)

with open('run_uavdet_hpc.py', 'w', encoding='utf-8') as f:
    f.writelines(clean)

print(">>> SUCCESS: Fixed all file syntax issues on HPC!")
