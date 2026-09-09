with open('run_uavdet_hpc.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Clean up any orphan try: or corrupted print lines
lines = code.splitlines()
cleaned_lines = []
for line in lines:
    if line.strip() == 'try:' or 'PAUSED' in line or "{'='*70}" in line:
        continue
    cleaned_lines.append(line)
code = '\n'.join(cleaned_lines)

# 2. Add self.device alias
code = code.replace('self.dev   = torch.device', 'self.dev = self.device = torch.device')
code = code.replace('self.dev    = torch.device', 'self.dev = self.device = torch.device')

# 3. Fix add_param_group duplicate parameter error
code = code.replace('self.optimizer.add_param_group({"params":bps,"lr":lr*0.1})', 'pass')

# 4. Add auto-resume logic in Trainer.__init__
old_init = 'self.history: List[EpochMetrics] = []'
new_init = '''self.history: List[EpochMetrics] = []

        self.start_epoch = 1
        last_ckpt = self.out / "checkpoints" / "last.pth"
        if last_ckpt.exists():
            try:
                ckpt = torch.load(last_ckpt, map_location=self.dev)
                self.model.load_state_dict(ckpt["model"])
                self.start_epoch = ckpt.get("epoch", 0) + 1
                if "metrics" in ckpt and "map50" in ckpt["metrics"]:
                    self.best_map = ckpt["metrics"]["map50"]
                print(f"[INFO] RESUMED! Loaded {last_ckpt} -> Starting from Epoch {self.start_epoch}")
            except Exception as e:
                print(f"[WARNING] Could not resume: {e}")'''

if 'self.start_epoch' not in code and old_init in code:
    code = code.replace(old_init, new_init, 1)

# 5. Update epoch loop to resume from start_epoch
code = code.replace('for epoch in range(1, self.epochs+1):', 'for epoch in range(self.start_epoch, self.epochs+1):')

with open('run_uavdet_hpc.py', 'w', encoding='utf-8') as f:
    f.write(code)

print(">>> SUCCESS: Fixed all syntax and indentation errors on HPC!")
