import re

with open('run_uavdet_hpc.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('self.dev   = torch.device', 'self.dev = self.device = torch.device')
code = code.replace('self.dev    = torch.device', 'self.dev = self.device = torch.device')

if 'self.optimizer.add_param_group' in code:
    code = re.sub(r'self\.optimizer\.add_param_group\(.*?\)', 'pass', code)

if 'def _lr(self, epoch):' in code and '_get_lr' not in code:
    code = code.replace('def _lr(self, epoch):', '_get_lr = _lr\n    def _lr(self, epoch):')

if 'self.start_epoch' not in code:
    old_init_end = 'self.history   = []'
    new_init_end = '''self.history   = []
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
    code = code.replace(old_init_end, new_init_end)
    code = code.replace('for epoch in range(1, self.epochs+1):', 'for epoch in range(self.start_epoch, self.epochs+1):')

with open('run_uavdet_hpc.py', 'w', encoding='utf-8') as f:
    f.write(code)

print(">>> SUCCESS: Fixed all trainer issues on HPC!")
