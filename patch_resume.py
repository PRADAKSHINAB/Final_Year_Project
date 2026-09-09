with open('run_uavdet_hpc.py', 'r') as f:
    code = f.read()

init_target = "self.history: List[EpochMetrics] = []"
resume_code = """self.history: List[EpochMetrics] = []

        self.start_epoch = 1
        last_ckpt = self.out / "checkpoints" / "last.pth"
        if last_ckpt.exists():
            try:
                ckpt = torch.load(last_ckpt, map_location=self.dev)
                self.model.load_state_dict(ckpt["model"])
                self.start_epoch = ckpt.get("epoch", 0) + 1
                if "metrics" in ckpt and "map50" in ckpt["metrics"]:
                    self.best_map = ckpt["metrics"]["map50"]
                print(f"[INFO] RESUMED SUCCESSFULLY! Loaded {last_ckpt} -> Jumping to Epoch {self.start_epoch}")
            except Exception as e:
                print(f"[WARNING] Could not resume: {e}")"""

if "self.start_epoch = 1" not in code and init_target in code:
    code = code.replace(init_target, resume_code, 1)

code = code.replace("for epoch in range(1, self.epochs+1):", "for epoch in range(self.start_epoch, self.epochs+1):")

with open('run_uavdet_hpc.py', 'w') as f:
    f.write(code)

print(">>> RESUME PATCH APPLIED!")
