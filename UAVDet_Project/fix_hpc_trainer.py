import re

with open('run_uavdet_hpc.py', 'r') as f:
    code = f.read()

pattern = r'class Trainer:.*?def train_epoch'
clean_trainer_start = '''class Trainer:
    def __init__(self, model, train_loader, val_loader, val_coco,
                 experiment_name, epochs=50, lr=0.005,
                 img_size=1280, freeze_epochs=5, device="cuda"):
        self.dev   = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
        self.model = model.to(self.dev)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.val_coco     = val_coco
        self.exp_name     = experiment_name
        self.epochs       = epochs
        self.base_lr      = lr
        self.img_size     = img_size
        self.freeze_epochs= freeze_epochs
        self.out          = Path("results") / experiment_name
        for d in ["checkpoints","logs","metrics","predictions"]:
            (self.out/d).mkdir(parents=True, exist_ok=True)

        if self.freeze_epochs > 0:
            for p in self.model.backbone.parameters():
                p.requires_grad = False

        tr = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = SGD(tr, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        self.scaler    = GradScaler(enabled=(self.dev.type=="cuda"))
        self.best_map  = -1.
        self.history   = []

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
                print(f"[WARNING] Could not resume from {last_ckpt}: {e}")

    def train_epoch'''

code = re.sub(pattern, clean_trainer_start, code, flags=re.DOTALL)
with open('run_uavdet_hpc.py', 'w') as f:
    f.write(code)

print(">>> SUCCESS: Trainer init fixed cleanly!")
