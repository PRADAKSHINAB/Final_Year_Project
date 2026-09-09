import re, sys

with open('run_uavdet_hpc.py', 'r', encoding='utf-8') as f:
    code = f.read()

pattern = r'class Trainer:.*?(?=\n# ───|\ndef generate_comparison_summary)'

clean_trainer_class = '''class Trainer:
    def __init__(self, model, train_loader, val_loader, val_coco,
                 experiment_name, epochs=50, lr=0.005,
                 img_size=1280, freeze_epochs=5, device="cuda"):
        self.dev    = torch.device(device if torch.cuda.is_available() and device=="cuda" else "cpu")
        self.device = self.dev
        self.model  = model.to(self.dev)
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

    def _lr(self, epoch):
        wu = 3
        if epoch <= wu: return self.base_lr*(0.1+0.9*epoch/wu)
        prog = (epoch-wu)/max(1,self.epochs-wu)
        return 1e-5 + 0.5*(self.base_lr-1e-5)*(1+math.cos(math.pi*prog))

    _get_lr = _lr

    def train_epoch(self, epoch):
        self.model.train()
        lr = self._lr(epoch)

        if self.freeze_epochs > 0 and epoch == self.freeze_epochs + 1:
            for p in self.model.backbone.parameters():
                p.requires_grad = True
            print(f"[E{epoch:03d}] Backbone unfrozen (requires_grad=True).")

        running = {}
        for step,(images,targets) in enumerate(self.train_loader,1):
            images  = [im.to(self.dev,non_blocking=True) for im in images]
            targets = [{k:v.to(self.dev,non_blocking=True) for k,v in t.items()} for t in targets]
            with autocast(enabled=(self.dev.type=="cuda")):
                ld = self.model(images,targets)
                loss = sum(ld.values())
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.scaler.step(self.optimizer); self.scaler.update()
            for k,v in ld.items(): running[k] = running.get(k,0.)+float(v.detach())
            running["total"] = running.get("total",0.)+float(loss.detach())
            if step%20==0 or step==len(self.train_loader):
                print(f"[{self.exp_name}][E{epoch:03d}/S{step:04d}] lr={lr:.5f} loss={running['total']/step:.4f}")
        return {k:v/len(self.train_loader) for k,v in running.items()}

    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        preds = []
        for images,targets in self.val_loader:
            images = [im.to(self.dev) for im in images]
            outs   = self.model(images)
            for t,o in zip(targets,outs):
                img_id = int(t["image_id"][0].item())
                meta   = self.val_coco.loadImgs([img_id])[0]
                sx     = meta.get("width",self.img_size)/self.img_size
                sy     = meta.get("height",self.img_size)/self.img_size
                for box,lbl,scr in zip(o["boxes"].cpu(),o["labels"].cpu(),o["scores"].cpu()):
                    x1,y1,x2,y2 = box.tolist()
                    preds.append({"image_id":img_id,"category_id":int(lbl),
                                  "bbox":[x1*sx,y1*sy,(x2-x1)*sx,(y2-y1)*sy],
                                  "score":float(scr)})
        return compute_coco_eval(preds, self.val_coco)

    def run(self):
        print(f"\n{'='*70}\nStarting Training: {self.exp_name} ({self.epochs} epochs) on {self.device}\n{'='*70}")

        try:
            for epoch in range(self.start_epoch, self.epochs+1):
                tl = self.train_epoch(epoch)
                vm = self.validate()
                m  = EpochMetrics(epoch=epoch,
                    loss_total=tl.get("total",0.),loss_rpn_cls=tl.get("loss_objectness",0.),
                    loss_rpn_box=tl.get("loss_rpn_box_reg",0.),
                    loss_roi_cls=tl.get("loss_classifier",0.),
                    loss_roi_box=tl.get("loss_box_reg",0.),
                    lr=self._lr(epoch), map50=vm["map50"], map50_95=vm["map50_95"],
                    ap_small=vm["ap_small"], ap_medium=vm["ap_medium"], ap_large=vm["ap_large"],
                    precision=vm["precision"], recall=vm["recall"], f1=vm["f1"])
                self.history.append(m)
                print(f"[{self.exp_name}][E{epoch:03d}] mAP50={m.map50:.4f} "
                      f"mAP50:95={m.map50_95:.4f} AP_s={m.ap_small:.4f} F1={m.f1:.4f}")

                state = {
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scaler": self.scaler.state_dict(),
                    "metrics": vm,
                    "history": [h.__dict__ for h in self.history]
                }
                torch.save(state, self.out/"checkpoints"/"last.pth")
                if m.map50 > self.best_map:
                    self.best_map = m.map50
                    torch.save(state, self.out/"checkpoints"/"best.pth")
                    print(f"  ★ Best mAP@50: {self.best_map:.4f}")
        except KeyboardInterrupt:
            print(f"\n[PAUSED] Training paused by user (Ctrl+C). Checkpoint saved to {self.out/'checkpoints'/'last.pth'}")
            print("[INFO] You can safely resume anytime by re-running the python training command!")
            sys.exit(0)

        with open(self.out/"metrics"/"metrics.json","w",encoding="utf-8") as f: json.dump(vm,f,indent=2)
        with open(self.out/"logs"/"training_log.csv","w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(self.history[0].__dict__.keys()))
            w.writeheader()
            for r in self.history: w.writerow(r.__dict__)

'''

code = re.sub(pattern, clean_trainer_class, code, flags=re.DOTALL)
with open('run_uavdet_hpc.py', 'w', encoding='utf-8') as f:
    f.write(code)

print(">>> SUCCESS: Trainer class updated cleanly on HPC!")
