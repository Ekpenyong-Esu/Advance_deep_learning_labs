import os
import wandb
from typing import Optional, Dict, Any


class WandbLogger:
    """
    Production-ready Weights & Biases logger.

    Features:
    - Safe initialization
    - Config tracking
    - Graceful disable (for local/debug runs)
    - Structured logging
    - Model watching support
    """

    def __init__(
        self,
        project: str,
        config: Optional[Dict[str, Any]] = None,
        run_name: Optional[str] = None,
        entity: Optional[str] = None,
        mode: str = "online",  # "online", "offline", "disabled"
        save_dir: str = "./wandb_logs"
    ):
        self.enabled = mode != "disabled"
        self.run = None

        if not self.enabled:
            print("[WandbLogger] Running in DISABLED mode")
            return

        try:
            os.environ["WANDB_DIR"] = save_dir

            self.run = wandb.init(
                project=project,
                name=run_name,
                config=config,
                entity=entity,
                mode=mode
            )

            print(f"[WandbLogger] Initialized run: {self.run.name}")

        except Exception as e:
            print(f"[WandbLogger] Initialization failed: {e}")
            self.enabled = False

    # -----------------------------
    # Logging Methods
    # -----------------------------
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not self.enabled:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"[WandbLogger] Logging failed: {e}")

    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        self.log({name: value}, step)

    def log_metrics(self, **kwargs):
        self.log(kwargs)

    # -----------------------------
    # Model Tracking
    # -----------------------------
    def watch(self, model, log: str = "all", log_freq: int = 100):
        if not self.enabled:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            print(f"[WandbLogger] Model watch failed: {e}")

    # -----------------------------
    # Artifact Logging
    # -----------------------------
    def log_model(self, model_path: str, name: str = "model"):
        if not self.enabled:
            return

        try:
            artifact = wandb.Artifact(name, type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
        except Exception as e:
            print(f"[WandbLogger] Model logging failed: {e}")

    # -----------------------------
    # Finish Run
    # -----------------------------
    def finish(self):
        if not self.enabled:
            return

        try:
            wandb.finish()
            print("[WandbLogger] Run finished successfully")
        except Exception as e:
            print(f"[WandbLogger] Finish failed: {e}")