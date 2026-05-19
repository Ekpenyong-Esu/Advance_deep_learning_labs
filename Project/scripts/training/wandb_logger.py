"""
wandb_logger.py
---------------
Thin wandb wrapper used by all trainers.
All calls are safe no-ops when wandb is not installed or WANDB_MODE=disabled.
"""
try:
    import wandb as _w
    _OK = True
except ImportError:
    _OK = False


def _active() -> bool:
    return _OK and _w.run is not None


def init_run(project, name, config=None):
    if not _OK:
        return None
    if _w.run is not None:
        _w.finish()
    return _w.init(
        project=project,
        name=name,
        config=config or {},
        resume="allow",
    )


def log(metrics, *, step=None):
    if _active():
        _w.log(metrics, step=step)


def log_batch_loss(loss, global_step):
    if _active():
        _w.log({"train/batch_loss": loss}, step=global_step)


def log_yolo_epoch(box_loss, cls_loss, dfl_loss, map50, map50_95, epoch):
    if _active():
        _w.log({
            "train/box_loss": box_loss,
            "train/cls_loss": cls_loss,
            "train/dfl_loss": dfl_loss,
            "val/map50":      map50,
            "val/map50_95":   map50_95,
        }, step=epoch)


def log_eval(metrics):
    if _active():
        _w.log({
            "eval/map50":     metrics.map50,
            "eval/map50_95":  metrics.map50_95,
            "eval/precision": metrics.precision,
            "eval/recall":    metrics.recall,
            "eval/fps":       metrics.fps,
        })


def log_eval_summary(metrics):
    if _active():
        _w.run.summary.update({
            "eval/map50":     metrics.map50,
            "eval/map50_95":  metrics.map50_95,
            "eval/precision": metrics.precision,
            "eval/recall":    metrics.recall,
            "eval/fps":       metrics.fps,
        })


def log_model(checkpoint_path, name="best-model"):
    if _active():
        import os
        artifact = _w.Artifact(name, type="model")
        if os.path.isdir(str(checkpoint_path)):
            artifact.add_dir(str(checkpoint_path))
        else:
            artifact.add_file(str(checkpoint_path))
        _w.run.log_artifact(artifact)


def log_test_summary(test_loss, test_acc, test_f1, test_prec, test_rec):
    if _active():
        _w.run.summary.update({
            "test_loss":      test_loss,
            "test_accuracy":  test_acc,
            "test_f1":        test_f1,
            "test_precision": test_prec,
            "test_recall":    test_rec,
        })


def finish():
    if _active():
        _w.finish()
