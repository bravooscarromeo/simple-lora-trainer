from flask import Blueprint, redirect, url_for, jsonify
from config import BASE_DIR
import os
import signal

from utils.launch_training import launch_training, TrainingConfigError

training_bp = Blueprint("training", __name__)

def is_process_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


@training_bp.route("/train/<project>", methods=["POST"])
def start_training(project):
    try:
        launch_training(project)
    except TrainingConfigError as e:
        from flask import session
        session["ui_issues"] = [{
            "field": "__global__",
            "level": "fatal",
            "message": str(e),
        }]
    return redirect(url_for("ui.index", project=project))


@training_bp.route("/stop/<project>", methods=["POST"])
def stop_training(project):
    pid_file = BASE_DIR / project / "training.pid"
    if pid_file.exists():
        pgid = int(pid_file.read_text())
        try:
            os.killpg(pgid, signal.SIGTERM)
        except Exception:
            pass
        pid_file.unlink(missing_ok=True)
    return redirect(url_for("ui.index", project=project))


@training_bp.route("/train_logs/<project>")
def train_logs(project):
    log_path = BASE_DIR / project / "logs" / "train.log"
    if not log_path.exists():
        return jsonify({"logs": ""})
    return jsonify({"logs": log_path.read_text()[-10000:]})