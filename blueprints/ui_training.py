from flask import Blueprint, render_template, request, redirect, url_for, session, current_app
from pathlib import Path
from utils.paths import PROJECTS_DIR, project_dir, project_config_path, MODELS_DIR
from utils.project_file import open_folder
from utils.risk_analysis import analyze_training_risk
import utils.dataset as dataset
import utils.training as training
import utils.lora as lora
import utils.precision as precision
import utils.optimizer as optimizer
import utils.model as model
from utils.project_config import load_config, save_config
from flask import jsonify
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

nvmlInit()
_gpu_handle = nvmlDeviceGetHandleByIndex(0)

ui_training_bp = Blueprint("ui", __name__)

@ui_training_bp.route("/", methods=["GET", "POST"])
def index():
    projects = [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]
    selected = request.values.get("project")

    issues: list[dict] = []

    issues.extend(session.pop("ui_issues", []))

    config = load_config(selected) if selected else None

    REQUIRED_TOP_LEVEL = {
        "dataset", "training", "lora",
        "optimizer", "scheduler",
        "precision", "output",
    }

    def has_fatal():
        return any(i["level"] == "fatal" for i in issues)

    def has_warn():
        return any(i["level"] == "warn" for i in issues)

    if request.method == "POST" and config:
        action = request.form.get("action")

        issues.clear()

        missing = REQUIRED_TOP_LEVEL - set(config.keys())
        if missing:
            issues.append({
                "field": "__global__",
                "level": "fatal",
                "message": f"Config missing sections: {', '.join(sorted(missing))}",
            })
        else:
            dataset.apply(request.form, config, issues)
            model.apply(request.form, config, issues)
            training.apply(request.form, config, issues)
            lora.apply(request.form, config, issues)
            precision.apply(request.form, config, issues)
            optimizer.apply(request.form, config, issues)

            issues.extend(analyze_training_risk(config["training"]))

        if action == "cancel":
            issues.clear()
            config = load_config(selected)

        elif action == "proceed":
            if not has_fatal():
                save_config(selected, config)
                issues.clear()

        elif action == "save":
            if not has_fatal() and not has_warn():
                save_config(selected, config)
                issues.clear()

        elif action == "train":
            if not has_fatal():
                save_config(selected, config)
                issues.clear()
                return redirect(
                    url_for("training.start_training", project=selected)
                )

        if issues:
            session["ui_issues"] = issues.copy()

    training_status = "idle"
    if selected:
        pid_file = project_dir(selected) / "training.pid"
        if pid_file.exists():
            training_status = "running"

    available_models = sorted(p.name for p in MODELS_DIR.glob("*.safetensors"))

    return render_template(
        "index.html",
        projects=projects,
        selected=selected,
        config=config,
        issues=issues,
        training_status=training_status,
        available_models=available_models,
    )


@ui_training_bp.route("/vram")
def vram_status():
    try:
        mem = nvmlDeviceGetMemoryInfo(_gpu_handle)
        used = mem.used // (1024 * 1024)
        total = mem.total // (1024 * 1024)

        return jsonify({
            "text": f"VRAM: {used}/{total}"
        })
    except Exception:
        return jsonify({
            "text": None
        })
    
@ui_training_bp.route("/api/open_dataset_folder", methods=["POST"])
def open_dataset_folder():
    selected = request.args.get("project")

    if not selected:
        return jsonify({"success": False, "error": "No project selected"}), 400

    project_path = PROJECTS_DIR / selected

    if not project_path.is_dir():
        return jsonify({"success": False, "error": "Project folder not found"}), 404

    ok, err = open_folder(project_path)
    if not ok:
        return jsonify({"success": False}), 400

    return jsonify({"success": True})
