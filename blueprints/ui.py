from flask import Blueprint, render_template, request, redirect, url_for, session
import yaml
from utils.paths import PROJECTS_DIR, project_dir, project_config_path, MODELS_DIR

from utils.risk_analysis import analyze_training_risk
import utils.dataset as dataset
import utils.training as training
import utils.lora as lora
import utils.precision as precision
import utils.optimizer as optimizer
import utils.model as model

ui_bp = Blueprint("ui", __name__)

def load_config(project):
    path = project_config_path(project)
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())

def save_config(project, config):
    path = project_config_path(project)
    path.write_text(yaml.dump(config, sort_keys=False))


@ui_bp.route("/", methods=["GET", "POST"])
def index():
    projects = [p.name for p in PROJECTS_DIR.iterdir() if p.is_dir()]
    selected = request.values.get("project")

    issues = []
    danger_fields = []

    session_issues = session.pop("ui_issues", [])
    issues.extend(session_issues)

    config = load_config(selected) if selected else None

    REQUIRED_TOP_LEVEL = {
        "dataset", "training", "lora",
        "optimizer", "scheduler",
        "precision", "output"
    }

    if request.method == "POST" and config:
        missing = REQUIRED_TOP_LEVEL - set(config.keys())
        if missing:
            issues.append({
                "field": "__global__",
                "level": "fatal",
                "message": f"Config missing sections: {', '.join(sorted(missing))}"
            })
        else:
            issues.clear()
            danger_fields.clear()

            dataset.apply(request.form, config, issues)
            model.apply(request.form, config, issues)
            training.apply(request.form, config, issues)
            lora.apply(request.form, config, issues)
            precision.apply(request.form, config, issues)
            optimizer.apply(request.form, config, issues)

            issues += analyze_training_risk(config["training"])
            danger_fields = [i["field"] for i in issues if i["level"] == "danger"]

            save_config(selected, config)

            if request.form.get("action") == "train" and not danger_fields:
                return redirect(url_for("training.start_training", project=selected))

    available_models = sorted(p.name for p in MODELS_DIR.glob("*.safetensors"))

    training_status = "idle"
    pid_file = project_dir(selected) / "training.pid" if selected else None
    if pid_file and pid_file.exists():
        training_status = "running"

    return render_template(
        "index.html",
        projects=projects,
        selected=selected,
        config=config,
        issues=issues,
        danger_fields=danger_fields,
        training_status=training_status,
        available_models=available_models,
    )