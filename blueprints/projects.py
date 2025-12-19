from flask import Blueprint, request, redirect, url_for
from utils.create_lora_project import create_project
from config import BASE_DIR
import shutil

projects_bp = Blueprint("projects", __name__)

@projects_bp.route("/create_project", methods=["POST"])
def create_project_route():
    name = request.form.get("project_name", "").strip()
    if name:
        try:
            create_project(name)
        except FileExistsError:
            pass
    return redirect(url_for("ui.index", project=name))


@projects_bp.route("/delete_project", methods=["POST"])
def delete_project_route():
    name = request.form.get("project_name")
    confirm = request.form.get("confirm_name")

    if name == confirm:
        shutil.rmtree(BASE_DIR / name, ignore_errors=True)

    return redirect(url_for("ui.index"))