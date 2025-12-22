from flask import Blueprint, render_template, request, send_file
from threading import Thread
import torch

from utils.dataset_io import (
    set_dataset_root,
    list_dataset_images,
    resolve_dataset_image,
    read_caption_for_image,
    write_caption_for_image,
)

from utils.paths import (
    PROJECTS_DIR,
    project_dir,
)

from utils.project_config import (
    load_config,
    save_config,
)

from utils.autocaption_progress import start, step, finish
from utils.blip import load as blip_load, generate_caption, unload as blip_unload

ui_dataset_bp = Blueprint("ui_dataset", __name__)

@ui_dataset_bp.route("/dataset")
def dataset():
    return render_template("dataset.html")

@ui_dataset_bp.route("/api/projects")
def api_projects():
    projects = sorted(
        p.name for p in PROJECTS_DIR.iterdir()
        if p.is_dir()
    )
    return {"projects": projects}

@ui_dataset_bp.route("/api/project/config/<project>")
def api_project_config(project):
    config = load_config(project)
    if not config:
        return {"error": "Config not found"}, 404

    return {
        "dataset_path": config["dataset"]["path"]
    }

@ui_dataset_bp.route("/api/dataset/load", methods=["POST"])
def api_dataset_load():
    data = request.get_json() or {}
    project = data.get("project")
    dataset_path = data.get("dataset_path")

    if not project or not dataset_path:
        return {"error": "Missing project or dataset path"}, 400

    config = load_config(project)
    if not config:
        return {"error": "Project config not found"}, 404

    config["dataset"]["path"] = dataset_path
    save_config(project, config)

    dataset_root = project_dir(project) / dataset_path
    if not dataset_root.exists():
        return {
            "error": "Dataset path does not exist",
            "path": str(dataset_root),
        }, 400

    set_dataset_root(dataset_root)

    images = []
    for img in list_dataset_images():
        images.append({
            "name": img["name"],
            "rel_path": img["rel_path"],
            "caption": read_caption_for_image(img["name"]),
        })

    return {
        "project": project,
        "dataset_path": dataset_path,
        "images": images,
    }

@ui_dataset_bp.route("/api/dataset/image/<path:rel_path>")
def api_dataset_image(rel_path):
    img_path = resolve_dataset_image(rel_path)
    if img_path is None:
        return "Not found", 404

    return send_file(img_path)

@ui_dataset_bp.route("/api/dataset/save", methods=["POST"])
def api_dataset_save():
    data = request.get_json(silent=True) or {}
    images = data.get("images", [])

    if not images:
        return {"status": "nothing_to_save", "saved": 0}

    saved = 0
    for img in images:
        name = img.get("name")
        caption = img.get("caption", "")
        if name and write_caption_for_image(name, caption):
            saved += 1

    return {"status": "ok", "saved": saved}

@ui_dataset_bp.route("/api/dataset/delete", methods=["POST"])
def api_dataset_delete():
    data = request.get_json(silent=True) or {}
    name = data.get("name")

    if not name:
        return {"error": "Missing image name"}, 400

    img_path = resolve_dataset_image(name)
    if not img_path:
        return {"error": "Image not found"}, 404

    img_path.unlink(missing_ok=True)
    img_path.with_suffix(".txt").unlink(missing_ok=True)

    return {"status": "ok", "deleted": name}

@ui_dataset_bp.route("/api/dataset/autocaption", methods=["POST"])
def api_dataset_autocaption():
    data = request.get_json() or {}
    images = data.get("images", [])
    overwrite = bool(data.get("overwrite", False))

    if not images:
        return {"error": "No images provided"}, 400

    Thread(
        target=_run_autocaption,
        args=(images, overwrite),
        daemon=True
    ).start()

    return {"status": "started"}

@ui_dataset_bp.route("/api/dataset/autocaption/progress")
def api_dataset_autocaption_progress():
    from utils.autocaption_progress import get
    return get()

def _run_autocaption(images, overwrite: bool):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start(total=len(images))
    blip_load(device)

    try:
        for img in images:
            name = img.get("name")
            if not name:
                step()
                continue

            path = resolve_dataset_image(name)
            if not path:
                step()
                continue

            if not overwrite:
                existing = read_caption_for_image(name)
                if existing.strip():
                    step()
                    continue

            caption = generate_caption(path)
            write_caption_for_image(name, caption)
            step()

    finally:
        blip_unload()
        finish()

@ui_dataset_bp.route("/api/dataset/autocaption/one", methods=["POST"])
def api_dataset_autocaption_one():
    data = request.get_json() or {}
    name = data.get("name")
    overwrite = bool(data.get("overwrite", False))

    if not name:
        return {"error": "Missing image name"}, 400

    path = resolve_dataset_image(name)
    if not path:
        return {"error": "Image not found"}, 404

    if not overwrite:
        existing = read_caption_for_image(name)
        if existing.strip():
            return {
                "status": "skipped",
                "reason": "caption_exists"
            }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    start(total=1)

    try:
        from utils.blip import load, generate_caption, unload
        load(device)
        caption = generate_caption(path)
        write_caption_for_image(name, caption)

        return {
            "status": "ok",
            "name": name,
            "caption": caption
        }

    finally:
        finish()
        unload()

@ui_dataset_bp.route("/api/dataset/crop", methods=["POST"])
def api_dataset_crop():
    data = request.get_json() or {}
    size = int(data.get("size", 512))
    name = data.get("name")

    if size < 64:
        return {"error": "Invalid crop size"}, 400

    from PIL import Image

    def crop_image(path):
        with Image.open(path) as im:
            w, h = im.size
            side = min(w, h)

            left = (w - side) // 2
            top = (h - side) // 2
            right = left + side
            bottom = top + side

            cropped = im.crop((left, top, right, bottom))

            if side != size:
                cropped = cropped.resize((size, size), Image.BICUBIC)

            cropped.save(path)

    if name:
        img_path = resolve_dataset_image(name)
        if not img_path:
            return {"error": "Image not found"}, 404

        crop_image(img_path)
        return {"status": "ok", "mode": "single"}

    for img in list_dataset_images():
        path = resolve_dataset_image(img["name"])
        if path:
            crop_image(path)

    return {"status": "ok", "mode": "all"}
