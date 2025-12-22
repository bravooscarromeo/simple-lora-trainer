const datasetState = {
images: [],
selectedIndex: null,
dirty: false,
filter: ""
};

let autocaptionPolling = false;

let saveTimer = null;
const SAVE_DEBOUNCE_MS = 600;

function scheduleSave() {
        if (!datasetState.dirty) return;

        if (saveTimer) {
        clearTimeout(saveTimer);
        }

        saveTimer = setTimeout(() => {
        saveTimer = null;
        saveCaptions();
        }, SAVE_DEBOUNCE_MS);
}

function isCaptionFlagged(caption) {
        if (!caption) return true;

        if (/[_\-|\\/<>[\]{}()*^$#@~=`+]/.test(caption)) return true;
        if (/[^\x00-\x7F]/.test(caption)) return true;

        const words = caption
        .toLowerCase()
        .trim()
        .split(/\s+/)
        .filter(Boolean);

        const weirdTokens = words.filter(w =>
        w.length === 1 && /[^a-z0-9]/i.test(w)
        );
        if (weirdTokens.length >= 2) return true;

        if (words.length > 100) return true;

        for (let i = 1; i < words.length; i++) {
        if (words[i] === words[i - 1]) return true;
        }

        if (words.length === 2 && words[0] === words[1]) return true;

        return false;
}

function renderImageList() {
        const listEl = document.getElementById("image-list");
        if (!listEl) return;

        const q = datasetState.filter.trim().toLowerCase();
        listEl.innerHTML = "";

        datasetState.images.forEach((img, index) => {
        if (q && !img.name.toLowerCase().includes(q)) return;

        const li = document.createElement("li");
        li.style.display = "flex";
        li.style.alignItems = "center";
        li.style.justifyContent = "space-between";
        li.style.padding = "4px 6px";
        li.style.borderRadius = "4px";
        li.style.cursor = "pointer";

        const nameSpan = document.createElement("span");
        nameSpan.textContent = img.name;
        li.appendChild(nameSpan);

        if (img.flagged) {
        li.style.background = "rgba(192, 57, 43, 0.15)";

        const flag = document.createElement("span");
        flag.textContent = "🚩";
        flag.style.fontSize = "13px";
        flag.style.opacity = "0.8";
        li.appendChild(flag);
        }

        if (index === datasetState.selectedIndex) {
        li.style.outline = "2px solid var(--accent)";
        }

        li.addEventListener("click", () => {
        selectImage(index);
        li.scrollIntoView({ block: "nearest" });
        });

        listEl.appendChild(li);
        });
}

function renderImagePreview() {
        const previewEl = document.getElementById("image-preview");
        if (!previewEl) return;

        if (datasetState.selectedIndex === null) {
        previewEl.textContent = "No image selected";
        return;
        }

        const img = datasetState.images[datasetState.selectedIndex];
        previewEl.innerHTML = `
        <img
        src="/api/dataset/image/${encodeURIComponent(img.path)}"
        alt="${img.name}"
        style="max-width:100%; max-height:100%; object-fit:contain; border-radius:6px;"
        />
        `;
}

function renderCaptionEditor() {
        const textarea = document.getElementById("caption-editor");
        if (!textarea) return;

        if (datasetState.selectedIndex === null) {
        textarea.value = "";
        textarea.disabled = true;
        return;
        }

        textarea.disabled = false;
        textarea.value = datasetState.images[datasetState.selectedIndex].caption || "";
}

function renderAll() {
        renderImageList();
        renderImagePreview();
        renderCaptionEditor();
        }

function selectImage(index) {
        if (index < 0 || index >= datasetState.images.length) return;
        if (datasetState.dirty) {
        saveCaptions();
        }
        datasetState.selectedIndex = index;
        renderAll();
}

function loadImages(imageList) {
        datasetState.images = imageList.map(img => {
        const caption = img.caption ?? "";

        return {
        name: img.name,
        path: img.rel_path,
        caption,
        flagged: isCaptionFlagged(caption)
        };
        });

        datasetState.images.sort((a, b) => b.flagged - a.flagged);

        datasetState.selectedIndex = datasetState.images.length ? 0 : null;
        datasetState.dirty = false;
        renderAll();
}

function loadProjects() {
        fetch("/api/projects")
        .then(r => r.json())
        .then(data => {
        const sel = document.getElementById("project-select");
        sel.innerHTML = "";

        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "-- Select --";
        placeholder.disabled = true;
        placeholder.selected = true;
        sel.appendChild(placeholder);

        data.projects.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        sel.appendChild(opt);
        });
        });
}

function loadDatasetFromUI() {
        const project = document.getElementById("project-select").value;
        const datasetPath = document.getElementById("dataset-path-input").value.trim();
        const errorBox = document.getElementById("dataset-error");

        errorBox.style.display = "none";

        if (!project || !datasetPath) {
        errorBox.textContent = "Project and dataset path are required";
        errorBox.style.display = "block";
        return;
        }

        fetch("/api/dataset/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project, dataset_path: datasetPath })
        })
        .then(async r => {
        const data = await r.json();
        if (!r.ok) throw data;
        return data;
        })
        .then(data => {
        loadImages(data.images);
        })
        .catch(err => {
        errorBox.textContent = err.error || "Failed to load dataset";
        errorBox.style.display = "block";
        });
}

function saveCaptions() {
        if (!datasetState.dirty) return;

        fetch("/api/dataset/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        images: datasetState.images.map(img => ({
        name: img.name,
        caption: img.caption
        }))
        })
        }).then(() => {
        datasetState.dirty = false;
        resortImagesKeepSelection();
        renderImageList();
        });
}


function deleteSelectedImage() {
        if (datasetState.selectedIndex === null) return;

        const img = datasetState.images[datasetState.selectedIndex];

        fetch("/api/dataset/delete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: img.name })
        }).then(() => {
        datasetState.images.splice(datasetState.selectedIndex, 1);
        datasetState.selectedIndex =
        datasetState.images.length ? Math.min(datasetState.selectedIndex, datasetState.images.length - 1) : null;
        renderAll();
        });
}

function autoCaptionAll() {
        const overwrite = document.getElementById("autocaption-overwrite").checked;

        const images = datasetState.images
        .filter(img => overwrite || !img.caption.trim())
        .map(img => ({ name: img.name }));

        if (!images.length) return;

        pollAutocaptionProgress();

        fetch("/api/dataset/autocaption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images, overwrite })
        });
}

function autoCaptionSingle() {
        if (datasetState.selectedIndex === null) return;

        const overwrite = document.getElementById("autocaption-overwrite").checked;
        const img = datasetState.images[datasetState.selectedIndex];

        pollAutocaptionProgress();

        fetch("/api/dataset/autocaption/one", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        name: img.name,
        overwrite
        })
        })
        .then(r => r.json())
        .then(data => {
        if (data.status === "ok") {
        img.caption = data.caption;
        datasetState.dirty = false;
        renderCaptionEditor();
        }
        })
        .catch(err => {
        console.error("[dataset] single autocaption failed", err);
        });
}

function pollAutocaptionProgress() {
        if (autocaptionPolling) return;
        autocaptionPolling = true;

        const statusEl = document.getElementById("autocaption-status");
        if (!statusEl) return;

        const interval = setInterval(() => {
        fetch("/api/dataset/autocaption/progress")
        .then(r => r.json())
        .then(p => {
        if (!p.running) {
        clearInterval(interval);
        autocaptionPolling = false;
        statusEl.textContent = "";
        loadDatasetFromUI();
        return;
        }
        statusEl.textContent = `Auto-captioning: ${p.current} / ${p.total}`;
        });
        }, 500);
}

function resortImagesKeepSelection() {
        if (datasetState.selectedIndex === null) return;

        const current = datasetState.images[datasetState.selectedIndex];

        datasetState.images.sort((a, b) =>
        b.flagged === a.flagged ? 0 : b.flagged ? 1 : -1
        );

        datasetState.selectedIndex = datasetState.images.indexOf(current);
}

function cropAllImages() {
        const size = parseInt(
        document.getElementById("crop-size-input")?.value,
        10
        );

        if (!size || size < 64) return;

        fetch("/api/dataset/crop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        size
        })
        }).then(() => {
        renderImagePreview();
        });
}

function cropSingleImage() {
        if (datasetState.selectedIndex === null) return;

        const size = parseInt(
        document.getElementById("crop-size-input")?.value,
        10
        );

        if (!size || size < 64) return;

        const img = datasetState.images[datasetState.selectedIndex];

        fetch("/api/dataset/crop", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
        name: img.name,
        size
        })
        }).then(() => {
        renderImagePreview();
        });
}

document.addEventListener("DOMContentLoaded", () => {
loadProjects();

const loadBtn = document.getElementById("load-dataset-btn");
if (loadBtn) loadBtn.addEventListener("click", loadDatasetFromUI);

const filterEl = document.getElementById("image-filter");
if (filterEl) {
filterEl.addEventListener("input", (e) => {
datasetState.filter = e.target.value || "";
renderImageList();
});
}
const cropBtn = document.getElementById("crop-btn");

if (cropBtn) {
cropBtn.addEventListener("click", (e) => {
if (e.shiftKey) {
cropSingleImage();
} else {
cropAllImages();
}
});
}

const captionEl = document.getElementById("caption-editor");
if (captionEl) {
captionEl.addEventListener("input", (e) => {
if (datasetState.selectedIndex === null) return;

const img = datasetState.images[datasetState.selectedIndex];
img.caption = e.target.value;
img.flagged = isCaptionFlagged(img.caption);

datasetState.dirty = true;
scheduleSave();

document.getElementById("caption-editor")
?.addEventListener("input", e => {
if (datasetState.selectedIndex === null) return;

const img = datasetState.images[datasetState.selectedIndex];
img.caption = e.target.value;
img.flagged = isCaptionFlagged(img.caption);

datasetState.dirty = true;
scheduleSave();


renderImageList();
});
});
}

const saveBtn = document.getElementById("save-captions-btn");
if (saveBtn) saveBtn.addEventListener("click", saveCaptions);

const deleteBtn = document.getElementById("delete-image-btn");
if (deleteBtn) deleteBtn.addEventListener("click", deleteSelectedImage);

const autoBtn = document.getElementById("autocaption-btn");
if (autoBtn) {
autoBtn.addEventListener("click", (e) => {
if (e.shiftKey) autoCaptionSingle();
else autoCaptionAll();
});
}

const cropHint = document.getElementById("autocrop-hint");

let cropHintTimer = null;

if (cropBtn && cropHint) {
  cropBtn.addEventListener("mouseenter", () => {
    cropHintTimer = setTimeout(() => {
      cropHint.style.opacity = "1";
    }, 1000);
  });

  cropBtn.addEventListener("mouseleave", () => {
    if (cropHintTimer) clearTimeout(cropHintTimer);
    cropHint.style.opacity = "0";
  });
}

const hint = document.getElementById("autocaption-hint");
let hintTimer = null;

if (autoBtn && hint) {
autoBtn.addEventListener("mouseenter", () => {
hintTimer = setTimeout(() => {
hint.style.opacity = "1";
}, 1000);
});

autoBtn.addEventListener("mouseleave", () => {
if (hintTimer) clearTimeout(hintTimer);
hint.style.opacity = "0";
});
}

const projectSelect = document.getElementById("project-select");
if (projectSelect) {
projectSelect.addEventListener("change", (e) => {
const project = e.target.value;
if (!project) return;

fetch(`/api/project/config/${project}`)
.then(r => r.json())
.then(data => {
const pathInput = document.getElementById("dataset-path-input");
if (pathInput) pathInput.value = data.dataset_path || "";
});
});
}
});
