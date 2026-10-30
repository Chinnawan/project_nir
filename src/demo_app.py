"""Local demo: apple quality grading with the locked formula (nir_ss + TTA + 3-seed soft vote).

  .venv\\Scripts\\python.exe -m src.demo_app      ->  http://127.0.0.1:7860   (or run_demo.bat)

Runs ONLY on this machine (127.0.0.1, share=False): the CAPA images are licensed by ULG and must not
leave the machine. Gradio's temp/cache folder is moved to outputs/demo_cache/ (git-ignored).

Mode 1 - CAPA image (out-of-fold, honest): the checkpoints of the 3 seeds of the fold in which the image
         was a TEST image (fold_assignment.json) -> these models never saw the image. Models of other folds
         are never used for it. A model switch (N7) picks NIR (final formula) / Visible (BL+GR) / all 4 bands,
         all trained on the same SingleSpectra images (visible_ss, full_ss from run_n5); "compare all models"
         shows every model on the same apple with its Grad-CAM; presentation examples are picked
         automatically from the recorded out-of-fold predictions.
Mode 2 - upload demo: all 15 checkpoints (5 folds x 3 seeds); not an evaluation mode.
Belt   - N random CAPA images (mode 1 rule per image) sorted into sell fresh / processing / reject.
G2/G3  - Kleynen 2005 (replicated) vs CNN NIR in 2 groups: step by step for one apple, and in batches (src/demo_kleynen.py).
"""
import os

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = os.path.join(_PROJECT, "outputs", "demo_cache")
os.makedirs(_CACHE, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _CACHE            # must be set before gradio is imported
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

import json  # noqa: E402
import random  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from pytorch_grad_cam import GradCAM  # noqa: E402
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: E402

from .config import CLASSES, OUTPUT_DIR  # noqa: E402
from .consent import require_consent  # noqa: E402
from .copy_paste import load_defect_mask  # noqa: E402
from .dataset import list_files  # noqa: E402
from .infer import DEVICE, load_checkpoint, predict_probs, preprocess  # noqa: E402
from .preprocess_ss import pad_square_resize  # noqa: E402
from .train_dl_kfold import SS_DIR  # noqa: E402

CKPT_DIR = os.path.join(OUTPUT_DIR, "dl", "confirm_ckpt")
FOLD_ASSIGN = os.path.join(OUTPUT_DIR, "dl", "fold_assignment.json")
CONFIRM_SUMMARY = os.path.join(OUTPUT_DIR, "dl", "confirm_summary.json")
SEEDS = [3, 4, 5]
FOLDS = range(5)
TH = {"Healthy": "ปกติ", "Internal_Defect": "ตำหนิภายใน", "Surface_Defect": "ตำหนิภายนอก"}
ACTION = {"Healthy": "ขายผลสด", "Surface_Defect": "ส่งแปรรูป", "Internal_Defect": "คัดทิ้ง"}
BIN_ORDER = ["Healthy", "Surface_Defect", "Internal_Defect"]
GRADCAM_LAYER = 5  # model.features[5]: last 14x14 block
CAM_NOTE = ("heatmap บอกบริเวณคร่าวๆ เท่านั้น ไม่ใช่ตำแหน่งตำหนิที่แม่นยำ — เมื่อวัดกับ mask ตำหนิจริง (โมเดล nir_SE_ss, "
            "ชั้น 14×14) ตำหนิภายนอกชี้ถูกจุด 13% เทียบกับสุ่ม 6% และตำหนิภายในชี้ถูกจุด 40% เทียบกับสุ่ม 31%")
# mode-1 model switch (N7): all on SingleSpectra images, seeds 3-5, TTA + soft vote, always the image's own test fold
MODELS = {
    "nir_ss": {"label": "NIR (RE+IR) — สูตรสุดท้าย", "short": "NIR (RE+IR)", "results": "confirm_results.json",
               "bands": [(2, "แบนด์ RE (750 nm)"), (3, "แบนด์ IR (800 nm)")], "cam_band": (3, "IR")},
    "visible_ss": {"label": "Visible (BL+GR)", "short": "Visible (BL+GR)", "results": "n5_results.json",
                   "bands": [(0, "แบนด์ BL (450 nm)"), (1, "แบนด์ GR (500 nm)")], "cam_band": (1, "GR")},
    "full_ss": {"label": "4 แบนด์ (BL+GR+RE+IR)", "short": "4 แบนด์", "results": "n5_results.json",
                "bands": [(1, "แบนด์ GR (500 nm)"), (3, "แบนด์ IR (800 nm)")], "cam_band": (3, "IR")},
}
DEFAULT_MODEL = "nir_ss"
DOMAIN_NOTE = ("ภาพจากกล้อง/แสง/สายพันธุ์อื่นอาจให้ผลต่างจากที่วัดบน CAPA (domain shift) — "
               "ผลนี้ไม่ใช่การรับรองความแม่นยำบนภาพของคุณ")
UPLOAD_NOTE = ("โหมดอัปโหลดใช้สาธิตภาพใหม่เท่านั้น ไม่ใช่การประเมินความแม่นยำ: ระบบตรวจไม่ได้ว่าไฟล์นี้เคยอยู่ใน CAPA "
               "หรือไม่ ถ้าเป็นภาพ CAPA ให้เลือกภาพในโหมด out-of-fold แทน มิฉะนั้นโมเดลบาง fold อาจเคยเห็นภาพแล้ว")


class Demo:
    def __init__(self, ckpt_dir=CKPT_DIR):
        require_consent()
        self.ckpt_dir = ckpt_dir
        with open(FOLD_ASSIGN, encoding="utf-8") as f:
            self.images = json.load(f)["images"]  # "RawA0001" -> {fold, tissue, label, defect_no}
        self.paths = {os.path.basename(fp).replace(".npy", ""): fp
                      for split in ("train", "val", "test") for fp, _ in list_files(split, SS_DIR)}
        assert set(self.paths) == set(self.images)
        self._cache = {}  # (variant, seed, fold) -> (model, meta)
        self._oof = None

    # ---- models -------------------------------------------------------------------------------
    def ckpt_path(self, seed, fold, variant=DEFAULT_MODEL):
        return os.path.join(self.ckpt_dir, f"{variant}_seed{seed}_fold{fold}.pt")

    def available(self, variant=DEFAULT_MODEL):
        return [(s, f) for f in FOLDS for s in SEEDS if os.path.exists(self.ckpt_path(s, f, variant))]

    def variants(self):
        """models with all 15 checkpoints (the default model first)"""
        return [v for v in MODELS if len(self.available(v)) == len(SEEDS) * len(FOLDS)]

    def get(self, seed, fold, variant=DEFAULT_MODEL):
        if (variant, seed, fold) not in self._cache:
            model, meta = load_checkpoint(self.ckpt_path(seed, fold, variant))
            assert meta["seed"] == seed and meta["fold"] == fold and meta["variant"] == variant, meta
            self._cache[(variant, seed, fold)] = (model, meta)
        return self._cache[(variant, seed, fold)]

    # ---- prediction ---------------------------------------------------------------------------
    def vote(self, cube, keys, variant=DEFAULT_MODEL):
        rows = []
        for seed, fold in keys:
            model, meta = self.get(seed, fold, variant)
            rows.append({"seed": seed, "fold": fold, "probs": predict_probs(model, meta, cube, tta=True)})
        p = np.mean([r["probs"] for r in rows], axis=0)
        return rows, p, int(p.argmax())

    def keys_for_known(self, rawid, variant=DEFAULT_MODEL):
        fold = self.images[rawid]["fold"]
        keys = [(s, fold) for s in SEEDS]
        missing = [k for k in keys if not os.path.exists(self.ckpt_path(*k, variant))]
        if missing:
            raise FileNotFoundError(f"missing {variant} checkpoints for fold {fold}: {missing} "
                                    f"(run {'run_confirm.bat' if variant == 'nir_ss' else 'run_n5.bat'} first)")
        assert all(f == fold for _, f in keys)  # never a model that saw this image
        return keys

    def predict_known(self, rawid, variant=DEFAULT_MODEL):
        cube = np.load(self.paths[rawid])
        rows, p, pred = self.vote(cube, self.keys_for_known(rawid, variant), variant)
        return cube, rows, p, pred

    def predict_new(self, cube):
        keys = self.available()
        if not keys:
            raise FileNotFoundError("no checkpoints found in " + self.ckpt_dir)
        rows, p, pred = self.vote(cube, keys)
        return rows, p, pred

    # ---- recorded out-of-fold results (model summary numbers + presentation examples) ------------
    def oof(self):
        """{variant: {rawid: 3-seed mean of the recorded TTA probabilities}} from the result files"""
        if self._oof is None:
            from .round2 import Context
            names = [os.path.basename(fp).replace(".npy", "") for fp, _ in Context().pool_raw]
            self._oof = {}
            for v, spec in MODELS.items():
                path = os.path.join(OUTPUT_DIR, "dl", spec["results"])
                if not os.path.exists(path):
                    continue
                acc = {}
                for r in json.load(open(path, encoding="utf-8"))["runs"].values():
                    if r["variant"] == v and r["seed"] in SEEDS:
                        for i, p in zip(r["test_idx"], r["tta_test_probs"]):
                            acc.setdefault(names[i], []).append(p)
                if len(acc) == len(self.images) and all(len(p) == len(SEEDS) for p in acc.values()):
                    self._oof[v] = {k: np.mean(p, axis=0) for k, p in acc.items()}
        return self._oof

    def model_summary(self, variant):
        o = self.oof().get(variant)
        if o is None:
            return None
        truth = np.array([CLASSES.index(self.images[k]["label"]) for k in sorted(o)])
        pred = np.array([int(o[k].argmax()) for k in sorted(o)])
        f1 = np.mean([2 * np.sum((pred == c) & (truth == c)) / (np.sum(pred == c) + np.sum(truth == c)) for c in range(3)])
        return {"accuracy": float(np.mean(pred == truth)), "macro_f1": float(f1)}

    def examples(self, n_max=5):
        """presentation examples picked automatically from the recorded out-of-fold predictions:
        (a) Bruise: NIR right, Visible wrong  (b) every model wrong  (c) Visible right, NIR wrong (for honesty)"""
        o = self.oof()
        if not {"nir_ss", "visible_ss"} <= set(o):
            return []
        ok = {v: {k: CLASSES[int(p.argmax())] == self.images[k]["label"] for k, p in o[v].items()} for v in o}
        ids = sorted(self.images)
        groups = [("(ก) Bruise: NIR ถูก / Visible ผิด", [k for k in ids if self.images[k]["tissue"] == "Bruise"
                                                       and ok["nir_ss"][k] and not ok["visible_ss"][k]]),
                  ("(ข) ทุกโมเดลผิด", [k for k in ids if not any(ok[v][k] for v in ok)]),
                  ("(ค) Visible ถูก / NIR ผิด", [k for k in ids if ok["visible_ss"][k] and not ok["nir_ss"][k]])]
        out = []
        for title, members in groups:
            # the most convincing first: highest recorded confidence of the model that is right (or, for (b), of the final model)
            ref = "visible_ss" if title.startswith("(ค)") else "nir_ss"
            members = sorted(members, key=lambda k: -o[ref][k].max())[:n_max]
            out += [(f"{title} — {k} {self.images[k]['tissue']} (จริง: {TH[self.images[k]['label']]})", k) for k in members]
        return out

    # ---- Grad-CAM -----------------------------------------------------------------------------
    def cam(self, cube, keys, class_idx, variant=DEFAULT_MODEL):
        cams = []
        for seed, fold in keys:
            model, meta = self.get(seed, fold, variant)
            x = preprocess(cube, meta).to(DEVICE)
            with GradCAM(model=model, target_layers=[model.features[GRADCAM_LAYER]]) as cam:
                cams.append(cam(input_tensor=x, targets=[ClassifierOutputTarget(class_idx)])[0])
        return np.mean(cams, axis=0)


def to_gray_rgb(band, size=448):
    return cv2.cvtColor(cv2.resize(band, (size, size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2RGB)


def overlay(ir_band, cam, mask=None, size=448):
    base = to_gray_rgb(ir_band, size)
    heat = cv2.cvtColor(cv2.applyColorMap((cv2.resize(cam, (size, size)) * 255).astype(np.uint8), cv2.COLORMAP_JET),
                        cv2.COLOR_BGR2RGB)
    out = (0.55 * base + 0.45 * heat).astype(np.uint8)
    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (0, 255, 0), 2)
    return out


def read_upload_bands(paths):
    """.npy (H,W,4) BL/GR/RE/IR, or 4 single-band images named *_BL / *_GR / *_RE / *_IR (tif/png/bmp).
    Returns the 4 bands as uint8 at their own resolution."""
    require_consent()
    paths = [p if isinstance(p, str) else p.name for p in (paths or [])]
    if not paths:
        raise ValueError("ยังไม่ได้เลือกไฟล์")
    if len(paths) == 1 and paths[0].lower().endswith(".npy"):
        arr = np.load(paths[0])
        if arr.ndim != 3 or arr.shape[2] != 4:
            raise ValueError(f".npy ต้องเป็น H×W×4 (BL,GR,RE,IR) แต่ได้รูปร่าง {arr.shape}")
        bands = [arr[..., i] for i in range(4)]
    else:
        by_band = {}
        for p in paths:
            stem = os.path.splitext(os.path.basename(p))[0].upper()
            for b in ("BL", "GR", "RE", "IR"):
                if stem.endswith("_" + b):
                    by_band[b] = p
        if set(by_band) != {"BL", "GR", "RE", "IR"}:
            raise ValueError("ต้องอัปโหลด 4 ไฟล์ที่ชื่อลงท้าย _BL _GR _RE _IR (เช่น Apple01_BL.tif) หรือไฟล์ .npy เดียว; "
                             f"พบแบนด์: {sorted(by_band) or 'ไม่มี'}")
        bands = []
        for b in ("BL", "GR", "RE", "IR"):
            img = cv2.imread(by_band[b], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"อ่านไฟล์ไม่ได้: {by_band[b]}")
            bands.append(img[..., 0] if img.ndim == 3 else img)
    shapes = {b.shape for b in bands}
    if len(shapes) != 1:
        raise ValueError(f"4 แบนด์ต้องมีขนาดเท่ากัน แต่ได้ {sorted(shapes)}")
    out = []
    for b in bands:
        if b.dtype != np.uint8:
            b = b.astype(np.float64)
            b = b * 255.0 if b.max() <= 1.0 else b * (255.0 / 65535.0) if b.max() > 255 else b
            b = np.clip(b, 0, 255).astype(np.uint8)
        out.append(b)
    return out


def load_upload(paths):
    """read_upload_bands -> uint8 (224,224,4); non-square inputs are zero-padded to a square first (like the SingleSpectra images)."""
    out = []
    for b in read_upload_bands(paths):
        if b.shape != (224, 224):
            b = pad_square_resize(b) if b.shape[0] != b.shape[1] else cv2.resize(b, (224, 224), interpolation=cv2.INTER_AREA)
        out.append(b)
    return np.stack(out, axis=-1)


def prob_table(rows, p, fold_note):
    table = [[f"seed {r['seed']} (fold {r['fold']})"] + [f"{x:.3f}" for x in r["probs"]] + [TH[CLASSES[int(r["probs"].argmax())]]]
             for r in rows]
    table.append([f"โหวตรวม (เฉลี่ยความน่าจะเป็น) — {fold_note}"] + [f"{x:.3f}" for x in p] + [TH[CLASSES[int(p.argmax())]]])
    return table


def result_card(p, pred, truth=None):
    """Left: prediction, action and a large confidence number with a bar. Right (mode 1 only): the answer key --
    big check mark on green if correct, big X on red if wrong."""
    cls = CLASSES[pred]
    # class colours avoid green/red, which are reserved for correct/wrong in the answer box
    colour = {"Healthy": "#2f80ed", "Surface_Defect": "#e08a00", "Internal_Defect": "#9b51e0"}[cls]
    conf = p[pred] * 100
    left = (f"<div style='flex:1;padding:14px 18px;border-radius:10px;border:3px solid {colour}'>"
            f"<div style='font-size:24px;font-weight:700;color:{colour}'>ทำนาย: {TH[cls]} → {ACTION[cls]}</div>"
            f"<div style='display:flex;align-items:baseline;gap:12px;margin-top:6px'>"
            f"<span style='font-size:20px'>ความมั่นใจ</span>"
            f"<span style='font-size:56px;font-weight:800;line-height:1;color:{colour}'>{conf:.1f}%</span></div>"
            f"<div style='margin-top:10px;height:14px;border-radius:7px;background:#3a3a3a;overflow:hidden'>"
            f"<div style='width:{conf:.1f}%;height:100%;background:{colour}'></div></div></div>")
    if truth is None:
        return f"<div style='display:flex;gap:14px'>{left}</div>"
    ok = truth["label"] == cls
    bg, mark, word = ("#1a7f37", "&#10004;", "ถูก") if ok else ("#c62828", "&#10008;", "ผิด")
    right = (f"<div style='flex:0 0 340px;display:flex;align-items:center;gap:16px;padding:14px 18px;border-radius:10px;"
             f"background:{bg};color:#ffffff'>"
             f"<div style='font-size:84px;font-weight:900;line-height:1'>{mark}</div>"
             f"<div><div style='font-size:40px;font-weight:800;line-height:1.1'>{word}</div>"
             f"<div style='font-size:17px;margin-top:6px'>คำตอบจริง: <b>{TH[truth['label']]}</b></div>"
             f"<div style='font-size:15px;opacity:0.9'>ชนิด: {truth['tissue']}</div></div></div>")
    return f"<div style='display:flex;gap:14px;align-items:stretch'>{left}{right}</div>"


def final_formula_note():
    if os.path.exists(CONFIRM_SUMMARY):
        s = json.load(open(CONFIRM_SUMMARY, encoding="utf-8")).get("final_formula") or {}
        if s:
            return (f"ผลวัดจริงของสูตรนี้ (out-of-fold ทั้ง 819 ภาพ, seed 3–5): accuracy {s['accuracy'] * 100:.1f}%, "
                    f"macro-F1 {s['macro_f1']:.3f}")
    return "ผลวัดจริงของสูตรนี้: ดู confirm_summary.json (ยังไม่มี)"


# ----------------------------------------------------------------------------- UI callbacks
def model_note(demo, variant):
    s = demo.model_summary(variant)
    nums = (f"ผลรวมของโมเดลนี้ (out-of-fold ทั้ง 819 ภาพ, 3 seed + TTA): accuracy **{s['accuracy'] * 100:.1f}%**, "
            f"macro-F1 **{s['macro_f1']:.3f}**") if s else "ผลรวมของโมเดลนี้: ยังไม่มีไฟล์ผล"
    return f"**โมเดล: {MODELS[variant]['label']}** — ภาพ SingleSpectra · {nums}"


def classify_known(demo, rawid, variant=DEFAULT_MODEL):
    if not rawid:
        raise ValueError("เลือกภาพก่อน")
    info = demo.images[rawid]
    cube, rows, p, pred = demo.predict_known(rawid, variant)
    keys = demo.keys_for_known(rawid, variant)
    mask = load_defect_mask(info["defect_no"]) if info["defect_no"] is not None else None
    cam = demo.cam(cube, keys, pred, variant)
    (b1, _), (b2, _) = MODELS[variant]["bands"]
    cb, cname = MODELS[variant]["cam_band"]
    fold_note = f"ใช้เฉพาะโมเดล fold {info['fold']} (ภาพนี้เป็น test ของ fold นี้ โมเดลไม่เคยเห็น)"
    return (to_gray_rgb(cube[..., b1]), to_gray_rgb(cube[..., b2]), result_card(p, pred, info),
            {TH[c]: float(p[i]) for i, c in enumerate(CLASSES)}, prob_table(rows, p, fold_note),
            overlay(cube[..., cb], cam, mask),
            model_note(demo, variant) + f"  \nGrad-CAM ซ้อนบนแบนด์ {cname} · " + CAM_NOTE +
            ("  (เส้นเขียว = ขอบตำหนิจริง)" if mask is not None else ""))


def compare_all(demo, rawid):
    """Same apple, every available model (each with its own fold-matched checkpoints): table + Grad-CAM side by side."""
    if not rawid:
        raise ValueError("เลือกภาพก่อน")
    info = demo.images[rawid]
    mask = load_defect_mask(info["defect_no"]) if info["defect_no"] is not None else None
    rows_html, cams, table = "", [], []
    for v in MODELS:
        if v not in demo.variants():
            cams.append(None)
            rows_html += f"<tr><td style='padding:8px 12px'>{MODELS[v]['label']}</td><td colspan='3'>ยังไม่มี checkpoint</td></tr>"
            continue
        cube, _, p, pred = demo.predict_known(rawid, v)
        cb, cname = MODELS[v]["cam_band"]
        cams.append(overlay(cube[..., cb], demo.cam(cube, demo.keys_for_known(rawid, v), pred, v), mask, size=336))
        ok = CLASSES[pred] == info["label"]
        bg, mark = ("#1a7f37", "&#10004; ถูก") if ok else ("#c62828", "&#10008; ผิด")
        table.append({"model": v, "pred": CLASSES[pred], "conf": float(p[pred]), "correct": ok})
        rows_html += (f"<tr style='font-size:20px'><td style='padding:8px 12px;font-weight:700'>{MODELS[v]['label']}</td>"
                      f"<td style='padding:8px 12px'>{TH[CLASSES[pred]]}</td>"
                      f"<td style='padding:8px 12px;font-size:28px;font-weight:800'>{p[pred] * 100:.1f}%</td>"
                      f"<td style='padding:8px 16px;background:{bg};color:#fff;font-size:24px;font-weight:800;text-align:center'>{mark}</td></tr>")
    html = (f"<div style='font-size:18px;margin-bottom:6px'>{rawid} · คำตอบจริง: <b>{TH[info['label']]}</b> · ชนิด: {info['tissue']} · "
            f"ทุกโมเดลใช้เฉพาะ fold {info['fold']} (ไม่เคยเห็นภาพนี้)</div>"
            f"<table style='border-collapse:collapse;width:100%'><tr style='text-align:left;opacity:0.8'><th style='padding:6px 12px'>โมเดล</th>"
            f"<th style='padding:6px 12px'>คำทาย</th><th style='padding:6px 12px'>ความมั่นใจ</th><th style='padding:6px 12px'>ถูก/ผิด</th></tr>"
            f"{rows_html}</table>")
    return (html, *cams, table)


def classify_upload(demo, files):
    cube = load_upload(files)
    rows, p, pred = demo.predict_new(cube)
    keys = [(r["seed"], r["fold"]) for r in rows]
    cam = demo.cam(cube, keys, pred)
    note = f"ใช้ ensemble {len(keys)} โมเดล (ทุก fold × seed) — {UPLOAD_NOTE}"
    return (to_gray_rgb(cube[..., 2]), to_gray_rgb(cube[..., 3]), result_card(p, pred),
            {TH[c]: float(p[i]) for i, c in enumerate(CLASSES)}, prob_table(rows, p, note),
            overlay(cube[..., 3], cam), CAM_NOTE + "  " + DOMAIN_NOTE + "  " + UPLOAD_NOTE)


def run_belt(demo, n, seed, progress=None):
    n = int(n)
    ids = sorted(demo.images)
    chosen = random.Random(int(seed)).sample(ids, n)
    results = []
    for k, rawid in enumerate(chosen):
        if progress is not None:
            progress((k, n), desc=f"ทำนายภาพที่ {k + 1}/{n}")
        _, _, p, pred = demo.predict_known(rawid)
        info = demo.images[rawid]
        results.append({"id": rawid, "tissue": info["tissue"], "true": info["label"], "pred": CLASSES[pred],
                        "probs": p, "fold": info["fold"]})
    cm = np.zeros((3, 3), dtype=int)
    for r in results:
        cm[CLASSES.index(r["true"]), CLASSES.index(r["pred"])] += 1
    acc = float(np.trace(cm)) / n
    cards = ""
    for c in BIN_ORDER:
        members = [r for r in results if r["pred"] == c]
        good = sum(r["true"] == c for r in members)
        wrong = [f"{TH[r['true']]}" for r in members if r["true"] != c]
        cards += (f"<div style='flex:1;padding:10px;border:2px solid #888;border-radius:8px;margin:4px'>"
                  f"<div style='font-size:18px;font-weight:600'>{ACTION[c]}</div><div>{len(members)} ผล</div>"
                  f"<div>ถูก {good}" + (f" · ผิด {len(members) - good} ({', '.join(sorted(set(wrong)))} ที่ถูกจัดมาผิดถัง)"
                                          if len(members) > good else "") + "</div></div>")
    leaked = sum(1 for r in results if r["true"] != "Healthy" and r["pred"] == "Healthy")
    summary = (f"<div style='display:flex'>{cards}</div><div style='margin-top:8px'>ทายถูก <b>{int(np.trace(cm))}/{n}</b> "
               f"= <b>{acc * 100:.1f}%</b> · ตำหนิที่หลุดไปขายผลสด: <b>{leaked}</b> ผล</div>")
    cm_rows = [[f"จริง: {TH[CLASSES[i]]}"] + [int(cm[i, j]) for j in range(3)] for i in range(3)]
    wrong_rows = [[r["id"], r["tissue"], TH[r["true"]], TH[r["pred"]], f"fold {r['fold']}"] +
                  [f"{x:.2f}" for x in r["probs"]] for r in results if r["true"] != r["pred"]]
    return summary, cm_rows, wrong_rows


def build_ui(demo):
    import gradio as gr

    ids = sorted(demo.images)
    choices = [(f"{i} — {demo.images[i]['tissue']} ({TH[demo.images[i]['label']]})", i) for i in ids]
    out_cols = ["seed (fold)"] + [f"P({TH[c]})" for c in CLASSES] + ["ทาย"]

    def _guard(fn):
        def run(*a):
            try:
                return fn(*a)
            except (ValueError, FileNotFoundError) as e:
                raise gr.Error(str(e))
        return run

    def pick_random(kind):
        pool = [i for i in ids if kind == "ทุกประเภท" or demo.images[i]["label"] == {"ปกติ": "Healthy", "ตำหนิภายใน": "Internal_Defect",
                                                                                   "ตำหนิภายนอก": "Surface_Defect"}[kind]]
        return random.choice(pool)

    variants = demo.variants() or [DEFAULT_MODEL]
    model_choices = [(MODELS[v]["label"], v) for v in variants]
    examples = demo.examples()

    def classify_ui(rawid, variant):
        out = list(classify_known(demo, rawid, variant))
        (_, l1), (_, l2) = MODELS[variant]["bands"]
        out[0] = gr.Image(value=out[0], label=l1)
        out[1] = gr.Image(value=out[1], label=l2)
        out[5] = gr.Image(value=out[5], label=f"Grad-CAM (14×14) ซ้อนแบนด์ {MODELS[variant]['cam_band'][1]}")
        return out

    def classify_if_picked(rawid, variant):
        if not rawid:  # clearing the dropdown or switching model before picking an image: nothing to do
            return [gr.skip()] * 7
        return classify_ui(rawid, variant)

    with gr.Blocks(title="Apple quality grading demo") as ui:
        gr.Markdown("## ระบบคัดคุณภาพแอปเปิลด้วยภาพ NIR (RE 750 nm + IR 800 nm)\n"
                    "สูตรที่ล็อกไว้: โมเดล EfficientNet-B0 บนภาพ SingleSpectra · TTA 6 มุมมอง × 3 seed → โหวตเฉลี่ยความน่าจะเป็น  \n"
                    f"{final_formula_note()}  \nโหมด 1 สลับเป็นโมเดล Visible / 4 แบนด์ เพื่อเทียบได้ (ภาพเดียวกัน preprocessing เดียวกัน) · "
                    "แท็บ \"ทีละขั้น\" / \"คัดทีละชุด\" เทียบกับวิธีต้นฉบับของ dataset (Kleynen 2005) แบบ 2 กลุ่ม  \n"
                    "รันบนเครื่องนี้เท่านั้น (ข้อมูล CAPA มีลิขสิทธิ์ ULG)")
        with gr.Tabs() as tabs:
            with gr.Tab("โหมด 1 — ภาพ CAPA (ซื่อสัตย์: โมเดลไม่เคยเห็นภาพนี้)"):
                with gr.Row():
                    model_dd = gr.Dropdown(model_choices, value=variants[0], label="โมเดล", scale=2)
                    # empty first choice: Gradio pre-selects the first choice, and re-picking it would fire no change event
                    ex_dd = gr.Dropdown([("— เลือกตัวอย่าง —", "")] + examples, value="",
                                        label="ตัวอย่างสำหรับพรีเซนต์ (คัดอัตโนมัติจากผล out-of-fold)", scale=4,
                                        interactive=bool(examples))
                with gr.Row():
                    dd = gr.Dropdown(choices, label="เลือกภาพ (RawA id — ชนิดตำหนิจริง)", filterable=True, scale=4)
                    kind = gr.Dropdown(["ทุกประเภท", "ปกติ", "ตำหนิภายใน", "ตำหนิภายนอก"], value="ทุกประเภท", label="สุ่มจาก", scale=1)
                    rnd = gr.Button("สุ่มภาพ", scale=1)
                    go = gr.Button("ทำนาย", variant="primary", scale=1)
                    cmp_btn = gr.Button("เทียบทุกโมเดล", variant="secondary", scale=1)
                card = gr.HTML()
                with gr.Row():
                    re_img = gr.Image(label="แบนด์ RE (750 nm)", interactive=False)
                    ir_img = gr.Image(label="แบนด์ IR (800 nm)", interactive=False)
                    cam_img = gr.Image(label="Grad-CAM (14×14) ซ้อนภาพ IR", interactive=False)
                with gr.Row():
                    lab = gr.Label(label="ความน่าจะเป็น (โหวตรวม)", num_top_classes=3)
                    tbl = gr.Dataframe(headers=out_cols, label="ตารางโหวต", interactive=False)
                cam_note = gr.Markdown()
                with gr.Accordion("เทียบทุกโมเดลบนแอปเปิลลูกนี้", open=True):
                    cmp_html = gr.HTML()
                    with gr.Row():
                        cmp_cams = [gr.Image(label=f"Grad-CAM — {MODELS[v]['short']} (ซ้อนแบนด์ {MODELS[v]['cam_band'][1]})",
                                             interactive=False) for v in MODELS]
                    cmp_state = gr.JSON(visible=False)
                outs = [re_img, ir_img, card, lab, tbl, cam_img, cam_note]
                rnd.click(pick_random, kind, dd, api_name="random_pick")
                go.click(_guard(classify_ui), [dd, model_dd], outs, api_name="predict_known")
                dd.change(_guard(classify_if_picked), [dd, model_dd], outs, api_name=False)
                model_dd.change(_guard(classify_if_picked), [dd, model_dd], outs, api_name=False)
                cmp_btn.click(_guard(lambda r: compare_all(demo, r)), dd, [cmp_html, *cmp_cams, cmp_state], api_name="compare_all")
                ex_dd.change(lambda r: r or gr.skip(), ex_dd, dd, api_name=False).then(
                    _guard(lambda r: compare_all(demo, r) if r else [gr.skip()] * (len(MODELS) + 2)), dd,
                    [cmp_html, *cmp_cams, cmp_state], api_name=False)
            with gr.Tab("โหมด 2 — อัปโหลดภาพใหม่"):
                gr.Markdown(f"อัปโหลด `.npy` (H×W×4: BL,GR,RE,IR) หรือภาพแบนด์เดี่ยว 4 ไฟล์ที่ชื่อลงท้าย `_BL _GR _RE _IR`  \n**{UPLOAD_NOTE}**  \n{DOMAIN_NOTE}")
                up = gr.File(file_count="multiple", label="ไฟล์ภาพ")
                go2 = gr.Button("ทำนาย", variant="primary")
                card2 = gr.HTML()
                with gr.Row():
                    re2 = gr.Image(label="แบนด์ RE (750 nm)", interactive=False)
                    ir2 = gr.Image(label="แบนด์ IR (800 nm)", interactive=False)
                    cam2 = gr.Image(label="Grad-CAM (14×14) ซ้อนภาพ IR", interactive=False)
                with gr.Row():
                    lab2 = gr.Label(label="ความน่าจะเป็น (โหวตรวม)", num_top_classes=3)
                    tbl2 = gr.Dataframe(headers=out_cols, label="ตารางโหวต", interactive=False)
                cam_note2 = gr.Markdown()
                go2.click(_guard(lambda f: classify_upload(demo, f)), up, [re2, ir2, card2, lab2, tbl2, cam2, cam_note2],
                          api_name="predict_upload")
            with gr.Tab("โหมดสายพาน"):
                gr.Markdown("สุ่มภาพ CAPA จัดเข้าถัง **ขายผลสด / ส่งแปรรูป / คัดทิ้ง** (แต่ละภาพใช้โมเดล fold ที่ไม่เคยเห็นภาพนั้น)")
                with gr.Row():
                    n_in = gr.Slider(10, 200, value=50, step=1, label="จำนวนผลแอปเปิล")
                    seed_in = gr.Number(value=0, precision=0, label="seed การสุ่ม (ผลซ้ำได้)")
                    go3 = gr.Button("เริ่มคัด", variant="primary")
                belt_html = gr.HTML()
                belt_cm = gr.Dataframe(headers=["", f"ทาย: {TH[CLASSES[0]]}", f"ทาย: {TH[CLASSES[1]]}", f"ทาย: {TH[CLASSES[2]]}"],
                                       label="Confusion matrix", interactive=False)
                belt_wrong = gr.Dataframe(headers=["ภาพ", "ชนิดจริง", "คลาสจริง", "ทาย", "fold"] + [f"P({TH[c]})" for c in CLASSES],
                                          label="ภาพที่ทายผิด", interactive=False)
                go3.click(_guard(lambda n, s, progress=gr.Progress(): run_belt(demo, n, s, progress)), [n_in, seed_in],
                          [belt_html, belt_cm, belt_wrong], api_name="belt")
            from .demo_kleynen import build_tabs
            build_tabs(demo, tabs, _guard)
    return ui


def main():
    require_consent()
    demo = Demo()
    for v in MODELS:
        print(f"{v}: checkpoints found {len(demo.available(v))}/15 in {demo.ckpt_dir}")
    from .kleynen_models import MODEL_DIR, all_present
    print(f"Kleynen GUI models in {MODEL_DIR}: {'all present' if all_present() else 'MISSING -> python -m src.kleynen_models'}")
    print(f"device: {DEVICE}")
    ui = build_ui(demo)
    ui.queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True,
                      allowed_paths=[_CACHE])


if __name__ == "__main__":
    main()
