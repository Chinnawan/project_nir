"""Local demo: apple quality grading with the locked formula (nir_ss + TTA + 3-seed soft vote).

  .venv\\Scripts\\python.exe -m src.demo_app      ->  http://127.0.0.1:7860   (or run_demo.bat)

Runs ONLY on this machine (127.0.0.1, share=False): the CAPA images are licensed by ULG and must not
leave the machine. Gradio's temp/cache folder is moved to outputs/demo_cache/ (git-ignored).

Mode 1 - CAPA image (out-of-fold, honest): the checkpoints of the 3 seeds of the fold in which the image
         was a TEST image (fold_assignment.json) -> these models never saw the image. Models of other folds
         are never used for it.
Mode 2 - new image (upload): all 15 checkpoints (5 folds x 3 seeds).
Belt   - N random CAPA images (mode 1 rule per image) sorted into sell fresh / processing / reject.
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
DOMAIN_NOTE = ("ภาพที่อัปโหลดจากกล้อง/แสง/สายพันธุ์อื่นอาจให้ผลต่างจากที่วัดบน CAPA (domain shift) — "
               "ผลนี้ไม่ใช่การรับรองความแม่นยำบนภาพของคุณ")


class Demo:
    def __init__(self, ckpt_dir=CKPT_DIR):
        self.ckpt_dir = ckpt_dir
        with open(FOLD_ASSIGN, encoding="utf-8") as f:
            self.images = json.load(f)["images"]  # "RawA0001" -> {fold, tissue, label, defect_no}
        self.paths = {os.path.basename(fp).replace(".npy", ""): fp
                      for split in ("train", "val", "test") for fp, _ in list_files(split, SS_DIR)}
        assert set(self.paths) == set(self.images)
        self._cache = {}

    # ---- models -------------------------------------------------------------------------------
    def ckpt_path(self, seed, fold):
        return os.path.join(self.ckpt_dir, f"nir_ss_seed{seed}_fold{fold}.pt")

    def available(self):
        return [(s, f) for f in FOLDS for s in SEEDS if os.path.exists(self.ckpt_path(s, f))]

    def get(self, seed, fold):
        if (seed, fold) not in self._cache:
            model, meta = load_checkpoint(self.ckpt_path(seed, fold))
            assert meta["seed"] == seed and meta["fold"] == fold and meta["variant"] == "nir_ss", meta
            self._cache[(seed, fold)] = (model, meta)
        return self._cache[(seed, fold)]

    # ---- prediction ---------------------------------------------------------------------------
    def vote(self, cube, keys):
        rows = []
        for seed, fold in keys:
            model, meta = self.get(seed, fold)
            rows.append({"seed": seed, "fold": fold, "probs": predict_probs(model, meta, cube, tta=True)})
        p = np.mean([r["probs"] for r in rows], axis=0)
        return rows, p, int(p.argmax())

    def keys_for_known(self, rawid):
        fold = self.images[rawid]["fold"]
        keys = [(s, fold) for s in SEEDS]
        missing = [k for k in keys if not os.path.exists(self.ckpt_path(*k))]
        if missing:
            raise FileNotFoundError(f"missing checkpoints for fold {fold}: {missing} (run run_confirm.bat first)")
        assert all(f == fold for _, f in keys)  # never a model that saw this image
        return keys

    def predict_known(self, rawid):
        cube = np.load(self.paths[rawid])
        rows, p, pred = self.vote(cube, self.keys_for_known(rawid))
        return cube, rows, p, pred

    def predict_new(self, cube):
        keys = self.available()
        if not keys:
            raise FileNotFoundError("no checkpoints found in " + self.ckpt_dir)
        rows, p, pred = self.vote(cube, keys)
        return rows, p, pred

    # ---- Grad-CAM -----------------------------------------------------------------------------
    def cam(self, cube, keys, class_idx):
        cams = []
        for seed, fold in keys:
            model, meta = self.get(seed, fold)
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


def load_upload(paths):
    """.npy (H,W,4) BL/GR/RE/IR, or 4 single-band images named *_BL / *_GR / *_RE / *_IR (tif/png/bmp).
    Returns uint8 (224,224,4); non-square inputs are zero-padded to a square first (like the SingleSpectra images)."""
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
def classify_known(demo, rawid):
    if not rawid:
        raise ValueError("เลือกภาพก่อน")
    info = demo.images[rawid]
    cube, rows, p, pred = demo.predict_known(rawid)
    keys = demo.keys_for_known(rawid)
    mask = load_defect_mask(info["defect_no"]) if info["defect_no"] is not None else None
    cam = demo.cam(cube, keys, pred)
    fold_note = f"ใช้เฉพาะโมเดล fold {info['fold']} (ภาพนี้เป็น test ของ fold นี้ โมเดลไม่เคยเห็น)"
    return (to_gray_rgb(cube[..., 2]), to_gray_rgb(cube[..., 3]), result_card(p, pred, info),
            {TH[c]: float(p[i]) for i, c in enumerate(CLASSES)}, prob_table(rows, p, fold_note),
            overlay(cube[..., 3], cam, mask), CAM_NOTE + ("  (เส้นเขียว = ขอบตำหนิจริง)" if mask is not None else ""))


def classify_upload(demo, files):
    cube = load_upload(files)
    rows, p, pred = demo.predict_new(cube)
    keys = [(r["seed"], r["fold"]) for r in rows]
    cam = demo.cam(cube, keys, pred)
    note = f"ใช้ ensemble {len(keys)} โมเดล (ทุก fold × seed) — ภาพนี้ไม่อยู่ใน CAPA จึงไม่มีปัญหา leakage"
    return (to_gray_rgb(cube[..., 2]), to_gray_rgb(cube[..., 3]), result_card(p, pred),
            {TH[c]: float(p[i]) for i, c in enumerate(CLASSES)}, prob_table(rows, p, note),
            overlay(cube[..., 3], cam), CAM_NOTE + "  " + DOMAIN_NOTE)


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

    with gr.Blocks(title="Apple quality grading demo") as ui:
        gr.Markdown("## ระบบคัดคุณภาพแอปเปิลด้วยภาพ NIR (RE 750 nm + IR 800 nm)\n"
                    "สูตรที่ล็อกไว้: โมเดล EfficientNet-B0 บนภาพ SingleSpectra · TTA 6 มุมมอง × 3 seed → โหวตเฉลี่ยความน่าจะเป็น  \n"
                    f"{final_formula_note()}  \nรันบนเครื่องนี้เท่านั้น (ข้อมูล CAPA มีลิขสิทธิ์ ULG)")
        with gr.Tabs():
            with gr.Tab("โหมด 1 — ภาพ CAPA (ซื่อสัตย์: โมเดลไม่เคยเห็นภาพนี้)"):
                with gr.Row():
                    dd = gr.Dropdown(choices, label="เลือกภาพ (RawA id — ชนิดตำหนิจริง)", filterable=True, scale=4)
                    kind = gr.Dropdown(["ทุกประเภท", "ปกติ", "ตำหนิภายใน", "ตำหนิภายนอก"], value="ทุกประเภท", label="สุ่มจาก", scale=1)
                    rnd = gr.Button("สุ่มภาพ", scale=1)
                    go = gr.Button("ทำนาย", variant="primary", scale=1)
                card = gr.HTML()
                with gr.Row():
                    re_img = gr.Image(label="แบนด์ RE (750 nm)", interactive=False)
                    ir_img = gr.Image(label="แบนด์ IR (800 nm)", interactive=False)
                    cam_img = gr.Image(label="Grad-CAM (14×14) ซ้อนภาพ IR", interactive=False)
                with gr.Row():
                    lab = gr.Label(label="ความน่าจะเป็น (โหวตรวม)", num_top_classes=3)
                    tbl = gr.Dataframe(headers=out_cols, label="ตารางโหวต", interactive=False)
                cam_note = gr.Markdown()
                outs = [re_img, ir_img, card, lab, tbl, cam_img, cam_note]
                rnd.click(pick_random, kind, dd, api_name="random_pick")
                go.click(_guard(lambda r: classify_known(demo, r)), dd, outs, api_name="predict_known")
                dd.change(_guard(lambda r: classify_known(demo, r)), dd, outs, api_name=False)
            with gr.Tab("โหมด 2 — อัปโหลดภาพใหม่"):
                gr.Markdown(f"อัปโหลด `.npy` (H×W×4: BL,GR,RE,IR) หรือภาพแบนด์เดี่ยว 4 ไฟล์ที่ชื่อลงท้าย `_BL _GR _RE _IR`  \n**{DOMAIN_NOTE}**")
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
    return ui


def main():
    demo = Demo()
    have = demo.available()
    print(f"checkpoints found: {len(have)}/15 in {demo.ckpt_dir}; device: {DEVICE}")
    ui = build_ui(demo)
    ui.queue().launch(server_name="127.0.0.1", server_port=7860, share=False, inbrowser=True,
                      allowed_paths=[_CACHE])


if __name__ == "__main__":
    main()
