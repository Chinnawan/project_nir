"""G2 / G3: Kleynen 2005 (replicated, K3 pipeline = G1 models) vs CNN (nir_ss, final formula) in 2 groups (sound / defective).

Two extra tabs of the local demo (src/demo_app.py):
  G2 "ทีละขั้น: Kleynen vs CNN"  one apple, every step of both methods side by side
  G3 "คัดทีละชุด"                a set of apples (all / fold / tissue / severity), both methods out-of-fold

Honesty rule as in mode 1: a dataset image is only ever processed by the models of the fold in which it was a TEST image
(fold_assignment.json) - Kleynen: models/kleynen_<bands>_fold<f>.pkl (= the K3 model of that fold, reproduced exactly),
CNN: the 3 seeds of that fold. The all-data Kleynen model is used for uploaded (non-dataset) images only.
2 groups for the CNN: defective if P(Internal) + P(Surface) > 0.5 (as K2).
Times: processing only (from the loaded image to the decision; no file reading, model loading or visualisation).
"""
import csv
import json
import os
import re
import time

import cv2
import numpy as np

from . import demo_app as d
from . import kleynen as K
from . import kleynen_models as KM
from .config import CLASSES, OUTPUT_DIR

GROUP_TH = {0: "ดี", 1: "มีตำหนิ"}
SEV_TH = {"reject": "คัดทิ้ง (reject)", "bruise": "รอยช้ำใหม่ (bruise)", "serious": "ตำหนิรุนแรง (serious)",
          "slight": "ตำหนิเล็กน้อย (slight)"}
EXAMPLE_TISSUES = {"Healthy": "ผลดี", "Stem": "ขั้ว", "Calix": "ก้น", "Bruise": "รอยช้ำ", "Russet": "russet", "Rot": "เน่า"}
BAND_LABELS = ["BL 450 nm", "GR 500 nm", "RE 750 nm", "IR 800 nm"]
GALLERY_MAX = 120
STEM_NOTE = ("ขั้ว/ก้น: เปเปอร์ Kleynen ใช้ตัวตรวจขั้ว/ก้นแยก (template matching) ซึ่งไม่ได้ทำซ้ำ — LDA ของ Kleynen "
             "เทรนบนผลดี + ผลมีตำหนิเท่านั้น (แบบ A ตามเปเปอร์) จึงมักเห็นขั้ว/ก้นเป็นตำหนิ")


def kleynen_row(band_set):
    return f"Kleynen replicated {band_set} [A: paper protocol]"


class Compare:
    def __init__(self, demo):
        self.demo = demo
        self.mapping = json.load(open(os.path.join(OUTPUT_DIR, "ss_mapping.json")))
        self.severity = self._severity()
        self._models = {}
        self._kle, self._cnn = {}, {}  # batch caches: (band_set, rawid) -> dict, rawid -> dict
        self._names = None

    # ---- labels -------------------------------------------------------------------------------
    def _severity(self):
        """{rawid: severity} from outputs/severity_labels.csv (K1); falls back to the dataset txt it was made from"""
        if os.path.exists(K.SEVERITY_CSV):
            with open(K.SEVERITY_CSV, encoding="utf-8") as f:
                return {r["image"]: r["severity"] for r in csv.DictReader(f)}
        sev = K.load_severity()
        return {k: sev[v["defect_no"]] for k, v in self.demo.images.items() if v["defect_no"] is not None}

    def truth(self, rawid):
        info = self.demo.images[rawid]
        return {**info, "group": int(info["label"] != "Healthy"), "severity": self.severity.get(rawid, "")}

    def names(self):
        if self._names is None:
            from .round2 import Context
            self._names = [os.path.basename(fp).replace(".npy", "") for fp, _ in Context().pool_raw]
        return self._names

    # ---- models -------------------------------------------------------------------------------
    def kmodel(self, band_set, fold):
        if (band_set, fold) not in self._models:
            if not os.path.exists(KM.model_path(band_set, fold)):
                raise FileNotFoundError(f"ไม่พบ {KM.model_path(band_set, fold)} — รัน python -m src.kleynen_models ก่อน (run_demo.bat ทำให้อัตโนมัติ)")
            m = KM.load_model(band_set, fold)
            assert m["fold"] == fold and m["band_set"] == band_set and m["erode_px"] == 5
            self._models[(band_set, fold)] = m
        return self._models[(band_set, fold)]

    def kmodel_for(self, rawid, band_set):
        fold = self.demo.images[rawid]["fold"]
        m = self.kmodel(band_set, fold)
        assert rawid in m["test_images"]  # never a model that saw this image
        return m

    # ---- the two methods on one dataset image ---------------------------------------------------
    def kleynen(self, rawid, band_set):
        bands = KM.load_ss_bands(rawid, self.mapping)
        return bands, KM.analyze(bands, self.kmodel_for(rawid, band_set))

    def cnn(self, cube, keys):
        for s, f in keys:  # load first: the time is the processing time only
            self.demo.get(s, f)
        t0 = time.perf_counter()
        rows, p, pred = self.demo.vote(cube, keys)
        ms = (time.perf_counter() - t0) * 1000
        p_def = float(p[1] + p[2])
        return {"rows": rows, "probs": p, "pred3": pred, "p_defect": p_def, "pred": int(p_def > 0.5), "ms": ms,
                "n_models": len(keys)}

    def kleynen_decision(self, rawid, band_set):
        if (band_set, rawid) not in self._kle:
            _, r = self.kleynen(rawid, band_set)
            self._kle[(band_set, rawid)] = {"pred": r["pred"], "p_defect": r["p_defect"], "ms": r["ms"]}
        return self._kle[(band_set, rawid)]

    def cnn_decision(self, rawid):
        if rawid not in self._cnn:
            r = self.cnn(np.load(self.demo.paths[rawid]), self.demo.keys_for_known(rawid))
            self._cnn[rawid] = {"pred": r["pred"], "p_defect": r["p_defect"], "ms": r["ms"], "pred3": r["pred3"]}
        return self._cnn[rawid]

    # ---- presentation examples ----------------------------------------------------------------
    def examples(self):
        """per tissue (sound / stem / calyx / bruise / russet / rot), from the recorded out-of-fold results (K3 450/750/800 A,
        CNN nir_ss vote): both right, CNN right + Kleynen wrong, Kleynen right + CNN wrong -- the most confident one of each"""
        k3 = json.load(open(os.path.join(K.DL, "k3_kleynen.json"), encoding="utf-8"))["predictions"]
        kp = dict(zip(self.names(), k3[f"{KM.DEFAULT_SET}|A_paper|p_defect"]))
        kd = dict(zip(self.names(), k3[f"{KM.DEFAULT_SET}|A_paper"]))
        oof = self.demo.oof().get("nir_ss")
        if oof is None:
            return []
        cp = {k: float(v[1] + v[2]) for k, v in oof.items()}
        out = []
        for tissue, th in EXAMPLE_TISSUES.items():
            ids = sorted(k for k, v in self.demo.images.items() if v["tissue"] == tissue)
            g = {k: int(self.demo.images[k]["label"] != "Healthy") for k in ids}
            k_ok = {k: kd[k] == g[k] for k in ids}
            c_ok = {k: int(cp[k] > 0.5) == g[k] for k in ids}
            for title, members, conf in (("ทั้งคู่ถูก", [k for k in ids if k_ok[k] and c_ok[k]], cp),
                                         ("CNN ถูก / Kleynen ผิด", [k for k in ids if c_ok[k] and not k_ok[k]], cp),
                                         ("Kleynen ถูก / CNN ผิด", [k for k in ids if k_ok[k] and not c_ok[k]], kp)):
                if members:
                    k = max(members, key=lambda m: (abs(conf[m] - 0.5), m))
                    out.append((f"{th} — {title} — {k}", k))
        return out


# --------------------------------------------------------------------------------------------- visuals
def gray_rgb(band):
    return cv2.cvtColor(band, cv2.COLOR_GRAY2RGB)


def contours(img, mask, colour, thick=1):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, colour, thick)
    return img


def fruit_mask_vis(ir, res):
    out = gray_rgb(ir).astype(np.float32)
    out[res["fruit_mask"] == 0] *= 0.25
    out = out.astype(np.uint8)
    contours(out, res["fruit_mask"], (0, 255, 0), 2)
    return contours(out, res["eroded"], (255, 220, 0), 1)


def defect_map_vis(ir, res, gt=None):
    out = gray_rgb(ir)
    m = res["defect_map"] == 1
    out[m] = (0.35 * out[m] + 0.65 * np.array([230, 30, 30])).astype(np.uint8)
    contours(out, res["eroded"], (255, 220, 0), 1)
    if gt is not None:
        contours(out, gt, (0, 230, 255), 2)
    return out


def load_gt_full(defect_no):
    if defect_no is None:
        return None
    return cv2.imread(os.path.join(K.MASK_DIR, f"SBW_Defect{defect_no:04d}.bmp"), cv2.IMREAD_GRAYSCALE) > 127


def mark(ok):
    return ("#1a7f37", "&#10004; ถูก") if ok else ("#c62828", "&#10008; ผิด")


def group_card(title, pred, p_def, ms, truth_group=None, sub=""):
    colour = "#e08a00" if pred else "#2f80ed"  # green/red are reserved for right/wrong
    conf = (p_def if pred else 1 - p_def) * 100
    right = ""
    if truth_group is not None:
        bg, m = mark(pred == truth_group)
        right = (f"<div style='flex:0 0 150px;display:flex;align-items:center;justify-content:center;border-radius:10px;"
                 f"background:{bg};color:#fff;font-size:30px;font-weight:900'>{m}</div>")
    return (f"<div style='display:flex;gap:10px'><div style='flex:1;padding:10px 14px;border-radius:10px;border:3px solid {colour}'>"
            f"<div style='font-size:15px;opacity:0.8'>{title}</div>"
            f"<div style='font-size:26px;font-weight:800;color:{colour}'>{GROUP_TH[pred]}</div>"
            f"<div style='font-size:15px'>P(มีตำหนิ) = <b style='font-size:22px'>{p_def:.3f}</b> · มั่นใจ {conf:.1f}%</div>"
            f"<div style='font-size:14px;opacity:0.85'>เวลาประมวลผล {ms:.0f} ms{sub}</div></div>{right}</div>")


def features_html(res, model):
    nb = len(model["bands"])
    names = model["band_names"]
    f = res["features"]
    rows = "".join(f"<tr><td style='padding:2px 8px'>{n}</td><td style='padding:2px 8px'>{f[j * 3]:.1f}</td>"
                   f"<td style='padding:2px 8px'>{f[3 * nb + j * 3]:.1f}</td></tr>" for j, n in enumerate(names))
    return (f"<div style='font-size:16px'><div>สัดส่วนพิกเซลตำหนิ: <b style='font-size:24px'>{res['defect_ratio'] * 100:.1f}%</b> ของผล</div>"
            f"<div>จำนวนบริเวณตำหนิ (≥{model['min_region_px']} px): <b style='font-size:24px'>{res['n_regions']}</b></div>"
            f"<div>บริเวณใหญ่สุด: <b>{res['largest_region'] * 100:.1f}%</b> ของผล</div>"
            f"<table style='margin-top:6px;font-size:14px'><tr style='opacity:0.8'><th style='padding:2px 8px'>แบนด์</th>"
            f"<th style='padding:2px 8px'>mean เนื้อดี</th><th style='padding:2px 8px'>mean เนื้อตำหนิ</th></tr>{rows}</table>"
            f"<div style='font-size:13px;opacity:0.8'>LDA ใช้ {len(f)} ฟีเจอร์ (mean/median/std ต่อแบนด์ บนเนื้อดีและเนื้อตำหนิ + 3 ฟีเจอร์ข้างบน)</div></div>")


def header_html(rawid, t, kres, cres, band_set):
    sev = f" · ความรุนแรง: <b>{SEV_TH.get(t['severity'], t['severity'])}</b>" if t["severity"] else ""
    kb, km = mark(kres["pred"] == t["group"])
    cb, cm = mark(cres["pred"] == t["group"])
    badge = lambda bg, txt: f"<span style='background:{bg};color:#fff;padding:3px 10px;border-radius:6px;font-weight:700'>{txt}</span>"
    return (f"<div style='font-size:18px'>{rawid} · ป้ายจริง: <b>{GROUP_TH[t['group']]}</b> ({d.TH[t['label']]}) · ชนิด: <b>{t['tissue']}</b>{sev}"
            f" · ใช้เฉพาะโมเดล fold {t['fold']} ทั้งสองวิธี (ไม่เคยเห็นภาพนี้)</div>"
            f"<div style='font-size:18px;margin-top:6px'>Kleynen ({band_set}) {badge(kb, km)} {kres['ms']:.0f} ms &nbsp;·&nbsp; "
            f"CNN NIR {badge(cb, cm)} {cres['ms']:.0f} ms ({cres['n_models']} โมเดล × TTA 6, {d.DEVICE})</div>"
            + (f"<div style='font-size:14px;opacity:0.85;margin-top:4px'>{STEM_NOTE}</div>" if t["tissue"] in ("Stem", "Calix") else ""))


def step_outputs(cmp, bands, cube, kmodel, kres, cres, cam, truth=None, gt_full=None, gt224=None, head=""):
    """the 10 outputs of the G2 tab"""
    gallery = [(gray_rgb(bands[..., i]), BAND_LABELS[i] + (" (ใช้)" if i in kmodel["bands"] else "")) for i in range(4)]
    tg = None if truth is None else truth["group"]
    kcard = group_card(f"Kleynen 2005 ทำซ้ำ — LDA ({kmodel['band_set']})", kres["pred"], kres["p_defect"], kres["ms"], tg,
                       " (CPU)")
    ccard = group_card("CNN NIR — 2 กลุ่ม: P(ตำหนิภายใน) + P(ตำหนิภายนอก) > 0.5", cres["pred"], cres["p_defect"], cres["ms"], tg,
                       f" ({d.DEVICE})")
    state = {"kleynen": {"pred": kres["pred"], "p_defect": kres["p_defect"], "ms": kres["ms"], "defect_ratio": kres["defect_ratio"],
                         "n_regions": kres["n_regions"], "band_set": kmodel["band_set"], "fold": kmodel["fold"]},
             "cnn": {"pred": cres["pred"], "p_defect": cres["p_defect"], "probs": [float(x) for x in cres["probs"]], "ms": cres["ms"],
                     "folds": sorted({r["fold"] for r in cres["rows"]}), "n_models": cres["n_models"]},
             "truth": None if truth is None else {"group": truth["group"], "tissue": truth["tissue"], "severity": truth["severity"]}}
    return (head, gallery, fruit_mask_vis(bands[..., 3], kres), defect_map_vis(bands[..., 3], kres, gt_full),
            features_html(kres, kmodel), kcard, d.overlay(cube[..., 3], cam, gt224),
            {d.TH[c]: float(cres["probs"][i]) for i, c in enumerate(CLASSES)}, ccard, state)


def step_known(cmp, rawid, band_set=KM.DEFAULT_SET):
    if not rawid:
        raise ValueError("เลือกภาพก่อน")
    t = cmp.truth(rawid)
    bands, kres = cmp.kleynen(rawid, band_set)
    cube = np.load(cmp.demo.paths[rawid])
    keys = cmp.demo.keys_for_known(rawid)
    cres = cmp.cnn(cube, keys)
    cam = cmp.demo.cam(cube, keys, cres["pred3"])
    gt224 = d.load_defect_mask(t["defect_no"]) if t["defect_no"] is not None else None
    return step_outputs(cmp, bands, cube, cmp.kmodel_for(rawid, band_set), kres, cres, cam, t, load_gt_full(t["defect_no"]),
                        gt224, header_html(rawid, t, kres, cres, band_set))


def step_upload(cmp, files, band_set=KM.DEFAULT_SET):
    """image outside the dataset: all-data Kleynen model + all 15 CNN checkpoints; no answer key"""
    bands = np.stack(d.read_upload_bands(files), axis=-1)
    cube = d.load_upload(files)
    kmodel = cmp.kmodel(band_set, None)
    kres = KM.analyze(bands, kmodel)
    keys = cmp.demo.available()
    cres = cmp.cnn(cube, keys)
    cam = cmp.demo.cam(cube, keys, cres["pred3"])
    head = (f"<div style='font-size:18px'>ภาพอัปโหลด {bands.shape[1]}×{bands.shape[0]} · Kleynen: โมเดลที่ fit บนข้อมูลทั้งหมด (280+246) · "
            f"CNN: ensemble {len(keys)} โมเดล · ไม่มีคำตอบจริง</div><div style='font-size:14px;opacity:0.85'>{d.DOMAIN_NOTE} "
            "Kleynen ตั้งค่า (erosion 5 px, closing 15 px) สำหรับภาพ SingleSpectra 430×560 — ภาพความละเอียดอื่นอาจแบ่งผลได้ไม่ดี</div>")
    return step_outputs(cmp, bands, cube, kmodel, kres, cres, cam, head=head)


# --------------------------------------------------------------------------------------------- G3 batch
def scopes(cmp):
    imgs = cmp.demo.images
    out = [(f"ทั้งหมด ({len(imgs)})", "all"),
           (f"ผลดี + ผลมีตำหนิ แบบเปเปอร์ ({sum(v['tissue'] not in ('Stem', 'Calix') for v in imgs.values())})", "paper")]
    out += [(f"fold {f} ({sum(v['fold'] == f for v in imgs.values())})", f"fold:{f}") for f in range(5)]
    out += [(f"ชนิด: {t} ({sum(v['tissue'] == t for v in imgs.values())})", f"tissue:{t}") for t in K.TISSUES
            if any(v["tissue"] == t for v in imgs.values())]
    out += [(f"ความรุนแรง: {SEV_TH[s]} ({sum(v == s for v in cmp.severity.values())})", f"sev:{s}") for s in K.SEV_ORDER]
    return out


def select(cmp, scope):
    imgs = cmp.demo.images
    if scope == "all":
        keep = lambda k: True
    elif scope == "paper":
        keep = lambda k: imgs[k]["tissue"] not in ("Stem", "Calix")
    else:
        kind, val = scope.split(":", 1)
        keep = {"fold": lambda k: imgs[k]["fold"] == int(val), "tissue": lambda k: imgs[k]["tissue"] == val,
                "sev": lambda k: cmp.severity.get(k) == val}[kind]
    ids = [k for k in sorted(imgs) if keep(k)]
    if not ids:
        raise ValueError(f"ชุด {scope} ไม่มีภาพ")
    return ids


def confusion(y, p):
    return [[int(np.sum((y == 0) & (p == 0))), int(np.sum((y == 0) & (p == 1)))],
            [int(np.sum((y == 1) & (p == 0))), int(np.sum((y == 1) & (p == 1)))]]


def run_batch(cmp, scope, band_set=KM.DEFAULT_SET, progress=None):
    from .nir_proof import mcnemar_exact
    ids = select(cmp, scope)
    n = len(ids)
    rows = []
    for k, rawid in enumerate(ids):
        if progress is not None:
            progress((k, n), desc=f"ภาพที่ {k + 1}/{n}")
        t = cmp.truth(rawid)
        kd, cd = cmp.kleynen_decision(rawid, band_set), cmp.cnn_decision(rawid)
        rows.append({"image": rawid, "tissue": t["tissue"], "severity": t["severity"], "fold": t["fold"], "true_group": t["group"],
                     "kleynen_pred": kd["pred"], "kleynen_p_defect": round(kd["p_defect"], 5), "kleynen_ms": round(kd["ms"], 1),
                     "cnn_pred": cd["pred"], "cnn_p_defect": round(cd["p_defect"], 5), "cnn_class3": CLASSES[cd["pred3"]],
                     "cnn_ms": round(cd["ms"], 1), "agree": int(kd["pred"] == cd["pred"])})
    y = np.array([r["true_group"] for r in rows])
    pk, pc = np.array([r["kleynen_pred"] for r in rows]), np.array([r["cnn_pred"] for r in rows])
    tissue = np.array([r["tissue"] for r in rows])
    sev = np.array([r["severity"] for r in rows])
    paper = ~np.isin(tissue, ["Stem", "Calix"])

    def pct(m, p):
        return f"{(p[m] == y[m]).mean() * 100:.1f}% ({int((p[m] == y[m]).sum())}/{int(m.sum())})" if m.any() else "–"
    stats = {"scope": scope, "band_set": band_set, "n": n}
    cards = ""
    for name, p, key in ((f"Kleynen ทำซ้ำ ({band_set})", pk, "kleynen"), ("CNN NIR (nir_ss)", pc, "cnn")):
        missed = int(np.sum((y == 1) & (p == 0)))
        s = {"accuracy": float((p == y).mean()), "acc_526_subset": float((p[paper] == y[paper]).mean()) if paper.any() else None,
             "rejected_pct": float(p.mean() * 100), "missed_defects": missed, "n_defective": int(y.sum()),
             "false_rejects": int(np.sum((y == 0) & (p == 1))), "confusion_2x2": confusion(y, p),
             "mean_ms": float(np.mean([r[f"{key}_ms"] for r in rows]))}
        stats[key] = s
        cards += (f"<div style='flex:1;padding:10px 14px;border:2px solid #888;border-radius:10px'>"
                  f"<div style='font-size:19px;font-weight:700'>{name}</div>"
                  f"<div style='font-size:16px'>ถูก <b style='font-size:26px'>{s['accuracy'] * 100:.1f}%</b> ({int((p == y).sum())}/{n})"
                  + (f" · บนผลดี+ตำหนิ (ไม่รวมขั้ว/ก้น) <b>{s['acc_526_subset'] * 100:.1f}%</b>" if s["acc_526_subset"] is not None else "")
                  + f"</div><div style='font-size:16px'>คัดออก (ทายว่ามีตำหนิ) <b>{s['rejected_pct']:.1f}%</b> · "
                  f"ตำหนิที่หลุด <b>{missed}</b>/{s['n_defective']} · ผลดีที่ถูกคัดทิ้ง <b>{s['false_rejects']}</b></div>"
                  f"<div style='font-size:14px;opacity:0.85'>เวลาเฉลี่ย {s['mean_ms']:.0f} ms/ภาพ</div></div>")
    mc = mcnemar_exact(y, pk, pc)
    stats["mcnemar"] = mc
    summary = (f"<div style='display:flex;gap:12px'>{cards}</div><div style='font-size:16px;margin-top:8px'>{n} ภาพ · "
               f"ตัดสินไม่ตรงกัน {int(np.sum(pk != pc))} ภาพ · Kleynen ถูกคนเดียว {mc['only_a_correct']} · CNN ถูกคนเดียว "
               f"{mc['only_b_correct']} · McNemar p = {mc['p_value']:.1e}</div>")
    if scope == "all":  # must equal K5
        k5 = json.load(open(os.path.join(K.DL, "k5_summary.json"), encoding="utf-8"))["rows"]
        ref_k = k5[kleynen_row(band_set)]["paper_subset_280_246"]["accuracy"] if kleynen_row(band_set) in k5 else None
        ref_c = k5["CNN merged NIR (nir_ss)"]["paper_subset_280_246"]["accuracy"]
        match = (ref_k is None or abs(ref_k - stats["kleynen"]["acc_526_subset"]) < 1e-9) and abs(ref_c - stats["cnn"]["acc_526_subset"]) < 1e-9
        stats["k5_check"] = {"kleynen_k5": ref_k, "cnn_k5": ref_c, "match": bool(match)}
        summary += (f"<div style='font-size:16px;margin-top:4px'>เทียบ K5 (526 ลูก): Kleynen {ref_k * 100:.1f}% · CNN {ref_c * 100:.1f}% → "
                    + ("<b style='color:#1a7f37'>&#10004; ตรงกัน</b>" if match else "<b style='color:#c62828'>&#10008; ไม่ตรง</b>") + "</div>")
    cm_k = [[f"จริง: {GROUP_TH[i]}"] + stats["kleynen"]["confusion_2x2"][i] for i in range(2)]
    cm_c = [[f"จริง: {GROUP_TH[i]}"] + stats["cnn"]["confusion_2x2"][i] for i in range(2)]
    groups = [("ผลดี (Healthy)", tissue == "Healthy"), ("ขั้ว + ก้น", np.isin(tissue, ["Stem", "Calix"])), ("ผลมีตำหนิทั้งหมด", y == 1)]
    groups += [(SEV_TH[s], sev == s) for s in K.SEV_ORDER]
    table = [[g, int(m.sum()), pct(m, pk), pct(m, pc)] for g, m in groups if m.any()]
    stats["per_group"] = {g: {"n": int(m.sum()), "kleynen": float((pk[m] == y[m]).mean()), "cnn": float((pc[m] == y[m]).mean())}
                          for g, m in groups if m.any()}
    # disagreements: defective first, then sound, then stem/calyx
    order = {1: 0, 0: 1}
    dis = sorted([r for r in rows if not r["agree"]],
                 key=lambda r: (order[r["true_group"]] + (1 if r["tissue"] in ("Stem", "Calix") else 0), r["image"]))
    gallery, dis_ids = [], []
    for r in dis[:GALLERY_MAX]:
        cube = np.load(cmp.demo.paths[r["image"]])
        ok = lambda p: "✔" if p == r["true_group"] else "✘"
        gallery.append((cv2.resize(gray_rgb(cube[..., 3]), (180, 180)),
                        f"{r['image']} {r['tissue']}{' ' + r['severity'] if r['severity'] else ''} · Kleynen {ok(r['kleynen_pred'])} / CNN {ok(r['cnn_pred'])}"))
        dis_ids.append(r["image"])
    stats["n_disagree"] = len(dis)
    note = (f"ภาพที่สองวิธีตัดสินไม่ตรงกัน {len(dis)} ภาพ" + (f" (แสดง {GALLERY_MAX} ภาพแรก ทั้งหมดอยู่ใน CSV)" if len(dis) > GALLERY_MAX else "")
            + " — กดภาพเพื่อดูทีละขั้นในแท็บ \"ทีละขั้น: Kleynen vs CNN\"")
    path = os.path.join(d._CACHE, f"g3_{re.sub(r'[^A-Za-z0-9_]+', '_', scope)}_{KM.GUI_SETS[band_set]}.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return summary, cm_k, cm_c, table, gallery, note, dis_ids, path, stats


# --------------------------------------------------------------------------------------------- UI
def build_tabs(demo, tabs, guard):
    """adds the G2 and G3 tabs inside an open gr.Tabs() context"""
    import gradio as gr

    cmp = Compare(demo)
    ids = sorted(demo.images)
    choices = [(f"{i} — {demo.images[i]['tissue']}" + (f" · {cmp.severity[i]}" if i in cmp.severity else "") +
                f" ({GROUP_TH[int(demo.images[i]['label'] != 'Healthy')]})", i) for i in ids]
    band_choices = list(KM.GUI_SETS)
    examples = cmp.examples()
    cm_headers = ["", "ทาย: ดี", "ทาย: มีตำหนิ"]

    with gr.Tab("ทีละขั้น: Kleynen vs CNN", id="g2"):
        gr.Markdown("แอปเปิลลูกเดียว ดูทุกขั้นของ **Kleynen 2005 (ทำซ้ำ, K3)** เทียบกับ **CNN NIR (nir_ss)** แบบ 2 กลุ่ม ดี / มีตำหนิ · "
                    "ภาพในชุดข้อมูลใช้เฉพาะโมเดลของ fold ที่ภาพนั้นเป็น test ทั้งสองวิธี · เวลา = ประมวลผลอย่างเดียว "
                    "(ไม่รวมอ่านไฟล์/โหลดโมเดล/วาดภาพ) · ตัวเลขประเมินผลอยู่ที่ K3/K5 (โมเดลในแท็บนี้ใช้แสดงผล)")
        with gr.Row():
            g2_ex = gr.Dropdown([("— เลือกตัวอย่าง —", "")] + examples, value="", scale=4, interactive=bool(examples),
                                label="ตัวอย่างพรีเซนต์ (ดี / ขั้ว / ก้น / รอยช้ำ / russet / เน่า — คัดจากผล out-of-fold)")
            g2_bs = gr.Dropdown(band_choices, value=KM.DEFAULT_SET, label="ชุดแบนด์ Kleynen", scale=2)
        with gr.Row():
            g2_dd = gr.Dropdown(choices, label="เลือกภาพ (RawA id — ชนิด · ความรุนแรง (ป้ายจริง))", filterable=True, scale=5)
            g2_go = gr.Button("วิเคราะห์", variant="primary", scale=1)
        g2_head = gr.HTML()
        gr.Markdown("### Kleynen 2005 (ทำซ้ำ): 4 แบนด์ → ตัดพื้นหลัง (threshold 800 nm) → Bayes ทีละพิกเซล → ฟีเจอร์ → LDA")
        g2_bands = gr.Gallery(label="① ภาพ 4 แบนด์ (SingleSpectra)", columns=4, height=200, allow_preview=False)
        with gr.Row():
            g2_mask = gr.Image(label="② mask ผล: threshold 800 nm + fill holes + closing (เขียว) · erosion 5 px (เหลือง)", interactive=False)
            g2_dmap = gr.Image(label="③ พิกเซลตำหนิจาก Bayes (แดง) · ขอบตำหนิเฉลย SBW_Defect (ฟ้า)", interactive=False)
        with gr.Row():
            with gr.Column():
                gr.Markdown("**④ ฟีเจอร์หลัก**")
                g2_feat = gr.HTML()
            with gr.Column():
                gr.Markdown("**⑤ ผล LDA**")
                g2_kcard = gr.HTML()
        gr.Markdown("### CNN NIR (nir_ss, สูตรสุดท้าย): Grad-CAM → ความน่าจะเป็น 3 คลาส → 2 กลุ่ม")
        with gr.Row():
            g2_cam = gr.Image(label="① Grad-CAM (14×14) ซ้อนแบนด์ IR · ขอบตำหนิเฉลย (เขียว)", interactive=False)
            with gr.Column():
                g2_lab = gr.Label(label="② ความน่าจะเป็น 3 คลาส (โหวต 3 seed + TTA)", num_top_classes=3)
                g2_ccard = gr.HTML(label="③ ผล 2 กลุ่ม")
        g2_state = gr.JSON(visible=False)
        g2_outs = [g2_head, g2_bands, g2_mask, g2_dmap, g2_feat, g2_kcard, g2_cam, g2_lab, g2_ccard, g2_state]
        with gr.Accordion("ภาพนอกชุดข้อมูล (อัปโหลด) — Kleynen โมเดลที่ fit บนข้อมูลทั้งหมด + CNN 15 โมเดล", open=False):
            g2_up = gr.File(file_count="multiple", label=".npy (H×W×4: BL,GR,RE,IR) หรือ 4 ไฟล์ *_BL *_GR *_RE *_IR")
            g2_up_go = gr.Button("วิเคราะห์ภาพอัปโหลด")

        def step_if_picked(rawid, bs):
            return step_known(cmp, rawid, bs) if rawid else [gr.skip()] * len(g2_outs)
        g2_go.click(guard(lambda r, b: step_known(cmp, r, b)), [g2_dd, g2_bs], g2_outs, api_name="kleynen_vs_cnn")
        g2_dd.change(guard(step_if_picked), [g2_dd, g2_bs], g2_outs, api_name=False)
        g2_bs.change(guard(step_if_picked), [g2_dd, g2_bs], g2_outs, api_name=False)
        g2_ex.change(lambda r: r or gr.skip(), g2_ex, g2_dd, api_name=False)
        g2_up_go.click(guard(lambda f, b: step_upload(cmp, f, b)), [g2_up, g2_bs], g2_outs, api_name="kleynen_vs_cnn_upload")

    with gr.Tab("คัดทีละชุด", id="g3"):
        gr.Markdown("รันทั้ง Kleynen และ CNN NIR แบบ out-of-fold กับชุดภาพที่เลือก (ครั้งแรกทั้ง 819 ภาพใช้ ~3–4 นาที — CNN 3 โมเดล × TTA 6 ต่อภาพ — แล้วจำผลไว้ ครั้งต่อไปเร็ว) · "
                    "ชุด \"ทั้งหมด\" ต้องได้ตัวเลขตรงกับ K5 (526 ลูก: Kleynen 82.3% / CNN 93.2%)")
        with gr.Row():
            g3_scope = gr.Dropdown(scopes(cmp), value="all", label="ชุดภาพ", scale=3)
            g3_bs = gr.Dropdown(band_choices, value=KM.DEFAULT_SET, label="ชุดแบนด์ Kleynen", scale=2)
            g3_go = gr.Button("เริ่มคัด", variant="primary", scale=1)
        g3_sum = gr.HTML()
        with gr.Row():
            g3_cmk = gr.Dataframe(headers=cm_headers, label="Confusion 2×2 — Kleynen", interactive=False)
            g3_cmc = gr.Dataframe(headers=cm_headers, label="Confusion 2×2 — CNN NIR", interactive=False)
        g3_tbl = gr.Dataframe(headers=["กลุ่ม", "n", "Kleynen ถูก", "CNN ถูก"], label="แยกกลุ่ม / ความรุนแรง", interactive=False)
        g3_note = gr.Markdown()
        g3_gal = gr.Gallery(label="ภาพที่สองวิธีตัดสินไม่ตรงกัน", columns=6, height=420, allow_preview=False)
        g3_ids = gr.State([])
        g3_csv = gr.File(label="export CSV (ผลรายภาพของชุดนี้)", interactive=False)
        g3_stats = gr.JSON(visible=False)
        g3_go.click(guard(lambda s, b, progress=gr.Progress(): run_batch(cmp, s, b, progress)), [g3_scope, g3_bs],
                    [g3_sum, g3_cmk, g3_cmc, g3_tbl, g3_gal, g3_note, g3_ids, g3_csv, g3_stats], api_name="batch_compare")

        def open_in_g2(ids_, evt: gr.SelectData):
            return ids_[evt.index], gr.Tabs(selected="g2")
        # picking a gallery image sets the G2 dropdown, whose change event runs the analysis
        g3_gal.select(open_in_g2, g3_ids, [g2_dd, tabs], api_name=False)
    return cmp
