# Apple Quality Classification using Multispectral Imaging (Vis+NIR) and CNN-based Deep Learning

ปริญญานิพนธ์ มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ (KMUTNB)
ผู้จัดทำ: นายชินวัณ ศรีประสงค์, นางสาวธัญลักษณ์ นามโส · อาจารย์ที่ปรึกษา: รศ.ดร.ชาญวิทย์ ตั้งสิริวรกุล

> **ไม่มีข้อมูลภาพใน repo นี้** — ดูหัวข้อ [ข้อมูล](#ข้อมูล-ไม่อยู่ใน-repo) ด้านล่าง

## โปรเจกต์นี้คืออะไร

งานนี้ตอบคำถามเดียว: **ภาพถ่ายแสงอินฟราเรดใกล้ (NIR: 750 + 800 nm) แยกแอปเปิลดีกับแอปเปิลมีตำหนิได้ดีกว่าภาพแสงที่ตามองเห็น (Visible: 450 + 500 nm) หรือไม่ และเพราะอะไร**

- ใช้ภาพแอปเปิล 819 ภาพ 4 แบนด์ จากฐานข้อมูล CAPA (ULG, Gembloux Agro-Bio Tech)
- แบ่ง 3 คลาส: ปกติ (573), ตำหนิภายใน (102: รอยช้ำ 55, เนื้อเสีย 24, เน่า 23), ตำหนิภายนอก (144)
- โมเดล EfficientNet-B0 (pretrained ImageNet) แบบเดียวกันทุกการทดลอง ต่างกันแค่แบนด์ที่ป้อนเข้า
- ทดสอบแบบ stratified 5-fold (ทุกภาพถูกทดสอบโดยโมเดลที่ไม่เคยเห็นภาพนั้น) × 3 seed แล้วเทียบแบบจับคู่ (seed+fold)
- เกณฑ์ "ต่างจริง": 95% bootstrap CI ไม่คร่อม 0 **และ** ≥ 12/15 คู่ไปทางเดียวกัน; ตัววัดหลัก macro-F1 คู่กับ accuracy

## ผลหลัก

ภาพ SingleSpectra ที่ปรับแสง (flat-field) และซ้อนแบนด์ตรงกันแล้ว, seeds 3–5 (จาก `outputs/logs/nir_proof_report.txt` หัวข้อ N5)

| แบนด์ | macro-F1 (โมเดลเดี่ยว, เฉลี่ย 15 รัน) | accuracy | macro-F1 (TTA + โหวต 3 seed) | accuracy |
|---|---|---|---|---|
| **NIR (RE 750 + IR 800)** | **0.858** | **91.0%** | **0.885** | **92.6%** |
| ทั้ง 4 แบนด์ | 0.842 | 90.2% | 0.866 | 92.1% |
| Visible (BL 450 + GR 500) | 0.767 | 85.3% | 0.797 | 87.5% |

- **NIR − Visible = +0.091 macro-F1**, CI [+0.072, +0.109], 15/15 คู่ → ต่างจริง; McNemar (819 ภาพ) p = 3.4×10⁻⁵
- **NIR − 4 แบนด์ = +0.016**, CI [−0.005, +0.038], 9/15 คู่ → ไม่ต่างเกินกว่า noise ของการเทรน (Visible ไม่ได้เพิ่มข้อมูลเมื่อมี NIR แล้ว)
- ส่วนต่างใหญ่ที่สุดอยู่ที่ **รอยช้ำ**: NIR จับได้ 90.9% เทียบกับ Visible 63.6% (55 ลูก)
- ภาพดิบ (`Processed_Dataset`) แต่ละแบนด์เหลื่อมกัน 14–22 px ที่ 224×224 ทำให้โมเดล 4 แบนด์บนภาพดิบดูแย่เกินจริง (83.0%); บนภาพที่ซ้อนตรงแล้วเท่ากับ NIR — สอดคล้องกับ Kleynen et al. (2005) ที่ระบุว่าภาพย่อย 4 แบนด์ต้องจัดแนวก่อนใช้
- หลักฐานประกอบ (รายละเอียดใน `nir_proof_report.txt`): วัด contrast ตำหนิจาก mask โดยไม่ใช้ AI (N1), ผลรายชนิดตำหนิ (N2), โมเดลแบบดั้งเดิม SVM/RF/LightGBM (N3), late fusion (N4), error analysis (N8)

**ข้อจำกัด:** "Visible" ในข้อมูลนี้มีแค่ 450/500 nm (ไม่มีแบนด์สีแดง) จึงสรุปได้ว่า NIR 750/800 nm ดีกว่า visible 450/500 nm ไม่ใช่ดีกว่ากล้อง RGB · ใช้ข้อมูลชุดเดียว ไม่มีชุดทดสอบภายนอก · คลาสเน่าทายถูกเพียง 56.5% (23 ภาพ)

## ข้อมูล (ไม่อยู่ใน repo)

ฐานข้อมูล **CAPA Apple Quality Grading Image Database เป็นลิขสิทธิ์ของ ULG (Gembloux Agro-Bio Tech), Belgium** และห้ามใช้โดยไม่ได้รับอนุญาตจาก ULG (รายละเอียดและช่องทางติดต่อขออนุญาตอยู่ใน `ReadMe.txt`) repo นี้จึง **ไม่มี** ภาพต้นฉบับ, ภาพที่แปลงแล้ว (`Processed_Dataset*`), mask, ภาพ Grad-CAM หรือภาพใดๆ ที่มีพิกเซลจากข้อมูลนี้ และไม่มีไฟล์โมเดล `.pt` — มีเฉพาะโค้ดและผลเชิงตัวเลข

งานที่ใช้ข้อมูลนี้ต้องระบุว่าข้อมูลได้มาจาก ULG (Gembloux Agro-Bio Tech), Belgium และอ้างอิง:

> Kleynen, O., Leemans, V., & Destain, M.-F. (2005). Development of a multi-spectral vision system for the detection of defects on apples. *Journal of Food Engineering, 69*(1), 41–49. https://doi.org/10.1016/j.jfoodeng.2004.07.008

## ควรอ่านไฟล์ไหนก่อน

1. `README.md` (ไฟล์นี้)
2. `outputs/logs/nir_proof_report.txt` — ตารางผลทั้งหมดที่ใช้ในเล่ม (N1–N5, N8)
3. `PROJECT_PLAN.md` หัวข้อ "พิสูจน์ NIR ดีกว่า Visible" — เหตุผล การตัดสินใจ และเกณฑ์ที่ล็อกไว้ก่อนรันแต่ละรอบ (ภาษาไทย; ส่วนต้นของไฟล์เป็นประวัติการทดลองตามลำดับเวลา บางผลช่วงแรกถูกหักล้างภายหลัง — ยึดหัวข้อท้ายๆ เป็นหลัก)
4. โค้ดหลัก: `src/train_dl_kfold.py` (pipeline, fold), `src/run_confirm.py` + `src/run_n5.py` (การทดลองสุดท้าย), `src/nir_proof.py` (สถิติ)

## โครงสร้างโฟลเดอร์

```
├── README.md, ReadMe.txt (เงื่อนไขการใช้ CAPA), PROJECT_PLAN.md, requirements.txt
├── preprocess_pipeline.py   # ภาพดิบ CAPA → Processed_Dataset (.npy 4 แบนด์, 224x224)
├── run_*.bat                # คำสั่งรันแต่ละรอบ (Windows)
├── src/
│   ├── config.py            # path โปรเจกต์ (PROJECT_ROOT), คลาส, แบนด์
│   ├── preprocess_ss.py     # ภาพ SingleSpectra (flat-field, ซ้อนตรง) → Processed_Dataset_SS
│   ├── dataset.py, models.py, repro.py   # data loader, EfficientNet-B0 + ปรับชั้นแรก, seed/deterministic
│   ├── train_dl_kfold.py    # 5-fold training pipeline
│   ├── run_multiseed.py, run_round2.py, run_confirm.py, run_fusion_models.py, run_n5.py   # การทดลองแต่ละรอบ
│   ├── train_classical.py, features.py   # SVM / Random Forest / LightGBM
│   ├── band_contrast.py, nir_proof.py, error_analysis.py, *_report.py   # วิเคราะห์และเขียนรายงาน
│   ├── gradcam_mask_eval.py, explain.py  # Grad-CAM
│   └── demo_app.py, infer.py, demo_selftest.py   # demo (Gradio, local only)
└── outputs/
    ├── dl/*.json            # ผลทุกรัน (prediction, probability ต่อภาพ), summary ทุกรอบ
    ├── classical/           # ผล classical ML + SHAP
    ├── logs/nir_proof_report.txt
    └── figures/band_contrast.png
```

## วิธีรัน

ทดสอบบน Windows + NVIDIA GPU (CUDA 12.1), Python venv ที่ `.venv`

1. **ขอข้อมูล CAPA จาก ULG** แล้ววางใน `images/` (RawImages, SingleSpectraImages) และ `segmentations/`
2. แก้ `PROJECT_ROOT` ใน `src/config.py` และ path ใน `preprocess_pipeline.py` / `run_*.bat` ให้ตรงเครื่อง (ตอนนี้ตั้งเป็น `D:\nir+rgb_project\...`)
3. ติดตั้ง
   ```
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
4. เตรียมข้อมูล
   ```
   .venv\Scripts\python preprocess_pipeline.py
   .venv\Scripts\python -m src.preprocess_ss
   ```
5. รันการทดลอง (แต่ละไฟล์ใช้หลายชั่วโมง, หยุดด้วย Ctrl+C แล้วรันซ้ำเพื่อทำต่อได้)
   - `run_overnight.bat` — multi-seed ทุกชุดแบนด์ (seeds 0–2) + classical ML
   - `run_round2.bat` — เทคนิคเพิ่มความแม่นยำ (รอบสำรวจ)
   - `run_confirm.bat` — ยืนยันสูตรสุดท้าย (nir_ss, seeds 3–5) + checkpoint
   - `run_n5.bat` — visible_ss / full_ss บนภาพที่ซ้อนตรง (seeds 3–5) + checkpoint
6. รายงาน: `.venv\Scripts\python -m src.nir_proof_report` → `outputs/logs/nir_proof_report.txt`
7. Demo (เปิดได้เฉพาะเครื่องตัวเอง `127.0.0.1:7860` เพราะแสดงภาพ CAPA): `run_demo.bat`

ทุกการเทรนตั้ง seed และ deterministic kernel (`src/repro.py`, `CUBLAS_WORKSPACE_CONFIG`) รันซ้ำบนเครื่องเดิมได้ผลเท่าเดิมทุกบิต

## อ้างอิงที่ใช้ในการอภิปรายผล

- Kleynen, O., Leemans, V., & Destain, M.-F. (2003). Selection of the most efficient wavelength bands for 'Jonagold' apple sorting. *Postharvest Biology and Technology, 30*(3), 221–232. https://doi.org/10.1016/S0925-5214(03)00112-1
- Kleynen, O., Leemans, V., & Destain, M.-F. (2005). ดูด้านบน
