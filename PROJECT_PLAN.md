# Apple Quality Classification using Multispectral Imaging and CNN-based Deep Learning

โปรเจกปริญญานิพนธ์ — สรุปแผนงานทั้งหมดจนถึงอัพเดตล่าสุด (ใช้ไฟล์นี้เป็น context ให้ Claude Code ทำงานต่อ)

## ข้อมูลทั่วไป

- **ผู้ทำ**: นายชินวัณ ศรีประสงค์ และ นางสาวธัญลักษณ์ นามโส
- **อาจารย์ที่ปรึกษา**: รองศาสตราจารย์ ดร.ชาญวิทย์ ตั้งสิริวรกุล
- **สถานะการคุยกับอาจารย์**: ยังไม่ได้คุยเรื่องแผนที่ขยาย (SE block, classical ML baseline, SHAP, Grad-CAM) — ต้องทำให้โปรเจกเป็นรูปเป็นร่างก่อนเพื่อเอาไปคุยกับอาจารย์
- **Dataset**: CAPA Apple Quality Grading Image Database
- **Model หลัก**: EfficientNet-B0
- **อ้างอิงหลัก**: Jiang et al. 2019 (ต่างกันที่ขนาด dataset, สถาปัตยกรรม, และรูปแบบการศึกษา)

## เป้าหมายหลัก

พิสูจน์ว่าข้อมูล multispectral (รวม Visible + NIR) ให้ผลดีกว่าการใช้ channel เดียว ในการตรวจจับตำหนิทั้งแบบผิวนอก (surface defect) และภายใน (internal defect) ของแอปเปิ้ล

## คำอธิบายแนวคิดแบบละเอียด (สำหรับความเข้าใจเชิงลึก)

ส่วนนี้อธิบาย "ทำไม" เบื้องหลังทุกจุดตัดสินใจของโปรเจก ไม่ใช่แค่ "ทำอะไร" — เขียนแบบง่ายๆ ไม่ทางการ เพื่อให้เข้าใจภาพรวมได้เร็ว

### ปัญหาที่โปรเจกนี้แก้

ตำหนิแอปเปิ้ลมี 2 แบบ: **surface defect** (เห็นด้วยตา เช่น รอยขูด รอยแผลบนผิว) และ **internal defect** (มองไม่เห็นด้วยตา เช่น รอยช้ำ เน่าข้างในเนื้อ ผิวนอกดูปกติเป๊ะ) ถ้าใช้กล้องธรรมดา (visible light) อย่างเดียว จะจับ internal defect ไม่ได้เลย เพราะมันซ่อนอยู่ข้างใน

### ทำไมแสง NIR ช่วยได้

แสง visible จะสะท้อนกลับที่ผิววัตถุเป็นหลัก ส่วนแสง NIR (Near-Infrared) มีคุณสมบัติทะลุเข้าไปในเนื้อเยื่อได้ลึกกว่าก่อนสะท้อนกลับ ถ้าเนื้อข้างในมีรอยช้ำหรือเน่า แสง NIR จะสะท้อนกลับมาผิดปกติ (มักดูจางลง เพราะแสงถูกดูดกลืนในเนื้อเยื่อที่เสียหาย) ในขณะที่ภาพ visible ธรรมดาจะดูปกติดี — นี่คือเหตุผลที่ต้องใช้ multispectral imaging (ถ่ายหลายช่วงความยาวคลื่นพร้อมกัน)

### ทำไมต้องใช้ CNN ไม่ใช้กฎเกณฑ์ธรรมดา

ตำหนิแต่ละแบบมีลักษณะซับซ้อน (รูปทรง ขนาด ความกระจายตัว texture) กฎเกณฑ์ตายตัว (เช่น "ถ้าค่าสีต่ำกว่า X ให้ถือว่าเสีย") ดักไม่ได้หมด CNN (Convolutional Neural Network) เก่งเรื่องเรียนรู้ pattern จากภาพเองโดยไม่ต้องบอกตรงๆ ว่าต้องดูอะไร เลือก **EfficientNet-B0** เพราะให้ผลดีและใช้ทรัพยากรน้อยกว่า CNN ตัวใหญ่ เหมาะกับข้อมูลจำนวนน้อย (819 ภาพ ถือว่าน้อยมากในโลก deep learning)

### แผนเดิม vs ทำไมอาจารย์บอกว่าแคบไป

แผนเดิม: เอา EfficientNet-B0 ตัวเดียวกันมาเทรน 3 รอบด้วย input ต่างกัน (visible only / NIR only / ครบ 4 channel) แล้วเทียบผล — ปัญหาคือเป็นแค่การทดลองซ้ำๆ กันด้วยข้อมูลต่างกัน ไม่มีความลึกทางเทคนิคพอสำหรับปริญญานิพนธ์ ไม่มีอะไรให้วิเคราะห์ลึกๆ

### SE Block คือตัวสำคัญที่สุด — แก้ปัญหาอะไร

วัตถุประสงค์ธีสิสข้อ 3.1 เขียนไว้ว่าจะ "วิเคราะห์ว่า channel ไหนสำคัญแค่ไหน" (Blue/Green/Red Edge/IR สำคัญแค่ไหนแบบรายตัว) แต่แผนเดิม (3 รอบ) เทียบได้แค่ระดับกลุ่ม (visible ทั้งกลุ่ม vs NIR ทั้งกลุ่ม) ไม่สามารถบอกได้ว่าใน 4 channel ตัวไหนสำคัญที่สุดแบบรายตัว

SE Block คือชิ้นส่วนเล็กๆ ที่แปะเข้าโมเดล CNN ได้เลย ทำงาน 3 ขั้น: **Squeeze** (บีบข้อมูลแต่ละ channel ให้เหลือค่าตัวแทนตัวเดียว) → **Excitation** (คำนวณน้ำหนักความสำคัญของแต่ละ channel) → **Reweight** (คูณน้ำหนักนั้นกลับเข้าข้อมูลเดิม) พอเทรนเสร็จ ดึงค่าน้ำหนักที่ SE Block เรียนรู้มาดูได้เลยว่า channel ไหนสำคัญสุด — ตอบ objective 3.1 ตรงเป้า ไม่ใช่การเพิ่มงานให้ดูเยอะ แต่อุดช่องโหว่ของแผนเดิมจริงๆ

### Classical ML baseline — แก้ปัญหาอะไร

คำถามที่ต้องตอบได้: "ทำไมต้องใช้ deep learning ที่ซับซ้อน ทั้งที่ข้อมูลมีแค่ 819 ภาพ ทำไมไม่ใช้ ML แบบดั้งเดิมที่เบากว่า" วิธีตอบคือต้องมีโมเดลดั้งเดิมมาเทียบจริง: สกัดฟีเจอร์ด้วยมือ (color statistics, GLCM texture, band ratio) แล้วเทรน SVM/RF/LightGBM เทียบผลกับ EfficientNet-B0 ว่าใครแม่นกว่า — มีตัวเลขจริงมาตอบ ไม่ใช่เดา

### Grad-CAM — แก้ปัญหาอะไร

ต่อให้โมเดลทำนายถูก เราจะรู้ได้ยังไงว่ามันมองเห็นตำหนิจริง ไม่ใช่บังเอิญเดาถูกจากอะไรที่ไม่เกี่ยวข้อง (เช่น มองพื้นหลังผิดๆ) Grad-CAM สร้าง heatmap ทับภาพ บอกว่าโมเดลโฟกัสตรงไหนตอนตัดสินใจ ถ้าโฟกัสตรงตำแหน่งตำหนิจริง คือหลักฐานว่าโมเดล "เข้าใจ" ปัญหาจริง

### Statistical significance testing — แก้ปัญหาอะไร

ถ้าโมเดล A ได้ 85% โมเดล B ได้ 87% ต่างกัน 2% นี่ดีกว่าจริงหรือบังเอิญ? ถ้าเทรนใหม่ตัวเลขอาจสลับกันได้ Bootstrap Confidence Interval หรือ McNemar's Test บอกได้ว่าความต่างนั้นมีนัยสำคัญทางสถิติจริง หรือแค่ความบังเอิญจากการสุ่มแบ่งข้อมูล

### SHAP — มาจากไหน แก้ปัญหาอะไร

มาจาก paper เปรียบเทียบ "Explainable AI for Fruit Quality Classification" ที่เขาใช้ SHAP อธิบายโมเดล ensemble (คู่กับ Grad-CAM อธิบาย CNN) เอามาใส่ใน classical ML baseline ของเรา ทำให้มีโครงสร้างอธิบายผลครบคู่กัน: SHAP อธิบายโมเดล ML แบบดั้งเดิม, Grad-CAM อธิบายโมเดล deep learning — เหมือน paper ต้นแบบ

### ทำไมแบ่งข้อมูล 70:15:15 และทำไมใช้ Macro F1

**Split**: Train (70%) ให้โมเดลเรียนรู้ / Validation (15%) เช็คระหว่างเทรนว่า overfitting หรือยัง จะได้หยุดเทรนตอนดีที่สุด / Test (15%) เก็บไว้วัดผลจริงท้ายสุด ไม่ให้โมเดลเห็นก่อน — ถ้าใช้ test set ปรับจูนโมเดลด้วย จะไม่ยุติธรรม (เหมือนดูเฉลยก่อนสอบ)

**Macro F1**: ข้อมูล imbalance (Healthy 573 ภาพ มากกว่า Internal Defect 102 ภาพ เยอะมาก) ถ้าใช้ accuracy ธรรมดา โมเดลอาจเดาว่า Healthy ตลอดแล้วได้ accuracy สูงทั้งที่แยกตำหนิไม่ได้เลย Macro F1 คำนวณ F1 ของแต่ละ class แยกกันก่อนแล้วเฉลี่ยแบบให้น้ำหนักเท่ากันทุก class ทำให้ถ้าแยก Internal Defect ไม่เก่ง ตัวเลขจะแย่ลงชัดเจน ซ่อนไม่ได้

## ที่มาของการขยายสโคป

แผนเดิมคือเทรนโมเดลเดิมซ้ำ 3 รอบด้วยชุด channel ต่างกัน (Visible only / NIR only / ครบ 4 channel) แต่อาจารย์ที่ปรึกษาให้ feedback ว่าสโคปแคบไป จึงขยายเป็นแผนที่ครอบคลุมกว่านี้ ประกอบด้วย 4 องค์ประกอบเพิ่มเติม:

1. **Classical ML baseline** — SVM, Random Forest, LightGBM ด้วย hand-crafted features (color statistics, GLCM texture, band ratios) เพื่อพิสูจน์ความจำเป็นของ deep learning บน dataset ขนาดเล็ก (~820 samples)
2. **Grad-CAM visualization** — แสดงว่าโมเดลโฟกัสบริเวณภาพที่เกี่ยวข้องจริง
3. **Statistical significance testing** — Bootstrap CI หรือ McNemar's Test เพื่อยืนยันว่าความต่างของ performance ระหว่าง model variants ไม่ใช่เรื่องบังเอิญ
4. **Squeeze-and-Excitation (SE) block** — เพิ่มเข้าไปใน EfficientNet-B0 เป็นจุดที่ academically justified ที่สุด เพราะตอบ objective 3.1 ของธีสิสตรง (วิเคราะห์ contribution ของแต่ละ channel/wavelength) ผ่านการเรียนรู้และ output ค่าน้ำหนักความสำคัญของแต่ละ channel

**เหตุผลที่ต้องมี SE block**: การออกแบบ A/B/C แบบเดิม (เทียบกลุ่ม Visible vs NIR) ไม่สามารถตอบคำถามเรื่อง contribution ของ wavelength band แต่ละตัวแบบรายตัวได้ — นี่คือช่องว่างที่ทำให้ต้องเพิ่ม SE block เข้ามา

## รายละเอียด Dataset

- รวม 819 ไฟล์: Healthy (573), Internal Defect (102: Bruise/Rot/Flesh_damage), Surface Defect (144: Russet/Scald/Hail_damage variants/Frost_damage/Limb_rub/Other)
- Layout ภาพดิบ: IR (บนซ้าย), GR (บนขวา), BL (ล่างซ้าย), RE (ล่างขวา) ใน frame 1024×1280 สี่ quadrant
- Label ใช้ PyTorch folder structure ล้วนๆ ไม่ต้องมีไฟล์ label แยก
- มีสคริปต์ `lookup.py` map ชื่อไฟล์ CAPA กับประเภทตำหนิและ class ได้แล้ว

### สถานะ pipeline (อัพเดตล่าสุดจาก Claude Code)

- เดิม `rawfile_map` ครบ 819 รายการ (คัดลอกจาก `lookup.py`) แทนที่เวอร์ชันตัดทอนเดิม
- รัน pipeline สำเร็จ → ได้ `Processed_Dataset` เป็นไฟล์ `.npy` ครบ 819 ไฟล์ (แต่ละไฟล์คือ tensor 224×224×4, channel = BL/GR/RE/IR)
- ผลการแบ่ง stratified split (70/15/15):

| Split | Healthy | Internal_Defect | Surface_Defect | รวม |
|-------|---------|------------------|-----------------|-----|
| train | 401 | 71 | 101 | 573 |
| val | 86 | 15 | 22 | 123 |
| test | 86 | 16 | 21 | 123 |

### ข้อสังเกตที่ต้องรู้ก่อนเทรนต่อ

- **Class imbalance ชัดเจน** — Healthy มากกว่า Internal_Defect ถึง ~5-6 เท่า ควรใช้ class weighting หรือ weighted sampler ตอนเทรน ไม่งั้นโมเดลจะ bias ไปทาง Healthy
- ไฟล์ `notes.txt` ที่เจอใน `SingleSpectraImages` มีรายละเอียดเสริมเกี่ยวกับ Bruise/Flesh_damage บางเบอร์ (ระยะเวลาที่ทำให้เกิดรอยช้ำ) เผื่อจะใช้ทำ severity label ในอนาคต แต่ยังไม่ได้เอามาใช้ในสคริปต์นี้
- ข้อมูลนี้มีเงื่อนไขลิขสิทธิ์จาก ULG Gembloux Agro-Bio Tech (ตาม ReadMe.txt) — ถ้าจะเผยแพร่ผลงานต้องอ้างอิงตามที่ระบุไว้

## กลยุทธ์การเทรน

- Monitor Train Loss + Val Loss + Val Accuracy ทุก epoch
- Early Stopping (patience=10)
- เซฟโมเดลที่ Val Loss ดีที่สุด
- Split แบบ 70:15:15 (train/val/test) — ทำสำเร็จแล้ว (ดูตารางด้านบน)
- ใช้ **Macro-Average F1** แทน Overall Accuracy เพราะข้อมูลไม่สมดุล (imbalanced) — เกณฑ์ที่ยอมรับได้คือ 70%+ สำหรับ Proof of Concept

## Key learnings ที่ใช้ตอบ defense

- NIR จะดูจางกว่าตอนตรวจจับรอยช้ำ เพราะแสงทะลุเข้าไปในเนื้อเยื่อแทนที่จะสะท้อนที่ผิว — เป็นประเด็นสำคัญตอน defense
- ตำหนิภายใน (Bruise, Rot, Flesh_damage) มองไม่เห็นด้วย Visible channel อย่างเดียว — เป็นเหตุผลหลักที่ต้องใช้ multispectral

## Paper อ้างอิงสำหรับเปรียบเทียบ

**Explainable AI for Fruit Quality Classification (2025)** — International Journal of Research in Computer Applications and Robotics (IJRCAR), ISSN 2320-7345
🔗 https://www.academia.edu/144683271/

- เทียบ CNN (อ่านภาพตรงๆ) vs ensemble methods (Random Forest/XGBoost/LightGBM ที่ทำงานบนฟีเจอร์ที่ดึงจากภาพ) แล้วใช้ SHAP + Grad-CAM อธิบายว่าโมเดลตัดสินใจยังไง
- ผลที่ได้: LightGBM มี accuracy เฉลี่ยสูงสุด 99.29% บนฟีเจอร์ที่ดึงจากภาพ สูงกว่า CNN ในบริบทของเขา
- **เหตุผลที่เลือก paper นี้**: ยืนยันว่า classical ML baseline (phase 2 ของเรา) จำเป็นจริง เพราะมีงานวิจัยจริงที่ ensemble model ชนะ CNN ได้บนข้อมูลขนาดเล็ก
- **นำมาปรับใช้**: เพิ่ม **SHAP** เข้า phase 2 (classical ML baseline) เพื่ออธิบาย feature importance ของ SVM/RF/LightGBM คู่กับ **Grad-CAM** ที่ใช้อธิบาย EfficientNet-B0+SE ใน phase 5 — ทำให้งานมีโครงสร้างอธิบายผลที่ครบคู่กันเหมือน paper ต้นแบบ
- **ตอนเขียนเล่ม**: ใส่ตาราง comparison เทียบ accuracy ของเรากับ paper นี้ พร้อมอธิบายเหตุผลของความต่าง (ขนาด dataset, จำนวน class, ประเภทข้อมูล)

## เครื่องมือ/ไลบรารีที่ใช้

`scikit-image`, `scikit-learn`, `lightgbm`, `xgboost`, `shap`, `pytorch-grad-cam`, `scipy`/`statsmodels`, PyTorch

## Workflow ทั้งหมด (6 phase)

1. **เตรียมข้อมูล** — ตัด channel, จัด folder, split → **เสร็จแล้ว**
2. **Classical ML baseline** — SVM, RF, LightGBM + SHAP อธิบาย feature importance
3. **Deep learning baseline** — EfficientNet-B0 ทั้ง 3 variant (Visible only / NIR only / ครบ 4 channel)
4. **เพิ่ม SE block** — ดัดแปลง EfficientNet-B0 ให้มี channel attention, ดึงค่าน้ำหนักความสำคัญของแต่ละ channel
5. **Grad-CAM + Statistical testing** — ตรวจสอบว่าโมเดลโฟกัสถูกที่ + ยืนยันความต่างของผลลัพธ์ด้วย Bootstrap CI/McNemar's Test
6. **สรุปผล + เขียนเล่ม + เตรียม defense**

## โครงสร้างโค้ด (อัพเดตล่าสุดจาก Claude Code)

```
src/
  config.py              constants: classes, channel groups (visible/nir/full/re/ir), paths
  dataset.py             PyTorch Dataset/DataLoader, per-channel norm stats, weighted sampler, augmentation (train only)
  features.py            hand-crafted features (color stats + GLCM texture + band-ratio NDI) — Phase 2
  train_classical.py     SVM / RandomForest / LightGBM + SHAP — Phase 2
  models.py              EfficientNet-B0 (variable in_channels) + InputChannelSE — Phase 3/4
  train_dl.py            training loop (run_training), early stopping (patience=10), single-split variants — Phase 3/4
  train_dl_kfold.py      5-fold CV, VARIANT_SPECS (visible/nir/full/full_SE/nir_SE/nir_no_aug/nir_SE_ss/re_only/ir_only), --variant, merges into kfold_results.json
  ensemble_kfold.py      soft-vote ensemble of per-fold checkpoints (result: data leakage, not usable) + in-sample/OOF split
  preprocess_ss.py       build Processed_Dataset_SS from SingleSpectra images + RawA<->SingleSpectra mapping (verified by image content)
  gradcam_mask_eval.py   out-of-fold Grad-CAM vs pixel-level defect masks (Pointing Game, IoU, same-class null)
  band_ablation_report.py single-band ablation report (re_only / ir_only / nir: paired folds, per-class F1, McNemar)
  explain.py             Grad-CAM + Bootstrap CI + McNemar's test + per-class F1 — Phase 5
outputs/
  classical/             metrics.json, predictions.json, *_model.joblib, shap/*.png
  dl/                    metrics.json, kfold_results.json, ensemble_results.json, per_class_f1.json, gradcam_mask_eval.json,
                         band_ablation.json, {variant}_best.pt, kfold_ckpt/*.pt, gradcam/ + gradcam_mask/ (images, not in git)
  stats/                 results.json (bootstrap CI + pairwise McNemar's)
  ss_mapping.json, ss_mapping_verification.json   RawA <-> SingleSpectra mapping + its image-content check
Processed_Dataset/, Processed_Dataset_SS/        dataset cubes (proprietary, not in git)
```

Rerun any phase with `.venv/Scripts/python.exe -m src.<module_name>`.

## ขั้นตอนต่อไป — ทุก Phase (1-5) รันจบแล้ว

- [x] จัดการ class imbalance ก่อนเทรน — `WeightedRandomSampler` ใน `dataset.get_dataloaders()`
- [x] สร้าง DataLoader อ่านไฟล์ `.npy` จาก `Processed_Dataset` — `src/dataset.py`
- [x] Phase 2 — classical ML baseline พร้อม SHAP — `src/train_classical.py`
- [x] Phase 3 — EfficientNet-B0 3 variants (visible/nir/full) — `src/train_dl.py`
- [x] Phase 4 — SE block บน full 4-channel — `src/models.py` (`EfficientNetWithChannelSE`) + `train_dl.py`
- [x] Phase 5 — Grad-CAM + statistical testing — `src/explain.py`

## ผลลัพธ์สุดท้าย (test set, 123 samples)

| Model | Macro F1 | Accuracy | 95% Bootstrap CI (F1) |
|---|---|---|---|
| SVM (classical) | 0.618 | 0.707 | [0.506, 0.719] |
| RandomForest (classical) | 0.516 | 0.634 | [0.401, 0.622] |
| LightGBM (classical) | 0.537 | 0.724 | [0.417, 0.652] |
| EfficientNet-B0 Visible-only (BL,GR) | 0.711 | 0.837 | [0.594, 0.810] |
| **EfficientNet-B0 NIR-only (RE,IR)** | **0.867** | **0.919** | **[0.771, 0.944]** |
| EfficientNet-B0 Full (4-band) | 0.745 | 0.845 | [0.630, 0.840] |
| EfficientNet-B0 Full + Input-SE | 0.640 | 0.797 | [0.517, 0.752] |

**Learned SE channel importance (BL, GR, RE, IR)**: [0.483, 0.594, 0.454, 0.537] — GR และ IR ถูกให้น้ำหนักสูงสุด

### สรุปสถิติ (McNemar's test, ดูฉบับเต็มที่ `outputs/stats/results.json`)

- **NIR-only ดีกว่าทุก variant อื่นอย่างมีนัยสำคัญทางสถิติ**: vs visible (p=0.031), vs full (p=0.023), vs full_SE (p=0.0003)
- Deep learning (ทุก variant) ดีกว่า classical ML อย่างมีนัยสำคัญเกือบทุกคู่ (ยกเว้น full_SE vs SVM/LightGBM ที่ไม่ต่างกันนัยสำคัญ) → สนับสนุนความจำเป็นของ DL บน dataset เล็กนี้ได้บางส่วน แต่ full_SE เป็นข้อยกเว้นที่ต้องอภิปรายตอนเขียนเล่ม
- visible vs full: p=1.00 (ไม่ต่างกันเลย) — สอดคล้องกับที่ full มี visible bands ปนอยู่ทำให้ performance ใกล้เคียงกัน

### ข้อค้นพบสำคัญที่ต้องอภิปรายตอน defense

1. **NIR-only ชนะทุกตัวรวมถึง full 4-channel** — ผลตรงข้ามกับสมมติฐานตั้งต้นที่ว่า "ยิ่งรวมทุก band ยิ่งดี" ต้องอธิบายว่า visible bands (BL,GR) อาจเป็น noise สำหรับงานนี้มากกว่าจะช่วย เพราะตำหนิ (โดยเฉพาะ internal defect) ไม่ปรากฏใน visible channel ชัดเจน การรวมเข้ามาอาจทำให้โมเดล confuse
2. **SE block (Phase 4) ทำให้ผลแย่ลงกว่า full ธรรมดา** (0.640 vs 0.745) ทั้งที่ตั้งใจให้ช่วย — สาเหตุที่เป็นไปได้: (a) dataset เล็กเกินไปสำหรับพารามิเตอร์เพิ่ม เกิด overfitting เร็วกว่า (early stop ที่ epoch 23 vs 27), (b) SE weight ที่เรียนรู้ได้ (0.45-0.59 ทุกตัวใกล้เคียงกัน ไม่ discriminative มาก) ยังไม่ได้ช่วยแยก band ที่สำคัญจริงออกจากกันชัดเจน — เป็นประเด็นเปิดให้ทดลองต่อ (เช่น ปรับ `se_reduction`, เพิ่ม regularization, หรือรัน seed หลายค่าแล้วดู variance) **[แก้ 2026-10-13: ข้อสรุปนี้ไม่ robust — ผล k-fold และ 3 seed แสดงว่า SE ไม่ได้แย่ลงหรือดีขึ้นอย่างมีนัยสำคัญ (full_SE 0.737 vs full 0.754, ต่างไม่เกินความแกว่ง) ความต่างในรอบเดียวนี้เป็น noise]**
3. Grad-CAM ยืนยันว่าโมเดล full_SE โฟกัสตรงตำแหน่งรอยช้ำ/ตำหนิจริงในภาพตัวอย่างที่ทำนายถูก (ดู `outputs/dl/gradcam/`) — เป็นหลักฐานเชิงคุณภาพว่าโมเดล "เข้าใจ" ปัญหา แม้ตัวเลข F1 จะไม่ใช่ตัวที่ดีที่สุด **[แก้ 2026-10-13: วัดเชิงปริมาณกับ mask ตำหนิจริงทั้ง 246 ภาพแล้ว ไม่สนับสนุนข้อความนี้ — ดีกว่า null เฉพาะ Internal (Bruise/Flesh_damage), Surface ไม่ต่างจากสุ่ม ดูหัวข้อ "งานที่ 2 — ตรวจ Grad-CAM เชิงปริมาณ"]**

### ข้อจำกัดที่ควรระบุในเล่ม

- ผลทั้งหมดมาจาก **1 run ต่อ 1 variant** (ไม่ได้ average หลาย seed) — ความต่างเล็กๆ (เช่น full vs full_SE ที่ McNemar p=0.24 ไม่ significant) อาจเปลี่ยนได้ถ้ารันซ้ำด้วย seed อื่น ควรรันซ้ำ 3-5 seeds แล้วรายงาน mean±std ถ้ามีเวลา
- test set มีแค่ 123 ภาพ (Internal_Defect 16, Surface_Defect 21) — CI กว้าง สะท้อนความไม่แน่นอนจาก sample size เล็ก
- SHAP ของ SVM ใช้ KernelExplainer กับ background 50 samples + explain 40 test samples เพื่อความเร็ว ไม่ใช่ full dataset

## วิธีรันซ้ำทุกขั้นตอน

```
cd D:\nir+rgb_project\project\CAPA_AppleQualityGradingImageDatabase
.venv\Scripts\python.exe -m src.dataset            # ตรวจ class counts + channel stats
.venv\Scripts\python.exe -m src.train_classical    # Phase 2
.venv\Scripts\python.exe -m src.train_dl           # Phase 3+4 (ใช้ GPU, ~6 นาทีรวม)
.venv\Scripts\python.exe -m src.explain            # Phase 5 (+ per-class F1)
.venv\Scripts\python.exe -m src.train_dl_kfold --variant <name>   # k-fold: nir, nir_SE, nir_no_aug, nir_SE_ss, re_only, ir_only, ... (~8-10 นาที/variant; ข้ามถ้ามีผลครบ 5 fold แล้ว, --force เพื่อรันทับ)
.venv\Scripts\python.exe -m src.preprocess_ss --verify-content    # สร้าง Processed_Dataset_SS (ก่อนรัน nir_SE_ss)
.venv\Scripts\python.exe -m src.gradcam_mask_eval                  # ต้องมี checkpoint nir_SE_ss_fold0-4
.venv\Scripts\python.exe -m src.band_ablation_report               # ต้องมีผล re_only, ir_only, nir
```

## Data Augmentation + K-Fold Cross-Validation — เสร็จแล้ว

เหตุผล: dataset มีแค่ 819 ภาพ (test set แค่ 123 ภาพ, Internal_Defect แค่ 16 ภาพ) ทำให้ CI กว้างและผลแปรปรวนระหว่างรอบเทรน (เช่น visible-only F1 เปลี่ยนจาก 0.711 เป็น 0.616 ระหว่าง 2 รอบที่รันโค้ดเดียวกัน) เพิ่ม 2 เทคนิคนี้เพื่อให้ผลน่าเชื่อถือขึ้นก่อนเขียนเล่ม

### งานที่ 1 — Data Augmentation (ทำใน `src/dataset.py`) — [x] เสร็จแล้ว

เพิ่ม augmentation แบบง่าย apply **เฉพาะ train set เท่านั้น** (ห้าม apply กับ val/test เพราะจะวัดผลผิดเพี้ยน):
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.5)
- Random rotation 90/180/270 องศา (p=0.5)
- Random brightness scaling เล็กน้อย (factor 0.85-1.15, clip 0-1)

ต้องมี flag บอกว่า Dataset instance เป็น train หรือไม่ (เช่น `training: bool` parameter) เพื่อ apply augmentation แค่ตอน train เท่านั้น val/test ต้องได้ภาพต้นฉบับเสมอเพื่อวัดผลได้ตรงกับสถานการณ์จริง (ไม่ใช่ห่วงเรื่อง data leakage — โมเดลเริ่มเทรนใหม่จาก pretrained ImageNet ทุกรอบ ไม่มีความจำข้ามรอบ แต่ augment test set จะทำให้วัดผลไม่ตรงกับข้อมูลจริงที่ไม่เคย manipulate)

รันเช็คว่าไม่พังหลังแก้:
```
.venv\Scripts\python.exe -m src.dataset
```

### งานที่ 2 — K-Fold Cross-Validation (สร้างไฟล์ใหม่ `src/train_dl_kfold.py`) — [x] เสร็จแล้ว

ไม่แก้ `train_dl.py` เดิม สร้างสคริปต์แยกที่:
1. รวมข้อมูลทั้งหมด (train+val+test เดิม) กลับมาเป็น pool เดียว
2. ใช้ `StratifiedKFold` (n_splits=5, shuffle=True, random_state=42) แบ่งเป็น 5 fold ตามสัดส่วน class
3. เทรนทั้ง 4 variant (visible, nir, full, full_SE) แบบ fold-by-fold — แต่ละ fold ใช้ 4 ส่วนเป็น train (แบ่งย่อยเป็น train/val ภายในอีกทีสำหรับ early stopping) ส่วนที่เหลือเป็น test ของ fold นั้น
4. เก็บ test macro-F1 ของแต่ละ (variant, fold) ลงไฟล์ `outputs/dl/kfold_results.json`
5. คำนวณ mean ± std ของ Macro F1 ต่อ variant จากทั้ง 5 fold แล้ว print สรุปให้เห็นว่า NIR-only ยังชนะทุก fold สม่ำเสมอไหม

ใช้ฟังก์ชันเทรนที่มีอยู่แล้วใน `train_dl.py` ถ้าเรียกใช้ซ้ำได้ (import มาใช้) ไม่ต้องเขียนใหม่ทั้งหมด

รันตอนเสร็จ:
```
.venv\Scripts\python.exe -m src.train_dl_kfold
```

ใช้เวลานานกว่าเดิมมาก (4 variant × 5 fold = 20 รอบเทรน) ควรแจ้งเวลาประมาณที่ใช้ก่อนเริ่มรันจริง

### ผลลัพธ์ K-Fold (5 folds, test macro-F1 mean ± std) — รันเสร็จ 2026-09-15

| Model | Macro F1 | Accuracy | เทียบกับผลรอบเดียวเดิม |
|---|---|---|---|
| Visible-only (BL,GR) | 0.666 ± 0.027 | 79.2% ± 2.2% | เดิม 0.711 F1 / 83.7% acc |
| **NIR-only (RE,IR)** | **0.843 ± 0.023** | **89.9% ± 2.0%** | เดิม 0.867 F1 / 91.9% acc |
| Full (4-band) | 0.724 ± 0.054 | 83.3% ± 2.7% | เดิม 0.745 F1 / 84.5% acc |
| Full + Input-SE | 0.734 ± 0.034 | 83.5% ± 1.7% | เดิม 0.640 F1 / 79.7% acc |

ผลเต็ม (per-fold) อยู่ที่ `outputs/dl/kfold_results.json`

**ข้อสรุปสำคัญจาก k-fold:**

1. **NIR-only ชนะทุก fold (5/5) อย่างสม่ำเสมอ** (per-fold F1: 0.845, 0.843, 0.873, 0.803, 0.852) — ยืนยันว่าผลจาก single-split run ก่อนหน้าไม่ใช่เรื่องบังเอิญ เป็นข้อสรุปที่มั่นคง
2. **Full+SE กลับดีกว่า Full ธรรมดาเล็กน้อยใน k-fold** (0.734 vs 0.724) — สวนทางกับผลรอบเดียวเดิมที่บอกว่า SE แย่กว่า (0.640 vs 0.745) ชี้ว่าความต่างจาก single-split run เป็น noise จาก random split เป็นหลัก ไม่ใช่ผลจริงของ SE block ต่อ performance — **ต้องแก้ข้อสรุปเดิมในหัวข้อ "ข้อค้นพบสำคัญที่ต้องอภิปรายตอน defense" ด้านบน** เวลาเขียนเล่ม (ข้อ 2 เดิมที่บอกว่า "SE block ทำให้ผลแย่ลง" ไม่ robust ข้าม fold)
3. **SE channel importance สม่ำเสมอทุก fold**: BL≈0.487, GR≈0.599, RE≈0.458, IR≈0.540 — GR และ IR สูงสุดเสมอทุก fold ตรงกับผล single-split เดิม (GR, IR สูงสุด) → ~~ข้อสรุปเรื่อง channel importance นี้ robust ไม่เปลี่ยนตาม fold~~ **[แก้ 2026-10-13: ความ "นิ่ง" ของ weight ไม่ใช่หลักฐานว่าแบนด์นั้นสำคัญ — การทดสอบแบนด์เดี่ยวใน NIR พบว่า weight ที่ต่างกัน (RE 0.56 vs IR 0.38) ไม่สะท้อนความต่างของ performance จริง จึงอย่าใช้ SE weight เป็นหลักฐานความสำคัญของแบนด์ (รวมถึงผล full_SE นี้ด้วย)]**
4. Visible-only มี std ต่ำสุด (±2.2% acc) แต่ full/full_SE มี std สูงกว่า (±2.7%, จุด full ที่ ±5.4% F1) แสดงว่าเป็น variant ที่ผลแปรปรวนง่ายกว่า NIR-only และ visible-only

## Ensembling จาก K-Fold Models + เช็ค Per-Class F1 — เสร็จแล้ว

เหตุผล: k-fold ที่ทำไปแล้วมีโมเดล 5 ตัวต่อ variant อยู่แล้ว (แค่ต่างกันที่ fold) เอามา ensemble กันได้เลยโดยไม่ต้องเทรนใหม่ — และยังไม่เคยดู per-class F1 (แยก Healthy/Internal_Defect/Surface_Defect) เพื่อเช็คว่าจำเป็นต้องทำ Focal Loss เพิ่มไหม

### งานที่ 1 — Ensembling จาก 5-fold models — [x] เสร็จแล้ว (`src/ensemble_kfold.py`)

1. โหลด checkpoint ทั้ง 5 fold ของ variant ที่สนใจ (เริ่มจาก `nir` และ `full_SE` ก่อน เพราะเป็น 2 ตัวที่สำคัญที่สุดของธีสิส)
2. สำหรับแต่ละภาพใน test set (หรือ held-out set ที่ยังไม่เคยใช้เทรนใน fold ไหนเลย ถ้ามี — ถ้าไม่มีให้ใช้ test set เดิมจาก single-split ตอนต้น เพราะ k-fold ใช้ pool ข้อมูลทั้งหมดไปแล้ว ต้องระวังเรื่อง fold ไหนเคยเห็นภาพนี้ตอนเทรนบ้าง)
3. รันทำนายด้วยทั้ง 5 โมเดล เก็บค่า softmax probability ของแต่ละโมเดล
4. เฉลี่ย probability จากทั้ง 5 โมเดล (soft voting) แล้วเลือก class ที่มีค่าเฉลี่ยสูงสุดเป็นคำตอบสุดท้าย
5. คำนวณ Macro F1 ของผลลัพธ์ ensemble เทียบกับ F1 เฉลี่ยของโมเดลเดี่ยวแต่ละตัว (จาก k-fold results เดิม)
6. เซฟผลลง `outputs/dl/ensemble_results.json`

**ข้อควรระวัง**: ต้องเลือก evaluation set ที่ไม่มี fold ไหนใช้เทรนเลย ถ้าไม่มีข้อมูลเหลือแบบนั้น ให้ระบุใน output ตรงๆว่าเป็นการประเมินแบบ "in-sample" (บางภาพอาจถูกโมเดลบางตัวเห็นมาก่อนตอนเทรน) เพื่อความโปร่งใสตอนเขียนเล่ม

รันแล้ว:
```
.venv\Scripts\python.exe -m src.ensemble_kfold train <variant> <fold_idx>   # x10 (nir/full_SE x fold 0-4)
.venv\Scripts\python.exe -m src.ensemble_kfold evaluate
```

โมเดล checkpoint ทั้ง 10 ตัวอยู่ที่ `outputs/dl/kfold_ckpt/{variant}_fold{i}.pt` (เก็บ state_dict + mean/std ของ fold นั้นไว้ด้วย เพื่อ normalize ตอน inference ให้ตรงกับตอนเทรน)

#### ผลลัพธ์ ensembling (evaluate บน original single-split test set, 123 ภาพ) — รันเสร็จ 2026-09-15

**สคริปต์ยืนยันการแบ่ง fold ถูกต้อง**: ทุกภาพใน eval set เป็น out-of-fold (ไม่เคยเห็นตอนเทรน) สำหรับโมเดลพอดี 1 ใน 5 ตัวเสมอ (ตรวจสอบอัตโนมัติในโค้ด — ผ่าน)

| Model | Individual folds (บน eval set นี้) | Soft-vote ensemble (5 โมเดล) | **Unbiased OOF (1 โมเดล/ภาพ)** |
|---|---|---|---|
| NIR-only | 0.931 ± 0.031 F1 (95.1% ± 2.2% acc) | **0.969 F1 (97.6% acc)** ⚠️ | **0.801 F1 (87.8% acc)** |
| Full+SE | 0.846 ± 0.040 F1 (89.9% ± 2.8% acc) | **0.957 F1 (96.8% acc)** ⚠️ | **0.659 F1 (79.7% acc)** |

ผลเต็มที่ `outputs/dl/ensemble_results.json`

**⚠️ ข้อค้นพบสำคัญ — ตัวเลข ensemble สูงลิ่วเป็น artifact ของ data leakage ไม่ใช่ของจริง:**

เหตุผล: การทำ 5-fold CV ใช้ pool ภาพทั้งหมด 819 ภาพ (รวม original test set เข้าไปด้วย) ทำให้ไม่มีภาพไหนเหลือที่ "ไม่เคยถูกโมเดลไหนเห็นเลยตอนเทรน" — เวลาเอา original test set (123 ภาพ) มาประเมิน โมเดลแต่ละตัวจากทั้ง 5 fold จะเคยเห็นภาพส่วนใหญ่ (~4/5) มาแล้วตอนเทรน (เป็น in-sample) มีแค่ 1/5 เท่านั้นที่เป็นภาพที่ fold นั้นไม่เคยเห็น (out-of-fold) ตัวเลข **"individual folds"** และ **"soft-vote ensemble"** ในตารางข้างบนจึงสูงเกินจริงมาก (ensemble ได้ 96-98% acc!) เพราะโมเดลส่วนใหญ่ "จำ" คำตอบได้จากตอนเทรน ไม่ใช่เดาถูกจริง

ตัวเลขที่น่าเชื่อถือกว่าคือ **"Unbiased OOF"** — สำหรับแต่ละภาพ ใช้แค่โมเดลจาก fold เดียวที่ไม่เคยเห็นภาพนั้นตอนเทรนเลย (ไม่มีการ ensemble) ได้ผล NIR-only = 0.801 F1 / 87.8% acc, Full+SE = 0.659 F1 / 79.7% acc — **ต่ำกว่าผลเฉลี่ยจาก 5-fold CV เดิม** (NIR 0.843±0.023, Full+SE 0.734±0.034 จากหัวข้อด้านบน) เพราะที่นี่วัดจากภาพแค่ ~24-25 ภาพต่อโมเดล (เศษที่ overlap ระหว่าง original test set กับ fold test set นั้นๆ) ตัวอย่างน้อยกว่ามาก ผลเลยแปรปรวนกว่า

**สรุปสำหรับเขียนเล่ม**: ตัวเลข ensemble ในตารางนี้ **ห้ามใช้เป็นผลหลัก** เพราะมี data leakage ชัดเจน ตัวเลขที่ควรใช้อ้างอิงยังคงเป็นผล 5-fold CV แบบ mean±std จากหัวข้อ "ผลลัพธ์ K-Fold" ด้านบน (NIR-only 0.843±0.023 F1, 89.9%±2.0% acc) — การ ensembling ในโครงสร้างข้อมูลนี้ทำไม่ได้อย่างยุติธรรมเพราะ dataset เล็กเกินไปที่จะกันส่วนหนึ่งไว้ไม่ให้ทุก fold เห็นเลย ถ้าต้องการทำ ensembling จริงๆ ต้องมี held-out set แยกต่างหากตั้งแต่แรกที่ไม่เอาไปทำ k-fold เลย (ซึ่งจะทำให้ข้อมูลเทรนน้อยลงไปอีก ไม่คุ้มกับ dataset ขนาด 819 ภาพนี้)

### งานที่ 2 — เช็ค Per-Class F1 — [x] เสร็จแล้ว (`src/explain.py` ฟังก์ชัน `print_per_class_f1()`)

เพิ่ม `classification_report` หรือคำนวณ F1 แยกทีละ class (Healthy, Internal_Defect, Surface_Defect) สำหรับโมเดล NIR-only (และ full_SE) บน test set — print ออกมาให้เห็นชัดว่า class ไหนมี F1 ต่ำสุด

**เกณฑ์ตัดสินใจ**: ถ้า Internal_Defect (class ที่มีข้อมูลน้อยสุด 102 ภาพ) มี F1 ต่ำกว่า class อื่นอย่างชัดเจน (เช่น ต่ำกว่า 0.10 เทียบกับ class อื่น) → แนะนำให้ลอง Focal Loss ต่อ ถ้า F1 ใกล้เคียงกันทุก class แล้ว → ไม่ต้องทำ Focal Loss เพิ่ม ประหยัดเวลาไปทำ Phase 6 (เขียนเล่ม) ได้เลย

รันแล้ว: `.venv\Scripts\python.exe -m src.explain` — ผล (single-split test set, 123 ภาพ) บันทึกที่ `outputs/dl/per_class_f1.json`:

| Model | Healthy F1 | Internal_Defect F1 | Surface_Defect F1 | ต้องทำ Focal Loss ไหม |
|---|---|---|---|---|
| NIR-only | 0.955 | 0.867 | 0.737 | **ไม่ต้อง** — ห่างจาก class อื่นแค่ 0.088 (≤0.10) |
| Full+SE | 0.904 | 0.480 | 0.545 | **ควรทำ** — ห่างจาก Healthy ถึง 0.424 (>0.10) |

**ข้อสังเกตที่น่าสนใจ**: สำหรับ NIR-only จริงๆ แล้ว **Surface_Defect** (0.737) แย่กว่า Internal_Defect (0.867) เสียอีก แม้ Internal_Defect จะมีข้อมูลน้อยกว่า (102 vs 144 ภาพ) — แปลว่าปัญหาไม่ได้อยู่ที่ class imbalance อย่างเดียว อาจเป็นเพราะ Surface_Defect มีความหลากหลายของตำหนิย่อยเยอะ (Russet/Scald/Hail/Frost/Limb_rub/Other) ทำให้ pattern ไม่สม่ำเสมอ — **สำหรับ NIR-only (โมเดลหลักที่ผลดีที่สุด) ไม่จำเป็นต้องทำ Focal Loss เพิ่ม** ข้ามไปทำ Phase 6 (เขียนเล่ม) ได้เลย ส่วน full_SE ที่แย่กว่าเป็นเพราะภาพรวม performance ต่ำกว่าอยู่แล้ว ไม่ใช่ประเด็นหลักของธีสิส (NIR-only คือโมเดลที่ดีที่สุดและใช้เป็นข้อสรุปหลัก)

## รอบสุดท้าย: รวมเทคนิคที่พิสูจน์แล้วว่าดีที่สุด (Best-Combo Run) — เสร็จแล้ว 2026-09-15

เหตุผล: จากทุกรอบที่ทำมา (รอบ 1-4) สรุปได้ว่า NIR-only (RE,IR) เป็น input ที่ดีที่สุดเสมอ, Data Augmentation ช่วยลด variance, SE block ช่วยเล็กน้อยเมื่อวัดถูกวิธี (k-fold), Ensembling และ Focal Loss พิสูจน์แล้วว่าไม่คุ้ม/ใช้ไม่ได้บน dataset นี้ — รอบนี้คือการรวมเทคนิคที่ดีที่สุดเข้าด้วยกันเป็นครั้งแรก โดยเฉพาะ **SE block บน NIR-only ที่ยังไม่เคยทดลอง** (ที่ผ่านมาลอง SE บน full 4-channel เท่านั้น)

### สิ่งที่ต้องทำ

1. เพิ่ม variant ใหม่ชื่อ `nir_SE` ใน `src/train_dl_kfold.py` (หรือไฟล์ที่เกี่ยวข้อง): ใช้ input channel แค่ RE, IR (เหมือน `nir` เดิม) แต่ใส่ SE block เข้าไปด้วย (ใช้โครงสร้าง SE เดียวกับที่ทำกับ `full_SE` แต่ลด input channel เหลือ 2)
2. รันผ่าน pipeline เดิมที่มีอยู่แล้ว: Data Augmentation (มีอยู่แล้วใน `dataset.py`) + WeightedRandomSampler (มีอยู่แล้ว) + 5-Fold CV (มีอยู่แล้วใน `train_dl_kfold.py`)
3. **ไม่ต้องทำ**: ensembling (พิสูจน์แล้วว่า data leakage) และ focal loss (พิสูจน์แล้วว่าไม่จำเป็นสำหรับ NIR-based model)
4. เก็บผลลง `outputs/dl/kfold_results.json` เพิ่ม entry ของ `nir_SE` ต่อจากของเดิม (ไม่ทับของเก่า)
5. เปรียบเทียบ `nir_SE` กับ `nir` (ตัวเดิมที่ดีที่สุด, 0.843±0.023 F1) ว่า SE block ช่วยดันขึ้นไปได้อีกไหม หรือไม่ต่างกัน หรือแย่กว่า — รายงานทั้ง Macro F1 และ Accuracy พร้อม mean±std จากทั้ง 5 fold
6. ดึงค่า SE channel importance ของ `nir_SE` ออกมาด้วย (RE vs IR ตัวไหนสำคัญกว่ากัน) — เป็นข้อมูลเสริมที่น่าสนใจสำหรับ objective 3.1 ระดับที่ละเอียดขึ้น (ไม่ใช่แค่ visible vs NIR แต่ลงไปถึงใน NIR เอง)

**สถานะโค้ด: [x] พร้อมรันแล้ว** — เพิ่ม `nir_SE` เข้า `VARIANT_SPECS` ใน `src/train_dl_kfold.py` (channel_group="nir", use_se=True) ทดสอบแล้วว่า build_model + forward pass ผ่าน ยังไม่ได้รันจริง (รอผู้ใช้กดรันเอง) คำสั่งที่ต้องรัน (5 fold, ~10-15 นาที, checkpoint เดิมของ visible/nir/full/full_SE จะไม่ถูกแตะต้อง เพราะ merge เข้า `kfold_results.json` แบบไม่ทับของเก่า):
```
.venv\Scripts\python.exe -m src.train_dl_kfold --variant nir_SE
```

### ผลลัพธ์ nir_SE (การรันครั้งเดียว) — [x] รันเสร็จ 2026-09-15 ~~🏆 ตัวเลขที่ดีที่สุดของทั้งโปรเจก~~ **[ถูกแทนที่ 2026-10-13: เฉลี่ย 3 seed ได้ 0.847 ไม่ต่างจาก `nir` 0.843 — ตัวเลข 0.862 เป็นรอบที่โชคดี ดูหัวข้อ "ผล multi-seed (รันข้ามคืน)"]**

| Model | Macro F1 (5-fold mean±std) | Accuracy | Per-fold F1 |
|---|---|---|---|
| nir (เดิม, ไม่มี SE) | 0.843 ± 0.023 | 89.9% ± 2.0% | — |
| **nir_SE (ใหม่)** | **0.862 ± 0.025** | **91.7% ± 2.1%** | [0.892, 0.878, 0.859, 0.818, 0.862] |

~~**ข้อสรุป**: SE block บน NIR-only ช่วยดันขึ้นจริง +0.019 F1 — nir_SE คือโมเดลสุดท้ายที่ควรใช้เป็นผลหลักของธีสิส (F1 0.862±0.025, Accuracy 91.7%±2.1%)~~ **[ถูกหักล้าง 2026-10-13: ผล 3 seed × 5 fold: `nir_SE` − `nir` = +0.003 (95% CI −0.012..+0.018, เครื่องหมายเดียวกันเพียง 7/15 คู่) → SE ไม่ได้ช่วยเกินความแกว่งของการเทรน ห้ามใช้ 0.862 เป็นผลหลัก ใช้ตัวเลข multi-seed แทน]**

**SE channel importance (RE, IR)** — สม่ำเสมอมากทุก fold: RE ≈ 0.558-0.565, IR ≈ 0.375-0.379 ~~→ ภายใน NIR เอง Red Edge สำคัญกว่า Infrared เกือบเท่าตัว~~ **[ถูกหักล้างแล้ว 2026-10-13 โดยการเทรนแบนด์เดี่ยว (หัวข้อ "ทดสอบความสำคัญของแบนด์แบบตรง"): RE-only ≈ IR-only ≈ RE+IR ไม่ต่างกัน — ห้ามใช้ข้อสรุปนี้ในเล่ม; ค่า SE weight ที่นิ่งเกือบเท่ากันทุก fold/ทุกแหล่งภาพน่าจะสะท้อนการ initialize + scale ของ input มากกว่าความสำคัญของแบนด์]**

**ข้อควรระวังตอนรายงานผล**: fold ที่ดีที่สุดได้ถึง 0.892 F1 / 93.9% acc แต่**ห้ามหยิบตัวเลข fold เดียวไปรายงาน** (cherry-picking) — ใช้ mean±std เสมอ

### Ablation test เพิ่มเติม — แยกผลของ K-Fold ออกจาก Augmentation

เหตุผล: รอบ 1-2 (single-split, ไม่มี augmentation) ได้ NIR-only F1 = 0.867/0.853 แต่รอบ 3-4 (K-Fold + augmentation) ได้แค่ 0.843±0.023 — ยังบอกไม่ได้ว่าตัวเลขที่ลดลงมาจาก **วิธีวัดผลเปลี่ยน** (k-fold เฉลี่ยรวมทั้ง fold ง่ายและยาก) หรือจาก **augmentation ทำให้แย่ลงจริง** เพราะเปลี่ยน 2 อย่างพร้อมกัน ต้องแยกทดสอบให้ชัด

**สิ่งที่ต้องทำ**: เพิ่ม variant ใหม่ชื่อ `nir_no_aug` — ใช้ NIR-only channels เหมือนเดิม รันผ่าน 5-Fold CV เหมือนเดิม **แต่ปิด Data Augmentation** (ใช้ dataset แบบรอบ 1-2 ไม่มี flip/rotate/brightness) ทุกอย่างอื่นเหมือน `nir` เดิมในรอบ 3

**สถานะโค้ด: [x] พร้อมรันแล้ว** — เพิ่ม `nir_no_aug` เข้า `VARIANT_SPECS` (channel_group="nir", use_se=False, augment_train=False) — เพิ่ม parameter `augment_train` ใน `dataset.get_dataloaders_from_items()` เพื่อปิด augmentation ได้แบบไม่กระทบ variant อื่น ยังไม่ได้รันจริง (รอผู้ใช้กดรันเอง) คำสั่งที่ต้องรัน (5 fold, ~10-15 นาที):
```
.venv\Scripts\python.exe -m src.train_dl_kfold --variant nir_no_aug
```

**ตารางเทียบผลที่ต้องได้หลังรันครบ** — [x] รันเสร็จแล้ว 2026-09-15:

| Experiment | Augmentation | วิธีวัดผล | Macro F1 |
|---|---|---|---|
| รอบ 1-2 (มีอยู่แล้ว) | ไม่มี | Single-split | 0.867 / 0.853 |
| **nir_no_aug (ใหม่)** | ไม่มี | K-Fold (5-fold mean±std) | **0.817 ± 0.037** (acc 89.1%±2.9%, folds=[0.820, 0.841, 0.859, 0.751, 0.815]) |
| nir (รอบ 3, มีอยู่แล้ว) | มี | K-Fold (5-fold mean±std) | 0.843±0.023 |

**ข้อสรุปจาก ablation (สำคัญสำหรับเล่ม)**:
1. **Augmentation ช่วยจริง ไม่ได้ทำให้แย่ลง** — เทียบในเงื่อนไข K-Fold เดียวกัน: มี aug (0.843±0.023) > ไม่มี aug (0.817±0.037) และ std ก็ต่ำกว่าด้วย (เสถียรกว่า)
2. **ตัวเลขรอบ 1-2 ที่สูง (0.867) เป็นผลจากโชคของ single-split** ไม่เกี่ยวกับการไม่มี augmentation — เห็นจาก std ของ nir_no_aug ที่สูงถึง ±0.037 (fold ต่ำสุด 0.751, สูงสุด 0.859) การสุ่มแบ่งครั้งเดียวจึงได้ค่าโชคดีแบบรอบ 1-2 ได้ง่าย
3. ตัวเลข single-split (รอบ 1-2) **ห้ามใช้เทียบกับตัวเลข k-fold โดยตรง** — เป็นคนละมาตรฐานการวัด

## Version Control + Backup — เสร็จแล้ว (2026-09-15)

เหตุผล: ผลลัพธ์ปัจจุบัน (เช่น NIR-only 0.867 F1 ตอน single-split) อยู่ใน checkpoint ไฟล์ (`*_best.pt`) ที่**ไม่มี skip-logic ป้องกันการทับ** ถ้ารัน `train_dl.py` หรือ `train_classical.py` ซ้ำ ไฟล์เดิมจะถูกทับทันที และที่แย่กว่านั้นคือ **แม้ตั้ง seed=42 ไว้แล้ว ผลก็ยังไม่ตรงกันทุกรอบ** (มีบันทึกไว้แล้วว่า visible-only F1 เคยได้ 0.711 แล้วรันโค้ดเดียวกันซ้ำได้ 0.616 — เป็นผลจาก GPU/cuDNN ที่มีส่วน non-deterministic ที่ยังไม่ได้ล็อก) จึงต้องกันไว้ก่อนจะรัน `nir_SE`/`nir_no_aug` หรือ experiment อื่นต่อ

### สิ่งที่เสี่ยงถูกทับ (ไม่มี skip-logic) vs สิ่งที่ปลอดภัยแล้ว

| ไฟล์/โฟลเดอร์ | เสี่ยงทับ? | เหตุผล |
|---|---|---|
| `outputs/dl/metrics.json`, `{visible,nir,full,full_SE}_best.pt`, `*_history.json` | **เสี่ยง** | มาจาก `train_dl.py` main() — ทับทุกครั้งที่รัน ไม่เช็คของเดิม |
| `outputs/classical/*` | **เสี่ยง** | มาจาก `train_classical.py` — ทับทุกครั้งเหมือนกัน |
| `outputs/dl/gradcam/*.png`, `outputs/stats/results.json`, `outputs/dl/per_class_f1.json` | **เสี่ยง** | มาจาก `explain.py` — ทับทุกครั้ง และผลเปลี่ยนตาม `*_best.pt` ล่าสุดด้วย |
| `outputs/dl/kfold_results.json` | ปลอดภัย | `train_dl_kfold.py` เช็คก่อนว่า variant นั้นมีครบ 5 fold รึยัง ถ้ามีแล้ว **ข้าม** (ต้องใส่ `--force` ถึงจะทับ) |
| `outputs/dl/kfold_ckpt/*.pt` (checkpoint 10 ตัวจาก ensembling) | ปลอดภัย | `ensemble_kfold.py` เช็ค `if os.path.exists(ckpt_path): skip` |

### 1. Local backup (ทำแล้ว)

คัดลอกทุกอย่างที่เสี่ยงทับไปเก็บที่ `backups/best_results_20260915_181726/` (231MB) — รวม checkpoint ทั้งหมด (single-split + k-fold), metrics/history json, Grad-CAM images, classical models, และสำเนา `PROJECT_PLAN.md` ณ เวลานั้น โฟลเดอร์ `backups/` ไม่ถูก track ใน git (ดูข้อ 3)

### 2. ข้อจำกัดลิขสิทธิ์ของ Dataset — ต้องรู้ก่อนแชร์อะไรออกไปข้างนอก

เช็ค `ReadMe.txt` ของ dataset แล้วเจอเงื่อนไขชัดเจน:

> "The CAPA Apple Quality Grading Image Database is a **proprietary of ULG (Gembloux Agro-Bio Tech), Belgium — and cannot be used without the consent of ULG**"
> ติดต่อขอ consent ได้ที่ Devrim Unay (Bahcesehir University) หรือ Marie-France Destain (Gembloux Agro-Bio Tech)

เงื่อนไขเพิ่มเติมตอนเผยแพร่ผลงาน (defense/thesis เข้าข่ายนี้):
1. ต้องระบุในเล่มว่าข้อมูลได้มาจาก ULG (Gembloux Agro-Bio Tech)
2. ต้อง cite: Kleynen, O., Leemans, V., & Destain, M.-F. (2005). *Development of a multi-spectral vision system for the detection of defects on apples.* Journal of Food Engineering, 69(1), 41-49.

**นัยสำคัญ**: ห้ามอัปโหลดตัว dataset หรือสิ่งที่มีเนื้อภาพติดไปด้วย (Grad-CAM ที่ overlay บนภาพจริง, raw `.npy` cubes) ขึ้นที่ไหนที่คนนอกเข้าถึงได้ (รวมถึง private GitHub repo) โดยไม่มี consent — เป็นการ redistribute โดยไม่ได้รับอนุญาต

### 3. Git repo + GitHub (ทำแล้ว)

ตั้งค่า git ในเครื่อง แยกสิ่งที่ขึ้น GitHub ได้ (โค้ด + ผลตัวเลข) ออกจากสิ่งที่ห้ามขึ้น (dataset + ภาพที่มีเนื้อ dataset + ไฟล์โมเดลใหญ่) ผ่าน `.gitignore`:

**ขึ้น GitHub แล้ว** (ไม่มีเนื้อภาพจาก dataset, ไม่มีปัญหาลิขสิทธิ์):
- โค้ดทั้งหมดใน `src/`, `preprocess_pipeline.py`, `requirements.txt`
- `PROJECT_PLAN.md`, `ReadMe.txt` (ไฟล์ license/attribution ของ dataset — เก็บไว้อ้างอิง ไม่ใช่ตัว dataset)
- ผลตัวเลข json ล้วนๆ: `outputs/dl/metrics.json`, `kfold_results.json`, `per_class_f1.json`, `ensemble_results.json`, `outputs/stats/results.json`, `outputs/classical/metrics.json`, `predictions.json`
- SHAP chart (`outputs/classical/shap/*.png`) — เป็น bar chart แสดง feature importance ไม่ใช่ภาพจาก dataset

**ห้ามขึ้น GitHub** (กันไว้ใน `.gitignore`):
- `Processed_Dataset/`, `images/`, `segmentations/`, `outputs/features/` — dataset ดิบ/แปรรูป
- `outputs/dl/gradcam/*.png` — มีภาพจริงจาก dataset ซ้อน heatmap (derivative ของภาพต้นฉบับ)
- `outputs/dl/*.pt`, `outputs/dl/kfold_ckpt/*.pt`, `outputs/classical/*.joblib` — model weight ไฟล์ใหญ่ (16MB/ไฟล์ x 15+ ไฟล์) ไม่ใช่ปัญหาลิขสิทธิ์โดยตรง แต่ไม่คุ้มจะ track ใน git (เก็บที่ local backup พอ)
- `backups/`, `.venv/`, `__pycache__/`

**สถานะ**: `git init` แล้ว, commit แรก (31 ไฟล์, 6567 บรรทัด) push ขึ้น [github.com/Chinnawan/project_nir](https://github.com/Chinnawan/project_nir) branch `main` เรียบร้อยเมื่อ 2026-09-15

**ต่อจากนี้**: ทุกครั้งที่มีผลใหม่ (เช่น `nir_SE`/`nir_no_aug` รันเสร็จ) ให้ commit + push เพิ่มได้เลย — แค่ระวังอย่า `git add -A` แบบไม่เช็ค ถ้ามีการสร้างไฟล์โฟลเดอร์ใหม่ที่ไม่ได้อยู่ใน `.gitignore` (เช่น เพิ่ม output folder ใหม่ที่มีภาพ) ให้เพิ่มเข้า `.gitignore` ก่อน commit เสมอ

## งานที่ต้องทำต่อ — ใช้ข้อมูลใน CAPA ที่ยังไม่เคยใช้ (พบ 2026-10-12)

### สิ่งที่พบจากการสำรวจโฟลเดอร์

1. **`images/SingleSpectraImages/`** — 3,276 ไฟล์ = 819 ภาพ × 4 แบนด์ (`{Healthy|Stem|Calix|Defect}NNN_{BL|GR|RE|IR}.tif`, 560×430) เป็น**แอปเปิลชุดเดียวกับ RawA** (Healthy 280 + Stem 148 + Calix 145 + Defect 246 = 819) **ไม่ใช่ข้อมูลเพิ่ม** แต่ตาม `Info_ReadFirst.rtf` ภาพชุดนี้ **ผ่าน flat-field correction** (ใช้ `RefFile/BW8RefWhite*`, `BW8RefBlack*`) และครอป/แก้ rotation+scale แล้ว ขณะที่ `RawA*.tif` ที่ pipeline ปัจจุบันใช้คือ "raw images **without** flat field correction"
2. **`segmentations/Segmented_defects/jona/im/segmentedBW/SBW_DefectNNNN.bmp`** — **mask ตำหนิระดับพิกเซล 246 ภาพ** ขนาด 560×430 ตรงกับ `DefectNNNN_*.tif` + `segmentations/KleynenDefectClasses_Simplified.txt` (ระดับความรุนแรง Class 1-4)
3. `Multispectral images of apples/` (นอก CAPA) — dataset อื่น 16 แบนด์ label เป็น ripeness/sweetness/variety **ไม่มี label ตำหนิ ใช้กับงานนี้ไม่ได้** — ไม่ต้องแตะ

**ข้อควรระวัง**:
- mask อยู่ในเรขาคณิตของภาพ SingleSpectra เท่านั้น ใช้กับ RawA ไม่ได้
- `Defects.txt` ที่ rtf อ้างถึง (บอกชนิดตำหนิของ DefectNNNN) **ไม่มีในโฟลเดอร์** — ต้องสร้าง mapping DefectNNNN → ชนิดตำหนิ (→ Internal/Surface) เอง โดยอิงลำดับใน `RawFile.txt` และ**ตรวจกับเบาะแสใน `SingleSpectraImages/notes.txt`** (เช่น Bruise010–055 = defect181–226, Flesh_damage005–014 = defect227–236, Flesh_damage015–024 = defect237–246) ถ้า mapping ไม่ลงตัว ให้หยุดแจ้งผู้ใช้ก่อน ห้ามเดา
- Healthy/Stem/Calix ทั้งหมด → class Healthy (เหมือน label เดิม) ตรวจว่าจำนวนต่อ class ยังเป็น 573/102/144

### งานที่ 1 — เทรน nir_SE ด้วยภาพ SingleSpectra (flat-field corrected)

1. สร้าง dataset ใหม่แยกโฟลเดอร์ (เช่น `Processed_Dataset_SS/`) **ห้ามทับ `Processed_Dataset/` เดิม** — stack แบนด์ BL/GR/RE/IR, resize 224×224, เก็บ `.npy` + เก็บ mapping ไฟล์ → class
2. เพิ่ม variant `nir_SE_ss` ใน `train_dl_kfold.py` ทุกอย่างเหมือน `nir_SE` (RE+IR, SE, augmentation, WeightedRandomSampler, StratifiedKFold n=5 random_state=42) ต่างแค่แหล่งภาพ
3. รายงาน Macro F1 / Accuracy mean±std + per-fold เทียบกับ `nir_SE` (0.862±0.025) + SE weights (RE, IR)
4. ทุกอย่างวัดด้วยภาพจริงเท่านั้น

#### สถานะ: [x] เสร็จแล้ว 2026-10-12

**สิ่งที่ทำ / ไฟล์ใหม่**
- `src/preprocess_ss.py` — สร้าง `Processed_Dataset_SS/` (819 ไฟล์ `.npy` 224×224×4, BL/GR/RE/IR) โดย **mirror `Processed_Dataset/` ไฟล์ต่อไฟล์** (ชื่อ `RawA####.npy` เดิม, split train/val/test เดิม, โฟลเดอร์ class เดิม) → StratifiedKFold ให้ **แอปเปิลทุกลูกอยู่ fold เดียวกับ `nir_SE`** (เทียบแบบจับคู่ได้ และมี assert ใน `build_pool("ss")` กันพลาด) แปลงภาพ 560×430 เป็นสี่เหลี่ยมจัตุรัสด้วยการ **pad ดำบน/ล่างเป็น 560×560 แล้ว resize 224** (ไม่ตัดแอปเปิลทิ้ง — วัดกรอบแอปเปิล (รวมก้าน) ทั้ง 819 ภาพ พบว่า center-crop 430×430 จะตัดส่วนของแอปเปิลหรือก้านทิ้งในราว 36% ของภาพ) ผลคือแอปเปิลเล็กลงราว 9% เทียบ pipeline raw
- `outputs/ss_mapping.json` — mapping RawA → ชื่อไฟล์ SingleSpectra + ชนิดตำหนิ + class (ไม่มีเนื้อภาพ ขึ้น git ได้)
- `train_dl_kfold.py` — เพิ่ม variant `nir_SE_ss` (+ field `source`, `save_ckpt` ใน `VARIANT_SPECS`), checkpoint ต่อ fold เก็บที่ `outputs/dl/kfold_ckpt/nir_SE_ss_fold{0-4}.pt` (อยู่ใน `.gitignore`), `dataset.list_files(split, root=None)` รองรับ dataset root อื่น
- `.gitignore` — เพิ่ม `Processed_Dataset_SS/`, `outputs/dl/gradcam_mask/`
- Backup ก่อนรัน: `backups/pre_ss_20261012_231238/`

**Mapping DefectNNNN → ชนิดตำหนิ (ไม่ได้เดา — ตรวจ 2 ทาง)**: `Defects.txt` ไม่มีในโฟลเดอร์ จึงใช้สมมติฐาน "Defect k = รายการที่ k ใน RawFile.txt ที่ไม่ใช่ Healthy/Stem/Calix" แล้วตรวจ
1. เบาะแสใน `notes.txt` ตรงครบ **9/9 จุด** (Other 15/16/17 = defect106/107/125, Bruise010/055 = defect181/226, Flesh_damage005/014/015/024 = defect227/236/237/246)
2. **ตรวจด้วยเนื้อภาพทั้ง 819 ภาพ**: normalized cross-correlation ระหว่างแบนด์ IR ของภาพ SingleSpectra กับ quadrant IR ของ RawA ที่ map ได้ → RawA ที่ map ชนะ RawA สุ่ม 10 ภาพอื่น **819/819** (NCC เฉลี่ย 0.995, ต่ำสุด 0.984) (`outputs/ss_mapping_verification.json`) และ class ตรงกับ label เดิมทุกไฟล์ (573/102/144)

**ผล `nir_SE_ss` (5-fold, ภาพ flat-field corrected)** vs `nir_SE` (ภาพ raw, fold เดียวกัน):

| Variant | Macro F1 | Accuracy | per-fold F1 (fold 0-4) | SE weight (RE, IR) |
|---|---|---|---|---|
| `nir_SE` (raw) | 0.862 ± 0.025 | 91.7% ± 2.1% | 0.892, 0.878, 0.859, 0.818, 0.862 | 0.561, 0.377 |
| **`nir_SE_ss` (SingleSpectra)** | **0.859 ± 0.024** | **90.9% ± 1.7%** | 0.850, 0.831, 0.859, 0.853, 0.903 | 0.558, 0.377 |

**ข้อสรุป**: ไม่ต่างกัน — ผลต่างเฉลี่ย −0.003 F1 เล็กกว่า std (±0.024) มาก และผลต่างรายคู่ fold กลับเครื่องหมาย (−0.042, −0.047, 0.000, +0.035, +0.041) → **flat-field correction ไม่ได้ช่วยและไม่ได้ทำให้แย่ลงอย่างมีนัยสำคัญ** ใช้เขียนในเล่มได้ว่าผลไม่ขึ้นกับการแก้ flat-field และ pipeline จาก RawA เพียงพอ (ข้อควรระวัง: ต่างกันทั้ง flat-field + วิธีครอป/pad จึงไม่ใช่การทดสอบ flat-field ล้วนๆ) SE weight เกือบเหมือนกันทั้ง 2 แหล่งภาพ (RE ≈ 0.56 > IR ≈ 0.38) แต่ weight นี้ไม่ได้บอกว่า band ไหน "สำคัญกว่า" โดยตรง เพราะ SE ทำงานก่อน normalization ของ backbone และมีเพียง 2 ช่อง

รันซ้ำ: `.venv\Scripts\python.exe -m src.preprocess_ss --verify-content` แล้ว `.venv\Scripts\python.exe -m src.train_dl_kfold --variant nir_SE_ss` (~8.5 นาที)

### งานที่ 2 — ตรวจ Grad-CAM เชิงปริมาณกับ mask ตำหนิ

ทำบนโมเดลจากงานที่ 1 (เพราะ mask ตรงกับเรขาคณิต SingleSpectra) ใช้เฉพาะภาพ Defect ที่อยู่ใน **test fold ของโมเดลนั้น** (out-of-fold เท่านั้น ห้ามใช้ภาพที่โมเดลเคยเห็นตอนเทรน):
1. สร้าง Grad-CAM แล้ว resize mask ให้ตรงขนาด
2. วัด **Pointing Game** (จุดที่ Grad-CAM สูงสุดตกในตำหนิไหม → hit rate %) และ **IoU** ระหว่าง Grad-CAM ที่ threshold (เช่น top 20% ของค่า) กับ mask
3. รายงานแยกตาม Internal vs Surface defect และตามระดับความรุนแรง Kleynen Class 1-4
4. เซฟผลตัวเลขลง `outputs/dl/gradcam_mask_eval.json` และภาพตัวอย่าง (overlay) ลงโฟลเดอร์ที่อยู่ใน `.gitignore` (มีเนื้อภาพ dataset — ห้ามขึ้น GitHub)

เหตุผล: เปลี่ยน Grad-CAM จาก "ดูภาพแล้วบอกว่าดูถูกจุด" เป็นตัวเลขที่พิสูจน์ได้ — Makieu et al. (2025) หัวข้อ 2.7 เขียนว่าคำอธิบายควรตรวจกับขอบเขตตำหนิจริง แต่งานเขาไม่ได้ทำ

#### สถานะ: [x] เสร็จแล้ว 2026-10-12 — **ผลไม่สนับสนุนข้อสรุปเดิมที่ว่า "Grad-CAM ยืนยันว่าโมเดลโฟกัสตำหนิ"**

สคริปต์: `src/gradcam_mask_eval.py` (รัน: `.venv\Scripts\python.exe -m src.gradcam_mask_eval`) ผลเต็มราย 246 ภาพที่ `outputs/dl/gradcam_mask_eval.json`, overlay ตัวอย่าง 12 ภาพที่ `outputs/dl/gradcam_mask/` (อยู่ใน `.gitignore` — มีเนื้อภาพ dataset ห้ามขึ้น GitHub)

**วิธีวัด**: ใช้โมเดล `nir_SE_ss` 5 fold; Grad-CAM (เทียบ class จริง) ของภาพ Defect แต่ละภาพคำนวณด้วย **โมเดลของ fold ที่ภาพนั้นเป็น test เท่านั้น (out-of-fold)** ครบ 246/246 ภาพ ภาพละ 1 ครั้ง; Pointing Game = จุดสูงสุดของ CAM ตกใน mask; IoU = CAM top-20% ของพิกเซล vs mask; mask ผ่านการ pad+resize แบบเดียวกับภาพ (ตรวจ overlay แล้วว่า mask ตรงตำหนิจริง) **ต้องมี null baseline**: เอา CAM เดิมไปเทียบกับ mask ของภาพอื่นที่สุ่ม (20 ครั้ง/ภาพ) — แบบ "สุ่มจาก class เดียวกัน" (same-cls) คือตัวเทียบที่ยุติธรรม เพราะ mask Internal ใหญ่ (~9% ของภาพ) แต่ Surface เล็กมาก (~2%) การสุ่มจากทุก class จึงให้ null ที่สูงเกินจริงของ Surface (ผลรอบแรกที่ใช้ null แบบนั้นทำให้ดูเหมือน Surface "แย่กว่าสุ่ม" ซึ่งเป็นการเทียบที่ไม่ยุติธรรม — แก้แล้ว)

| กลุ่ม | n | Pointing hit [95% CI] | null same-cls | IoU | null same-cls |
|---|---|---|---|---|---|
| ทั้งหมด | 246 | 49.6% [43.4, 55.8] | 42.0% | 0.209 | 0.186 |
| **Internal_Defect** | 102 | **85.3%** [77.1, 90.9] | 71.5% | 0.390 | 0.339 |
| **Surface_Defect** | 144 | **24.3%** [18.0, 31.9] | 21.0% | 0.081 | 0.077 |
| Kleynen 1 (rejection) | 59 | 61.0% | 50.6% | 0.234 | 0.241 |
| Kleynen 2 (bruise) | 56 | 96.4% | 77.6% | 0.485 | 0.367 |
| Kleynen 3 (serious) | 55 | 27.3% | 22.8% | 0.082 | 0.087 |
| Kleynen 4 (slight) | 76 | 22.4% | 22.9% | 0.079 | 0.080 |
| Bruise | 55 | 96.4% | 78.1% | 0.493 | 0.368 |
| Flesh_damage | 24 | 79.2% | 61.2% | 0.287 | 0.297 |
| Rot | 23 | 65.2% | 66.5% | 0.254 | 0.311 |
| Russet | 42 | 28.6% | 21.5% | 0.146 | 0.076 |

(ทำนายถูก vs ผิด: Internal/ถูก 96.4% hit (n=84) vs Internal/ผิด 33.3% (n=18); Surface/ถูก 28.1% (n=114) vs Surface/ผิด 10.0% (n=30))

**ข้อสรุปที่ซื่อสัตย์ (ต้องปรับ narrative ในเล่ม/defense)**
1. **Surface defect: CAM ไม่ได้ชี้ตำแหน่งตำหนิดีกว่าสุ่ม** (24.3% vs null 21.0%, IoU 0.081 vs 0.077 — ต่างกันน้อยกว่า CI) แยกตามชนิดย่อย (n เพียง 7–42 ต่อชนิด CI กว้างมาก) hit rate ของบางชนิดสูงกว่า null เล็กน้อย (Hail, Frost, Scald, Other, Russet +6–10 จุด) แต่ IoU ไม่ต่างจาก null (ยกเว้น Russet ที่ IoU 0.146 vs 0.076) ส่วน Hail_damage_perf และ Limb_rub ต่ำกว่า null — ไม่มีชนิดไหนให้หลักฐานที่ชัดเจนว่าชี้ตำแหน่งได้
2. **Internal defect: ดีกว่า null แต่ไม่มากอย่างที่ตัวเลข 85% บอก** — ส่วนใหญ่มาจากกลุ่มที่ mask ใหญ่และอยู่กลางผล (null same-cls ก็ได้ 71.5% อยู่แล้ว) เหนือ null ราว +14 จุด (Bruise +18, Flesh_damage +18, **Rot +0 — ไม่ดีกว่า null เลย**)
3. จาก overlay: CAM เกือบทุกภาพเป็น **blob เรียบขนาดใหญ่ตรงกลางผล** (feature map สุดท้ายของ EfficientNet-B0 ละเอียดแค่ 7×7) — ละเอียดเกินกว่าจะชี้ตำหนิ Surface ที่เล็ก (<1-2% ของภาพ) ได้ และ Bruise/Flesh_damage ส่วนใหญ่เป็นตำหนิที่ **ทำให้เกิดขึ้นเองบริเวณกลางผล** (notes.txt) จึง "ถูก" ได้ง่ายแม้ CAM เป็นแค่ center prior
4. ผลนี้ **ไม่ได้แปลว่าโมเดลทำนายผิด** (F1 0.86 มาจากการวัดแยกต่างหาก) แต่แปลว่า **Grad-CAM แบบนี้ใช้เป็นหลักฐานว่า "โมเดลเข้าใจตำแหน่งตำหนิ" ไม่ได้** โดยเฉพาะ Surface — ข้อ 3 ใน "ข้อค้นพบสำคัญที่ต้องอภิปรายตอน defense" (Grad-CAM ยืนยันว่าโมเดลโฟกัสตรงตำแหน่งตำหนิ จากภาพตัวอย่างที่เลือก) **ต้องเปลี่ยนเป็น**: ตัวอย่างที่เลือกดูตรงจุดจริง แต่เมื่อวัดทั้ง 246 ภาพเทียบ null พบว่าได้ผลเหนือ null เฉพาะ Internal (Bruise/Flesh_damage) ส่วน Surface ไม่ต่างจากสุ่ม
5. ข้อจำกัด: วัดบนโมเดลเดียว (`nir_SE_ss`) ด้วย Grad-CAM ชั้นสุดท้ายเท่านั้น; ถ้าจะลองต่อให้ชี้ตำแหน่งได้ละเอียดขึ้น ลอง layer ก่อนหน้า (stride 16/ละเอียด 14×14), Grad-CAM++/ScoreCAM ได้ — ยังไม่ได้ทำ

### (ทำทีหลัง ถ้ามีเวลา) งานที่ 3 — Defect Copy-Paste augmentation

ใช้ mask ตัดตำหนิจากภาพ Defect ใน **training fold เท่านั้น** แปะทุกแบนด์ตำแหน่งเดียวกันลงภาพ Healthy ใน training fold เดียวกัน เทียบ K-Fold มี/ไม่มี ถ้าดีขึ้นไม่เกิน ± std ถือว่าไม่ช่วย — ยังไม่ต้องทำจนกว่าผู้ใช้สั่ง

**ก่อนเริ่ม**: backup ผลปัจจุบันตาม workflow หัวข้อ Version Control และแจ้งเวลาประมาณก่อนรันจริง

## งานที่ต้องทำต่อ — ทดสอบความสำคัญของแบนด์แบบตรง (Single-band ablation) + Commit

### เหตุผล

ข้อสรุป "RE สำคัญกว่า IR" ที่ใช้ตอบ objective 3.1 ตอนนี้อิงจาก SE weight อย่างเดียว แต่ SE weight ออกมา RE ≈ 0.56, IR ≈ 0.38 **เกือบเท่ากันทุก fold และทั้ง 2 แหล่งภาพ (raw/SingleSpectra)** ความคงที่ระดับนี้ชวนสงสัยว่า SE แทบไม่ได้เรียนรู้จากข้อมูล และหมายเหตุในงานที่ 1 ก็ระบุแล้วว่า weight นี้บอกความสำคัญของแบนด์โดยตรงไม่ได้ (SE อยู่ก่อน normalization ของ backbone) → ต้องพิสูจน์ด้วยการเทรนแบนด์เดียวแล้วเทียบ ซึ่งเป็นหลักฐานตรงที่สุด

### งานที่ 4 — เทรน RE-only และ IR-only (5-fold)

1. เพิ่ม channel group `re` (RE อย่างเดียว) และ `ir` (IR อย่างเดียว) ใน `CHANNEL_GROUPS` (`src/config.py`) — ตรวจว่า `build_model` รองรับ in_channels=1 (ปรับ conv แรกของ EfficientNet-B0 จาก pretrained weight แบบเดียวกับที่ทำกับ 2/4 channel)
2. เพิ่ม variant `re_only` และ `ir_only` ใน `VARIANT_SPECS` ของ `train_dl_kfold.py` — **ทุกอย่างเหมือน `nir` เดิม** (ไม่มี SE เพราะ SE บน 1 ช่องไม่มีความหมาย, มี augmentation, WeightedRandomSampler, StratifiedKFold n=5 random_state=42, แหล่งภาพ raw) ต่างแค่จำนวนแบนด์ → fold ตรงกับ `nir`/`nir_SE` ทุกลูก เทียบแบบจับคู่ได้
3. รายงาน Macro F1 / Accuracy mean±std + per-fold ของ `re_only`, `ir_only` เทียบกับ `nir` (RE+IR, 0.843±0.023) ในตารางเดียว + per-class F1 (Healthy/Internal/Surface) ของแต่ละ variant ถ้าทำได้จากผลเดิมโดยไม่ต้องเทรนใหม่
4. เทียบรายคู่ fold (เหมือนที่ทำกับ `nir_SE_ss`) ระบุว่าผลต่างแต่ละคู่เครื่องหมายเดียวกันทุก fold หรือไม่
5. **วิธีอ่านผล**:
   - `re_only` ชนะ `ir_only` ชัด (ทุก/เกือบทุก fold, ต่างเกิน std) → ข้อสรุป "RE สำคัญกว่า IR" ยืนได้ด้วยหลักฐานตรง (SE weight เป็นแค่ข้อมูลเสริม)
   - ต่างไม่เกิน std หรือ `ir_only` ชนะ → **ต้องตัดข้อสรุป "RE สำคัญกว่า IR" ออก** และหยุดใช้ SE weight เป็นหลักฐานความสำคัญของแบนด์
   - ถ้า RE+IR (`nir`) ดีกว่าแบนด์เดียวทั้งคู่ → สรุปได้ว่า 2 แบนด์ NIR เสริมกัน
6. หมายเหตุให้ใส่ผลไว้ในหัวข้อนี้ และอัปเดตข้อสรุปเรื่อง SE weight ในส่วนอื่นของไฟล์ที่อ้างถึง "RE สำคัญกว่า IR" ให้ตรงกับผล

รัน (ประมาณ 2 × 8–12 นาที): แจ้งเวลาประมาณก่อนรันจริง
```
.venv\Scripts\python.exe -m src.train_dl_kfold --variant re_only
.venv\Scripts\python.exe -m src.train_dl_kfold --variant ir_only
```

#### สถานะงานที่ 4: [x] เสร็จแล้ว 2026-10-13 — **ผล: RE-only ≈ IR-only ≈ RE+IR → ต้องตัดข้อสรุป "RE สำคัญกว่า IR" ออก**

โค้ด: `config.py` เพิ่ม channel group `re`/`ir`; `train_dl_kfold.py` เพิ่ม `re_only`/`ir_only` (และเก็บ per-class F1 + confusion matrix + prediction ต่อ fold ไว้ใน `kfold_results.json` สำหรับ variant ใหม่ๆ); `src/band_ablation_report.py` สร้างตารางเปรียบเทียบ → `outputs/dl/band_ablation.json` (รัน: `.venv\Scripts\python.exe -m src.band_ablation_report`) ทดสอบ `build_model` ที่ in_channels=1 ผ่าน, fold ตรงกับ `nir` ทุกลูก (ใช้ `build_pool("raw")` + StratifiedKFold n=5 seed 42 เหมือนเดิม) เวลารันจริง ~8.5 + 9.2 นาที

| Variant | Macro F1 | Accuracy | per-fold F1 (fold 0-4) |
|---|---|---|---|
| RE only | 0.839 ± 0.034 | 90.4% ± 2.8% | 0.880, 0.820, 0.859, 0.782, 0.852 |
| IR only | 0.844 ± 0.049 | 90.1% ± 3.2% | 0.864, 0.844, 0.915, 0.764, 0.835 |
| RE+IR (`nir`) | 0.843 ± 0.023 | 89.9% ± 2.0% | 0.845, 0.843, 0.873, 0.803, 0.852 |

ผลต่างรายคู่ fold (แอปเปิลชุดเดียวกันในแต่ละ fold): RE − IR เฉลี่ย **−0.006** (sd 0.033; RE ชนะ 3/5 fold; diffs = +0.016, −0.024, −0.056, +0.018, +0.017); RE+IR − RE เฉลี่ย +0.005 (4/5 fold, แต่ fold 0 = −0.035); RE+IR − IR เฉลี่ย −0.001 (2/5 fold) — paired-t p = 0.72 / 0.69 / 0.94 (n=5 บอกได้แค่แนวโน้ม) McNemar บน prediction out-of-fold รวม 819 ภาพ: RE vs IR p=0.90 (RE ถูกอย่างเดียว 34, IR ถูกอย่างเดียว 32); RE+IR vs RE p=0.79; RE+IR vs IR p=0.64

Per-class F1 (pooled out-of-fold ทั้ง 819 ภาพ):

| Variant | Healthy | Internal_Defect | Surface_Defect |
|---|---|---|---|
| RE only | 0.955 | 0.792 | 0.769 |
| IR only | 0.945 | 0.821 | 0.770 |
| RE+IR (`nir`)* | 0.951 | 0.808 | 0.800 |

\* **แถว `nir` และ McNemar ที่เกี่ยวกับ `nir` มาจากการเทรนอีกรอบ** (checkpoint 5 fold จากตอนทำ ensembling — recipe/fold เดียวกัน แต่ไม่ใช่รอบเดียวกับตัวเลข 0.843±0.023 ในตารางแรก เพราะตอนนั้นไม่ได้เก็บ prediction) F1 ต่อ fold ของ 2 รอบนี้ต่างกัน: รอบ `kfold_results.json` = 0.845, 0.843, 0.873, 0.803, 0.852 vs รอบ checkpoint = 0.909, 0.885, 0.854, 0.801, 0.817 (**ต่างสูงสุด 0.065 ใน fold เดียว** ทั้งที่ recipe เหมือนกัน ที่มา: GPU kernel ที่ไม่ deterministic และ augmentation ใช้ `random` ของ Python ที่ไม่ได้ seed) ความแปรปรวนจากการรันซ้ำนี้ใหญ่กว่าความต่าง RE vs IR (−0.006) กว่า 10 เท่า — ข้อความ "matches kfold_results.json" ที่เคยพิมพ์ใน log ของ `ensemble_kfold.py` ไม่เป็นจริง (แก้ข้อความแล้ว)

**ข้อสรุป (ตามเกณฑ์วิธีอ่านผลที่วางไว้)**
1. **ไม่มีหลักฐานว่า RE สำคัญกว่า IR** — ต่างกันเพียง −0.006 F1 (เล็กกว่า std 0.034/0.049 มาก), ผลต่างรายคู่ fold กลับเครื่องหมาย, McNemar p=0.90 → **ตัดข้อสรุป "RE สำคัญกว่า IR" ออกจากเล่มและ defense** และ **เลิกใช้ SE weight เป็นหลักฐานความสำคัญของแบนด์** (weight RE 0.56 vs IR 0.38 ที่นิ่งทุก fold ไม่สะท้อนความต่างของ performance จริง) ข้อสรุปเรื่อง objective 3.1 ระดับ "ภายใน NIR" ที่ตอบได้จริงคือ: แต่ละแบนด์ NIR เดี่ยวๆ ให้ผลระดับเดียวกัน
2. **RE+IR ไม่ได้ดีกว่าแบนด์เดียว** (0.843 vs 0.839/0.844 — ผลต่างอยู่ในระดับ noise) จึง **สรุปไม่ได้ว่า 2 แบนด์ NIR "เสริมกัน"** มีเพียงข้อสังเกตอ่อนๆ ว่า RE+IR std ต่ำสุด (0.023 vs 0.034/0.049) และ Surface F1 สูงกว่าเล็กน้อย (0.800 vs 0.77 — แต่มาจากรอบเทรนที่ 2 และต่างกันระดับ noise) อย่านำไปอ้างเป็นผล
3. สิ่งที่ยังยืนอยู่: NIR (ไม่ว่า RE, IR หรือทั้งคู่ ≈0.84) ดีกว่า visible-only (0.666±0.027) ชัดเจนในระดับเกิน std — ใช้ตอบ objective หลักได้ แต่การทดสอบนี้ **ไม่ได้ทดสอบแบนด์เดี่ยวของ visible (BL, GR)** จึงยังบอกไม่ได้ว่า visible ตัวไหนด้อยกว่า
4. ข้อจำกัด: 5 fold, 1 รอบเทรนต่อ variant (ยกเว้น `nir` 2 รอบ) และความแปรปรวนจากการรันซ้ำต่อ fold ใหญ่ถึง ~0.06 — ความต่างระดับ ±0.02 จึงแยกจาก noise ไม่ได้ ถ้าจะสรุปความต่างเล็กๆ ระหว่างแบนด์ต้องรันหลาย seed และ seed `random` ของ augmentation ให้ครบก่อน (ยังไม่ได้ทำ)

### งานที่ 5 — Commit + push ขึ้น GitHub (ค้าง 2 ชุด + ชุดนี้)

ทำ**หลัง**งานที่ 4 เสร็จ ตาม workflow หัวข้อ Version Control:
1. ก่อน add: รัน `git status` ตรวจว่าไม่มีไฟล์ที่มีเนื้อภาพ dataset (`Processed_Dataset*/`, `outputs/dl/gradcam*/`, `*.pt`, `*.joblib`, `backups/`) หลุดเข้ามา — ถ้ามีโฟลเดอร์ output ใหม่ที่มีภาพ ให้เพิ่มเข้า `.gitignore` ก่อน
2. **ห้าม `git add -A`** — add ทีละไฟล์/โฟลเดอร์ที่ตั้งใจ: `src/`, `.gitignore`, `PROJECT_PLAN.md`, `outputs/dl/kfold_results.json`, `outputs/dl/gradcam_mask_eval.json`, `outputs/ss_mapping*.json` และไฟล์ผล json ใหม่จากงานที่ 4
3. แยก commit ตามชุดงานให้อ่าน history ง่าย (เช่น 1: nir_SE + nir_no_aug, 2: SingleSpectra + Grad-CAM mask eval, 3: single-band ablation) — ถ้าแยกไม่ได้เพราะแก้ไฟล์เดียวกัน ให้รวมเป็น commit เดียวพร้อมข้อความที่ระบุครบ
4. push ขึ้น `main` ของ github.com/Chinnawan/project_nir แล้วรายงาน hash ของ commit

## งานที่ต้องทำต่อ — รันข้ามคืน: Multi-seed ทุก variant ด้วย protocol เดียว (2026-10-13)

### เหตุผล

งานที่ 4 พบว่าเทรน `nir` สูตรเดิม 2 รอบ F1 ต่อ fold ต่างกันได้ถึง **0.065** (GPU non-deterministic + `random` ของ Python ที่ augmentation ใช้ไม่ได้ seed — ตรวจแล้ว: `dataset.py` ใช้ `random.random()`/`random.choice()` แต่มีแค่ `torch.manual_seed`) ความแกว่งนี้ **ใหญ่กว่า** ความต่างหลายตัวที่เคยสรุปว่า "ช่วย":

| ข้อสรุป | ความต่าง | สถานะ |
|---|---|---|
| NIR > Visible | +0.18 | น่าจะจริง (ใหญ่กว่า noise มาก) |
| NIR > Vis+NIR (full) | +0.12 | น่าจะจริง |
| DL > classical ML | +0.2 ขึ้นไป | น่าจะจริง แต่ classical ML ยังวัดแบบ single split |
| Augmentation ช่วย (nir vs nir_no_aug) | +0.026 | **ยังไม่แน่** → ผล 3 seed: +0.025 ก้ำกึ่ง ยังไม่ยืนยัน (ดู "ผล multi-seed") |
| SE ช่วย (nir_SE vs nir) | +0.019 | **ยังไม่แน่** → ผล 3 seed: +0.003 ไม่ต่าง (ดู "ผล multi-seed") |
| RE vs IR | −0.006 | ไม่ต่าง |

เป้าหมายรอบนี้: ทุกตัวเลขในเล่มมาจาก protocol เดียว ทำซ้ำได้ และแยก "ผลจริง" ออกจาก "ความแกว่งของการเทรน" ได้ + เติมแบนด์เดี่ยวให้ครบ 4 แบนด์ (ตอบ objective 3.1 ทีละแบนด์)

### ขั้นที่ A — ทำให้เทรนซ้ำได้ (ทำก่อนทุกอย่าง)

1. ฟังก์ชัน `set_seed(seed)` เรียกตอนเริ่มทุก (variant, seed, fold): seed `random`, `numpy`, `torch`, `torch.cuda` (all), ตั้ง `torch.backends.cudnn.deterministic=True`, `benchmark=False`, `os.environ["PYTHONHASHSEED"]`; DataLoader ใช้ `generator` ที่ seed แล้ว + `worker_init_fn` ที่ seed `random`/`numpy` ในแต่ละ worker; WeightedRandomSampler ใช้ generator ที่ seed แล้ว
2. ลอง `torch.use_deterministic_algorithms(True, warn_only=True)` (ถ้าต้องใช้ ให้ตั้ง `CUBLAS_WORKSPACE_CONFIG=:4096:8`) — ถ้ามี op ที่ไม่รองรับ ให้ใช้ warn_only และบันทึกชื่อ op ไว้
3. **Fold split คงที่** (StratifiedKFold n=5, random_state=42 เหมือนเดิม) — seed ใหม่เปลี่ยนแค่การเทรน (init head, ลำดับ batch, augmentation) fold จึงยังเทียบรายคู่กับผลเดิมได้
4. **ทดสอบความทำซ้ำได้**: รัน `nir` fold 0 seed 0 **2 ครั้ง** (จำกัด epoch ให้สั้นได้ เช่น 5 epoch) เทียบ val loss/test F1 → บันทึกผลว่าตรงกันเป๊ะหรือต่างเท่าไร ถ้ายังต่าง ให้บันทึกขนาดความต่างไว้เป็นข้อจำกัด แล้วไปต่อ (multi-seed จะครอบคลุมส่วนที่เหลือ)

### ขั้นที่ B — เพิ่ม variant ใหม่

- `bl_only` (BL), `gr_only` (GR): recipe เหมือน `re_only`/`ir_only` (ไม่มี SE, มี aug, raw) → ได้แบนด์เดี่ยวครบ 4 แบนด์
- `gr_nir` (GR+RE+IR, ไม่มี SE, มี aug, raw): ตอบคำถามที่จะถามอาจารย์ว่า "เลือก visible บางแบนด์แล้วช่วย NIR ไหม"
- เพิ่ม channel group `bl`, `gr`, `gr_nir` ใน `CHANNEL_GROUPS`

### ขั้นที่ C — สคริปต์รันข้ามคืน

สร้าง `src/run_multiseed.py` + `run_overnight.bat` (ให้ผู้ใช้เปิดรันเองใน terminal แยก ไม่ผูกกับ session ของ Claude Code):
1. **Seeds = 0, 1, 2** × 5 fold × variant ตามลำดับความสำคัญ (ถ้าเครื่องดับกลางคืน ของสำคัญจะเสร็จก่อน):
   1. classical ML (SVM/RF/LightGBM) 5-fold ด้วย fold เดียวกัน — เร็ว, ทำครั้งเดียวพอ (deterministic) เก็บ per-fold + prediction
   2. `nir`, `nir_SE`, `nir_no_aug`
   3. `re_only`, `ir_only`, `bl_only`, `gr_only`
   4. `visible`, `full`, `full_SE`, `gr_nir`
2. **ห้ามทับ `kfold_results.json` เดิม** — เขียนผลลง `outputs/dl/multiseed_results.json` **ทันทีหลังจบแต่ละ fold** (เก็บ F1, acc, per-class F1, confusion matrix, epochs, SE weights ถ้ามี, และ prediction ต่อภาพ)
3. **Resume ได้**: ถ้า (variant, seed, fold) มีในไฟล์แล้ว ให้ข้าม — สั่งรันซ้ำหลังเครื่องดับก็ต่อจากจุดเดิม
4. ถ้า fold ไหน error ให้ log แล้วไปต่อ ไม่ให้ทั้ง batch ตาย log ทั้งหมดลง `outputs/logs/overnight_YYYYMMDD.log` (พร้อมเวลาเริ่ม/จบแต่ละ fold)
5. ไม่ต้องเซฟ checkpoint ทุกตัว (เปลืองที่) ยกเว้น `nir_SE` seed 0 (เผื่อ demo / Grad-CAM)
6. ก่อนผู้ใช้รันจริง: **smoke test** (2 variant × 1 seed × 1 fold × 2 epoch) ว่ารันจบ เขียน json ถูก และ resume ข้ามได้ แล้วลบผล smoke test ออกจาก json
7. ประเมินเวลาจริงจาก smoke test แล้วแจ้งผู้ใช้ (คาดราว 12 variant × 3 seed × ~9 นาที ≈ 5–7 ชม. ถ้า deterministic mode ช้าลง ให้บอก)

### ขั้นที่ D — สคริปต์สรุปผล (เขียนไว้ก่อน รันตอนเช้า)

`src/multiseed_report.py` → `outputs/dl/multiseed_summary.json` + พิมพ์ตาราง:
1. ต่อ variant: Macro F1 / Accuracy = mean ± std ของค่าเฉลี่ย 5-fold ข้าม 3 seed, และ mean ± std ข้ามทั้ง 15 run + per-class F1 (pooled out-of-fold)
2. **การเปรียบเทียบรายคู่** (จับคู่ด้วย seed+fold เดียวกัน): nir_SE−nir, nir−nir_no_aug, nir−full, nir−visible, re−ir, gr_nir−nir, แบนด์เดี่ยวทั้ง 4 แบนด์ไล่คู่, DL ตัวดีสุด−classical ตัวดีสุด → รายงานผลต่างเฉลี่ย, 95% CI (bootstrap ข้าม 15 คู่), และนับว่ากี่คู่จาก 15 ที่เครื่องหมายเดียวกัน
3. **เกณฑ์ตัดสิน**: ถือว่า "ต่างจริง" เมื่อ CI ไม่คร่อม 0 **และ** เครื่องหมายเดียวกัน ≥ 12/15 คู่ — ไม่เข้าเกณฑ์ให้เขียนว่า "ไม่ต่างเกินความแกว่งของการเทรน"
4. ความแกว่งจาก seed: ต่อ fold, std ข้าม 3 seed (ใช้ตอบว่าการเทรนซ้ำแกว่งแค่ไหนหลังล็อก seed)

**ห้ามสรุปผลใน PROJECT_PLAN.md เอง** — รอสรุปกับผู้ใช้ตอนเช้า

### ขั้นที่ E — (ถ้า batch หลักเสร็จก่อนเช้า) Grad-CAM ชั้นละเอียดขึ้น

บน checkpoint `nir_SE_ss` เดิม รัน `gradcam_mask_eval` เพิ่มที่ layer ก่อนสุดท้าย (14×14) เทียบ Pointing/IoU กับ null แบบเดิม → `outputs/dl/gradcam_mask_eval_14x14.json` ทำเป็นขั้นสุดท้ายของ `run_overnight.bat`

### สถานะการเตรียม (2026-10-13): ขั้น A–D เสร็จ + smoke test ผ่าน — batch จริงรันเสร็จแล้ว (ดู "ผล multi-seed" ด้านล่าง)

**ขั้น A (ทำซ้ำได้)** — `src/repro.py` (`set_seed`: random/numpy/torch/cuda + `cudnn.deterministic` + `use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG`; generator/worker_init_fn); `dataset.get_dataloaders_from_items(seed=)` ใช้ generator ที่ seed แล้วกับ WeightedRandomSampler/DataLoader; `train_dl_kfold.run_fold(seed=, epochs=, with_history=)` fold split และ inner train/val split ยังคงที่ (random_state=42) seed เปลี่ยนแค่ head init / ลำดับ batch / augmentation
- **ทดสอบความทำซ้ำได้ (`python -m src.run_multiseed --repro-test`, ผลที่ `outputs/dl/repro_test.json`)**: `nir` และ `nir_SE` fold 0 seed 0 รัน 2 ครั้งใน**โปรเซสแยก** (5 epoch) → **ตรงกันแบบ bit-identical** (train/val loss ทุก epoch ต่าง 0.0, prediction ต่าง 0 ภาพ, test F1 0.813853 = 0.813853 และ 0.808653 = 0.808653), ไม่มี warning จาก deterministic mode (ไม่มี op ที่ไม่รองรับ) — ข้อจำกัด: ทดสอบเพียง 5 epoch / 1 fold / 2 โครงสร้างบน GPU เครื่องนี้; ผลไม่ได้ยืนยันว่าเครื่องอื่น/เวอร์ชัน torch อื่นจะได้ตัวเลขเดียวกัน
- ราคาของ deterministic mode: ~4.1–5.0 วินาที/epoch (re_only 4.14, nir 4.77, nir_SE 4.87, gr_nir 4.94, full 5.04) ช้ากว่าแบบเดิม (~4.2 วินาที/epoch ของ nir) ราว 13%

**ขั้น B** — channel group `bl`, `gr`, `gr_nir` ใน `config.py`; variant `bl_only`, `gr_only`, `gr_nir` ใน `VARIANT_SPECS` (recipe เหมือน `re_only`: ไม่มี SE, มี aug, raw) รวม DL 11 variant (นับ classical เป็นกลุ่มแยก — แผนเดิมเขียน 12)

**ขั้น C** — `src/run_multiseed.py` + `run_overnight.bat`: seeds 0,1,2 × 5 fold; ลำดับ classical → [nir, nir_SE, nir_no_aug] → [re, ir, bl, gr] → [visible, full, full_SE, gr_nir] (ในแต่ละกลุ่ม: seed → variant → fold); เขียน `outputs/dl/multiseed_results.json` ด้วย atomic write หลังจบทุก fold (ไม่แตะ `kfold_results.json`); resume ได้ (ข้าม key `variant|seedN|foldK` ที่มีแล้ว); fold ที่ error ถูก log + บันทึกใน `errors` แล้ว batch ไปต่อ (ทดสอบด้วยการจำลองความล้มเหลวแล้ว; รันซ้ำจะลอง fold ที่ล้มใหม่); log ลง `outputs/logs/overnight_YYYYMMDD.log` พร้อมเวลาเริ่ม/จบและ ETA ต่อ fold; checkpoint เก็บเฉพาะ `nir_SE` seed 0 (`outputs/dl/kfold_ckpt/nir_SE_seed0_fold*.pt`); classical (SVM/RF/LightGBM, hyperparameter เดิมของ Phase 2, LightGBM ตั้ง `deterministic=True`) รัน 5 fold ด้วย fold เดียวกันครั้งเดียว เก็บเป็น seed 0 (deterministic) และใช้ซ้ำทุก seed ตอนจับคู่; ขอให้ Windows ไม่ idle-sleep ระหว่างรัน (`SetThreadExecutionState` ระดับโปรเซส หมดเมื่อโปรเซสจบ — ไม่แก้ power setting ของเครื่อง; ปิดฝาโน้ตบุ๊กยังทำให้เครื่องหลับ)
- **Smoke test (ผ่าน)** — `run_overnight.bat --smoke`: classical fold 0 + `nir`/`nir_SE` seed 0 fold 0 2 epoch เขียน json ถูก, รันซ้ำแล้วข้ามทั้งหมด ("0 runs to do"), ไม่มี checkpoint หลุดจาก smoke; ผล smoke อยู่ใน json แยกและลบทิ้งแล้ว (`multiseed_results.json` จริงยังไม่ถูกสร้าง) — ตัวเลข F1 ของ `nir` fold 0 2 epoch (0.7455) ตรงกันทุกครั้งที่รัน
- ส่วนเบี่ยงจากแผน: bat รัน `multiseed_report` ต่อท้ายอัตโนมัติ (บันทึกที่ `outputs/logs/overnight_report.txt`) เพื่อให้ตอนเช้ามีตารางพร้อม — สคริปต์นี้แค่พิมพ์ตัวเลข/บอก verdict ตามเกณฑ์ ไม่ได้สรุปผลลงแผน; **ขั้น E (Grad-CAM 14×14) ยังไม่ได้ทำ** (อยู่นอกขอบเขตที่สั่ง "ขั้น A ถึง D")

**เวลาประมาณ** (วัดจริง: วินาที/epoch × epoch เฉลี่ยของแต่ละ variant จากรอบ k-fold ก่อนหน้า + overhead ~3 วินาที/fold; `bl_only`, `gr_only`, `gr_nir` ใช้ค่า epoch ของ variant ใกล้เคียง): ~8–11.5 นาที/(variant, seed) × 11 variant × 3 seed ≈ **5.1 ชั่วโมง** (ช่วงที่เป็นไปได้ ~4.1–6.6 ชม. ขึ้นกับจำนวน epoch ที่ early stopping หยุด และอุณหภูมิ/พลังงานของโน้ตบุ๊ก GPU RTX 3050 Ti 4GB) + classical ~1 นาที ถ้ารันเป็น 165 fold-run ทั้งหมด

**ขั้น D** — `src/multiseed_report.py` (รัน: `.venv\Scripts\python.exe -m src.multiseed_report` → `outputs/dl/multiseed_summary.json`): ตารางต่อ variant, การเปรียบเทียบรายคู่ (จับคู่ seed+fold, bootstrap 95% CI 10,000 ครั้ง, นับคู่ที่เครื่องหมายเดียวกัน, เกณฑ์ "ต่างจริง" = CI ไม่คร่อม 0 และ ≥ 12/15), seed noise ต่อ fold — ทดสอบกับข้อมูลจำลองที่ฝังความต่างจริงไว้ (ตัดสินถูก) และกับข้อมูล smoke ที่ไม่ครบ (แก้ข้อบกพร่องที่พบ: คู่ที่มีไม่ครบ 15 คู่จะไม่ถูกตัดสิน แสดงเป็น INCOMPLETE แทนที่จะอ้างว่า "REAL") ข้อควรระวังในการอ่าน: 15 คู่ใช้ 5 fold ร่วมกัน CI จึงเป็นค่าประมาณ; classical ถูก broadcast ข้าม seed ตอนจับคู่

**คำสั่งเปิดรัน** (เปิด terminal แยก; รันซ้ำคำสั่งเดิมเพื่อทำต่อหลังเครื่องดับ):
```
cd /d D:\nir+rgb_project\project\CAPA_AppleQualityGradingImageDatabase
run_overnight.bat
```

### ผล multi-seed (รันข้ามคืน เสร็จ 2026-10-13) — **ตัวเลขอ้างอิงสำหรับเล่ม (แทนตัวเลขรอบเดียวด้านบนทั้งหมด)**

รันจริง 01:15–06:29 น. = **5.2 ชม.** (ประมาณไว้ 5.1) สำเร็จ **180/180 run** (DL 11 variant × 3 seed × 5 fold = 165 + classical 3 โมเดล × 5 fold = 15), **error 0**, ไม่มี run ชนเพดาน 100 epoch, warning ใน log มีเพียง deprecation ของ sklearn (`SVC(probability=True)`) ผลดิบ `outputs/dl/multiseed_results.json`, สรุป `outputs/dl/multiseed_summary.json`, รายงาน `outputs/logs/overnight_report.txt` (log ไม่ขึ้น git) protocol: seed ล็อก + deterministic kernels (ตรวจแล้วว่าทำซ้ำได้ bit-identical, `outputs/dl/repro_test.json`), fold split คงที่, ภาพ raw, ไม่มี ensemble/focal loss

**ตารางหลัก** (macro-F1 / accuracy = ค่าเฉลี่ยของ mean 5-fold ข้าม 3 seed; ± แรก = std ระหว่าง seed; "std ทุก run" = std ของ F1 ระดับ fold-run ทั้ง 15 ค่า สะท้อนความต่างระหว่าง fold + seed; per-class = F1 รวม out-of-fold เฉลี่ยข้าม seed)

| Variant | Macro F1 | Accuracy | std ทุก run | Healthy / Internal / Surface F1 |
|---|---|---|---|---|
| RE only | 0.853 ± 0.007 | 90.6% | 0.032 | 0.949 / 0.806 / 0.803 |
| NIR + SE (`nir_SE`) | 0.847 ± 0.008 | 90.6% | 0.034 | 0.953 / 0.790 / 0.798 |
| NIR (RE+IR) | 0.843 ± 0.008 | 90.2% | 0.044 | 0.950 / 0.794 / 0.786 |
| IR only | 0.843 ± 0.006 | 89.8% | 0.036 | 0.943 / 0.825 / 0.761 |
| NIR ไม่มี augmentation | 0.818 ± 0.011 | 88.9% | 0.050 | 0.940 / 0.751 / 0.769 |
| GR + NIR | 0.810 ± 0.009 | 88.4% | 0.044 | 0.938 / 0.729 / 0.765 |
| full (4 แบนด์) | 0.754 ± 0.012 | 85.0% | 0.043 | 0.914 / 0.625 / 0.729 |
| BL only | 0.752 ± 0.009 | 84.1% | 0.023 | 0.905 / 0.623 / 0.728 |
| GR only | 0.752 ± 0.019 | 84.2% | 0.041 | 0.907 / 0.645 / 0.709 |
| full + SE | 0.737 ± 0.026 | 83.2% | 0.031 | 0.903 / 0.614 / 0.699 |
| visible (BL+GR) | 0.688 ± 0.015 | 80.3% | 0.042 | 0.885 / 0.526 / 0.651 |
| classical RandomForest | 0.589 (1 run) | 71.3% | 0.045 | 0.809 / 0.552 / 0.406 |
| classical SVM | 0.573 (1 run) | 66.9% | 0.063 | 0.771 / 0.530 / 0.417 |
| classical LightGBM | 0.546 (1 run) | 72.3% | 0.049 | 0.826 / 0.524 / 0.290 |

(classical deterministic จึงรันครั้งเดียวต่อ fold; Surface_Defect เป็น class ที่อ่อนที่สุดในทุกโมเดล)

**การเปรียบเทียบรายคู่** (A − B ใน macro-F1, จับคู่ seed+fold, 15 คู่; CI 95% และเกณฑ์ "ต่างจริง" = CI ไม่คร่อม 0 และ ≥12/15 คู่เครื่องหมายเดียวกัน; คอลัมน์ "fold" = ตรวจซ้ำโดยเฉลี่ย 3 seed ต่อ fold แล้วนับ 5 fold เพราะ 15 คู่ใช้ 5 fold ร่วมกัน CI ของสคริปต์จึงแคบกว่าความจริง)

| คู่ | ผลต่างเฉลี่ย | CI 95% (สคริปต์) | คู่ที่เครื่องหมายเดียวกัน | fold ที่เครื่องหมายเดียวกัน | สรุป |
|---|---|---|---|---|---|
| NIR − full | +0.089 | +0.066..+0.116 | 15/15 | 5/5 | **ต่างจริง** |
| NIR − visible | +0.156 | +0.127..+0.182 | 15/15 | 5/5 | **ต่างจริง** |
| RE − BL / IR − GR / NIR − GR (แบนด์เดี่ยว) | +0.101 / +0.091 / +0.091 | ไม่คร่อม 0 | 15/15, 14/15 | 5/5 | **ต่างจริง** (NIR band ≫ visible band) |
| visible − BL only | −0.065 | — | — | 5/5 | BL+GR รวมกัน**แย่กว่า** BL เดี่ยว |
| RE only − RandomForest (DL ดีสุด − classical ดีสุด) | +0.264 | +0.239..+0.290 | 15/15 | 5/5 | **ต่างจริง** (+0.19..+0.33 ทุก fold) |
| RE − IR | +0.010 | −0.012..+0.030 | 10/15 | 4/5 | ไม่ต่างเกินความแกว่ง |
| BL − GR | −0.000 | −0.020..+0.019 | 5/15 | — | ไม่ต่าง |
| `nir_SE` − `nir` | +0.003 | −0.012..+0.018 | 7/15 | 3/5 | **ไม่ต่าง (SE ไม่ช่วย)** |
| `nir` − `nir_no_aug` (augmentation) | +0.025 | −0.000..+0.047 | 12/15 | 4/5 (fold 0 = −0.013) | **ก้ำกึ่ง ยังไม่ยืนยัน** (CI แตะ 0; ระดับ fold CI −0.009..+0.059) |
| GR+NIR − `nir` | −0.034 | −0.054..−0.012 | 12/15 | 4/5 (CI ระดับ fold −0.074..+0.007) | สัญญาณว่าเพิ่ม GR ทำให้แย่ลง แต่ **ยังไม่ใช่ข้อสรุป** |

**ข้อสรุปสำหรับเล่ม/defense (ตัวเลขนี้แทนข้อสรุปเดิม)**
1. **ยืนยันแล้ว**: (ก) NIR ดีกว่า visible และ full 4 แบนด์อย่างชัดเจนทุก fold (ข) แบนด์ NIR เดี่ยว (RE หรือ IR) ให้ผลระดับเดียวกับ RE+IR (≈0.84–0.85) และดีกว่า BL หรือ GR เดี่ยวราว +0.09–0.10 (ค) DL ดีกว่า classical ML (ที่ดีที่สุด RF 0.589) ราว +0.26 ทุก fold จึงตอบได้ว่า "ทำไมต้อง deep learning" ด้วยตัวเลขที่วัดด้วย protocol เดียวกัน
2. **ตัดออก**: "SE block ช่วย" และ "nir_SE คือโมเดลที่ดีที่สุด (0.862)" — ไม่มีโมเดลไหนในกลุ่ม NIR (RE, IR, RE+IR, +SE) ที่ดีกว่ากันเกินความแกว่ง ตัวเลข 0.862 เป็นรอบเดียวที่โชคดี (เฉลี่ย 3 seed = 0.847) ถ้าต้องระบุ "โมเดลหลัก" ให้ใช้ `nir` (RE+IR) หรือ `nir_SE` พร้อม mean±std ของ 3 seed และบอกตรงๆ ว่าไม่ต่างจากแบนด์เดี่ยว; SE weight ใช้เป็นหลักฐานความสำคัญของแบนด์ไม่ได้ (ดูงานที่ 4)
3. **ยังสรุปไม่ได้**: augmentation ช่วยหรือไม่ (แนวโน้มบวก +0.025 แต่ CI แตะ 0 และ fold 0 ย้อนทาง — เขียนเล่มว่า "แนวโน้มช่วยเล็กน้อย ยังไม่มีนัยสำคัญ" ไม่ใช่ "ช่วยจริง" ซึ่งเป็นข้อสรุปเดิมจาก ablation รอบเดียวที่ต้องแก้) และ GR ทำให้ NIR แย่ลงหรือไม่
4. **ข้อสังเกตใหม่ที่ยังไม่ทราบสาเหตุ**: F1 ลดลงตามจำนวนช่อง — 1 ช่อง NIR 0.853 → 2 ช่อง 0.843 → 3 ช่อง (GR+NIR) 0.810 → 4 ช่อง 0.754 และ BL+GR (0.688) แย่กว่าแต่ละแบนด์เดี่ยว (0.752, 5/5 fold) สมมุติฐาน (**ยังไม่ได้ทดสอบ**): conv ชั้นแรกถูก init ด้วยน้ำหนัก RGB เฉลี่ยแล้วทำซ้ำข้ามช่อง ทำให้สเกลสัญญาณเข้าโตตามจำนวนช่อง หรืออาจเป็น overfitting จากข้อมูลน้อย — อย่าเขียนในเล่มเป็นข้อเท็จจริงจนกว่าจะทดสอบ (เช่น ปรับสเกล init เป็น 3/จำนวนช่อง)
5. **ความแกว่งหลังล็อก seed ยังมีอยู่**: std ของ F1 ต่อ fold ข้าม 3 seed เฉลี่ย 0.020–0.036 (สูงสุด 0.061, `gr_only` fold 2) การล็อก seed ทำให้ทำซ้ำได้ ไม่ได้ทำให้ผลนิ่ง — ความต่างระหว่าง variant ที่เล็กกว่า ~0.03 อย่าเชื่อจากการรันครั้งเดียว และ F1 ระดับ fold-run มี std 0.03–0.06 (ต้องรายงานคู่กับค่าเฉลี่ย)
6. ตัวเลข k-fold รอบเดียวเดิมใกล้เคียงค่าเฉลี่ย 3 seed (nir 0.843→0.843, nir_no_aug 0.817→0.818, ir_only 0.844→0.843, full_SE 0.734→0.737) ยกเว้นที่ได้ค่าสูง/ต่ำกว่าปกติ (nir_SE 0.862→0.847, re_only 0.839→0.853, full 0.724→0.754, visible 0.666→0.688)

**ข้อจำกัด**: (ก) 3 seed × 5 fold บน 819 ภาพ; fold ถูกใช้ร่วมกันข้ามคู่ จึงตีความ CI ของสคริปต์ว่าเป็นค่าประมาณ (ข) classical รันครั้งเดียว (ค) ทำซ้ำได้ bit-identical ตรวจเพียง 5 epoch/1 fold/2 โครงสร้างบน GPU เครื่องนี้ (RTX 3050 Ti) ไม่รับประกันข้ามเครื่อง/เวอร์ชัน torch (ง) ขั้น E (Grad-CAM 14×14) ยังไม่ได้ทำ

### หลังรันเสร็จ (ตอนเช้า) — ทำแล้ว

commit + push ตาม workflow Version Control (ไม่ใช้ `git add -A`, ตรวจ `gh auth status` ว่า active เป็น Chinnawan) รวมการแก้ PROJECT_PLAN.md — ดู git log

## งานที่ต้องทำต่อ — รอบ 2: เพิ่มความแม่นยำ (รันข้ามคืน 2026-10-13 → 14)

### เหตุผลและข้อมูลที่ใช้วางแผน (วิเคราะห์จาก prediction ใน `multiseed_results.json`)

- **Ensemble ภายใน fold ได้ผลทันที (ไม่ leak)**: ทุกโมเดลที่โหวตมาจาก fold เดียวกัน เทรนด้วย training fold เดียวกัน ไม่มีตัวไหนเห็น test fold — ต่างจาก ensemble ข้าม fold ที่เคยพบ leakage: `nir` เดี่ยว 0.843 → 3-seed vote 0.857 → RE-only 3-seed vote 0.876 → **กลุ่ม NIR 12 โมเดล (nir, nir_SE, re_only, ir_only × 3 seed) vote = F1 0.888 / acc 92.9%** (เลือกกลุ่มหลังเห็นผล จึงอาจสูงเกินจริงเล็กน้อย ต้องยืนยันด้วย seed ใหม่)
- **จุดที่ `nir` ผิด** (เฉลี่ย 3 seed): Rot ผิด 51% (ส่วนใหญ่ทาย Surface), Hail_damage_perf 27% (→ Internal), Other 27%, Russet 22% (→ Healthy), **Stem 7% (→ Surface 23 ครั้ง)**, Healthy 6% — มี 30 ภาพที่ `nir` (vote) ผิดแต่ `full` (vote) ถูก → visible มีข้อมูลที่ NIR ไม่มี
- **ข้อสงสัยเรื่องวิธีรวมแบนด์**: `models._adapt_first_conv` ใช้ `mean_weight.repeat(1, C, 1, 1)` → ทุกแบนด์ได้ filter เหมือนกันเป๊ะและไม่ปรับสเกลตามจำนวนช่อง ตอนเริ่มเทรนโมเดลเห็นแค่ "ผลรวมของแบนด์" และพบว่า F1 ลดลงตามจำนวนช่อง (1 → 0.853, 2 → 0.843, 3 → 0.810, 4 → 0.754; BL+GR แย่กว่า BL/GR เดี่ยว 5/5 fold) — ต้องตรวจว่าข้อสรุป "NIR > Vis+NIR" ไม่ได้เกิดจากวิธี init นี้

### Protocol (เหมือนรอบ multi-seed ทุกอย่าง)

seed ล็อก + deterministic, seeds 0,1,2 × StratifiedKFold n=5 random_state=42 (fold เดิม), เขียนผลลง **ไฟล์ใหม่** `outputs/dl/round2_results.json` หลังจบทุก fold (resume ได้, error ไม่ล้ม batch), เก็บ prediction **และ softmax probability** ต่อภาพ (ใช้ทำ ensemble/TTA ภายหลัง), วัดผลบน **ภาพจริงเท่านั้น**, รายงาน 3 class เสมอ (Healthy/Internal/Surface) baseline เปรียบเทียบ = `nir` จาก `multiseed_results.json` (protocol เดียวกัน ไม่ต้องรันใหม่) ยกเว้นงานที่ใช้ภาพ SingleSpectra ให้เทียบกับ `nir_ss`

### Variant ที่ต้องรัน (เปลี่ยนทีละอย่างจาก `nir` = RE+IR, aug, ไม่มี SE, 224, raw)

| # | variant | เปลี่ยนอะไร | แก้ปัญหา |
|---|---|---|---|
| 1 | `nir_5cls` | เทรน 5 class (Healthy / Stem / Calix / Internal / Surface — label จาก `ss_mapping.json` tissue) ตอนทายรวม Healthy+Stem+Calix กลับเป็น Healthy (รวม prob ของ 3 class) | ขั้วผล/ก้นผลถูกทายเป็นตำหนิ |
| 2 | `nir_ls` | label smoothing 0.1 ใน CrossEntropyLoss | overconfidence บนข้อมูลน้อย |
| 3 | `nir_320` | input 320×320 (resize จาก raw quadrant ที่ความละเอียดสูงกว่า 224 — ต้องสร้าง `Processed_Dataset_320/` แยก ไม่ทับของเดิม) ถ้า VRAM 4GB ไม่พอให้ลด batch size และบันทึกไว้ | ตำหนิผิวขนาดเล็ก |
| 4 | `nir_ss` | baseline บนภาพ SingleSpectra (= `nir` แต่ source=ss) | ตัวเทียบของ #5, #6 |
| 5 | `nir_ss_plus` | เทรนด้วยภาพ raw **และ** SingleSpectra ของแอปเปิลลูกเดียวกัน (ภาพจริง 2 เวอร์ชันต่อลูก) ทดสอบบน raw — ใช้ `ss_mapping.json` ให้ทั้ง 2 เวอร์ชันอยู่ fold เดียวกันเสมอ | ข้อมูลจริงเพิ่มโดยไม่สังเคราะห์ (เทียบกับ `nir`) |
| 6 | `nir_ss_cp` | **Defect Copy-Paste** บนภาพ SingleSpectra (เทียบกับ `nir_ss`) — ดูรายละเอียดด้านล่าง | ตำหนิมีตัวอย่างน้อย (Rot, Russet) |
| 7 | `full_fix`, `visible_fix`, `nir_fix` | แก้ init conv แรก: (ก) สเกล weight × 3/C และ (ข) ให้แต่ละแบนด์มี filter ต่างกัน (เช่น ใช้ weight RGB ของ ImageNet วนตามช่อง R,G,B,R… หรือ mean + noise เล็กน้อย) — เลือก 1 วิธีที่สมเหตุสมผลที่สุดแล้วบันทึกเหตุผล | ตรวจว่า "NIR > Vis+NIR" มาจาก init หรือไม่ + ให้ visible ช่วย NIR ได้ |

**TTA (ไม่ต้องเทรนเพิ่ม)**: ทุก run ให้ประเมินทั้งแบบปกติและแบบ TTA (เฉลี่ย softmax ของภาพเดิม + flip แนวนอน/แนวตั้ง + หมุน 90/180/270) แล้วเก็บผลทั้งคู่

**Defect Copy-Paste (#6) — กฎ**:
1. ทำบนภาพ SingleSpectra (mask อยู่ในเรขาคณิตนี้เท่านั้น) ใช้ตำหนิจาก `segmentations/.../SBW_DefectNNNN.bmp`
2. **ตำหนิต้นทางและแอปเปิลปลายทางต้องมาจาก training fold เดียวกันเท่านั้น** (ห้ามใช้ภาพจาก val/test ของ fold นั้น)
3. ปลายทาง = ภาพ Healthy (ไม่ใช้ Stem/Calix เป็นพื้น) วางในบริเวณผลที่ห่างขอบผล ≥ 20 px สุ่มตำแหน่ง/หมุน/ย่อขยาย 0.8–1.2
4. วิธีวาง: คูณ "อัตราส่วนความเข้มของตำหนิต่อผิวรอบๆ ตำหนิในภาพต้นทาง" (ต่อแบนด์) ลงบนผิวปลายทาง + ขอบนุ่ม (Gaussian) — ทำทุกแบนด์ที่ตำแหน่งเดียวกัน (ต้นแบบที่ทดสอบแล้ว: `outputs/synthetic_demo/copy_paste_demo.png` ดูสมจริงในแบนด์ IR; แบนด์ visible มีปัญหาผิวสองสี ซึ่งไม่กระทบ #6 เพราะใช้แค่ RE+IR)
5. label ภาพสังเคราะห์ = class ของตำหนิต้นทาง สร้างแบบ on-the-fly ด้วยความน่าจะเป็น ~0.3 ต่อภาพ Healthy ใน batch (ไม่ต้องเขียนไฟล์ภาพลงดิสก์) เพิ่มน้ำหนักให้ Rot/Russet/Other/Hail_damage_perf เป็น 2 เท่า
6. ห้ามมีภาพสังเคราะห์ใน val (early stopping) และ test
7. ก่อนรันจริง: เซฟตัวอย่างภาพสังเคราะห์ 12 ภาพ (IR band) ลง `outputs/synthetic_demo/` ให้ผู้ใช้ดู (อยู่ใน `.gitignore` แล้ว)

### ลำดับใน batch (ถ้าเครื่องดับ ของสำคัญเสร็จก่อน)

1. `nir_5cls`, `nir_ls` → 2. `full_fix`, `nir_fix`, `visible_fix` → 3. `nir_ss`, `nir_ss_cp`, `nir_ss_plus` → 4. `nir_320` → 5. (ถ้าเหลือเวลา) Grad-CAM 14×14 บน checkpoint `nir_SE_ss` เดิม → `outputs/dl/gradcam_mask_eval_14x14.json`

### สคริปต์สรุป `src/round2_report.py` → `outputs/dl/round2_summary.json` + `outputs/logs/round2_report.txt`

1. ตารางต่อ variant (ปกติ และ TTA): Macro F1 / acc (mean ± std ข้าม seed), per-class F1, และ **error rate ราย tissue** (Rot, Russet, Stem ฯลฯ)
2. เทียบรายคู่กับ baseline ของตัวเอง (จับคู่ seed+fold, เกณฑ์ "ต่างจริง" เดิม: CI ไม่คร่อม 0 และ ≥12/15 คู่) + ตรวจระดับ fold (เฉลี่ย 3 seed แล้วนับ 5 fold)
3. Ensemble ภายใน fold: 3-seed vote ของแต่ละ variant + soft-vote (เฉลี่ย probability) ของกลุ่มที่ผ่านเกณฑ์
4. **ห้ามสรุปผลลง PROJECT_PLAN.md เอง** — รอสรุปกับผู้ใช้

### หลังจากนี้ (พรุ่งนี้ ไม่อยู่ใน batch คืนนี้)

รวมเทคนิคที่ "ต่างจริง" เป็นสูตรเดียว → **ล็อกสูตร** → รันยืนยันด้วย **seed ใหม่ 3, 4, 5** (ห้ามใช้เลือกสูตร) + ensemble ภายใน fold → ตัวเลขนั้นคือตัวเลขที่ใส่เล่ม

### ขั้นตอนก่อนผู้ใช้กดรัน

ทำโค้ดทั้งหมด + smoke test (ทุก variant × 1 seed × 1 fold × 2 epoch — ตรวจ VRAM ของ `nir_320`, ตรวจว่า copy-paste ใช้แค่ training fold) + ทดสอบ resume + ประเมินเวลาจริงจาก benchmark แล้วสร้าง `run_round2.bat` (แบบเดียวกับ `run_overnight.bat`) **ไม่ต้องรันจริงเอง** ให้ผู้ใช้เปิดรันใน terminal แยก แล้ว commit โค้ด (ห้าม `git add -A`)

### สถานะการเตรียมรอบ 2 (2026-10-13): โค้ดครบ + smoke test ผ่านทุก variant — **ยังไม่ได้รัน batch จริง รอผู้ใช้เปิด `run_round2.bat` เอง**

**ไฟล์ใหม่/ที่แก้**: `src/round2.py` (variant ทั้ง 9 + เทรน/ประเมินต่อ fold + TTA), `src/copy_paste.py`, `src/run_round2.py` (batch, smoke, self-check), `src/round2_report.py`, `src/preprocess_320.py`, `run_round2.bat`; แก้ `models.py` (ตัวเลือก `init="timm"` ค่าเริ่มต้น `legacy` ไม่เปลี่ยนพฤติกรรมเดิม), `train_dl_kfold.py` (source `320`), `gradcam_mask_eval.py` (`--layer/--out/--overlay-dir`), `.gitignore` (`Processed_Dataset_320/`, `outputs/dl/gradcam_mask_14x14/`) ข้อมูล: `Processed_Dataset_320/` (819 ไฟล์ 320×320×4 สร้างจาก quadrant ของ RawA ด้วย pipeline เดียวกับ 224 — ย่อกลับ 224 แล้วต่างจากของเดิมเฉลี่ย 0.25 ระดับเทา, 324 MB, mirror ชื่อ/split เดิม ไม่ทับ `Processed_Dataset/`)

**การตัดสินใจออกแบบ (ที่แผนเปิดให้เลือกหรือไม่ได้ระบุ — บันทึกไว้เพื่อให้ตรวจสอบได้)**
- **init conv แรก (#7)**: ใช้สูตรเดียวกับ `timm.adapt_input_conv` — วนใช้ filter RGB ของ ImageNet ตามช่อง (R,G,B,R,…) แล้วคูณ 3/จำนวนช่อง (1 ช่อง = ผลรวม 3 filter) เหตุผล: เป็นสูตรมาตรฐานที่ใช้กันแพร่หลาย ทำให้ (ก) แต่ละแบนด์ได้ filter ต่างกัน (ข) สเกลสัญญาณรวมตรงกับโครงข่าย RGB ที่ pretrain มา (เดิม `mean` ซ้ำทุกช่องโดยไม่ปรับสเกล — ตรวจแล้วผลรวม |น้ำหนัก| ของ nir เดิม 122.6 vs timm 283.1) ข้อจำกัด: `nir_fix` ให้ RE=filter R, IR=filter G (×1.5) การจับคู่แบนด์กับสีเป็นการเลือกโดยพลการ
- **loss ตรวจ early stopping (val)** ใช้ cross-entropy ปกติ (ไม่ smooth) ทุก variant เพื่อให้กฎหยุดเทรนเหมือนกัน; label smoothing 0.1 ใช้กับ train loss เท่านั้น
- **`nir_5cls`**: sampler ใช้น้ำหนักจากจำนวน 3 class จริงเหมือน baseline (สัดส่วนการสุ่ม Healthy:Internal:Surface ไม่เปลี่ยน มีแค่ label ละเอียดขึ้น) ทายโดยรวม prob Healthy+Stem+Calix; เก็บ prob 5 class ไว้ด้วย
- **`nir_ss_plus`**: เทรนด้วยภาพ raw + SingleSpectra ของแอปเปิลชุดเดียวกัน (ใน training fold เท่านั้น) แต่ละ epoch สุ่มจำนวนเท่าเดิม (= จำนวน raw ใน train) เพื่อคุมงบเวลา/จำนวน step ให้เท่า `nir`; normalization ใช้สถิติของ raw train (ภาพที่ใช้ทดสอบ); val/test เป็นภาพ raw
- **`nir_ss_cp`**: donor และเป้าหมายมาจาก **training ส่วนที่ใช้เทรนจริงเท่านั้น (ไม่รวม inner val)**; วางแบบ multiplicative ต่อแบนด์ `target*(1+alpha*(ratio-1))` (ratio = ความเข้มตำหนิ ÷ median ผิวรอบๆ), alpha = mask นุ่ม (Gaussian), ตำแหน่งสุ่มห่างขอบผล ≥ 20 พิกเซล, หมุน 0–360°, ย่อขยาย 0.8–1.2, p=0.3 ต่อภาพ Healthy-tissue ใน batch, donor Rot/Russet/Other/Hail_damage_perf น้ำหนัก ×2, label = class ของ donor; val/test ไม่มีภาพสังเคราะห์ ตัวอย่างภาพ 12 ภาพ (IR band, ซ้าย=ต้นฉบับ ขวา=สังเคราะห์) อยู่ที่ `outputs/synthetic_demo/round2_cp_samples.png` (อยู่ใน `.gitignore`) — จากภาพ ตำหนิดูสมจริงในแบนด์ IR; ข้อสังเกต: ตำหนิขนาดใหญ่วางบนผลเล็กไม่ได้ (ต้องห่างขอบ ≥ 20 px) สัดส่วน label ที่วางจริงจึงเป็น Internal ~27–34% / Surface ~66–73% (ตามน้ำหนัก donor ได้ 35%)
- **TTA**: 6 มุมมอง (เดิม, พลิกแนวนอน, พลิกแนวตั้ง, หมุน 90/180/270) เฉลี่ย softmax ทุก run เก็บทั้งผลปกติและ TTA + softmax ต่อภาพ
- ลำดับ batch ตามแผน (ข้อ 5 Grad-CAM 14×14 = `features[5]` ซึ่งยืนยันแล้วว่าให้แผนที่ 14×14 รันต่อท้ายใน `run_round2.bat` ถ้ายังไม่มี `gradcam_mask_eval_14x14.json` แล้วจึงรัน `round2_report`)

**ผลตรวจสอบก่อนรัน (ผ่านทั้งหมด)** — `python -m src.run_round2 --selfcheck` / `--smoke`:
- inner train/val split ของทุก variant ตรงกับ baseline `run_fold` ทุกรายการและทุกลำดับ (ตรวจ fold 0, 3) → เทียบแบบจับคู่กับ `nir`/`full`/`visible` ใน `multiseed_results.json` ได้
- copy-paste: donor (162 ภาพ/fold) และเป้าหมายอยู่ใน training fold ทั้งหมด ไม่มีตัวไหนอยู่ใน val/test ของ fold นั้น ตรวจครบ **5/5 fold**; val/test dataset ไม่มี paster; อัตราวางจริง 26.0% จาก 400 ครั้ง (เป้า ~30%), หาตำแหน่งวางสำเร็จ 104/104; seed เดียวกันให้ภาพสังเคราะห์ตรงกันทุกพิกเซล
- **smoke test**: ทุก variant × seed 0 × fold 0 × 2 epoch รันจบ 9/9, error 0, json ถูกต้อง (prob รวมเป็น 1, ไม่มี NaN), รันซ้ำแล้วข้ามทั้งหมด ("0 runs to do"), ทดสอบ fold ล้มแล้ว batch ไปต่อ (กลไกเดียวกับรอบ multi-seed) และทดสอบตัวสำรอง CUDA out-of-memory (จำลองข้อผิดพลาด → ลอง batch 8 อัตโนมัติ และบันทึกใน `batch_size`) ตัวเลข F1 ของ smoke (2 epoch) **ไม่มีความหมายเชิงผล** และถูกลบแล้ว
- **VRAM**: `nir_320` peak **2.9 GB จาก 4 GB** ที่ batch 16 (พอ ไม่ต้องลด batch) ส่วน variant 224 ราว 1.5 GB
- **ทำซ้ำได้**: `nir_ss_cp`, `nir_320`, `nir_5cls` รัน 2 ครั้งในโปรเซสแยก (3 epoch) ได้ softmax (ปกติ+TTA) และ best val loss ตรงกันแบบ bit-identical (hash เท่ากัน) และผลจากการรันผ่าน `run_round2.bat --smoke` ตรงกับรันตรงทุกตัวเลข
- รายงาน (`round2_report`) ทดสอบกับข้อมูลไม่ครบ (แสดง INCOMPLETE ไม่พัง) และกับข้อมูลจำลองครบ 135 run ที่ฝังความต่างไว้ (การปรับปรุงที่ฝังไว้ถูกตัดสินว่า REAL, การทำให้แย่ลงที่ฝังไว้ถูกตัดสินว่า REAL ติดลบ, ที่ไม่ต่างถูกตัดสินว่าไม่เกินความแกว่ง); hard-vote 3 seed ของ `nir` จากสคริปต์ = **0.857** ตรงกับตัวเลขที่ใช้วางแผน (แปลว่ากติกาเสมอโหวตตรงกัน)

**เวลาประมาณ** (วัดจริง วินาที/epoch: nir-type 4.3–4.8, full 5.1, copy-paste 5.2, ss_plus 5.1, **nir_320 9.3**; จำนวน epoch ใช้ค่าเฉลี่ยของ `nir`/`full`/`visible` ใน multi-seed แล้วคูณ 1.2 ให้ `nir_ss_cp` และ `nir_ss_plus` เพราะคาดว่าต้องเทรนนานขึ้น): 9 variant × 3 seed × 5 fold = 135 fold-run ≈ **5.2 ชั่วโมง** (ช่วง 4.3–6.3 ชม.) แยก: `nir_320` ~58 นาที, `nir_ss_cp` ~39, `nir_ss_plus` ~38, `full_fix` ~34, ที่เหลือ ~28 นาที/ตัว + Grad-CAM 14×14 ~2 นาที + report ~1 นาที (ตัวเลขของ `nir_ss_cp`/`ss_plus`/`nir_320` เป็นการคาด ไม่เคยรันเต็มมาก่อน)

**คำสั่งเปิดรัน** (terminal แยก; รันซ้ำคำสั่งเดิมเพื่อทำต่อหลังเครื่องดับ; ผลลง `outputs/dl/round2_results.json`, log `outputs/logs/round2_YYYYMMDD.log`, รายงาน `outputs/logs/round2_report.txt`):
```
cd /d D:\nir+rgb_project\project\CAPA_AppleQualityGradingImageDatabase
run_round2.bat
```
ข้อควรรู้: เสียบปลั๊ก, โค้ดขอให้ Windows ไม่ idle-sleep ระหว่างรันแต่ปิดฝาเครื่องยังทำให้หลับ; **ยังไม่ได้ push** (commit ในเครื่องเท่านั้น)

## ผลรอบ 2 (รันเสร็จ 2026-10-14 00:54) — สรุปร่วมกับผู้ใช้ 2026-10-14

135/135 run (9 variant × 3 seed × 5 fold), error 0 รายงาน: `outputs/logs/round2_report.txt`, สรุป `outputs/dl/round2_summary.json`, ผลดิบ `outputs/dl/round2_results.json` (baseline `nir`/`full`/`visible` = จาก `multiseed_results.json`, seed+fold เดียวกัน)

| Variant | F1 (ปกติ) | Acc | F1 (TTA) | Acc TTA | เทียบ baseline | ผล |
|---|---|---|---|---|---|---|
| **`nir_ss`** (ภาพ SingleSpectra/flat-field, RE+IR) | **0.872 ± 0.012** | **91.9%** | **0.880 ± 0.013** | **92.5%** | vs `nir` +0.028 (13/15, CI +0.010..+0.046; ระดับ fold 4/5, CI −0.005..+0.061) | **ต่างจริง (ตัวเดียวที่ผ่านเกณฑ์)** |
| `nir_ls` (label smoothing 0.1) | 0.861 ± 0.013 | 91.3% | 0.855 | 90.8% | vs `nir` +0.018 (11/15) | แนวโน้มบวก ยังไม่ผ่าน |
| `nir_ss_cp` (copy-paste) | 0.859 ± 0.019 | 90.7% | 0.872 | 91.7% | vs `nir_ss` −0.013 (ระดับ fold ลบ 5/5) | **ไม่ช่วย แนวโน้มแย่ลง** |
| `nir_ss_plus` (raw+SS เป็นภาพเพิ่ม) | 0.857 ± 0.009 | 90.9% | 0.861 | 91.1% | vs `nir` +0.013 (10/15) | ไม่ต่างเกิน noise |
| `nir_5cls` (แยก Stem/Calix) | 0.848 ± 0.003 | 90.6% | 0.859 | 91.1% | vs `nir` +0.004 (7/15) | ไม่ต่าง |
| `nir_fix` (init timm) | 0.838 ± 0.010 | 89.7% | 0.852 | 90.6% | vs `nir` −0.005 | ไม่ต่าง |
| `nir_320` | 0.830 ± 0.007 | 89.2% | 0.852 | 90.5% | vs `nir` −0.014; TTA−plain +0.023 (ต่างจริง) | ไม่ช่วย (TTA ช่วยเฉพาะตัวนี้) |
| `full_fix` (Vis+NIR, init timm) | 0.714 ± 0.014 | 82.2% | 0.732 | 83.6% | vs `full` −0.041 (ระดับ fold ลบ 5/5) | **แก้ init แล้ว full ไม่ดีขึ้น** |
| `visible_fix` | 0.702 ± 0.016 | 80.6% | 0.708 | 81.2% | vs `visible` +0.014 | ไม่ต่าง |
| (อ้างอิง) `nir` / `full` / `visible` | 0.843 / 0.754 / 0.688 | 90.2 / 85.0 / 80.3% | — | — | — | — |

**Ensemble ภายใน fold (EXPLORATORY — เลือกกลุ่มหลังเห็นผล ต้องยืนยันด้วย seed ใหม่)**: `nir_ss` 3-seed soft vote 0.884 / 92.6%, **`nir_ss`[TTA] 3-seed soft vote 0.894 ± 0.034 / 93.3%** (สูงสุด), `nir` 3-seed hard vote 0.857 / 91.1%

**Error ราย tissue**: `nir_ss` ลด Rot จาก 50.7% → 37.7% และ Scald 11.8% → 7.8%, Healthy 5.7% → 3.6%; copy-paste ลด Hail_damage_perf (21.5% → 15.1%) และ Bruise แต่ Other/Russet แย่ลง

**Grad-CAM 14×14 (`features[5]`, โมเดล `nir_SE_ss` เดิม)**: Surface pointing hit 13.2% vs null 6.0% (7×7 เดิม: 24.3% vs 21.0%) → ละเอียดขึ้นแล้วชี้ตำหนิผิวได้เหนือ null ชัดกว่าเดิม (≈2.2 เท่าของ null) แต่ค่าสัมบูรณ์ยังต่ำ และ IoU ยังแทบเท่า null (0.033 vs 0.028); Internal 40.2% vs 30.7% (IoU 0.171 vs 0.146)

**ข้อสรุป**
1. **สูตรใหม่ที่ดีที่สุด = `nir_ss`** (ภาพที่ผ่าน flat-field correction ของผู้สร้าง dataset, RE+IR, augmentation, ไม่มี SE) — ข้อสรุปเดิมจากรอบเดียว (`nir_SE_ss` ≈ `nir_SE`, "flat-field ไม่ช่วย") **ถูกหักล้างโดย multi-seed** ข้อควรระวัง: SingleSpectra ต่างจาก raw ทั้ง flat-field และเรขาคณิต (crop/pad, แอปเปิลเล็กลง ~9%) จึงยังแยกไม่ได้ว่าปัจจัยไหนช่วย และ CI ระดับ fold แตะ 0
2. **"NIR > Vis+NIR" ไม่ได้เกิดจากวิธี init conv แรก** — ใช้ init มาตรฐาน (timm: filter ต่างกันต่อแบนด์ + สเกล 3/C) แล้ว full ยังแย่กว่า nir (และแย่กว่า full เดิม) → ข้อสรุปหลักแข็งแรงขึ้น
3. ไม่ช่วยเกิน noise: 5 class, 320 px, init ใหม่, ภาพ SS เป็นภาพเพิ่ม; label smoothing แนวโน้มบวกแต่ยังไม่ผ่าน; **copy-paste ไม่ช่วย** (ผลลบที่ต้องรายงานตรงๆ)
4. TTA ช่วยเล็กน้อยแทบทุกตัว (+0.01) ผ่านเกณฑ์เฉพาะ `nir_320`

## งานที่ต้องทำต่อ — ยืนยันสูตรสุดท้ายด้วย seed ใหม่ (confirmatory run)

**ล็อกสูตรก่อนรัน (ห้ามเปลี่ยนหลังเห็นผล)**: `nir_ss` + TTA (6 แบบเดิม) + soft-vote 3 seed ภายใน fold
1. รัน **seed ใหม่ 3, 4, 5** × 5 fold เดิม: `nir_ss` (สูตรสุดท้าย) และ `nir` (baseline, จับคู่ seed+fold) — เขียนลง `outputs/dl/confirm_results.json` (ไฟล์ใหม่, resume ได้, เก็บ probability)
2. รายงาน (ไม่ใช้ seed 0–2 เลย): ต่อ variant mean ± std (ปกติ/TTA), per-class F1, เทียบรายคู่ `nir_ss` − `nir` ด้วยเกณฑ์เดิม, และ **ensemble 3-seed soft vote (+TTA) ของแต่ละ variant** → ตัวเลขเหล่านี้คือตัวเลขที่ใส่เล่ม
3. (ทางเลือก, ระบุเป็น exploratory) `nir_ss_ls` = `nir_ss` + label smoothing 0.1 ด้วย seed 3–5 — ถ้าดีกว่าก็รายงานเป็นข้อสังเกต ไม่ใช่ผลหลัก
4. เวลาโดยประมาณ ~1–1.5 ชม. (+30 นาทีถ้ารวมข้อ 3) ใช้ runner/protocol เดียวกับรอบ 2 สร้าง `run_confirm.bat` + smoke test แล้วให้ผู้ใช้เปิดรันเอง
6. **เซฟ checkpoint ของ `nir_ss` ทุก fold × seed 3,4,5 (15 ไฟล์)** ลง `outputs/dl/confirm_ckpt/nir_ss_seed{S}_fold{K}.pt` (state_dict + mean/std ที่ใช้ normalize ของ fold นั้น) และเซฟ `outputs/dl/fold_assignment.json` = ภาพแต่ละภาพ (RawA id) อยู่ใน test ของ fold ไหน — ใช้กับ demo app; checkpoint อยู่ใน `.gitignore` (ไฟล์ใหญ่) ไม่ต้องเซฟ checkpoint ของ `nir`
5. commit ผลรอบ 2 ที่ยังไม่ได้ commit (`round2_results.json`, `round2_summary.json`, `gradcam_mask_eval_14x14.json`, PROJECT_PLAN.md) ก่อนเริ่ม (ห้าม `git add -A`)

### สถานะ (2026-10-14): รันเสร็จ 17:17 (45/45 fold-run, error 0, ใช้เวลา 1.5 ชม. ตรงที่ประมาณไว้) — **ผลไม่ยืนยันข้อค้นพบของรอบ 2** (ดู "ผลยืนยัน" ด้านล่าง)

**ลำดับที่ทำ**: (ข้อ 5) commit ผลรอบ 2 = `66ddf92` → โค้ดยืนยัน = `de4be01` (commit ก่อนรันจริง เพื่อให้ผลอ้างอิงโค้ดที่ commit แล้ว; ยังไม่ push) ไฟล์ใหม่: `src/run_confirm.py`, `src/confirm_report.py`, `src/infer.py` (โหลด checkpoint + ทำนายด้วย TTA แบบเดียวกับตอนเทรน), `run_confirm.bat`; แก้ `src/round2.py` (`run_round2_fold(variants=, ckpt_path=)` ค่าเริ่มต้นเดิมไม่เปลี่ยน, คืน mean/std ของ fold)

**การตัดสินใจที่ผมเลือกเอง (แผนไม่ได้ระบุ — บันทึกเพื่อตรวจสอบ)**
1. **รันครบทุกข้อรวมข้อ 3 (ทางเลือก)**: `nir_ss_ls` อยู่เป็นกลุ่มสุดท้ายของ batch (ถ้าเครื่องดับ แกนหลัก `nir_ss`+`nir` เสร็จก่อน) และถูกติดป้าย **exploratory** ในรายงาน ไม่ใช่ส่วนของสูตรที่ล็อก
2. **`nir` baseline รันพร้อม TTA ด้วย** (รอบก่อน baseline ไม่มี TTA/probability) จึงเทียบได้ทั้ง `nir_ss − nir` (ปกติ), `nir_ss[TTA] − nir[TTA]` (ยุติธรรมที่สุด), และสูตรสุดท้ายเทียบ baseline เก่า `nir_ss[TTA] − nir`
3. **ห้ามปนข้อมูล seed 0–2**: `run_confirm` มี assert seed ≥ 3, `confirm_report` มี assert เดียวกัน; ผล seed 0 ที่ใช้ตรวจ pipeline (ด้านล่าง) ใช้ตรวจ "ความเหมือนของขั้นตอน" เท่านั้น ไม่ถูกนำมาคำนวณผลใดๆ
4. **checkpoint** = `outputs/dl/confirm_ckpt/nir_ss_seed{S}_fold{K}.pt` 15 ไฟล์ เก็บ weight ที่ val loss ต่ำสุด + mean/std 4 แบนด์ของ fold นั้น + variant/seed/fold/channels/source (อยู่ใน `.gitignore`); `outputs/dl/fold_assignment.json` (819 ภาพ → fold ที่เป็น test, tissue, label, defect_no) **ขึ้น git ได้** เพราะไม่มีพิกเซลภาพ
5. **เกณฑ์ตรวจ checkpoint 2 ระดับ** (แก้เกณฑ์ตัวเองหลัง smoke test ครั้งแรกไม่ผ่าน): ครั้งแรกผมตั้ง "ความต่างของ probability < 1e-3" เมื่อทำนายทีละ 1 ภาพ แต่ได้ 2.2e-3 → ตรวจแยกสาเหตุแล้วพบว่า (ก) ทางเดียวกับตอนเทรน (batch 16) ให้ผลตรงกับที่บันทึกไว้ที่ 5e-6 (= การปัดเศษ 5 ตำแหน่ง) และ preprocessing ตรงกัน 100% แปลว่า checkpoint + สถิติ normalize ถูกต้อง (ข) ทีละ 1 ภาพ (ทางของ demo) ต่างที่ ~2e-3 จากการเลือก kernel ของ cuDNN ต่างกัน โดยคำตัดสินตรงกันทั้ง 164/164 ภาพ จึงตั้งเกณฑ์: batch-16 < 1e-4 และ batch-1 < 1e-2 + คำตัดสินต้องตรงกันทั้งหมด (ไม่ได้ผ่อนเกณฑ์เพื่อให้ผ่าน — แยกสิ่งที่ต้องเท่ากันเป๊ะ (ขั้นตอนเดียวกัน) ออกจากสิ่งที่ต่างได้ตามธรรมชาติ (batch ต่างขนาด))
6. smoke ผ่านทั้ง 3 variant × seed 3 × fold 0 × 2 epoch + ตรวจ resume ("0 runs to do") + ตรวจ `fold_assignment.json` ตรงกับ fold ที่ทดสอบจริง; ผล smoke ลบแล้ว
7. **ตรวจความต่อเนื่องของ baseline (full-length, fold 0, seed 0 — ตรวจขั้นตอน ไม่ใช่ผลวัด)**: `nir` ผ่าน pipeline รอบนี้ = `nir` ของรอบ multi-seed **เหมือนเป๊ะ** (17 epoch, F1 0.801754, best val loss เท่ากัน, prediction ตรงกันทุกภาพ) และ `nir_ss` ในโปรเซสใหม่ = ผลรอบ 2 เหมือนเป๊ะ (probability ปกติ+TTA ตรงกันทุกค่า) → baseline เทียบข้ามรอบได้ และการทำซ้ำได้ใช้ข้ามวัน/โปรเซสได้จริง
8. **เวลา**: ประมาณ ~1.5 ชม. (แกนหลัก `nir_ss`+`nir` ~0.9 ชม. + `nir_ss_ls` ~0.6 ชม.; คำนวณจากเวลาจริงต่อ fold ของรอบก่อน: `nir_ss` 98 วินาที, `nir` 116 วินาที, `nir_ls` 148 วินาทีเป็นตัวแทนของ `nir_ss_ls`) ใช้ `run_confirm.bat` (smoke: `run_confirm.bat --smoke`) รายงานอัตโนมัติ `outputs/logs/confirm_report.txt` → `outputs/dl/confirm_summary.json`


### ผลยืนยัน (seed 3, 4, 5 เท่านั้น; รายงาน `outputs/logs/confirm_report.txt`, สรุป `outputs/dl/confirm_summary.json`, ผลดิบ `outputs/dl/confirm_results.json`)

| Variant | F1 ปกติ | Acc | F1 + TTA | Acc TTA | per-class F1 (TTA: ปกติ / ภายใน / ภายนอก) |
|---|---|---|---|---|---|
| **`nir_ss`** (สูตรที่ล็อก) | 0.858 ± 0.003 | 91.0% | **0.869 ± 0.010** | **91.7%** | 0.956 / 0.846 / 0.806 |
| `nir` (baseline, raw) | 0.849 ± 0.006 | 90.4% | 0.855 ± 0.002 | 90.9% | 0.951 / 0.812 / 0.802 |
| `nir_ss_ls` (exploratory) | 0.868 ± 0.008 | 91.7% | 0.877 ± 0.011 | 92.2% | 0.960 / 0.858 / 0.814 |

การเปรียบเทียบรายคู่ (กติกาเดิม: CI ไม่คร่อม 0 และ ≥12/15 คู่เครื่องหมายเดียวกัน; + ตรวจระดับ fold):

| คู่ | ผลต่างเฉลี่ย | CI 95% | คู่/fold เครื่องหมายเดียวกัน | ผล |
|---|---|---|---|---|
| **`nir_ss − nir` (ปกติ) — การเปรียบเทียบหลักที่แผนกำหนด** | +0.009 | −0.004..+0.025 | 8/15; 4/5 | **ไม่ยืนยัน** (รอบ 2 ได้ +0.028) |
| `nir_ss[TTA] − nir[TTA]` (ยุติธรรมที่สุด) | +0.014 | −0.004..+0.030 | 11/15; 4/5 | ไม่เกินความแกว่ง |
| `nir[TTA] − nir` (ผลของ TTA เฉพาะ baseline) | +0.006 | −0.003..+0.015 | 10/15; 4/5 | ไม่เกินความแกว่ง |
| `nir_ss[TTA] − nir_ss` (ผลของ TTA เฉพาะ nir_ss) | +0.011 | +0.002..+0.020 | 10/15; 3/5 | ไม่เกินความแกว่ง (เกณฑ์ ≥12/15 ไม่ผ่าน) |
| `nir_ss[TTA] − nir` (สูตรสุดท้ายเทียบ baseline เดิม) | +0.020 | +0.005..+0.037 | 12/15; 5/5 | ผ่านเกณฑ์ (พอดีขอบ ≥12/15) |
| [exploratory] `nir_ss_ls − nir_ss` | +0.010 | −0.002..+0.024 | 8/15; 3/5 | ไม่ต่าง |

**Ensemble 3 seed ภายใน fold (ตัวเลขสำหรับเล่ม)**: `nir_ss`[TTA] soft vote F1 **0.885 ± 0.024** (ระหว่าง fold) · **accuracy 92.6%**; `nir_ss` soft vote ไม่ใช้ TTA 0.877 / 92.1%; `nir`[TTA] soft vote 0.871 ± 0.031 / 91.7% (ต่างจาก `nir_ss` ราย fold: +0.006, −0.008, −0.002, +0.028, +0.050 → `nir_ss` ดีกว่าเพียง 3/5 fold)

**สูตรสุดท้าย (nir_ss + TTA + 3-seed soft vote) รวมทุกภาพ out-of-fold 819 ภาพ**: accuracy **92.6%**, macro-F1 **0.885**

| จริง \ ทาย | ปกติ | ตำหนิภายใน | ตำหนิภายนอก | precision | recall | F1 |
|---|---|---|---|---|---|---|
| ปกติ (573) | 554 | 6 | 13 | 0.950 | 0.967 | 0.958 |
| ตำหนิภายใน (102) | 8 | 82 | 12 | 0.921 | 0.804 | 0.859 |
| ตำหนิภายนอก (144) | 21 | 1 | 122 | 0.830 | 0.847 | 0.838 |

(baseline `nir`[TTA] ensemble: accuracy 91.7%, macro-F1 0.871) Error ราย tissue ของ `nir_ss` (ปกติ): Rot 42.0%, Frost_damage 36.4%, Russet 25.4%, Other 25.0%, Flesh_damage 22.2%, Hail_damage_perf 19.4%, Bruise 10.9%, Stem 7.0%, Healthy 2.9%

**ข้อสรุปที่ซื่อสัตย์ (ต้องใช้ตามนี้ในเล่ม/defense)**
1. **ข้อค้นพบหลักของรอบ 2 ("ภาพ SingleSpectra ช่วย +0.028") ไม่ถูกยืนยัน**: เมื่อวัดบน seed ใหม่ได้ +0.009 (CI คร่อม 0, 8/15 คู่) — ผลถอยเข้าหาศูนย์ตามที่คาดเมื่อเลือกผู้ชนะจาก 27 การเปรียบเทียบ ("winner's curse") ห้ามรายงานว่า "SingleSpectra/flat-field ช่วยให้ดีขึ้นอย่างมีนัยสำคัญ" — รายงานได้เพียงว่า "แนวโน้มบวกเล็กน้อย (+0.01) ไม่ผ่านการยืนยัน" ข้อสรุปเดิมจากรอบ multi-seed ที่ว่า "flat-field ไม่ช่วย" กลับมาใกล้ความจริงมากกว่าข้อสรุปรอบ 2
2. **สูตรที่ล็อกเทียบ baseline เดิมผ่านเกณฑ์ที่ +0.020** แต่ต้องอ่านอย่างระวัง: เป็นผลรวมของ 2 ส่วน (ภาพ SS + TTA) ซึ่งแต่ละส่วนเดี่ยวๆ ไม่ผ่านเกณฑ์ และเป็นการเปรียบเทียบรองที่ผมเพิ่มในรายงาน (รายงานนี้ทดสอบ 7 คู่) ผ่านแบบพอดีขอบ (12/15) จึงใช้เป็นแนวโน้มบวกเล็กน้อย (≈ +2 จุด F1, accuracy 90.4% → 91.7%) ไม่ใช่ข้อสรุปแข็ง
3. TTA ให้ผลบวกเล็กน้อยสม่ำเสมอ (+0.006 ถึง +0.011 ที่นี่, +0.010 เฉลี่ยในรอบ 2) ใช้ได้เพราะไม่เสียต้นทุนตอนเทรน แต่แต่ละครั้งไม่เกินความแกว่ง
4. ตัวเลขที่ใส่เล่มได้อย่างปลอดภัย: **EfficientNet-B0 บนแบนด์ NIR (RE+IR) ได้ F1 ≈ 0.85–0.87 และ accuracy ≈ 90–92% ต่อโมเดล, ensemble 3 seed ≈ 0.87–0.885 / 91.7–92.6%** โดยภาพ raw กับ SingleSpectra ไม่ต่างกันอย่างมีนัยสำคัญ; ความอ่อนแอที่ชัดเจนคือ Surface_Defect (F1 ≈ 0.81) และตำหนิชนิด Rot / Frost_damage / Russet (error 25–42%)
5. `nir_ss_ls` (exploratory) +0.010 ไม่ต่าง — ไม่ใช่ผลหลัก
6. baseline `nir` นิ่ง: seed 0–2 ได้ 0.843, seed 3–5 ได้ 0.849 (std ระหว่าง seed ≤ 0.008); `nir_ss` seed 0–2 ได้ 0.872 → seed 3–5 ได้ 0.858 (ลดลง 0.014 ซึ่งใหญ่กว่า std ระหว่าง seed มาก สนับสนุนการอธิบายด้วย winner's curse)

**ข้อควรบันทึกจากการตรวจ checkpoint ตอนรันจริง**: ผลตรวจ 15 checkpoint — fold_assignment ตรง 15/15; ทางเดียวกับตอนเทรน (batch 16) ให้ probability ตรงที่ 5e-6 ทั้ง 15 ไฟล์; ทางของ demo (ทีละภาพ) คำตัดสินตรงกันทั้ง 2,457 ภาพ (ผิดตรงกัน 0) แต่ probability ต่างสูงสุด 1.1e-2 ที่ 1 ไฟล์ (`nir_ss` seed 4 fold 0) เกินเกณฑ์ 1e-2 ที่ผมเดาไว้ → ตรวจสอบสาเหตุ: บน CPU ผลไม่ขึ้นกับขนาด batch (ต่างกัน 1.1e-6) แต่บน GPU batch 1 กับ 16 ต่างกัน 1.1e-2 และ GPU กับ CPU ต่างกัน 2.3e-3 (สอดคล้องกับ TF32 ของ GPU ตระกูล Ampere ซึ่งปัดเศษหยาบกว่า float32 ปกติ — เป็นข้อสันนิษฐานจากรูปแบบความต่าง ไม่ได้ปิด TF32 เพื่อพิสูจน์) จึงปรับเกณฑ์เป็น 2e-2 พร้อมคงเงื่อนไขเข้มคือคำตัดสินต้องตรงกันทุกภาพ แล้วตรวจซ้ำผ่านครบ 15/15 — บันทึกไว้ตรงๆ ว่าเกณฑ์นี้ถูกแก้หลังเห็นข้อมูล (เหตุผลข้างต้น) และ log ของการรันเดิม (`confirm_20261014.log`) ยังแสดง "HAD FAILURES" 1 รายการจากเกณฑ์เก่า

## งานที่ต้องทำต่อ — Demo app สำหรับทดสอบเองและ defense (ทำหลัง confirmatory run เสร็จ)

**เครื่องมือ**: Gradio (Python) รันบนเครื่องนี้ `.venv\Scripts\python.exe -m src.demo_app` เปิดที่ `http://127.0.0.1:7860` **ห้าม `share=True` / ห้ามเปิดออกนอกเครื่อง** (มีภาพจาก CAPA ติดลิขสิทธิ์ ULG) — เพิ่ม `run_demo.bat`

**โหมด 1 — ทดสอบด้วยภาพ CAPA (out-of-fold, ซื่อสัตย์)**
- เลือกภาพจาก dropdown (RawA id + ชนิดตำหนิจริง) หรือปุ่มสุ่ม
- ใช้ `fold_assignment.json` หา fold ที่ภาพนั้นเป็น test → โหลด **เฉพาะโมเดล 3 seed ของ fold นั้น** (ไม่เคยเห็นภาพนี้) ห้ามใช้โมเดลของ fold อื่นกับภาพนี้
- ทำนายด้วยสูตรที่ล็อกไว้: TTA 6 แบบ × 3 seed → soft vote

**โหมด 2 — ภาพใหม่ (อัปโหลด)**: รับไฟล์ `.npy` (H×W×4 BL/GR/RE/IR) หรือ tif 4 แบนด์ ใช้ ensemble ทุก fold × seed (15 โมเดล) เพราะไม่มีปัญหา leakage กับภาพที่ไม่อยู่ใน CAPA ไม่แสดงคำตอบจริง ระบุบนหน้าจอว่าภาพจากกล้องอื่นอาจได้ผลต่างจาก CAPA (domain shift)

**สิ่งที่แสดงบนหน้าจอ** (ตาม mockup ที่ตกลงกับผู้ใช้)
1. ภาพแบนด์ RE (750 nm) และ IR (800 nm) ที่โมเดลใช้
2. ผลทาย 3 class + แถบความน่าจะเป็น (ปกติ / ตำหนิภายใน / ตำหนิภายนอก) + การกระทำ: ปกติ → ขายผลสด, ตำหนิภายนอก → ส่งแปรรูป, ตำหนิภายใน → คัดทิ้ง
3. (โหมด 1) คำตอบจริง + ชนิด tissue + ถูก/ผิด
4. ตารางโหวต: แต่ละ seed ทายอะไร ความน่าจะเป็นเท่าไร และผล soft vote — ระบุว่าใช้โมเดล fold ไหน
5. Grad-CAM ชั้น `features[5]` (14×14) ซ้อนภาพ IR + เส้นขอบ mask ตำหนิจริง (ถ้ามี mask) — มีข้อความกำกับว่า "heatmap บอกบริเวณคร่าวๆ เท่านั้น" (อ้างผลวัด Grad-CAM vs mask)
6. **โหมดสายพาน**: สุ่ม N ภาพ (ค่าเริ่ม 50, โหมด 1 เท่านั้น) จัดเข้าถัง ขายผลสด / ส่งแปรรูป / คัดทิ้ง พร้อมนับทายถูก, confusion matrix 3×3 และรายการภาพที่ทายผิด

**ข้อกำหนด**: โหลดโมเดลครั้งเดียวแล้ว cache, รันบน GPU ถ้ามี (CPU ก็ต้องทำงานได้), ไม่เขียนภาพ CAPA ลงนอกโฟลเดอร์ที่อยู่ใน `.gitignore`, ทดสอบว่าเปิดได้ + ทายภาพตัวอย่าง 3 ภาพ (Healthy / Bruise / Russet) ถูกโหลด fold ถูกต้อง แล้ว commit โค้ด (ห้าม `git add -A`)

### สถานะ Demo app (2026-10-14): เขียนเสร็จ + ทดสอบกับโมเดลจริงผ่านทั้งหมด

**ไฟล์**: `src/demo_app.py` (แอป), `src/demo_selftest.py` (ชุดทดสอบ), `run_demo.bat` (เปิดแอป → http://127.0.0.1:7860), `requirements.txt` เพิ่ม `gradio==6.30.0` (ติดตั้งลง `.venv` แล้ว — ตรวจด้วย dry-run ก่อนว่าไม่แก้แพ็กเกจเดิมเลย เพราะการเทรนกำลังรันอยู่)

**การตัดสินใจที่เลือกเอง (แผนไม่ได้ระบุ)**
1. **ที่เก็บ cache ของ Gradio** ย้ายไป `outputs/demo_cache/` (อยู่ใน `.gitignore`) ด้วย `GRADIO_TEMP_DIR` เพราะ Gradio เขียนภาพที่ส่งออกลงโฟลเดอร์ชั่วคราว — ค่าเริ่มต้นคือโฟลเดอร์ temp ของระบบซึ่งอยู่นอกโฟลเดอร์ที่ `.gitignore` คุม; ปิด analytics; `launch(server_name="127.0.0.1", share=False)`
2. **Grad-CAM แสดงสำหรับ "คลาสที่โหวตทาย"** เฉลี่ยข้ามโมเดลที่ใช้ (3 โมเดลในโหมด 1, 15 โมเดลในโหมด 2) ที่ `features[5]` (14×14) บนมุมมองเดิมเท่านั้น (ไม่ใช่ TTA); ข้อความกำกับอ้างผลวัด 14×14 ซึ่งวัดบนโมเดล `nir_SE_ss` ไม่ใช่ `nir_ss` — ระบุชื่อโมเดลไว้ในข้อความบนจอแล้ว
3. **โหมด 2**: รับ `.npy` (H×W×4 BL/GR/RE/IR) หรือภาพ 4 ไฟล์ที่ชื่อลงท้าย `_BL _GR _RE _IR` (ลำดับไฟล์ไม่สำคัญ); ค่าเลขทศนิยมช่วง [0,1] คูณ 255, ค่า 16 บิตย่อเป็น 8 บิต, ภาพไม่เป็นจัตุรัสถูก pad ดำแล้ว resize 224 (แบบเดียวกับภาพ SingleSpectra), แจ้งข้อผิดพลาดชัดเจนถ้าไฟล์ไม่ครบ/รูปร่างผิด; ภาพ CAPA ที่อัปโหลดเป็นโหมด 2 จะมีโมเดล 12 จาก 15 ตัวที่เคยเห็นภาพนั้น (ผลไม่ซื่อสัตย์ — ใช้โหมด 1 สำหรับภาพ CAPA)
4. **โหมดสายพาน** สุ่ม N ภาพ (10–200, ค่าเริ่ม 50, seed กำหนดได้ให้ผลซ้ำได้) แต่ละภาพใช้เฉพาะโมเดล fold ของมันเอง; แสดงจำนวนต่อถัง (ขายผลสด / ส่งแปรรูป / คัดทิ้ง) ถูก/ผิดต่อถัง, ตำหนิที่หลุดไปขายผลสด, confusion matrix, รายการภาพที่ทายผิด
5. หน้าจอแสดงผลวัดจริงของสูตร (accuracy 92.6%, macro-F1 0.885) อ่านจาก `confirm_summary.json` อัตโนมัติ

**ผลทดสอบกับโมเดลจริง** (`python -m src.demo_selftest`, GPU; และ `--cpu-only-quick` กับ `CUDA_VISIBLE_DEVICES=-1` — หมายเหตุ: บน Windows ตัวแปรว่าง `""` ถูกมองเป็นไม่ตั้งค่า ต้องใช้ `-1`):
- โหมด 1 ภาพตัวอย่าง Healthy (RawA0002), Bruise (RawA0004), Russet (RawA0001): โหลดเฉพาะ seed 3,4,5 ของ fold ที่ภาพนั้นเป็น test (fold 4, 4, 0) ตรงกับ `fold_assignment.json`; meta ใน checkpoint ตรงทั้ง fold/seed; คำตัดสินตรงกับที่บันทึกใน `confirm_results.json`; ทายถูกทั้ง 3 (ความน่าจะเป็น 0.992 / 0.958 / 0.976)
- สายพาน 50 ภาพ (seed 0): ใช้เฉพาะโมเดลของ fold ตัวเอง, คำตัดสินตรงกับที่บันทึก 50/50, ทายถูก 90.0% (ตัวอย่าง 50 ภาพเล็ก ค่าแกว่งได้มากจาก 92.6% รวม)
- โหมด 2: `.npy` ใช้โมเดลครบ 15 ตัว และ 4 ไฟล์ tif ให้ผลโหวตเท่ากับ `.npy` เป๊ะ
- เซิร์ฟเวอร์จริง: ผูกกับ 127.0.0.1 เท่านั้น (ยืนยันด้วย `netstat` ว่า LISTENING ที่ 127.0.0.1) API ตอบ และเปิดหน้าเว็บในเบราว์เซอร์ทดลองกดทำนาย ได้การ์ดผล ภาพ RE/IR/Grad-CAM แถบความน่าจะเป็น และตารางโหวตแสดงเฉพาะ fold 0 ถูกต้อง; เซิร์ฟเวอร์ทดสอบปิดแล้วและลบไฟล์ cache ชั่วคราว
- **ข้อสังเกตจาก Grad-CAM ในตัวอย่าง Russet**: จุดร้อนสุดอยู่ที่ก้านและขอบภาพ (พื้นที่ pad สีดำ) ไม่ใช่ที่ตำหนิ — สอดคล้องกับผลวัดว่าชี้ตำแหน่งตำหนิได้ต่ำ และบ่งว่า heatmap ที่ชั้นนี้มีผลของขอบภาพปนอยู่ ยังไม่ได้ตรวจเพิ่ม (อาจ mask พื้นหลังออกในอนาคต แต่จะเป็นการแก้หลักฐาน จึงไม่ทำโดยพลการ)
- เหตุการณ์ที่ควรรู้: หน้าต่าง `run_confirm.bat` ที่เปิดไว้ค้างอยู่ที่ `pause` (รอกดปุ่ม) กดปุ่มใดก็ได้เพื่อปิด

## (พักไว้ 2026-10-17 — ไม่ต้องทำ) — Demo app รอบ 2: สลับโมเดล + ถัง "ไม่แน่ใจ" (โฟกัสโหมด 1) (2026-10-17)

### ข้อมูลที่ใช้ตัดสินใจ (คำนวณจาก `confirm_results.json`, สูตรสุดท้าย nir_ss TTA 3-seed soft vote, 819 ภาพ out-of-fold)

ผู้ใช้ทดลองสุ่มแล้วเจอความมั่นใจ ~60% บ่อย ตรวจแล้วไม่ใช่ bug: ความมั่นใจ (max prob) median 0.958; <0.6 = 5.9% ของภาพ (ถูก 58%), 0.6–0.7 = 4.8% (ถูก 64%), 0.7–0.8 = 6.0% (76%), 0.8–0.9 = 12.8% (92%), 0.9–0.95 = 15.6% (97%), ≥0.95 = 54.9% (99.3%) — ความมั่นใจสะท้อนความถูกต้องจริง (calibrated ดี) ภาพความมั่นใจต่ำกระจุกในตำหนิ: Surface 27% มี conf<0.7, Internal 18%, Healthy 5% (median 0.89 / 0.94 / 0.97) — ถ้าสุ่มเฉพาะตำหนิจะเจอบ่อย ค่าเฉลี่ย 3 seed ต่ำลงเมื่อโมเดลเห็นไม่ตรงกัน (โมเดลเดี่ยว median 0.981)

### งาน A — ถัง "ไม่แน่ใจ → ส่งคนตรวจ" (ไม่ต้องเทรน)

1. โหมด 1 และโหมดสายพาน: เพิ่ม slider เกณฑ์ความมั่นใจ (ค่าเริ่ม 0.70, ช่วง 0.50–0.95) ถ้า max prob < เกณฑ์ → ผลเป็น "ไม่แน่ใจ — ส่งคนตรวจ" (ยังแสดงคลาสที่โหวตชนะไว้เป็นข้อมูลรอง)
2. โหมดสายพาน: เพิ่มถังที่ 4 "ส่งคนตรวจ" แสดง % ที่ส่งคนตรวจ และความแม่นยำของส่วนที่โมเดลตัดสินเอง
3. เพิ่มกราฟ/ตาราง (คำนวณจาก `confirm_results.json` ทั้ง 819 ภาพ ไม่ใช่แค่ภาพที่สุ่ม): ที่เกณฑ์ต่างๆ (0.5, 0.6, 0.7, 0.8, 0.9) → % ส่งคนตรวจ vs accuracy และ macro-F1 ของส่วนที่ตัดสินเอง + จำนวนตำหนิที่หลุดไปถังขายผลสด — คาดที่ 0.70: ส่งคนตรวจ ~10.7%, accuracy ส่วนที่เหลือ ~96% (ให้คำนวณจริงแล้วรายงาน)
4. บันทึกตัวเลขในข้อ 3 ลงหัวข้อนี้ (เป็นผลเชิงปฏิบัติสำหรับเล่ม — ระบุว่าเกณฑ์ 0.70 เลือกล่วงหน้า ไม่ได้ปรับจากผล)

### งาน B — สลับโมเดลในโหมด 1 (เทียบโมเดลบนแอปเปิลลูกเดียวกัน)

**กฎ**: ทุกโมเดลต้อง out-of-fold เหมือนเดิม (ใช้ fold split เดิม StratifiedKFold n=5 random_state=42 → ใช้ `fold_assignment.json` เดียวกัน) ห้ามใช้ `*_best.pt` ของ single-split เดิม (คนละ split)

1. **เทรนเพิ่มพร้อมเซฟ checkpoint** ด้วย protocol multi-seed เดิม **seed 0** (deterministic → ต้องได้ prediction ตรงกับ `multiseed_results.json` seed 0 ทุกภาพ ให้ assert ตรวจ): `visible`, `full`, `nir` (raw) → `outputs/dl/demo_ckpt/{variant}_seed0_fold{K}.pt` (~30 นาที, อยู่ใน `.gitignore`) — ถ้าไม่ตรงเป๊ะ ให้บันทึกความต่างและใช้ผลจาก checkpoint ใหม่เป็นตัวเลขของ demo พร้อมหมายเหตุ
2. **Classical**: เทรน RandomForest ต่อ fold ด้วย hyperparameter/feature เดิมของรอบ multi-seed (deterministic) → `outputs/dl/demo_ckpt/rf_fold{K}.joblib` ตรวจว่า prediction ตรงกับ `multiseed_results.json`
3. Dropdown "โมเดล" ในโหมด 1 (ค่าเริ่ม = สูตรสุดท้าย):
   - สูตรสุดท้าย: NIR (RE+IR) ภาพ SingleSpectra + TTA + 3 seed (มีอยู่แล้ว)
   - NIR (RE+IR) โมเดลเดี่ยว (`nir` seed 0)
   - Visible (BL+GR) (`visible` seed 0)
   - Vis+NIR 4 แบนด์ (`full` seed 0)
   - Classical: Random Forest บนฟีเจอร์ที่สกัดเอง (แสดง top SHAP features ของภาพนั้นแทน Grad-CAM ถ้าทำได้ง่าย ไม่งั้นแสดงแค่ผล)
   - แต่ละตัวแสดงชื่อ/แบนด์ที่ใช้ และตัวเลขผลรวมของโมเดลนั้นจากไฟล์ผล (macro-F1/acc) บนการ์ด
   - ภาพที่แสดงต้องเปลี่ยนตามแบนด์ของโมเดล (visible → BL/GR, ภาพ raw สำหรับโมเดล raw)
4. ปุ่ม **"เทียบทุกโมเดล"**: แอปเปิลลูกเดียวกัน → ตาราง 1 แถวต่อโมเดล (ผลทาย, ความมั่นใจ, ถูก/ผิด) — ไว้โชว์ว่าลูกเดียวกัน Visible ทายผิดแต่ NIR ทายถูก
5. **ภาพตัวอย่างสำหรับพรีเซนต์**: สคริปต์หา RawA ที่ (ก) สูตรสุดท้ายถูก + Visible ผิด (โดยเฉพาะ Bruise/Flesh_damage) 5 ภาพ (ข) ทุกโมเดลผิด (ภาพยากจริง เช่น Rot) 3 ภาพ (ค) สูตรสุดท้ายความมั่นใจต่ำ <0.6 3 ภาพ → บันทึกรายการ id + เหตุผลลงหัวข้อนี้ และทำ dropdown "ภาพตัวอย่างสำหรับพรีเซนต์"

### ทั่วไป

- ไม่เปลี่ยนตัวเลขผลหลักในเล่ม (demo อ่านจากไฟล์ผลเดิม) · ยังห้ามเปิดแอปออกนอกเครื่อง
- อัปเดต `src/demo_selftest.py` ให้ทดสอบทุกโมเดล + ถังไม่แน่ใจ แล้วรัน
- commit (ห้าม `git add -A`) แล้ว **push** (ตอนนี้ค้าง 4 commit) ตรวจ `gh auth status` เป็น Chinnawan

## งานที่ต้องทำต่อ — พิสูจน์ "NIR ดีกว่า Visible" ให้แน่น (โฟกัสหลักของธีสิส) (2026-10-17)

> ผู้ใช้ตัดสินใจ 2026-10-17: เป้าหมายธีสิสคือ **พิสูจน์ว่า NIR ดีกว่า Visible จริงหรือไม่ และเพราะอะไร** ไม่ใช่ทำระบบใช้งานจริง
> → หัวข้อ "Demo app รอบ 2" ด้านบน **พักไว้** (ไม่ต้องทำ) ใช้ demo เดิมพอ

### หลักฐานที่มีแล้ว (multiseed, 3 seeds x 5 folds, pipeline เดียวกันทุกอย่าง)
| variant | แบนด์ | Macro F1 | Acc | F1 Healthy / Internal / Surface |
|---|---|---|---|---|
| visible | BL450+GR500 | 0.688 ± 0.015 | 80.3% | 0.885 / 0.526 / 0.651 |
| full | ทั้ง 4 | 0.754 ± 0.012 | 85.0% | 0.914 / 0.625 / 0.729 |
| nir | RE750+IR800 | 0.843 ± 0.008 | 90.2% | 0.950 / 0.794 / 0.786 |
- nir − visible = +0.156 CI [+0.127,+0.182] 15/15 → REAL; nir − full = +0.089 15/15 → REAL
- single band: re/ir (~0.85) > bl/gr (~0.75) ทุกคู่ REAL; init fix ยืนยันว่าไม่ใช่ artifact (**แต่ 2026-10-18 พบว่าแบนด์ใน raw เหลื่อมกัน ~15–22 px → N5 ยืนยัน: บนภาพ SS ที่ตรงกัน full_ss ≈ nir_ss และ nir_ss − visible_ss = +0.091 ดู "ผล N5"**)
- ช่องว่างใหญ่สุดที่ Internal (+0.27) → สอดคล้องกับสมมติฐาน "NIR ทะลุผิวเห็นรอยช้ำใต้ผิว"

### ข้อจำกัดที่ต้องเขียนในเล่ม (ห้ามลืม)
- CAPA ไม่มีแบนด์สีแดง → "Visible" ของเราคือ **น้ำเงิน+เขียว (450/500)** ไม่ใช่ RGB เต็ม claim ต้องเขียนว่า "NIR 750/800 nm ดีกว่า visible 450/500 nm" ไม่ใช่ "ดีกว่ากล้อง RGB"
- ข้อมูลชุดเดียว (CAPA, 819 ลูก) ไม่มี external test (ค้นแล้ว 2026-10-17 ไม่มี dataset สาธารณะที่ใช้ร่วมได้)

### งาน (เรียงตามลำดับ; ส่วนใหญ่ไม่ต้องเทรนใหม่ — ใช้ json ที่มีอยู่)
**N1. หลักฐานทางฟิสิกส์ (ไม่ใช้ CNN เลย)** — สคริปต์ `src/band_contrast.py`
- ใช้ defect masks 246 ภาพ (`segmentations/.../SBW_DefectNNNN.bmp`) จับคู่กับภาพแต่ละแบนด์ (ตรวจ alignment/ขนาดก่อน ถ้า mask ไม่ตรงภาพให้บันทึกและใช้เฉพาะที่ตรง)
- ต่อภาพ ต่อแบนด์: mean intensity ใน defect vs ผิวดีรอบๆ (ring dilate mask, ตัดพื้นหลังออก) → contrast = (healthy−defect)/(healthy+defect) และ Cohen's d
- สรุปแยกตามชนิดตำหนิ (Bruise, Rot, Flesh_damage, Russet, Scald, Hail, ...) ต่อแบนด์ → ตาราง + กราฟแท่ง `outputs/figures/band_contrast.png`
- paired test ต่อภาพ: contrast(RE/IR) − contrast(BL/GR) bootstrap CI แยก Internal vs Surface
- คาดหวัง: Internal (โดยเฉพาะ Bruise) contrast ที่ NIR สูงกว่า visible ชัด; ถ้าไม่เป็นแบบนั้นให้รายงานตามจริง

**N2. per-class / per-tissue nir vs visible** — จาก `multiseed_results.json` (seeds 0-2)
- paired (seed+fold) F1 ต่อคลาส + recall ต่อ tissue (Bruise, Rot, Flesh_damage, ...) ของ nir vs visible พร้อม CI
- McNemar test บน pooled out-of-fold predictions (ต่อ seed) nir vs visible และ nir vs full (paper ใช้ McNemar)

**N3. ไม่ขึ้นกับชนิดโมเดล** — classical ML (RF, SVM; LightGBM ถ้ามี) ด้วย feature ชุดเดิมแต่แยก visible-only vs nir-only, fold เดิม (`fold_assignment.json`)
- ถ้า NIR > visible ทั้งใน CNN และ classical → ข้อสรุปมาจากข้อมูล ไม่ใช่จาก CNN

**N4. Visible มีข้อมูลเสริมไหม? (อธิบายว่าทำไม full < nir)** — จาก json ที่มี ไม่ต้องเทรน
- late fusion ภายใน fold เดียวกัน: softmax เฉลี่ย nir+visible (seed+fold เดียวกัน) และแบบถ่วงน้ำหนัก w∈{0.1..0.5} (เลือก w ด้วย inner rule ห้ามดู test; ถ้าทำไม่ได้ให้รายงานเฉพาะ w=0.5 กับ curve แบบ exploratory)
- ถ้า late fusion ≤ nir → visible ไม่เพิ่มข้อมูล; ถ้า > nir แต่ full (early fusion) < nir → ปัญหาอยู่ที่วิธีรวมแบนด์ (เขียนเป็น discussion)
- ตรวจด้วยว่าภาพแบนด์ visible กับ NIR ของลูกเดียวกัน align กันไหม (ถ้าเหลื่อมกันอาจอธิบาย full < nir) — วัด shift ด้วย phase correlation บนตัวอย่าง 50 ภาพ

**N5 (ขยาย 2026-10-18 — รันข้ามคืน) visible_ss + full_ss บนภาพ SingleSpectra + บันทึก checkpoint ให้ demo**
- เหตุผลใหม่จาก N4: ภาพ raw (Processed) แต่ละแบนด์เหลื่อมกัน ~14-22 px (ที่ 224px) ส่วน SingleSpectra เหลื่อม < 1 px (median) → full (raw) < nir อาจเป็นเพราะภาพซ้อนไม่ตรง ไม่ใช่เพราะ visible ไม่มีข้อมูล
- ก่อนรัน: ตรวจด้วยตาว่าการเหลื่อมใน raw เป็นจริง (overlay BL/RE ของ 3 ลูก บันทึกไว้ใน outputs/figures/ — gitignored เพราะเป็นภาพจาก dataset) เพื่อกันว่าเป็น artifact ของ phase correlation
- เทรน `visible_ss` (BL+GR) และ `full_ss` (4 แบนด์) seeds 3-5 x 5 folds pipeline เดียวกับ nir_ss ของ confirm run (TTA ด้วย) **และบันทึก checkpoint** ไปที่ `outputs/dl/confirm_ckpt/{variant}_seed{S}_fold{K}.pt` + เก็บ test_probs/tta_test_probs
- เปรียบเทียบแบบ paired (seed+fold) กับ `nir_ss` ของ confirm run: nir_ss − visible_ss, nir_ss − full_ss, full_ss − full(raw, จาก fusion_results), late fusion w=0.5 บน ss
- การตีความที่ล็อกไว้ก่อนรัน:
  - nir_ss > visible_ss ยังชัด → ข้อสรุปหลักแข็งแรงบน preprocessing เดียวกัน
  - full_ss ≈/> nir_ss → "4 แบนด์แย่กว่า" ใน raw มาจาก misalignment (เขียนเป็น discussion); full_ss < nir_ss ชัด → visible ทำให้แย่ลงจริงแม้ภาพตรงกัน
- เพิ่มผลเข้า `nir_proof_report.txt` (หัวข้อ N5) และหัวข้อ "ผล" ในแผนนี้ แล้ว commit/push (กฎเดิม)
- รันผ่าน `run_n5.bat` (เวลาโดยประมาณ 2-3 ชม.)

**N7. Demo: สลับโมเดล (ทำหลัง N5 เสร็จ ใช้ checkpoint จาก N5 ไม่ต้องเทรนเพิ่ม)** — ปลดพักบางส่วนของ "Demo app รอบ 2"
- โหมด 1 เพิ่ม dropdown: `NIR (RE+IR, สูตรสุดท้าย)` / `Visible (BL+GR)` / `4 แบนด์` — ทุกตัวเป็น ss, seeds 3-5 ของ fold ที่ลูกนั้นอยู่ (out-of-fold เสมอ), TTA + soft vote
- ปุ่ม "เทียบทุกโมเดล": ตาราง 3 แถว (โมเดล / คำทาย / ความมั่นใจ / ถูก-ผิด) + Grad-CAM ของแต่ละโมเดลข้างกัน
- dropdown "ตัวอย่างสำหรับพรีเซนต์": (ก) Bruise ที่ NIR ถูก Visible ผิด (ข) ลูกที่ทุกโมเดลผิด (ค) ลูกที่ Visible ถูก NIR ผิด (ต้องมีเพื่อความซื่อตรง) — คัดจาก out-of-fold predictions อัตโนมัติ
- ไม่ทำ: ถัง "ไม่แน่ใจ", Random Forest ใน demo, belt mode ใหม่
- อัปเดต demo_selftest (assert ว่าคำทายใน demo ตรงกับ test_predictions ใน json) แล้ว commit/push

**N8. Error analysis + ผลแบบ 2 กลุ่ม (ทำหลัง N5; ไม่ต้องเทรน — ใช้ json ที่มี)** — `src/error_analysis.py` → เพิ่มหัวข้อใน `nir_proof_report.txt`
- แยกความผิดของสูตรสุดท้าย (nir_ss TTA 3-seed soft vote, pooled OOF 819) เป็น 3 แบบ: (ก) มีตำหนิแต่ทายว่าดี (ข) ลูกดีถูกหาว่ามีตำหนิ (ค) เจอตำหนิแต่ใส่ผิดกลุ่ม — ค่าที่คำนวณเบื้องต้น 2026-10-18: ก=29/246, ข=19/573, ค=13 (รวม 61)
- ตาราง per-tissue: จำนวนที่ทายเป็น H/I/S (เบื้องต้น: Rot 23 → 3/13/7, Flesh_damage 24 → 1/19/4, Bruise 55 → 4/50/1) + ความมั่นใจเฉลี่ยตอนผิด
- ผลแบบ 2 กลุ่ม (ดี vs มีตำหนิ = รวม prob Internal+Surface): accuracy/F1/recall ตำหนิ ของ nir_ss เทียบ **visible_ss และ full_ss (จาก N5, seeds 3-5, ss เหมือนกัน → แฟร์)** + McNemar; เบื้องต้น nir_ss binary acc 94.1%
- คัดตัวอย่างภาพ "เจอตำหนิแต่ผิดกลุ่ม" 3 ลูก (Grad-CAM ตรง mask) ไว้ใน outputs/figures/ (gitignored)
- ห้าม: จูนโมเดลให้ลูกเหล่านี้ถูก / ย้าย Rot ไปกลุ่ม Surface หลังเห็นผล

### เหตุผลสำหรับเล่ม (Discussion): ทำไม 4 แบนด์ < NIR — สรุป 2026-10-18 (ข้อ 4 รอผล N5)
1. Visible ไม่มีข้อมูลใหม่ (late fusion +0.004 n.s.; visible ถูก/NIR ผิด 4.2%) — ยืนยันแล้ว
2. Visible พาความแปรปรวนสีผิวมา (null contrast visible ≈5× NIR) — Kleynen et al. 2003: 500 nm ใช้แยก blush/ground colour; bruise เห็นชัดสุดที่ 700–920 nm
3. ข้อมูลน้อย + แบนด์มาก (819 ภาพ; 0.843 → 0.810 → 0.754) — สอดคล้อง Hughes 1968
4. ภาพ raw ซ้อนไม่ตรง 14–22 px — Kleynen et al. 2005: MultiSpec Agro-Imager ฉาย 4 sub-images บน CCD เดียว "pixel coordinates did not match perfectly, alignment was necessary"; Zhang et al. 2019 (ICCV): shift 6 px → miss rate 15.18→25.10 — **ผล N5 (2026-10-18): บนภาพ SS ที่ตรงกัน full_ss 0.842 ≈ nir_ss 0.858 (ไม่ต่าง) และ full_ss − full(raw) = +0.111 REAL → นี่คือเหตุผลหลักที่ 4 แบนด์ < NIR ในภาพ raw** (ข้อจำกัด: SS แก้ทั้งการเหลื่อมและ flat-field พร้อมกัน แยกไม่ได้ 100%) → ข้อ 3 (Hughes) อ่อนลง: เมื่อภาพตรงกัน การเพิ่มแบนด์ไม่ได้ทำให้แย่ลง; ข้อ 1–2 ยังใช้อธิบายว่าทำไมการเพิ่ม visible ไม่ได้ช่วย
- ตัดออกแล้ว: init ของ conv แรก (full_fix 0.714 ยังแย่)
- อ้างอิง: Kleynen 2003 doi:10.1016/S0925-5214(03)00112-1; Kleynen 2005 doi:10.1016/j.jfoodeng.2004.07.008; Hughes 1968 IEEE TIT 14(1) doi:10.1109/TIT.1968.1054102; Zhang et al. ICCV 2019 (arXiv 1901.02645)
- ข้อจำกัดที่ต้องเขียน: Visible = 450/500 เท่านั้น (ไม่มีแดง), dataset เดียว, ภาพ CAPA ในเล่มต้องเครดิต ULG

**N6. สรุป** — `src/nir_proof_report.py` → `outputs/logs/nir_proof_report.txt` รวม N1-N5 เป็นตารางพร้อมใส่เล่ม และบันทึกผลกลับลงหัวข้อนี้ (หัวข้อย่อย "ผล") รวม % accuracy ทุกที่ที่มี F1
- commit เฉพาะ src/*.py, PROJECT_PLAN.md, outputs/logs/*.txt, json สรุป (ห้าม `git add -A`, ห้าม commit ภาพจาก dataset/figure ที่มีภาพแอปเปิล; กราฟตัวเลขล้วนได้) แล้ว push หลังเช็ค `gh auth status` เป็น Chinnawan

### ผล N1–N4 + N6 (ทำเสร็จ 2026-10-18; N5 ยังไม่ทำตามที่ผู้ใช้สั่ง)
รายงานรวม `outputs/logs/nir_proof_report.txt` (สร้างด้วย `python -m src.nir_proof_report`), สรุป `outputs/dl/nir_proof_summary.json`, ผลดิบ `outputs/dl/nir_proof_n1..n4.json`, กราฟ `outputs/figures/band_contrast.png` (ตัวเลขล้วน ไม่มีภาพแอปเปิล)

**N1 — ฟิสิกส์ (ไม่ใช้ CNN)** `src/band_contrast.py` — 244 ภาพตำหนิ (ข้าม 2 ภาพ mask เล็กเกิน, mask ไม่ตรงภาพ 0 ภาพ) ภาพ SingleSpectra ความละเอียดเต็ม
- **contrast ดิบ (|c|) visible สูงกว่า NIR** (NIR−vis = −0.087; NIR สูงกว่าแค่ 14.3% ของภาพ) — แต่เป็นเพราะผิว Jonagold สองสีในแบนด์ visible: วาง mask เดียวกันบนแอปเปิลปกติ (null) ได้ |c| visible ≈ 0.031 vs NIR ≈ 0.006 → **ผิวปกติใน visible แปรปรวนมากกว่า ~5 เท่า** contrast ดิบจึงวัด "การมองเห็นตำหนิ" ไม่ได้
- ตัววัดหลัก = **AUC แยก contrast ของตำหนิออกจาก contrast ของผิวปกติ** (0.5 = แยกไม่ได้)

| กลุ่ม | n | AUC BL | GR | RE | IR | AUC(NIR) − AUC(vis) [95% CI] |
|---|---|---|---|---|---|---|
| ตำหนิทั้งหมด | 244 | 0.90 | 0.92 | 0.96 | 0.93 | **+0.036 [+0.013, +0.061]** ✓ |
| Internal_Defect | 100 | 0.92 | 0.88 | 0.95 | 0.92 | +0.038 [−0.002, +0.083] (ก้ำกึ่ง) |
| Surface_Defect | 144 | 0.89 | 0.94 | 0.98 | 0.93 | **+0.039 [+0.010, +0.069]** ✓ |
| Bruise (ช้ำ) | 55 | 0.85 | 0.78 | 0.92 | 0.88 | **+0.084 [+0.012, +0.161]** ✓ |
| Scald | 17 | 0.86 | 0.97 | 1.00 | 0.98 | **+0.075 [+0.004, +0.167]** ✓ |
| Other | 20 | 0.75 | 0.86 | 0.98 | 0.93 | **+0.149 [+0.050, +0.254]** ✓ |
| Flesh_damage | 24 | 0.97 | 0.96 | 0.94 | 0.95 | −0.019 (ไม่ต่าง) |
| Russet | 42 | 0.85 | 0.89 | 0.92 | 0.81 | −0.008 (ไม่ต่าง) |
| Rot / Hail / Frost / Limb_rub | — | ≥0.94 ทุกแบนด์ | | | | ไม่ต่าง (เพดาน ~1.0) |
- Cohen's d (NIR−vis, ส่วนเกินจาก null): รวม −0.05 (ไม่ต่าง), **Bruise +0.50 [+0.34,+0.66]**, Other +0.97, แต่ Flesh_damage −1.76 และ Russet −0.45 (visible ชัดกว่า — Russet เป็นตำหนิสี/ผิว คาดได้)
- **สรุป N1:** ข้อมูลดิบยืนยันว่า NIR แยกตำหนิออกจากผิวปกติได้ดีกว่า โดยเฉพาะ **รอยช้ำ (Bruise)** ซึ่งตรงกับสมมติฐาน "NIR ทะลุผิว เห็นเนื้อช้ำใต้ผิว" — เหตุผลหลักคือผิวปกติใน NIR สม่ำเสมอ (ไม่มีลายสี anthocyanin/chlorophyll) ไม่ใช่เพราะตำหนิ "เข้ม" กว่า; ตำหนิบางชนิด (Flesh_damage, Russet) visible เห็นพอๆ กันหรือดีกว่า

**N2 — คลาส/ชนิดตำหนิที่ NIR ชนะ** (multiseed seeds 0–2; nir 0.843 / 90.2%, visible 0.688 / 80.3%, full 0.754 / 85.0%)

| คลาส | F1 nir | F1 visible | nir − visible | nir − full |
|---|---|---|---|---|
| Healthy | 0.950 | 0.886 | +0.064 REAL (15/15) | +0.036 REAL |
| Internal_Defect | 0.796 | 0.528 | **+0.268 REAL** (14/15) | +0.174 REAL |
| Surface_Defect | 0.785 | 0.649 | +0.136 REAL (14/15) | +0.058 (11/15 ไม่ผ่านเกณฑ์) |

- recall ต่อชนิด (เฉลี่ย 3 seed, CI bootstrap ข้ามลูก): **Bruise NIR 90.3% vs visible 37.0% (+53.3 จุด [+41.2,+64.8])**, Other +25.0 จุด [+6.7,+43.3], Healthy 94.3% vs 87.5% (+6.8 [+3.1,+10.5]) — ชนิดอื่นเป็นบวกทุกตัวแต่ CI คร่อม 0 (n น้อย); Flesh_damage 81.9% = 81.9% (เท่ากันพอดี สอดคล้อง N1)
- McNemar (819 ลูก out-of-fold ต่อ seed): nir vs visible p = 2.2e-10, 4.8e-10, 7.6e-10 (nir ถูกคนเดียว ~120 ลูก vs visible ถูกคนเดียว ~40 ลูก); nir vs full p = 1.4e-6, 8.7e-5, 0.018
- **สรุป N2:** ช่องว่างมาจาก Internal_Defect และเจาะจงคือ **รอยช้ำ** — ตรงกับ N1 (ฟิสิกส์) ทั้งทิศทางและชนิดตำหนิ

**N3 — ไม่ขึ้นกับชนิดโมเดล (classical ML, feature ทำมือ, fold เดิม)** — feature visible = BL_*/GR_* + NDI_BL_GR (31 ตัว), NIR = RE_*/IR_* + NDI_RE_IR (31 ตัว), all = 66 ตัว (ชุด all ได้ตัวเลขตรงกับ multiseed classical เดิมเป๊ะ)

| โมเดล | visible F1 (acc) | NIR F1 (acc) | all F1 (acc) | NIR − vis | fold ที่ NIR ชนะ | McNemar p |
|---|---|---|---|---|---|---|
| SVM | 0.483 (57.8%) | 0.553 (63.1%) | 0.573 (66.9%) | +0.071 [−0.007,+0.148] | 4/5 | **0.021** |
| RandomForest | 0.520 (66.5%) | 0.556 (68.4%) | 0.589 (71.3%) | +0.036 [−0.038,+0.111] | 4/5 | 0.37 |
| LightGBM | 0.487 (68.4%) | 0.545 (72.3%) | 0.546 (72.3%) | +0.058 [−0.037,+0.154] | 4/5 | **0.019** |
- **สรุป N3:** NIR > visible ทั้ง 3 โมเดล (ทิศทางเดียวกับ CNN; McNemar มีนัยสำคัญ 2/3) แต่ fold t-CI (5 fold) คร่อม 0 ทุกตัว → หลักฐาน "ทิศทางสอดคล้อง" ไม่ใช่หลักฐานหลัก; ต่างจาก CNN ตรงที่ใน classical การรวม 4 แบนด์ (all) ไม่แย่กว่า NIR

**N4 — visible เพิ่มข้อมูลไหม? (late fusion vs early fusion)** seed 3–5, 15 คู่ (nir จาก `confirm_results.json`, visible/full เทรนใหม่ → `fusion_results.json` 30/30 run error 0)

| วิธี | macro-F1 | accuracy |
|---|---|---|
| nir | 0.849 | 90.4% |
| visible | 0.678 | 79.3% |
| full (early fusion, 4 แบนด์ซ้อนกัน) | 0.731 | 83.0% |
| late fusion w=0.5 (หลัก) | 0.853 | 90.9% |
| late w=0.1 / 0.2 / 0.3 / 0.4 (exploratory) | 0.851 / 0.852 / 0.855 / 0.861 | 90.7 / 90.8 / 91.0 / 91.4% |
| nir [TTA] / late w=0.5 [TTA] | 0.855 / 0.871 | 90.9% / 91.9% |
- late w=0.5 − nir = +0.004 [−0.015,+0.023] 8/15 → **ไม่ต่าง**; แบบ TTA +0.016 10/15 → ไม่ผ่านเกณฑ์
- late w=0.5 − full = +0.122 REAL 15/15; nir − full = **+0.118 REAL 15/15** (ยืนยัน nir > full ด้วย seed ใหม่); nir − visible = +0.171 REAL 15/15; full − visible = +0.053 REAL
- complementarity: NIR ผิดแต่ visible ถูก 4.2% ของลูก, NIR ถูกแต่ visible ผิด 15.3%, ผิดทั้งคู่ 5.4%
- **สรุป N4:** visible **ไม่ได้เพิ่มข้อมูลที่ใช้ได้** เหนือ NIR (late fusion ≈ nir) — และ early fusion (full) แย่กว่าทั้ง nir และ late fusion ชัด → ปัญหาของ full อยู่ที่ **วิธีรวมแบนด์** (ดูข้อค้นพบ alignment ต่อไป)

**N4 (ต่อ) — ⚠️ ข้อค้นพบสำคัญ: แบนด์ในภาพ raw ไม่ตรงกัน (misregistration)** — phase correlation ของภาพ gradient, 50 ลูกสุ่ม, หน่วย px ที่ 224×224

| คู่แบนด์ | raw (4 quadrant) median shift | SingleSpectra median shift |
|---|---|---|
| BL–GR | 22.6 px (dx −15.8, dy +16.2) | 0.4 px |
| RE–IR | 20.7 px (dx +15.6, dy +13.7) | 0.4 px |
| BL–RE / BL–IR / GR–RE / GR–IR | 14–17 px | 0.4–0.6 px |
- shift ใน raw **เป็นระบบ** (ค่าเกือบคงที่ทุกลูก max ≈ median สำหรับคู่ BL–GR, RE–IR) — เกิดจากการตัด 4 quadrant ของภาพเดียวที่เลนส์แต่ละแบนด์มองจากตำแหน่งต่างกัน; SingleSpectra (ผ่าน geometric correction) ตรงกันในระดับ <1 px (max สูงบางภาพเป็นจุดที่ phase correlation ล้มเหลว ไม่ใช่ภาพเหลื่อมจริง)
- ผลกระทบต่อข้อสรุป:
  1. **การเทียบแบนด์เดี่ยว (RE/IR > BL/GR, REAL ทุกคู่) ไม่ได้รับผลกระทบ** — หลักฐานหลักของ "NIR > visible" ยังแน่น
  2. **nir vs visible (2 แบนด์ทั้งคู่) ยุติธรรม** — ทั้งสองคู่เหลื่อม ~21–23 px พอๆ กัน
  3. **nir > full และ "F1 ลดตามจำนวนช่อง" อาจเป็นผลจาก misregistration** (ตำหนิเดียวไปอยู่คนละตำแหน่งในแต่ละช่อง) ไม่ใช่เพราะ visible "รบกวน" — ข้อสรุปเดิม "NIR > Vis+NIR ไม่ใช่ artifact ของ init" ยังจริงในแง่ init แต่ **ตอนนี้มีตัวแปรกวนใหม่คือ alignment** ต้องเขียนในเล่มอย่างระวัง
  4. ข้อสังเกต: nir_ss (ตรงกัน) − nir (raw เหลื่อม) = +0.009 แค่นั้น → การเหลื่อมภายใน RE–IR แทบไม่กระทบ (RE กับ IR ข้อมูลคล้ายกันมาก) แต่ระหว่าง visible–NIR ยังไม่รู้
- **แนะนำ (= N5 ขยาย):** เทรน `visible_ss` และ `full_ss` บนภาพ SingleSpectra ที่ align แล้ว (3 seed × 5 fold, ~1.5 ชม.) — ถ้า full_ss ≥ nir_ss แปลว่า full < nir เดิมเป็นผลของ misregistration; ถ้ายัง < แปลว่า visible ไม่ช่วยจริงแม้ภาพตรงกัน

**ข้อสรุปรวม "NIR ดีกว่า Visible จริงไหม และเพราะอะไร"**
1. **จริง** — CNN: nir 0.843–0.849 (90.2–90.4%) vs visible 0.678–0.688 (79.3–80.3%) REAL 15/15 ใน seed สองชุดแยกกัน (0–2 และ 3–5), McNemar p < 1e-9; แบนด์เดี่ยว RE/IR > BL/GR ทุกคู่
2. **ไม่ใช่ artifact ของ CNN** — classical ML ทั้ง 3 โมเดลทิศทางเดียวกัน (แรงน้อยกว่า), และ **ฟิสิกส์ของภาพดิบ** (N1) ให้ AUC NIR > visible โดยไม่ใช้โมเดลเลย
3. **เพราะอะไร** — (ก) ผิวปกติใน NIR สม่ำเสมอกว่า ~5 เท่า (ไม่มีลายสีผิว) ตำหนิจึงโดดเด่นจากพื้นหลัง (ข) รอยช้ำใต้ผิวเห็นชัดใน NIR (AUC +0.084, recall +53 จุด) — ช่องว่าง F1 ใหญ่สุดที่ Internal_Defect (+0.27) มาจากตรงนี้
4. **visible ไม่เพิ่มข้อมูล** เหนือ NIR (late fusion ≈ nir) ส่วนที่ full < nir อาจมาจาก misregistration ของภาพ raw — ~~ยังพิสูจน์ไม่ได้จนกว่าจะทำ N5~~ **N5 ยืนยันแล้ว: full_ss ≈ nir_ss ดู "ผล N5"; และช่องว่าง NIR − visible บนภาพเดียวกัน = +0.091 (ไม่ใช่ +0.17)**
5. ข้อจำกัด: visible = 450/500 nm (ไม่มีแดง), dataset เดียว, Internal ใน N1 ก้ำกึ่ง (CI แตะ 0), ตำหนิบางชนิด (Flesh_damage, Russet) visible เห็นพอๆ กัน

### การตัดสินใจระหว่างทำ N1–N6 (2026-10-17/18 — เลือกเองตามคำสั่งผู้ใช้ "เลือกทางที่สมเหตุสมผลแล้วบันทึก")
1. **UI demo:** ตามคำขอผู้ใช้ — % ความมั่นใจตัวใหญ่ (56px) + แถบ progress หนาขึ้น, เฉลยย้ายไปกล่องขวา: ✔ "ถูก" พื้นเขียว / ✘ "ผิด" พื้นแดง; **เปลี่ยนสีคลาสในการ์ดทำนายเป็น น้ำเงิน (ปกติ) / ส้ม (ตำหนิผิว) / ม่วง (ตำหนิภายใน)** เพื่อสงวนเขียว/แดงไว้สำหรับถูก/ผิดเท่านั้น (ไม่ให้สับสน) — ต้องปิดแล้วเปิด `run_demo.bat` ใหม่ถึงจะเห็น
2. **N4 ต้องเทรนเพิ่ม** (แผนเขียนว่าไม่ต้อง): `multiseed_results.json` ไม่ได้เก็บ softmax probabilities จึงทำ late fusion ไม่ได้ → เทรน `visible` และ `full` seed 3–5 บนภาพ raw (`src/run_fusion_models.py`, 30 run) ให้จับคู่กับ `nir` ใน `confirm_results.json` (seed+fold เดียวกัน, ตรวจ test_idx ตรงกันด้วย assert) — ใส่ `full` ด้วยเพื่อเทียบ early fusion ใน seed เดียวกัน
3. **N1 ใช้ภาพ SingleSpectra ความละเอียดเต็ม** (mask อยู่ใน geometry ของ SS); ใช้ค่าสัมบูรณ์ (ตำหนิเข้มหรือสว่างกว่าผิวก็เห็นได้ทั้งคู่); **เพิ่ม null baseline + AUC เป็นตัววัดหลัก** เพราะ contrast ดิบถูกกวนด้วยลายสีผิว (Jonagold สองสี) — รายงาน contrast ดิบไว้ด้วยตามจริง
4. **N2:** CI ของ recall ต่อชนิดตำหนิใช้ bootstrap ข้ามลูก (recall ต่อลูกเฉลี่ย 3 seed ก่อน) เพราะ n ต่อชนิดน้อย; McNemar แยกต่อ seed (ไม่รวม seed เพราะไม่อิสระกัน)
5. **N3:** แบ่ง feature ตามชื่อแบนด์ — NDI ข้ามกลุ่ม (เช่น BL/IR) ตัดออกจากทั้งสองชุดเพื่อไม่ให้ชุดใดได้ข้อมูลอีกกลุ่ม; scaler fit ใน fold; LightGBM `deterministic=True`
6. **N4:** ไม่มี val probabilities ให้เลือก w แบบ inner rule → **w=0.5 เป็นผลหลัก** ค่าอื่นเป็น exploratory (w=0.4 สูงสุด 0.861 แต่ห้ามใช้เป็นผลหลักเพราะเลือกจาก test); TTA เป็นผลรอง
7. alignment วัดบนภาพ gradient (ไม่ใช่ intensity) เพราะความสว่างแต่ละแบนด์ต่างกันมาก; สุ่ม 50 ลูกด้วย seed 42
8. `outputs/logs/` อยู่ใน .gitignore → ใช้ `git add -f outputs/logs/nir_proof_report.txt` เฉพาะไฟล์นี้
9. N5 ยังไม่ทำ (ผู้ใช้สั่ง) แต่ **แนะนำให้ทำเป็นลำดับถัดไป และขยายเป็น `visible_ss` + `full_ss`** เพราะข้อค้นพบ alignment

### ผล N5 (ขยาย) + N7 + N8 + N6 รอบ 2 (ทำเสร็จ 2026-10-18)
รายงานรวม `outputs/logs/nir_proof_report.txt` (เพิ่มหัวข้อ N5, N8), ผลดิบ `outputs/dl/n5_results.json` (30/30 run, error 0, ~71 นาที), วิเคราะห์ `outputs/dl/nir_proof_n5.json`, `outputs/dl/n8_error_analysis.json`; checkpoint `outputs/dl/confirm_ckpt/{visible_ss,full_ss}_seed{3,4,5}_fold{0-4}.pt` (30 ไฟล์, gitignored)

**ตรวจ alignment ด้วยตาก่อนเทรน** (`src/alignment_visual.py` → `outputs/figures/private/alignment_overlay.png`, gitignored เพราะเป็นภาพ dataset) — overlay BL (ม่วง) กับ RE (เขียว) ของภาพ input จริงที่ใช้เทรน (npy 224×224) 3 ลูก (RawA0004, RawA0076, RawA0500):
- ภาพ raw: **ขอบม่วง/เขียวซ้อนเหลื่อมชัดเจนทุกลูก** ~16.5 px ทางแนวนอน (BL→RE shift −16.6/−16.6/−16.3 px) — เลื่อน RE กลับตามค่าที่ phase correlation วัดได้แล้วขอบหายหมด → shift เป็นของจริง ไม่ใช่ artifact ของ phase correlation; RE–IR ใน raw ก็เหลื่อมชัด (แนวทแยง)
- ภาพ SingleSpectra: ซ้อนตรงกันด้วยตา (shift ≤ 0.7 px)

**N5 — visible_ss / full_ss บนภาพ SingleSpectra (align แล้ว) เทียบ nir_ss** seed 3–5, 15 คู่ (seed, fold), recipe เดียวกับ nir_ss ทุกอย่าง (ต่างกันแค่แบนด์)

| โมเดล | ภาพ | macro-F1 | accuracy | F1 Healthy / Internal / Surface |
|---|---|---|---|---|
| **nir_ss** (RE+IR) | SS | **0.858** | **91.0%** | 0.953 / 0.841 / 0.781 |
| visible_ss (BL+GR) | SS | 0.767 | 85.3% | 0.914 / 0.667 / 0.720 |
| full_ss (4 แบนด์) | SS | 0.842 | 90.2% | 0.949 / 0.806 / 0.770 |
| nir | raw | 0.849 | 90.4% | 0.948 / 0.804 / 0.795 |
| visible | raw | 0.678 | 79.3% | 0.876 / 0.506 / 0.652 |
| full | raw | 0.731 | 83.0% | 0.901 / 0.595 / 0.697 |
| late fusion nir_ss+visible_ss w=0.5 | SS | 0.872 | 92.1% | 0.959 / 0.831 / 0.825 |
| 3-seed vote + TTA: nir_ss / visible_ss / full_ss / full(raw) | | 0.885 / 0.797 / 0.866 / 0.762 | 92.6% / 87.5% / 92.1% / 85.5% | |

| เปรียบเทียบ (paired, plain probs) | ผลต่าง F1 [95% CI] | คู่ทิศเดียวกัน | ผล |
|---|---|---|---|
| nir_ss − visible_ss | **+0.091 [+0.072, +0.109]** | 15/15, fold 5/5 | **REAL** (McNemar vote p = 3.4e-5) |
| nir_ss − full_ss | +0.016 [−0.005, +0.038] | 9/15 | ไม่ต่าง (McNemar p = 0.71) |
| full_ss − full (raw) | **+0.111 [+0.089, +0.132]** | 15/15 | **REAL** (McNemar p = 1.2e-9) |
| visible_ss − visible (raw) | **+0.089 [+0.063, +0.116]** | 15/15 | **REAL** |
| nir_ss − nir (raw) | +0.009 [−0.004, +0.024] | 8/15 | ไม่ต่าง (ตรงกับ confirm run) |
| full_ss − visible_ss | +0.075 | 15/15 | REAL |
| late_ss w=0.5 − nir_ss | +0.014 [+0.005, +0.023] | 9/15 | ไม่ผ่านเกณฑ์ (CI ไม่คร่อม 0 แต่ทิศไม่สม่ำเสมอ) |
| TTA: nir_ss − visible_ss / nir_ss − full_ss | +0.085 REAL / +0.016 ไม่ต่าง | | |
- per-class nir_ss − visible_ss: Healthy +0.038, **Internal +0.174**, Surface +0.060 — REAL ทุกคลาส; nir_ss − full_ss ไม่ต่างทุกคลาส; full_ss − full: Internal +0.211 REAL
- **ตีความตามกฎที่ล็อกไว้ก่อนรัน:**
  1. **nir_ss > visible_ss ยังชัด → ข้อสรุปหลัก "NIR > visible" แข็งแรงบน preprocessing เดียวกัน** (+0.091 F1, accuracy 91.0% vs 85.3%)
  2. **full_ss ≈ nir_ss → "4 แบนด์แย่กว่า NIR" ในภาพ raw มาจากภาพ raw (การเหลื่อมของแบนด์) ไม่ใช่เพราะ visible ทำให้แย่** — เขียนเป็น discussion
- ⚠️ ข้อควรระวัง (ต้องเขียนในเล่ม):
  - **ช่องว่าง NIR − visible บนภาพเดียวกันคือ +0.091 ไม่ใช่ +0.16/+0.17** (ตัวเลข raw พองเพราะภาพ raw ทำร้าย visible มากกว่า: visible ได้ +0.089 จาก SS แต่ NIR ได้แค่ +0.009) → ใช้ตัวเลข SS เป็นตัวเลขเทียบที่ยุติธรรมที่สุด แล้วรายงาน raw คู่กัน
  - SingleSpectra ต่างจาก raw **2 อย่าง** (geometric correction = แก้การเหลื่อม + flat-field correction = แก้แสงไม่สม่ำเสมอ) → ผลนี้แยกไม่ได้ว่าส่วนไหนมาจากอะไร; ข้อสังเกต: RE–IR ใน raw เหลื่อม ~21 px พอๆ กับ BL–GR (~23 px) แต่ NIR แทบไม่ได้อะไรจาก SS — อธิบายได้ว่า RE กับ IR ข้อมูลเกือบซ้ำกัน การเหลื่อมจึงเสียหายน้อย หรือ flat-field สำคัญกับ visible มากกว่า (ยังพิสูจน์ไม่ได้)
  - ถ้าต้องการแยก (ทางเลือก ไม่ได้ทำ): เทรนแบนด์เดี่ยว `bl_ss`/`gr_ss` เทียบ `bl_only`/`gr_only` raw (แบนด์เดี่ยวไม่มีปัญหาการเหลื่อม → ถ้ายังได้เพิ่มมาก = ผลของ flat-field) ~1 ชม.
  - late fusion บน SS (+0.014, CI ไม่คร่อม 0 แต่ 9/15) — มีสัญญาณอ่อนๆ ว่า visible อาจเพิ่มข้อมูลเล็กน้อยเมื่อภาพตรงกัน แต่ไม่ผ่านเกณฑ์
- checkpoint verification: ทุก checkpoint + mean/std ให้ prob ตรงกับที่บันทึก (batch-16 path ต่างสูงสุด 5e-6) และ path ภาพเดียวของ demo ให้คำทายตรงกันทั้งหมด ยกเว้น 2 ภาพที่ prob ที่บันทึกไว้ **เสมอกันพอดี** (RawA0796 0.4957/0.4957, RawA0165 0.4709/0.4710 → ปัดเศษ GPU batch-1 vs batch-16 พลิกได้) และ full_ss seed5 fold3 ต่าง prob 2.3e-2 (เกินเกณฑ์ 2e-2 นิดเดียว) แต่คำทายไม่เปลี่ยนเลย → ยอมรับ บันทึกไว้

**N7 — Demo สลับโมเดล** (`src/demo_app.py`; ทดสอบ `src/demo_selftest.py` ALL PASSED + ตรวจในเบราว์เซอร์)
- โหมด 1 มี dropdown "โมเดล": NIR (RE+IR) — สูตรสุดท้าย / Visible (BL+GR) / 4 แบนด์ — ทุกตัวเป็นภาพ SS, seed 3–5 ของ fold ที่ลูกนั้นเป็น test (out-of-fold เสมอ), TTA + soft vote; ภาพแบนด์เปลี่ยนตามโมเดล (Visible → BL/GR, 4 แบนด์ → GR/IR) และแสดงผลรวม out-of-fold ของโมเดลนั้น (NIR 92.6%/0.885, Visible 87.5%/0.797, 4 แบนด์ 92.1%/0.866)
- ปุ่ม "เทียบทุกโมเดล": ตาราง 3 แถว (โมเดล / คำทาย / ความมั่นใจ / ✔ ถูก เขียว · ✘ ผิด แดง) + Grad-CAM ของทั้ง 3 โมเดลข้างกัน
- dropdown "ตัวอย่างสำหรับพรีเซนต์" (คัดอัตโนมัติจากผล out-of-fold ที่บันทึกไว้, กลุ่มละ ≤5 เรียงตามความมั่นใจของโมเดลที่ถูก): (ก) Bruise NIR ถูก/Visible ผิด: RawA0550, 0784, 0787, 0647, 0243 · (ข) ทุกโมเดลผิด: RawA0091 (Other), 0203 (Rot), 0442, 0467 (Flesh_damage), 0141 (Rot) · (ค) Visible ถูก/NIR ผิด: RawA0158, 0288, 0267 (Russet), 0450, 0343 (Healthy) — เลือกแล้วจะเทียบทุกโมเดลให้ทันที
- selftest ใหม่: ทั้ง 3 โมเดล 30 ภาพสุ่ม → คำทายรายseed ตรง `tta_test_predictions` ใน json 90/90 และ vote ตรง 30/30 ทุกโมเดล; compare_all ตรงกับ vote; ตัวอย่างทุกตัวตรงกฎของกลุ่ม; API `/compare_all` ตอบ; server bind 127.0.0.1 เท่านั้น
- **ต้องปิดแล้วเปิด `run_demo.bat` ใหม่ถึงจะเห็น**

**N8 — Error analysis + ผลแบบ 2 กลุ่ม** (`src/error_analysis.py`) สูตรสุดท้าย nir_ss TTA 3-seed vote, 819 ลูก (assert ว่าตรง `confirm_summary.json`)
- ผิด 61 ลูก (accuracy 92.6%): (ก) มีตำหนิแต่ทายว่าดี **29/246** (ความมั่นใจเฉลี่ย 0.68) · (ข) ลูกดีถูกหาว่ามีตำหนิ **19/573** (0.60) · (ค) เจอตำหนิแต่ผิดกลุ่ม **13** (Internal→Surface 12, Surface→Internal 1; 0.77) — ตอนทายถูกความมั่นใจเฉลี่ย 0.92 → ทายผิดส่วนใหญ่มั่นใจต่ำ (ยกเว้นกลุ่ม ค)
- per tissue (ทายเป็น H/I/S): Bruise 55 → 4/50/1 (90.9%) · Flesh_damage 24 → 1/19/4 (79.2%, ผิดแบบมั่นใจสูง 0.88) · **Rot 23 → 3/13/7 (56.5% — แย่สุด, ส่วนใหญ่ถูกทายเป็น Surface)** · Russet 42 → 7/0/35 · Frost 11 → 3/0/8 · Hail 16 → 3/1/12 · Stem 148 → 139/3/6 · Calix 98.6% · Healthy 97.1%
- **ผลแบบ 2 กลุ่ม (ดี vs มีตำหนิ, P(Internal)+P(Surface) > 0.5), 3-seed vote + TTA:**

| โมเดล | accuracy | F1 (มีตำหนิ) | recall ตำหนิ | recall ลูกดี | ตำหนิหลุด / ลูกดีถูกคัดทิ้ง |
|---|---|---|---|---|---|
| **nir_ss** | **94.0%** | **0.899** | 89.0% | 96.2% | 27 / 22 |
| visible_ss | 90.4% | 0.829 | 78.0% | 95.6% | 54 / 25 |
| full_ss | 94.4% | 0.901 | 85.4% | 98.3% | 36 / 10 |
| nir / visible / full (raw) | 93.0 / 85.8 / 88.5% | 0.881 / 0.761 / 0.801 | 86.2 / 75.2 / 76.8% | 96.0 / 90.4 / 93.5% | |
- paired 15 run: nir_ss − visible_ss F1 +0.087 REAL (15/15), accuracy +0.046 REAL, McNemar p = 2.2e-3 · nir_ss − full_ss ไม่ต่าง (p = 0.79; full_ss คัดลูกดีทิ้งน้อยกว่าแต่ปล่อยตำหนิหลุดมากกว่า) · full_ss − full(raw) REAL
- ตัวอย่าง "เจอตำหนิแต่ผิดกลุ่ม" ที่ Grad-CAM ชี้ตรงตำหนิจริง: 4 จาก 13 ลูก (peak อยู่ใน mask) → เลือก **RawA0115 (Rot), RawA0528 (Flesh_damage), RawA0203 (Rot)** ทั้งหมด Internal ถูกทายเป็น Surface ความมั่นใจ 0.81–0.95 → `outputs/figures/private/n8_wrong_group_examples.png` (gitignored) — สื่อว่าโมเดล "เห็น" ตำหนิถูกที่ แต่แยก "ภายใน vs ผิว" ไม่ได้สำหรับ Rot/Flesh_damage ที่ลามถึงผิว
- ไม่ได้จูนโมเดลให้ลูกเหล่านี้ถูก และไม่ได้ย้าย Rot ไปกลุ่ม Surface (ตามกฎในแผน)

**ข้อสรุปรวมที่ปรับแล้ว (แทนข้อ 1 และ 4 ของ "ข้อสรุปรวม" ใน ผล N1–N4)**
1. **NIR > visible จริง แม้ใช้ภาพ preprocessing เดียวกัน (SS):** +0.091 F1 (91.0% vs 85.3%), REAL 15/15, McNemar p = 3.4e-5; Internal +0.17; แบบ 2 กลุ่ม 94.0% vs 90.4% — ตัวเลข raw (+0.17) ใหญ่เกินจริงเพราะภาพ raw ทำร้าย visible มากกว่า
2. **4 แบนด์ ≈ NIR เมื่อภาพตรงกัน** (0.842 vs 0.858, ไม่ต่าง) → ที่ full < nir ในภาพ raw มาจากภาพ raw ไม่ใช่ visible ทำให้แย่ — แต่การรวม visible ก็ **ไม่ได้ช่วยเพิ่มอย่างมีนัยสำคัญ** (full_ss ไม่ดีกว่า nir_ss, late fusion +0.014 ไม่ผ่านเกณฑ์) → NIR อย่างเดียวพอ (ใช้ 2 แบนด์แทน 4)
3. ความผิดพลาดหลักของสูตรสุดท้ายคือ Rot (56.5%) และตำหนิที่หลุดเป็น "ดี" 29 ลูก

### การตัดสินใจระหว่างทำ N5 / N7 / N8 / N6 รอบ 2 (2026-10-18 — เลือกเองตามคำสั่งผู้ใช้)
1. ตรวจ alignment ด้วยตาบน **npy 224×224 ที่ใช้เทรนจริง** (ไม่ใช่ tif ต้นฉบับ) เพราะคำถามคือ "โมเดลเห็นภาพเหลื่อมไหม"; 3 ลูกเลือกล่วงหน้า (2 ลูกที่ใช้ทดสอบ demo มาก่อน + 1 ลูกกลางชุด) ไม่ได้เลือกจากผล; เพิ่ม `outputs/figures/private/` ใน .gitignore สำหรับรูปที่มีพิกเซล dataset
2. N5 ใช้ recipe เดียวกับ nir_ss เป๊ะ (legacy init, aug, ไม่มี SE, 224, 3 class, TTA) เพื่อให้ต่างกันแค่แบนด์; รัน full_ss ก่อน (ตอบคำถามหลัก); เขียน `run_n5.bat` ไว้ด้วยแต่รันผ่าน python โดยตรงเพราะทำในรอบนี้เลย (~71 นาที ไม่ต้องข้ามคืน)
3. ผลหลักใช้ plain probabilities (สอดคล้อง N4/multiseed), TTA เป็นผลรอง; McNemar ใช้ 3-seed vote + TTA (รูปแบบเดียวกับสูตรสุดท้าย)
4. เกณฑ์ verify checkpoint: คำทายพลิกบนภาพที่ prob บันทึกไว้เสมอกัน (margin < 0.02 = ระดับ noise GPU เดียวกับเกณฑ์ 2e-2 ของ run_confirm) ไม่นับเป็น fail; full_ss seed5 fold3 เกินเกณฑ์ prob 2e-2 เล็กน้อย (2.3e-2) แต่คำทายไม่เปลี่ยน → ยอมรับและบันทึก (ไม่แก้เกณฑ์ให้ผ่าน)
5. ตีความ N5 ตามกฎที่ล็อกไว้ แต่บันทึกข้อจำกัดว่า SS = แก้การเหลื่อม + flat-field พร้อมกัน แยกไม่ได้ และเสนอการทดลองแยก (แบนด์เดี่ยว SS) เป็นทางเลือก
6. N7: ภาพแบนด์และพื้นของ Grad-CAM เปลี่ยนตามโมเดล (Visible ซ้อนบน GR, อื่นๆ ซ้อนบน IR); โหมดอัปโหลดและสายพานยังใช้สูตรสุดท้ายอย่างเดียว (ตามแผน: ไม่ทำ belt mode ใหม่); ตัวอย่างพรีเซนต์ กลุ่มละ ≤5 เรียงตามความมั่นใจของโมเดลที่ถูก, กลุ่ม (ข) "ทุกโมเดลผิด" = ทั้ง 3 โมเดล SS; ใส่ตัวเลือกว่างเป็นค่าเริ่มของ dropdown ตัวอย่าง (Gradio เลือกตัวแรกให้อัตโนมัติ → เลือกตัวแรกซ้ำจะไม่ทำงาน — พบตอนทดสอบในเบราว์เซอร์)
7. N8: กฎ 2 กลุ่มใช้ P(Internal)+P(Surface) > 0.5 ตามแผน = 94.0%; ตัวเลขเบื้องต้น 94.1% ในแผนเป็นแบบ argmax 3 คลาสแล้วยุบ — รายงานทั้งคู่; ตัวอย่าง Grad-CAM ใช้ Grad-CAM แบบเดียวกับ demo (ชั้น 14×14, เฉลี่ย 3 seed, target = คลาสที่ทาย), "ตรง mask" = peak อยู่ใน mask ที่ขยาย 9×9 px (ที่ 224) เรียงด้วยสัดส่วน top-10% CAM ใน mask
8. commit `n5_results.json` (prob ของ 30 run, ตัวเลขล้วน ~350 KB เหมือน fusion_results.json) เพื่อให้ demo/N8 คำนวณซ้ำได้; ไม่ commit checkpoint และรูปใน `outputs/figures/private/`

## K: ทำซ้ำ Kleynen 2005 + เทียบแบบ 2 กลุ่ม (ดี/มีตำหนิ) (2026-10-28)

> **ผลแบบ 3 กลุ่มของ nir_ss (สูตรสุดท้าย 92.6% / macro-F1 0.885) ยังเป็นผลหลัก ห้ามแก้หรือเขียนทับ** — หัวข้อนี้เป็นมุมมองเสริมแบบ 2 กลุ่มเพื่อเทียบกับงานต้นฉบับของ dataset (Kleynen et al. 2005, J. Food Eng. 69:41–49 ที่ README ของ CAPA บังคับให้อ้าง)

- **K1 ป้ายความรุนแรง**: ค้นทุกไฟล์ (txt/xls/csv/pdf/ชื่อไฟล์/โฟลเดอร์) หาป้าย slight/serious/reject/bruise ของแต่ละภาพตำหนิ → ถ้าเจอสร้าง `severity_labels.csv` แล้วนับเทียบ 55/56/76/59 (Agarla et al. 2023, Sensors 23:7893); ไม่เจอให้บันทึกว่าไม่เจอ
- **K2 CNN รวมคลาส (ไม่เทรนใหม่)**: OOF ของ nir_ss และ visible → รวม Internal+Surface = Defect → อัตราทายถูกแยก ลูกดี 280 / ขั้ว+ก้น / ตำหนิ 246 + แยกชนิดตำหนิ + แยกความรุนแรง + confusion 2×2
- **K3 ทำซ้ำวิธี Kleynen บนภาพ SingleSpectra (CPU)**: ตัดพื้นหลังด้วย threshold 800 nm + fill holes + closing (บันทึก mask ตัวอย่าง 20 ภาพรวมขั้ว/ก้นใน `outputs/kleynen_masks/` ห้าม commit) → Bayes บน histogram หลายมิติ (พิกเซลตำหนิจาก SBW_Defect, ใช้เฉพาะภาพ train fold) → ฟีเจอร์รายลูก → LDA ดี/มีตำหนิ; fold/seed เดิม; 3 ชุดแบนด์ 450/750/800 (เปเปอร์), 750/800 (NIR), 450/500 (Visible); รายงานแบบ K2 + แบบตาราง 2 ของเปเปอร์
- **K4 (GPU ว่าง รันพร้อม K3)**: เทรน nir_ss และ visible แบบ 2 คลาส (Healthy vs Defect) fold/seed/TTA เดิม → รายงานแบบ K2
- **K5 ตารางสรุป**: Kleynen (ตัวเลขเปเปอร์) | Kleynen ที่ทำซ้ำ (3 ชุดแบนด์) | CNN รวมคลาส (NIR, Visible) | CNN 2 คลาส (NIR, Visible) + McNemar Kleynen ทำซ้ำ (450/750/800) vs CNN NIR
- บันทึกผลทุกข้อเป็น JSON ใน outputs/ แล้วสรุปลงหัวข้อนี้; commit เฉพาะโค้ด, JSON, PROJECT_PLAN.md (ห้าม `git add -A`, ห้ามภาพ mask/ภาพ CAPA/.npy/.pt) แล้ว push

### ผล K (ทำเสร็จ 2026-10-28) — โค้ด `src/kleynen.py` (k1/k2/k3/k5), `src/run_k4.py`; JSON `outputs/dl/k1_severity.json`, `k2_merged.json`, `k3_kleynen.json`, `k3_kleynen_erode15.json`, `k4_results.json`, `k5_summary.json`

**K1 — เจอป้ายความรุนแรง** ใน `segmentations/KleynenDefectClasses_Simplified.txt` (Class 1 = reject, 2 = bruise, 3 = serious, 4 = slight; ครบ 246 ภาพ ตรงกับ defect_no ทุกภาพ) → `outputs/severity_labels.csv` (เก็บในเครื่อง ไม่ commit)
- จำนวน: **serious 55 / bruise 56 / slight 76 / reject 59** = ตรงกับ 55/56/76/59 ที่ให้มาทุกตัว (เรียงตามลำดับ serious/bruise/slight/reject; ข้อความใน Agarla 2023 ไม่ได้บอกจำนวนแยกหมวด มีแค่ในรูป จึงยืนยันลำดับจากเปเปอร์ไม่ได้)
- ตรวจไขว้กับ Kleynen 2005 ตาราง 2: bruise 98.2% = 55/56, serious 94.5% = 52/55, slight 55% = 42/76, รวม 84.6% = 208/246 → **เปเปอร์ใช้ 246 ภาพและป้ายชุดเดียวกับเราเป๊ะ**
- ไขว้ชนิด×ความรุนแรง: Bruise เกือบทั้งหมดเป็น bruise (54/55); Flesh_damage ส่วนใหญ่ reject (21/24); Rot reject 16/23; Russet ส่วนใหญ่ slight (34/42); Hail_damage_perf กระจาย (reject 14, serious 12, slight 5)

**K2 — CNN รวมคลาส (Internal+Surface → Defect, ไม่เทรนใหม่)** OOF 3-seed vote + TTA, defective ถ้า P(I)+P(S) > 0.5

| โมเดล | ลูกดี 280 | ขั้ว+ก้น 293 | ตำหนิ 246 | reject / bruise / serious / slight | acc 819 | 2×2 [[TN,FP],[FN,TP]] |
|---|---|---|---|---|---|---|
| **nir_ss** (NIR, SS) | 96.8% | 95.6% | **89.0%** | 94.9 / **96.4** / 85.5 / 81.6 | **94.0%** | [[551,22],[27,219]] |
| visible_ss (Visible, SS) | 94.6% | 96.6% | 78.0% | 94.9 / 69.6 / 74.5 / 73.7 | 90.4% | [[548,25],[54,192]] |
| nir (raw) | 96.8% | 95.2% | 86.2% | 88.1 / 91.1 / 83.6 / 82.9 | 93.0% | [[550,23],[34,212]] |
| visible (raw) | 87.1% | 93.5% | 75.2% | 96.6 / 53.6 / 76.4 / 73.7 | 85.8% | [[518,55],[61,185]] |
- ต่อชนิดตำหนิ (nir_ss vs visible_ss): **Bruise 96.4 vs 69.1%**, Scald 88.2 vs 64.7, Hail_perf 93.5 vs 77.4, Russet 83.3 vs 81.0; Flesh_damage 95.8 vs 100, Rot 87.0 vs 91.3 (visible ดีกว่าเล็กน้อย — ตำหนิที่เห็นได้จากสีผิว)

**K3 — ทำซ้ำ Kleynen 2005 บนภาพ SingleSpectra (CPU, deterministic)**
- วิธี: fruit mask = Otsu บน 800 nm + fill holes + closing 15 px + component ใหญ่สุด → erosion 5 px; Bayes priors เท่ากัน (ตามเปเปอร์) บน histogram 64 ระดับ/แบนด์ (3 แบนด์ = 262k ช่อง); พิกเซลดีจากผลดี (Healthy) พิกเซลตำหนิจาก SBW_Defect เฉพาะภาพ train fold; ฟีเจอร์ตามเปเปอร์ (mean/median/std ของแต่ละแบนด์ บนเนื้อดีและเนื้อตำหนิ + สัดส่วนตำหนิ) + จำนวนบริเวณ + บริเวณใหญ่สุด → StandardScaler + LDA; fold เดิม
- ตรวจ mask ด้วยตา 20 ภาพ (`outputs/kleynen_masks/contact_sheet.png`, gitignored): ขอบผลตรงดีทุกภาพรวมขั้ว/ก้น; แต่การแบ่งพิกเซลมี false positive ที่ขอบผล (erosion 5 px น้อยไป) และบนลายสีผิวบางลูก; ขั้ว/ก้นถูกแบ่งเป็นตำหนิ (คาดได้ — เปเปอร์ใช้ตัวตรวจขั้ว/ก้นแยก)
- ระดับพิกเซล (test fold): recall ของพิกเซลตำหนิ 84.9% (450/750/800), 85.1% (NIR), 72.9% (Visible); พิกเซลผลดีที่ถูกหาว่าตำหนิ 11.5% / 15.5% / 34.1%
- บางลูกในกลุ่ม RawA07xx (Bruise) ถูกแบ่งเป็นตำหนิ 80–90% ของผล → น่าจะเป็นความต่างของแสง/รอบถ่าย (บันทึกเป็นข้อสังเกต)

**K5 — ตารางสรุปแบบ 2 กลุ่ม** (รูปแบบตาราง 2 ของเปเปอร์: ผลดี 280 + ตำหนิ 246 = 526 ลูก; คอลัมน์ขั้ว/ก้น และ acc 819 = ทั้งชุด)

| วิธี | ผลดี | ตำหนิ | reject | bruise | serious | slight | acc 526 | ขั้ว/ก้น | acc 819 |
|---|---|---|---|---|---|---|---|---|---|
| **Kleynen 2005 (ตัวเลขเปเปอร์)** | 94.3 | 84.6 | 100 | 98.2 | 94.5 | 55 | 89.8 | (มีตัวตรวจแยก) | – |
| Kleynen ทำซ้ำ 450/750/800 [A ตามเปเปอร์] | 86.1 | 78.0 | 93.2 | 92.9 | 83.6 | 51.3 | 82.3 | 25.6 | 62.0 |
| Kleynen ทำซ้ำ 750/800 (NIR) [A] | 89.3 | 80.5 | 91.5 | 92.9 | 85.5 | 59.2 | 85.2 | 16.0 | 60.4 |
| Kleynen ทำซ้ำ 450/500 (Visible) [A] | 75.0 | 58.9 | 72.9 | 67.9 | 54.5 | 44.7 | 67.5 | 23.2 | 51.6 |
| Kleynen ทำซ้ำ 450/750/800 [B: ขั้ว/ก้น = ดี] | 93.6 | 56.9 | 74.6 | 87.5 | 43.6 | 30.3 | 76.4 | 91.1 | 81.7 |
| Kleynen ทำซ้ำ 450/750/800 [A, erosion 15 px — sensitivity] | 88.2 | 82.5 | 98.3 | 94.6 | 90.9 | 55.3 | 85.6 | 8.5 | 58.0 |
| **CNN รวมคลาส NIR (nir_ss)** | **96.8** | **89.0** | 94.9 | 96.4 | 85.5 | **81.6** | **93.2** | 95.6 | **94.0** |
| CNN รวมคลาส Visible (visible_ss) | 94.6 | 78.0 | 94.9 | 69.6 | 74.5 | 73.7 | 86.9 | 96.6 | 90.4 |
| CNN 2 คลาส NIR (nir_ss_2cls) | 96.4 | 88.6 | 94.9 | 96.4 | 87.3 | 78.9 | 92.8 | 96.6 | 94.1 |
| CNN 2 คลาส Visible (visible_ss_2cls) | 93.2 | 75.6 | 94.9 | 57.1 | 74.5 | 75.0 | 85.0 | 94.9 | 88.5 |
(B 750/800: 94.3/52.0, acc 819 80.7%; B 450/500: 97.1/11.8, acc 819 71.8% — แบบ B ทุกชุด LDA ยอมทิ้งตำหนิเพื่อรักษาขั้ว/ก้น)

- **McNemar หลัก: Kleynen ทำซ้ำ (450/750/800, A) vs CNN NIR รวมคลาส บน 526 ลูก: 82.3% vs 93.2%, Kleynen ถูกคนเดียว 24 vs CNN ถูกคนเดียว 81, p = 4.6e-8** · บนทั้ง 819 (แบบ B) 81.7% vs 94.0%, p = 4.0e-16 · เทียบ erosion 15 px: 85.6% vs 93.2%, p = 2.1e-5 · เทียบ CNN 2 คลาส NIR: 82.3% vs 92.8%, p = 5.7e-8
- ภายใน Kleynen ทำซ้ำ: 750/800 vs 450/500 = 85.2% vs 67.5%, p = 7.9e-13 (**NIR > Visible แม้ในวิธีคลาสสิก**); 450/750/800 vs 750/800 = 82.3% vs 85.2%, p = 0.064 (การเพิ่ม 450 ไม่ช่วย — ต่างจากเปเปอร์ที่ว่า 450 ช่วยแยก slight/russet; ในงานเรา slight 51.3 vs 59.2)
- **K4 CNN 2 คลาส vs รวมคลาส:** NIR 2 คลาส 94.1% vs รวมคลาส 94.0% (McNemar p = 1.0; paired 15 run accuracy +0.001 ไม่ต่าง) → **เทรนแบบ 3 คลาสแล้วรวม ไม่เสียอะไรเลยเมื่อเทียบเทรน 2 คลาสตรงๆ** (และได้ข้อมูลภายใน/ผิวเพิ่ม); Visible 2 คลาสแย่กว่ารวมคลาสเล็กน้อย (accuracy −0.016 REAL 12/15); NIR 2 คลาส vs Visible 2 คลาส 94.1% vs 88.5%, p = 4.4e-6

**สรุป K**
1. **CNN NIR ดีกว่าวิธีต้นฉบับของ dataset ชัดเจน** — บนโปรโตคอลเดียวกับเปเปอร์ (526 ลูก) CNN 93.2% vs Kleynen ที่เราทำซ้ำ 82.3% (p = 4.6e-8) และสูงกว่าตัวเลขในเปเปอร์ (89.8%) ด้วย; จุดที่ชนะมากสุดคือ **slight 81.6% vs 55%** (เปเปอร์) และ CNN จัดการขั้ว/ก้นได้เอง 95.6% โดยไม่ต้องมีตัวตรวจแยก
2. การทำซ้ำได้รูปแบบเดียวกับเปเปอร์ (reject/bruise สูง, serious กลาง, slight ต่ำสุด) แต่ตัวเลขต่ำกว่า ~5–8 จุด — เหตุผลที่เป็นไปได้: (ก) เราใช้ cross-fitting จริง (histogram ไม่เคยเห็นภาพ test และฟีเจอร์ train ก็ out-of-sample) ส่วนเปเปอร์ cross-validate แค่ LDA (ข) ขนาด erosion ไม่ระบุในเปเปอร์ (5 → 15 px ทำให้ใกล้เปเปอร์ขึ้นมาก: 85.6%, slight 55.3 = เปเปอร์) (ค) เปเปอร์เลือกพิกเซลตำหนิด้วยมือ เราใช้ mask SBW ทั้งก้อน
3. **NIR > Visible ซ้ำอีกครั้งในอีกวิธีหนึ่ง** (Kleynen ทำซ้ำ 750/800 vs 450/500: 85.2% vs 67.5%) — ใช้เสริมข้อสรุปหลักของธีสิส
4. ผลแบบ 3 กลุ่มของ nir_ss (92.6% / 0.885) ยังเป็นผลหลักตามเดิม — K เป็นการเทียบเสริมกับงานต้นฉบับเท่านั้น

### การตัดสินใจระหว่างทำ K (2026-10-28 — เลือกเองตามคำสั่งผู้ใช้)
1. K1: อ่านป้ายจากไฟล์ txt ของ dataset (ชื่อคลาสอยู่ในหัวไฟล์); severity_labels.csv เก็บในเครื่อง (ป้ายราย ภาพเป็นข้อมูลของ dataset — ผู้ใช้สั่ง commit เฉพาะโค้ด/JSON/แผน) — JSON มีแต่จำนวนรวม; ดาวน์โหลด PDF Kleynen 2005 (open access จาก ORBi ULiège) มาอ่านตัวเลขตาราง 2 และรายละเอียดวิธี (ไฟล์อยู่ใน scratchpad ไม่อยู่ในโปรเจก)
2. K2: "visible" ใช้ **visible_ss เป็นหลัก** (preprocessing เดียวกับ nir_ss → แฟร์ ตามผล N5) และรายงาน raw ไว้ด้วย; กฎรวมคลาส P(I)+P(S) > 0.5 เหมือน N8
3. K3 ตามเปเปอร์ได้: priors เท่ากัน, ใช้ histogram ไม่ใช่ parametric, ฟีเจอร์ mean/median/std บนเนื้อดี/เนื้อตำหนิ + สัดส่วน, LDA, พิกเซลดีจากผลดีเท่านั้น; ที่เปเปอร์ไม่ระบุ/ทำไม่ได้: histogram 64 ระดับ (เปเปอร์ 256 ระดับแต่มีพิกเซลตำหนิมากกว่า), threshold 800 nm = Otsu, erosion 5 px (เลือกก่อนเห็นผล — ใช้เป็นผลหลัก; 15 px เป็น sensitivity เพราะเห็นจากรูปว่าขอบยังติด ไม่ได้เลือกจากความแม่นยำ), สีที่ไม่เคยเห็นตอนเทรน → ตำหนิ (ตามกฎ "ดีเมื่อ P > 0.5"), เพิ่มฟีเจอร์จำนวนบริเวณ/บริเวณใหญ่สุด (ตามที่ผู้ใช้เสนอ), ไม่ทำตัวตรวจขั้ว/ก้น (template matching) → รายงาน 2 แบบ: A = ตามเปเปอร์ (เทรน/วัดบน 280+246) และ B = ทั้ง 819 โดยนับขั้ว/ก้นเป็นดี
4. K3 กันข้อมูลรั่ว: histogram ของ fold ทดสอบใช้เฉพาะภาพ train; ฟีเจอร์ที่ใช้เทรน LDA ของแต่ละลูก train มาจาก histogram ที่ไม่รวม fold ของลูกนั้น (cross-fitting) เพื่อให้ train/test มีลักษณะเดียวกัน; วิธีนี้ deterministic จึงไม่มีผลของ seed (ใช้ fold เดิม)
5. K4: variant `nir_ss_2cls`, `visible_ss_2cls` recipe เดียวกับ 3 คลาสทุกอย่าง (seed 3–5, fold เดิม, TTA) ต่างแค่ head/loss; sampler ใช้น้ำหนักสมดุล 3 คลาสจริงเหมือนเดิม (ให้ข้อมูลที่เห็นตอนเทรนเหมือนกันเป๊ะ); ไม่เซฟ checkpoint; เพิ่มรองรับ `n_classes = 2` ใน `round2.py` (ไม่กระทบ 3/5 คลาส); รันพร้อม K3 (GPU ว่าง) 30/30 run error 0, ~62 นาที
6. K5: McNemar หลักทำบน 526 ลูก (ชุดเดียวกับโปรโตคอลเปเปอร์) และบน 819 สำหรับแบบ B; รายงาน CNN 2 คลาสด้วย; ตัวเลข "acc 526" ของเปเปอร์คำนวณจาก (280×94.3 + 246×84.6)/526

## สิ่งที่ยังไม่ได้ทำ (Phase 6)

- [ ] เขียนเล่มวิทยานิพนธ์โดยใช้ผลลัพธ์ข้างต้น + รูป Grad-CAM + ตาราง SHAP + ผล k-fold (ปรับข้อสรุปเรื่อง SE block ตามผล k-fold ข้อ 2 ด้านบน) — **ห้ามเอาตัวเลข ensemble มาใช้เป็นผลหลัก เพราะมี data leakage (ดูหัวข้อ ensembling ด้านบน)**
- [ ] เตรียมสไลด์/ประเด็น defense จากหัวข้อ "ข้อค้นพบสำคัญ" ด้านบน (อัพเดตด้วยผล k-fold ที่ยืนยัน/แก้ข้อสรุปเดิม)
- [x] เพิ่ม Data Augmentation ใน `src/dataset.py`
- [x] สร้าง K-Fold Cross-Validation ใน `src/train_dl_kfold.py`
- [x] Ensembling จาก 5-fold models (NIR-only, full_SE) — พบว่าทำไม่ได้อย่างยุติธรรมบน dataset นี้ (data leakage) สรุปว่ายังคงใช้ผล 5-fold CV เดิมเป็นหลัก ไม่ใช้ ensemble
- [x] เช็ค Per-Class F1 — สรุปว่า NIR-only ไม่ต้องทำ Focal Loss เพิ่ม (full_SE ควรทำแต่ไม่ใช่โมเดลหลัก)
- [x] Backup ผลลัพธ์ปัจจุบันไว้ที่ `backups/best_results_20260915_181726/` ก่อนรัน experiment ใหม่ + ตั้ง git repo push ขึ้น [github.com/Chinnawan/project_nir](https://github.com/Chinnawan/project_nir) (โค้ด+ผลตัวเลขเท่านั้น ไม่มี dataset — ดูหัวข้อ "Version Control + Backup" ด้านบน)
- [x] รอบสุดท้าย — ทดลอง NIR-only + SE Block (`nir_SE`) ประเมินด้วย K-Fold — ~~ได้ผลดีที่สุดของโปรเจก 0.862~~ **แก้ 2026-10-13: เฉลี่ย 3 seed = 0.847±0.008 ไม่ต่างจาก `nir` 0.843±0.008 (SE ไม่ช่วย)**
- [x] Ablation test — แยกผลของ K-Fold ออกจาก Augmentation (`nir_no_aug`) — **สรุป: augmentation ช่วยจริง (+0.026 F1), ตัวเลขรอบ 1-2 เป็นโชคจาก single-split** (ดูตาราง ablation ด้านบน)
- [x] Commit + push ผลใหม่ (`nir_SE`, `nir_no_aug` ใน `kfold_results.json`) — **อยู่ใน commit แรก `74afa45` อยู่แล้ว** (ตรวจแล้วด้วย `git show 74afa45:outputs/dl/kfold_results.json`)
- [x] งานที่ 1 — เทรน `nir_SE_ss` ด้วยภาพ SingleSpectra (flat-field corrected) — **F1 0.859±0.024, acc 90.9%±1.7% ไม่ต่างจาก `nir_SE` raw (0.862±0.025)** mapping ตรวจด้วยเนื้อภาพ 819/819 (ดูหัวข้อ "ใช้ข้อมูลใน CAPA ที่ยังไม่เคยใช้")
- [x] งานที่ 2 — ตรวจ Grad-CAM เชิงปริมาณกับ mask ตำหนิ (Pointing Game + IoU) — **Surface ไม่ดีกว่าสุ่ม, Internal ดีกว่า null ~+14 จุด (Bruise/Flesh_damage) แต่ Rot ไม่ดีกว่า — ต้องแก้ narrative เรื่อง Grad-CAM ในเล่ม**
- [x] Commit งานชุด SingleSpectra + Grad-CAM mask eval (commit `7dca196`) — push แล้วพร้อมงานที่ 5
- [x] งานที่ 4 — Single-band ablation (`re_only`, `ir_only`) — **RE-only 0.839±0.034 ≈ IR-only 0.844±0.049 ≈ RE+IR 0.843±0.023 → ไม่มีหลักฐานว่า RE สำคัญกว่า IR ต้องตัดข้อสรุปนี้ออก และเลิกใช้ SE weight เป็นหลักฐานความสำคัญของแบนด์** (ดูหัวข้อ "ทดสอบความสำคัญของแบนด์แบบตรง")
- [x] งานที่ 5 — Commit + push ขึ้น GitHub เสร็จ (2026-10-13): `7dca196` (SingleSpectra + Grad-CAM mask eval), `467e428` (แบนด์เดี่ยว), `55e25fd` — push ครั้งแรกล้มเหลว 403 เพราะ `gh` active account เป็น `onyourprai` แก้โดย `gh auth switch --user Chinnawan` (ผู้ใช้ล็อกอินบัญชี Chinnawan ไว้แล้ว) หมายเหตุ: ถ้าจะ push ต่อจากเครื่องนี้ ตรวจ `gh auth status` ว่า active เป็น Chinnawan
- [x] รันข้ามคืน: ล็อก seed + multi-seed (3 seed) ทุก variant + แบนด์เดี่ยวครบ 4 + `gr_nir` + classical ML 5-fold — **เสร็จ 2026-10-13 (5.2 ชม., 180/180 run, error 0)** ผลหลัก: NIR ≫ visible/full, แบนด์ NIR เดี่ยว ≈ RE+IR ≫ BL/GR เดี่ยว, DL ≫ classical (+0.26), **SE ไม่ช่วย, augmentation ก้ำกึ่ง** (ดู "ผล multi-seed")
- [ ] (ทางเลือก) ทดสอบสมมุติฐาน "F1 ลดตามจำนวนช่อง" ด้วยการปรับสเกล init ของ conv ชั้นแรก; ขั้น E Grad-CAM 14×14
- [x] รอบ 2 เพิ่มความแม่นยำ (5 class, label smoothing, TTA, 320px, SingleSpectra เป็นภาพเพิ่ม, Defect Copy-Paste, แก้ init รวมแบนด์) + ensemble ภายใน fold — **เสร็จ 2026-10-14 (135/135 run, 4.9 ชม.)**: มีเพียง `nir_ss` ที่ผ่านเกณฑ์ (+0.028) แต่ **ไม่ถูกยืนยันด้วย seed ใหม่** (ดู "ผลยืนยัน"); copy-paste, 5 class, 320px, init ใหม่ไม่ช่วย
- [x] ล็อกสูตรสุดท้ายแล้วยืนยันด้วย seed 3,4,5 — **ผล: `nir_ss − nir` = +0.009 ไม่ผ่านการยืนยัน; สูตร (SS+TTA) เทียบ baseline เดิม +0.020 ผ่านแบบพอดีขอบ (แนวโน้มบวกเล็กน้อย); ensemble 3 seed accuracy 92.6% / macro-F1 0.885**
- [x] Confirmatory run (`nir_ss` vs `nir`, seed 3–5) + เซฟ checkpoint `nir_ss` (15 ไฟล์) + `fold_assignment.json` — เสร็จ ตรวจ checkpoint ผ่าน (ดูหัวข้อ "ผลยืนยัน")
- [x] Demo app (Gradio) — เขียนและทดสอบกับโมเดลจริงผ่านทั้งหมด เปิดด้วย `run_demo.bat` (http://127.0.0.1:7860) ดูหัวข้อ "สถานะ Demo app"
- [ ] Demo รอบ 2: ถัง "ไม่แน่ใจ → ส่งคนตรวจ" + สลับ/เทียบโมเดล + ภาพตัวอย่างพรีเซนต์ — ดูหัวข้อ "Demo app รอบ 2"
- [x] Demo UI: % ความมั่นใจใหญ่ชัด + กล่องเฉลยขวา (✔ เขียว / ✘ แดง) — เสร็จ 2026-10-18
- [x] พิสูจน์ NIR > Visible: N1 ฟิสิกส์ (AUC NIR−vis +0.036 REAL, Bruise +0.084), N2 per-class/tissue (Bruise recall +53 จุด, McNemar p<1e-9), N3 classical (ทิศทางเดียวกัน 3/3), N4 late fusion ≈ nir (visible ไม่เพิ่มข้อมูล) + **พบแบนด์ raw เหลื่อม ~15–22 px**, N6 รายงาน `outputs/logs/nir_proof_report.txt` — เสร็จ 2026-10-18 (ดู "ผล N1–N4 + N6")
- [x] N5 (ขยาย): ตรวจ alignment ด้วยตา (ยืนยันเหลื่อม ~16 px ใน raw) + เทรน `visible_ss` + `full_ss` (30/30, ~71 นาที, checkpoint 30 ไฟล์) — **nir_ss − visible_ss +0.091 REAL (91.0% vs 85.3%); full_ss 0.842 ≈ nir_ss 0.858 → full < nir ใน raw มาจากภาพ raw** — เสร็จ 2026-10-18 (ดู "ผล N5")
- [x] N7: demo สลับโมเดล NIR / Visible / 4 แบนด์ + ปุ่มเทียบทุกโมเดล + ตัวอย่างพรีเซนต์อัตโนมัติ — selftest ALL PASSED — เสร็จ 2026-10-18 (ต้องเปิด `run_demo.bat` ใหม่)
- [x] N8: error analysis (ผิด 61: ตำหนิหลุด 29, ลูกดีถูกคัด 19, ผิดกลุ่ม 13; Rot แย่สุด 56.5%) + ผล 2 กลุ่ม (nir_ss 94.0% vs visible_ss 90.4% REAL, ≈ full_ss 94.4%) — เสร็จ 2026-10-18
- [x] K: ทำซ้ำ Kleynen 2005 + เทียบ 2 กลุ่ม — เจอป้ายความรุนแรง (55/56/76/59 ตรง); Kleynen ทำซ้ำ 82.3% vs CNN NIR 93.2% บน 526 ลูก (p = 4.6e-8, เปเปอร์ 89.8%); CNN 2 คลาส ≈ รวมคลาส (94.1 vs 94.0%) — เสร็จ 2026-10-28 (ดู "ผล K")
- [ ] (ทางเลือก) แยกผลของ flat-field ออกจากการแก้เหลื่อม: เทรนแบนด์เดี่ยว `bl_ss`/`gr_ss` เทียบ `bl_only`/`gr_only` raw (~1 ชม.)
- [ ] นำผลไปคุยกับอาจารย์ที่ปรึกษาตามที่ระบุไว้ในหัวข้อ "สถานะการคุยกับอาจารย์" — **ตัวเลขสุดท้ายพร้อมแล้ว — ใช้ตาราง "ผล multi-seed" (NIR 0.843±0.008, nir_SE 0.847±0.008, RE-only 0.853±0.007; SE ไม่ช่วย) ไม่ใช่ 0.862 แต่ต้องปรับ narrative ก่อน (นอกจากข้อ SE ที่แก้แล้ว): (1) Grad-CAM — วัดเทียบ mask แล้วดีกว่า null เฉพาะ Internal (Bruise/Flesh_damage) Surface ไม่ต่างจากสุ่ม (2) ตัด "RE สำคัญกว่า IR" และอย่าใช้ SE weight เป็นหลักฐานความสำคัญของแบนด์ (ผลแบนด์เดี่ยว: RE≈IR≈RE+IR) — เพิ่มข้อจำกัดว่าการรันซ้ำให้ F1 ต่อ fold ต่างกันได้ถึง ~0.06**
