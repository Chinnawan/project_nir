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
  config.py            constants: classes, channel groups (visible/nir/full), paths
  dataset.py           PyTorch Dataset/DataLoader, per-channel norm stats, weighted sampler
  features.py          hand-crafted features (color stats + GLCM texture + band-ratio NDI) — Phase 2
  train_classical.py   SVM / RandomForest / LightGBM + SHAP — Phase 2
  models.py            EfficientNet-B0 (variable in_channels) + InputChannelSE — Phase 3/4
  train_dl.py          training loop, early stopping (patience=10), 3 variants + SE variant — Phase 3/4
  explain.py           Grad-CAM + Bootstrap CI + McNemar's test — Phase 5
outputs/
  classical/           metrics.json, predictions.json, *_model.joblib, shap/*.png
  dl/                  metrics.json, {variant}_best.pt, {variant}_history.json, gradcam/*.png
  stats/               results.json (bootstrap CI + pairwise McNemar's)
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
2. **SE block (Phase 4) ทำให้ผลแย่ลงกว่า full ธรรมดา** (0.640 vs 0.745) ทั้งที่ตั้งใจให้ช่วย — สาเหตุที่เป็นไปได้: (a) dataset เล็กเกินไปสำหรับพารามิเตอร์เพิ่ม เกิด overfitting เร็วกว่า (early stop ที่ epoch 23 vs 27), (b) SE weight ที่เรียนรู้ได้ (0.45-0.59 ทุกตัวใกล้เคียงกัน ไม่ discriminative มาก) ยังไม่ได้ช่วยแยก band ที่สำคัญจริงออกจากกันชัดเจน — เป็นประเด็นเปิดให้ทดลองต่อ (เช่น ปรับ `se_reduction`, เพิ่ม regularization, หรือรัน seed หลายค่าแล้วดู variance)
3. Grad-CAM ยืนยันว่าโมเดล full_SE โฟกัสตรงตำแหน่งรอยช้ำ/ตำหนิจริงในภาพตัวอย่างที่ทำนายถูก (ดู `outputs/dl/gradcam/`) — เป็นหลักฐานเชิงคุณภาพว่าโมเดล "เข้าใจ" ปัญหา แม้ตัวเลข F1 จะไม่ใช่ตัวที่ดีที่สุด

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
.venv\Scripts\python.exe -m src.explain            # Phase 5
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
3. **SE channel importance สม่ำเสมอทุก fold**: BL≈0.487, GR≈0.599, RE≈0.458, IR≈0.540 — GR และ IR สูงสุดเสมอทุก fold ตรงกับผล single-split เดิม (GR, IR สูงสุด) → ข้อสรุปเรื่อง channel importance นี้ robust ไม่เปลี่ยนตาม fold
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

## งานที่ต้องทำต่อ — รอบสุดท้าย: รวมเทคนิคที่พิสูจน์แล้วว่าดีที่สุด (Best-Combo Run)

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

### Ablation test เพิ่มเติม — แยกผลของ K-Fold ออกจาก Augmentation

เหตุผล: รอบ 1-2 (single-split, ไม่มี augmentation) ได้ NIR-only F1 = 0.867/0.853 แต่รอบ 3-4 (K-Fold + augmentation) ได้แค่ 0.843±0.023 — ยังบอกไม่ได้ว่าตัวเลขที่ลดลงมาจาก **วิธีวัดผลเปลี่ยน** (k-fold เฉลี่ยรวมทั้ง fold ง่ายและยาก) หรือจาก **augmentation ทำให้แย่ลงจริง** เพราะเปลี่ยน 2 อย่างพร้อมกัน ต้องแยกทดสอบให้ชัด

**สิ่งที่ต้องทำ**: เพิ่ม variant ใหม่ชื่อ `nir_no_aug` — ใช้ NIR-only channels เหมือนเดิม รันผ่าน 5-Fold CV เหมือนเดิม **แต่ปิด Data Augmentation** (ใช้ dataset แบบรอบ 1-2 ไม่มี flip/rotate/brightness) ทุกอย่างอื่นเหมือน `nir` เดิมในรอบ 3

**สถานะโค้ด: [x] พร้อมรันแล้ว** — เพิ่ม `nir_no_aug` เข้า `VARIANT_SPECS` (channel_group="nir", use_se=False, augment_train=False) — เพิ่ม parameter `augment_train` ใน `dataset.get_dataloaders_from_items()` เพื่อปิด augmentation ได้แบบไม่กระทบ variant อื่น ยังไม่ได้รันจริง (รอผู้ใช้กดรันเอง) คำสั่งที่ต้องรัน (5 fold, ~10-15 นาที):
```
.venv\Scripts\python.exe -m src.train_dl_kfold --variant nir_no_aug
```

**ตารางเทียบผลที่ต้องได้หลังรันครบ**:

| Experiment | Augmentation | วิธีวัดผล | Macro F1 |
|---|---|---|---|
| รอบ 1-2 (มีอยู่แล้ว) | ไม่มี | Single-split | 0.867 / 0.853 |
| **nir_no_aug (ใหม่)** | ไม่มี | K-Fold (5-fold mean±std) | ? |
| nir (รอบ 3, มีอยู่แล้ว) | มี | K-Fold (5-fold mean±std) | 0.843±0.023 |

**วิธีอ่านผล**: ถ้า `nir_no_aug` ได้ใกล้เคียง 0.867 (สูงกว่า `nir` ที่มี augmentation) → แปลว่า augmentation ทำให้แย่ลงจริง ควรพิจารณาตัดออกจาก combo สุดท้าย ถ้า `nir_no_aug` ได้ใกล้เคียง 0.843 (เท่ากับ `nir`) → แปลว่า augmentation ไม่มีผลอะไรเลย ตัวเลขที่ต่างจากรอบ 1-2 เป็นเพราะวิธีวัดผลเปลี่ยนอย่างเดียว ไม่เกี่ยวกับ augmentation

## สิ่งที่ยังไม่ได้ทำ (Phase 6)

- [ ] เขียนเล่มวิทยานิพนธ์โดยใช้ผลลัพธ์ข้างต้น + รูป Grad-CAM + ตาราง SHAP + ผล k-fold (ปรับข้อสรุปเรื่อง SE block ตามผล k-fold ข้อ 2 ด้านบน) — **ห้ามเอาตัวเลข ensemble มาใช้เป็นผลหลัก เพราะมี data leakage (ดูหัวข้อ ensembling ด้านบน)**
- [ ] เตรียมสไลด์/ประเด็น defense จากหัวข้อ "ข้อค้นพบสำคัญ" ด้านบน (อัพเดตด้วยผล k-fold ที่ยืนยัน/แก้ข้อสรุปเดิม)
- [x] เพิ่ม Data Augmentation ใน `src/dataset.py`
- [x] สร้าง K-Fold Cross-Validation ใน `src/train_dl_kfold.py`
- [x] Ensembling จาก 5-fold models (NIR-only, full_SE) — พบว่าทำไม่ได้อย่างยุติธรรมบน dataset นี้ (data leakage) สรุปว่ายังคงใช้ผล 5-fold CV เดิมเป็นหลัก ไม่ใช้ ensemble
- [x] เช็ค Per-Class F1 — สรุปว่า NIR-only ไม่ต้องทำ Focal Loss เพิ่ม (full_SE ควรทำแต่ไม่ใช่โมเดลหลัก)
- [ ] รอบสุดท้าย — ทดลอง NIR-only + SE Block (`nir_SE`) ประเมินด้วย K-Fold — ดูหัวข้อ "งานที่ต้องทำต่อ — รอบสุดท้าย" ด้านบน
- [ ] Ablation test — แยกผลของ K-Fold ออกจาก Augmentation (`nir_no_aug`) — ดูหัวข้อ "Ablation test เพิ่มเติม" ด้านบน
- [ ] นำผลไปคุยกับอาจารย์ที่ปรึกษาตามที่ระบุไว้ในหัวข้อ "สถานะการคุยกับอาจารย์"
