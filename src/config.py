import os

PROJECT_ROOT = r"D:\nir+rgb_project\project\CAPA_AppleQualityGradingImageDatabase"
DATA_DIR = os.path.join(PROJECT_ROOT, "Processed_Dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

CLASSES = ["Healthy", "Internal_Defect", "Surface_Defect"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
SPLITS = ["train", "val", "test"]

# Channel stack order fixed by preprocess_pipeline.py: BL(450nm), GR(500nm), RE(750nm), IR(800nm)
CHANNEL_NAMES = ["BL", "GR", "RE", "IR"]
CHANNEL_GROUPS = {
    "visible": [0, 1],   # BL, GR
    "nir": [2, 3],        # RE, IR
    "full": [0, 1, 2, 3],  # all 4 bands
    "re": [2],            # RE only (single-band ablation)
    "ir": [3],            # IR only (single-band ablation)
    "bl": [0],            # BL only (single-band ablation)
    "gr": [1],            # GR only (single-band ablation)
    "gr_nir": [1, 2, 3],  # GR + RE + IR (does one visible band help NIR?)
}

SEED = 42
