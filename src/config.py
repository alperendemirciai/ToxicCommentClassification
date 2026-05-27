"""Central configuration for the multi-label toxic comment classification project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset"
ORIGINAL_CSV = DATA_DIR / "original" / "train.csv"
SPLITS_DIR = DATA_DIR / "splits"
SUBSAMPLE_CSV = SPLITS_DIR / "subsample_8k.csv"
FOLDS_JSON = SPLITS_DIR / "folds.json"

RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = PROJECT_ROOT / "tables"
LOGS_DIR = PROJECT_ROOT / "logs"

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
NUM_LABELS = len(LABELS)

SEED = 42
SUBSAMPLE_SIZE = 8000
N_FOLDS = 5
INNER_VAL_RATIO = 0.10  # held out from each train-fold for best-epoch selection

# Text cleaning toggle (structural noise + Unicode hygiene). See src/text_clean.py.
# Applied uniformly to BERT inputs and LLM prompt comments when True.
CLEAN_TEXT = True

# BERT training defaults (used by 02_train_bert.py)
BERT_MAX_LEN = 512
BERT_MAX_LEN_FALLBACK = 256
BERT_BATCH_SIZE = 16
BERT_EVAL_BATCH_SIZE = 64
BERT_EPOCHS = 4
BERT_LR = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1
BERT_DECISION_THRESHOLD = 0.5

# Weighted loss (pos_weight on BCEWithLogitsLoss) to counter label imbalance.
# When True, pos_weight[c] = (#neg_c / #pos_c) computed on each fold's training set.
BERT_USE_WEIGHTED_LOSS = False
BERT_POS_WEIGHT_CLIP = 50.0  # cap to avoid extreme weights on very rare labels

# LLM defaults (used by 03_run_llm.py)
OLLAMA_HOST = "http://localhost:11434"
LLM_NUM_PREDICT = 64        # we only need a short JSON line
LLM_TEMPERATURE = 0.0
LLM_FEW_SHOT_K = 8          # 6 (one per label) + 2 clean negatives
LLM_REQUEST_TIMEOUT = 240   # seconds per inference (reasoning chains can be long)

# Reasoning toggle for reasoning-capable models (gpt-oss, deepseek-r1, qwq, o1).
# False = ask Ollama to skip CoT and emit JSON directly (fast).
# True  = let the model think first, then emit JSON (slower, sometimes better).
# Ignored for non-reasoning models.
LLM_THINK = False
LLM_NUM_PREDICT_THINK = 2048  # bigger budget when reasoning is on

MODEL_REGISTRY = {
    "distilbert": "distilbert-base-uncased",
    "roberta":    "roberta-base",
    "bert":       "bert-base-uncased",
}

LLM_REGISTRY = {
    "llama3":  "llama3:8b",
    "qwen":    "qwen2.5:7b-instruct",
    "phi3":    "phi3:mini",
    "mistral": "mistral:latest",
    "oss": "gpt-oss:latest"
}
