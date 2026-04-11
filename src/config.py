"""
Centralized configuration for the research experiment pipeline.

All models, budgets, tasks, and paths are defined here.
Edit this file to change experiment parameters.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]  # research-paper/
DATA_DIR = ROOT_DIR / "data" / "raw"
RESULTS_DIR = ROOT_DIR / "results"
DOCUMENTS_DIR = RESULTS_DIR / "documents"

ENV_PATHS = [
    ROOT_DIR.parents[2] / "SlideMaker-Backend" / ".env",
    Path.home() / "Projects" / "Slidemaker app" / "SlideMaker-Backend" / ".env",
]

# ──────────────────────────────────────────────────────────────
# Generation Models  (model_id, provider, display_name)
# ──────────────────────────────────────────────────────────────
GENERATION_MODELS = [
    {"id": "gpt-4o",                    "provider": "openai",    "name": "GPT-4o"},
    {"id": "claude-sonnet-4-20250514",  "provider": "anthropic", "name": "Claude Sonnet"},
    {"id": "gemini-2.0-flash",          "provider": "google",    "name": "Gemini Flash"},
]

# ──────────────────────────────────────────────────────────────
# Evaluation (Judge) Models
# ──────────────────────────────────────────────────────────────
JUDGE_MODELS = [
    {"id": "gpt-4o",                    "provider": "openai",    "name": "GPT-4o"},
    {"id": "claude-sonnet-4-20250514",  "provider": "anthropic", "name": "Claude Sonnet"},
    {"id": "gemini-2.0-flash",          "provider": "google",    "name": "Gemini Flash"},
]

# ──────────────────────────────────────────────────────────────
# Chunking Methods & Budgets
# ──────────────────────────────────────────────────────────────
CHUNKING_METHODS = [
    "truncation",
    "fixed_size_first_last",
    "semantic_breakpoint",
    "pac_position_aware",
]

# Default budget for main experiments (words)
DEFAULT_BUDGET = 2000

# Ablation budgets
ABLATION_BUDGETS = [1000, 2000, 3000, 5000]

# ──────────────────────────────────────────────────────────────
# Downstream Tasks
# ──────────────────────────────────────────────────────────────
TASKS = ["slides", "summary"]

# ──────────────────────────────────────────────────────────────
# W-curve Sensitivity Analysis Parameters
# ──────────────────────────────────────────────────────────────
# Default W-curve: intro=2.0, conclusion=2.0, results_peak=1.8, floor=0.6
# We vary each parameter independently to test sensitivity
SENSITIVITY_CONFIGS = [
    # Baseline
    {"name": "baseline",   "intro": 2.0, "conclusion": 2.0, "results_peak": 1.8, "floor": 0.6},
    # Vary intro weight
    {"name": "intro_low",  "intro": 1.0, "conclusion": 2.0, "results_peak": 1.8, "floor": 0.6},
    {"name": "intro_high", "intro": 3.0, "conclusion": 2.0, "results_peak": 1.8, "floor": 0.6},
    # Vary conclusion weight
    {"name": "concl_low",  "intro": 2.0, "conclusion": 1.0, "results_peak": 1.8, "floor": 0.6},
    {"name": "concl_high", "intro": 2.0, "conclusion": 3.0, "results_peak": 1.8, "floor": 0.6},
    # Vary results peak
    {"name": "res_low",    "intro": 2.0, "conclusion": 2.0, "results_peak": 1.0, "floor": 0.6},
    {"name": "res_high",   "intro": 2.0, "conclusion": 2.0, "results_peak": 2.5, "floor": 0.6},
    # Vary floor
    {"name": "floor_zero", "intro": 2.0, "conclusion": 2.0, "results_peak": 1.8, "floor": 0.0},
    {"name": "floor_high", "intro": 2.0, "conclusion": 2.0, "results_peak": 1.8, "floor": 1.2},
    # Flat (no position bias — ablation)
    {"name": "flat",       "intro": 1.0, "conclusion": 1.0, "results_peak": 1.0, "floor": 1.0},
    # U-curve (original, no results bump)
    {"name": "u_curve",    "intro": 2.0, "conclusion": 2.0, "results_peak": 0.6, "floor": 0.6},
]

# ──────────────────────────────────────────────────────────────
# Evaluation Metrics
# ──────────────────────────────────────────────────────────────
SLIDE_METRICS = [
    "completeness", "accuracy", "statistics_retention",
    "coherence", "relevance", "coverage_balance",
]

SUMMARY_METRICS = [
    "completeness", "accuracy", "statistics_retention",
    "coherence", "conciseness", "coverage_balance",
]

# ──────────────────────────────────────────────────────────────
# Batch API settings
# ──────────────────────────────────────────────────────────────
BATCH_ENABLED = True              # Use batch APIs where available (50% cost savings)
BATCH_POLL_INTERVAL = 60          # seconds between batch status checks
BATCH_MAX_WAIT = 86400            # max wait for batch completion (24h)

# Rate limiting (requests per minute) for non-batch calls
RATE_LIMITS = {
    "openai": 30,
    "anthropic": 30,
    "google": 15,
}

# ──────────────────────────────────────────────────────────────
# Human Evaluation
# ──────────────────────────────────────────────────────────────
HUMAN_EVAL_PAIRS = 30             # Number of doc pairs for human evaluation
HUMAN_EVAL_SEED = 42              # Random seed for reproducible pair selection
