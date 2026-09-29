"""End-to-end pipeline entry point: python -m ab.gpt.act.acc_predictor."""
from __future__ import annotations

from . import config
from .preprocess import data_preprocessing
from .dataset import prepare_llm_datasets
from .train import train_model
from .eval import test_model


def _run_tag() -> str:
    """Folder-safe '<model>_seed<seed>' tag for this run, read from config at call
    time so editing MODEL_NAME / SEED in config.py routes results to a new folder."""
    base = str(config.MODEL_NAME).rsplit("/", 1)[-1]  # drop the org/ prefix
    base = "".join(c if (c.isalnum() or c in "-.") else "_" for c in base)
    return f"{base}_seed{config.SEED}"


def main() -> None:
    # Per-run output dir: out/acc_predict/<model>_seed<seed>/  (datasets stay shared
    # in ACC_DIR since they depend on the toggles, not on the model or seed).
    # Self-contained run folder: raw data, datasets, checkpoints, predictions and
    # metrics for THIS model+seed all live here. Only the expensive, model/seed-
    # independent proxy cache stays shared (its values are already baked into the
    # per-run datasets below), so it is not rebuilt for every run.
    run_dir = config.ACC_DIR / _run_tag()
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = run_dir / "llm_finetuning_data.jsonl"
    raw_csv = run_dir / "llm_finetuning_data.csv"
    output_dir = run_dir / "tuned_model"
    test_output_path = run_dir / "test_predictions.csv"
    test_metrics_path = run_dir / "test_metrics.log"
    print(f"Run: model={config.MODEL_NAME}  seed={config.SEED}")
    print(f"Outputs -> {run_dir}")

    print("=" * 80)
    print("Step 1/4: Data preprocessing (nn_dataset → JSONL)")
    print("=" * 80)
    data_preprocessing(
        output_jsonl_path=raw_jsonl,
        output_csv_path=raw_csv,
    )

    print("\n" + "=" * 80)
    print("Step 2/4: Prepare LLM training datasets")
    print("=" * 80)
    train_path, val_path, test_path = prepare_llm_datasets(input_path=raw_jsonl, output_dir=run_dir)
    print(f"Train: {train_path}")
    print(f"Val:   {val_path}")
    print(f"Test:  {test_path}")

    print("\n" + "=" * 80)
    print("Step 3/4: Fine-tune model")
    print("=" * 80)
    train_model(train_path=train_path, val_path=val_path, output_dir=output_dir,
                model_name=config.MODEL_NAME, seed=config.SEED)
    print(f"Model saved to {output_dir}")

    print("\n" + "=" * 80)
    print("Step 4/4: Test model on held-out test set")
    print("=" * 80)
    test_model(
        model_path=output_dir,
        data_path=test_path,
        output_path=test_output_path,
        metrics_path=test_metrics_path,
    )
    print(f"\nDone. Pipeline complete. Results in {run_dir}")

if __name__ == "__main__":
    main()
