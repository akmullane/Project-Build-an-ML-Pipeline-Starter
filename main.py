import json
import os
import tempfile

import hydra
import mlflow
from omegaconf import DictConfig

_steps = [
    "download",
    "basic_cleaning",
    "data_check",
    "data_split",
    "train_random_forest",
    # NOTE: We do not include this in the steps so it is not run by mistake.
    # You first need to promote a model export to "prod" before you can run this,
    # then you need to run this step explicitly
    # "test_regression_model",
]


@hydra.main(version_base=None, config_name="config", config_path=".")
def go(config: DictConfig):

    # Setup W&B experiment grouping
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    steps_par = config["main"]["steps"]
    active_steps = steps_par.split(",") if steps_par != "all" else _steps

    # Use a temp dir for intermediate files
    with tempfile.TemporaryDirectory():

        # 1) Download
        if "download" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    config["main"]["components_repository"],
                    "get_data",
                ),
                "main",
                env_manager="conda",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded",
                },
            )

        # 2) Basic cleaning
        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                env_manager="conda",
                parameters={
                    "input_artifact": config["etl"]["input"],
                    # IMPORTANT: output artifact NAME cannot include ":latest"
                    "output_artifact": config["etl"]["cleaned"].split(":")[0],
                    "output_type": "clean_data",
                    "output_description": "Data with basic cleaning applied",
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        # 3) Data check (must be src/data_check)
        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                env_manager="conda",
                parameters={
                    # IMPORTANT: when READING artifacts, it SHOULD include ":latest"
                    "csv": config["etl"]["cleaned"],
                    "ref": config["data_check"]["ref"],
                    "kl_threshold": config["data_check"]["kl_threshold"],
                    "min_price": config["etl"]["min_price"],
                    "max_price": config["etl"]["max_price"],
                },
            )

        # 4) Train/val/test split
        if "data_split" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    config["main"]["components_repository"],
                    "train_val_test_split",
                ),
                "main",
                env_manager="conda",
                parameters={
                    "input": config["etl"]["cleaned"],
                    "test_size": config["modeling"]["test_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                },
            )

          # 5) Train random forest
        if "train_random_forest" in active_steps:

            # Serialize RF config into JSON (project requirement: DO NOT TOUCH)
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+", encoding="utf-8") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                env_manager="conda",
                parameters={
                    "trainval_artifact": "trainval_data.csv:latest",
                    "val_size": config["modeling"]["val_size"],
                    "random_seed": config["modeling"]["random_seed"],
                    "stratify_by": config["modeling"]["stratify_by"],
                    "rf_config": rf_config,
                    "max_tfidf_features": config["modeling"]["max_tfidf_features"],
                    "output_artifact": "random_forest_export",
                },
            )

        # Optional: test regression model (only after promoting model to prod)
        if "test_regression_model" in active_steps:
            _ = mlflow.run(
                os.path.join(
                    hydra.utils.get_original_cwd(),
                    config["main"]["components_repository"],
                    "test_regression_model",
                ),
                "main",
                env_manager="conda",
                parameters={
                    "mlflow_model": "model_export:prod",
                    "test_dataset": "test_data.csv:latest",
                },
            )


if __name__ == "__main__":
    go()
