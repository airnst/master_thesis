# README.MD

Before deploying the repository, the data folder must be downloaded. This is a ZIP file available [here](https://box.fu-berlin.de/s/9TZFLSdRSLzE8By). Please unzip it as /master_thesis/data.

The experiments can be found in experiments.ipynb. Please don't forget to activate the .venv in the folder named .venv-ex and to install the packages listed in the requirements.txt.

The notebooks in the folder ml document the conducted training process.
In folder ocr, the feature extraction can be found.
Folder ood contains the notebook documenting the evaluation-labelling pipeline for OOD records.

The large evaluator class among others can be found in the Classes folder. In the directory "models" all trained models can be found. create_training_dataset.ipynb uses the Evaluator and allows for evaluation of multiple transcription tools.
