# Arabic Text-to-SQL using Fine-tuned T5

## Overview

This project focuses on fine-tuning a pre-trained T5 model to translate natural language questions written in Arabic into corresponding SQL queries. It utilizes the Hugging Face Transformers library and PyTorch for the training and evaluation process. The project uses an Arabic version of the Spider dataset, likely derived from or similar to the Ar-Spider dataset [2][4].

## Features

*   **Arabic-to-SQL Translation:** Fine-tunes a T5 model specifically for converting Arabic questions into executable SQL queries.
*   **Model Fine-tuning:** Uses the `transformers` library (`Trainer` API) to adapt a pre-trained text-to-SQL model (`cssupport/t5-small-awesome-text-to-sql`) to the Arabic language dataset.
*   **Data Handling:** Loads, preprocesses, and splits data from a JSONL file (`AR_spider.jsonl`).
*   **Evaluation:** Measures performance using Exact Match Accuracy and BLEU scores [3].
*   **Context Similarity Relationship (CSR - Experimental):** Includes an experimental implementation inspired by the CSR approach described in the Ar-Spider paper [4] to potentially improve performance by incorporating schema context using sentence embeddings.
*   **Reproducibility:** Sets random seeds for consistent results.

## Requirements

*   Python 3.x
*   PyTorch (with CUDA support recommended for faster training)
*   Hugging Face Account (potentially, for downloading models)
*   The Arabic text-to-SQL dataset file (e.g., `AR_spider.jsonl`, based on Ar-Spider [2][4]). Place this file in an accessible location (e.g., `/kaggle/input/txttosql-nlp/` as per the script, or update the path in the script).

## Installation

1.  **Clone the repository (or save the script):**
    If this is in a repository:
    ```
    git clone <your-repository-url>
    cd <repository-directory>
    ```
    If you only have the script file, save it (e.g., `train_arabic_sql.py`) in a new directory and navigate into it.

2.  **Create a `requirements.txt` file** with the following content:
    ```
    torch
    transformers
    pandas
    numpy
    scikit-learn
    evaluate
    seaborn
    matplotlib
    tqdm
    sentence-transformers # For CSR implementation
    # Add accelerate if using multi-GPU or specific optimizations
    # accelerate
    # Add wandb if you want to enable WandB logging (currently disabled in script)
    # wandb
    ```

3.  **Create a virtual environment (recommended):**
    ```
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

4.  **Install the required libraries:**
    ```
    pip install -r requirements.txt
    ```
    *(Note: Ensure you have the correct PyTorch version installed for your CUDA setup if using GPU)*

## Dataset

This project requires an Arabic text-to-SQL dataset in JSONL format. The script is configured to load `/kaggle/input/txttosql-nlp/AR_spider.jsonl`. This dataset is likely based on the Ar-Spider project, which is an Arabic translation of the popular Spider dataset [2][4]. Ensure this file exists at the specified path or modify the script (`line 33`) to point to the correct location of your dataset file.

The Ar-Spider dataset contains pairs of Arabic natural language questions and their corresponding SQL queries over various database schemas [4].

## Usage

1.  **Configure:**
    *   Verify the dataset path in the script (`line 33`).
    *   Adjust training arguments (epochs, batch size, etc.) in the `TrainingArguments` within the `train_specialized_t5_model` function if needed.
    *   The script disables W&B logging (`os.environ["WANDB_DISABLED"] = "true"`). Remove or comment this line (`line 20`) if you want to use Weights & Biases.

2.  **Run the training script:**
    ```
    python train_arabic_sql.py
    ```
    (Replace `train_arabic_sql.py` with the actual filename).

3.  **Process:**
    *   The script will load and preprocess the data.
    *   It will split the data into training, validation, and test sets.
    *   It downloads the base model `cssupport/t5-small-awesome-text-to-sql` and fine-tunes it on the Arabic training data.
    *   Checkpoints and the best model (based on evaluation loss) will be saved to the specified `output_dir` (default: `./specialized_t5_arabic_sql`).
    *   After training, the model is evaluated on the test set, printing Exact Match Accuracy and Average BLEU score.
    *   Example translations from the test set are performed and printed.
    *   Finally, the best model and tokenizer are saved to `./best_arabic_sql_model` along with a sample `inference.py` script.

## Evaluation

The model's performance is evaluated using:

*   **Exact Match Accuracy:** Percentage of generated SQL queries that exactly match the target SQL query (after stripping whitespace).
*   **BLEU Score:** Measures the similarity between the generated and target SQL queries, considering n-gram overlap.

## Context Similarity Relationship (CSR)

The script includes functions (`apply_csr`, `translate_arabic_to_sql` with `use_csr=True`) demonstrating an experimental implementation of the CSR approach [4]. This technique aims to improve schema linking by calculating the similarity between the Arabic question tokens and potential schema items (tables/columns) using multilingual sentence embeddings (`paraphrase-multilingual-mpnet-base-v2`). The most relevant schema items are then appended to the input to provide context to the model during generation. Note that the inference implementation uses a simplified version with common SQL keywords, as actual schema access would be needed for a full implementation.

## Inference

After training, the best model is saved in the `./best_arabic_sql_model` directory. An example `inference.py` script is also saved there. You can use it as follows:

