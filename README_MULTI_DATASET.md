# Multi-Dataset Response Generation using Self-Familiarity Method

This project extends the existing self-familiarity method to generate responses on multiple datasets (TriviaQA, TruthfulQA, CoQA, TyDiQA) for different language models (Llama 2 7B, Llama 3.1 8B, Vicuna 7B).

## Overview

The self-familiarity method works by:
1. **Forward Generation**: Generating a detailed explanation of a concept
2. **Backward Generation**: Using the generated explanation to predict what concept it relates to
3. **Scoring**: Computing a familiarity score based on the backward generation quality

This approach helps evaluate how well a model understands concepts by measuring its ability to both explain and recognize them.

## Features

- **Multi-Model Support**: Works with Llama 2 7B, Llama 3.1 8B, and Vicuna 7B
- **Multi-Dataset Processing**: Processes TriviaQA, TruthfulQA, CoQA, and TyDiQA datasets
- **Local Model Support**: Works with locally downloaded models
- **Automatic Fallback**: Uses sample data if datasets are not accessible
- **Comprehensive Output**: Saves both intermediate and final results
- **Memory Optimization**: Efficient memory management for large models

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Setup

### Option 1: Download Models from Hugging Face

```bash
# Create models directory
mkdir -p models

# Download Llama 2 7B (requires Hugging Face access)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir models/llama2-7b-chat-hf

# Download Llama 3.1 8B (requires Hugging Face access)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir models/llama3.1-8b-instruct

# Download Vicuna 7B
huggingface-cli download lmsys/vicuna-7b-v1.5 --local-dir models/vicuna-7b-v1.5
```

### Option 2: Use Custom Model Paths

Update the model paths in `run_all_models.py`:

```python
models = {
    "llama2-7b": {
        "path": "/path/to/your/llama2-7b-model",
        "samples": 50
    },
    "llama3.1-8b": {
        "path": "/path/to/your/llama3.1-8b-model",
        "samples": 50
    },
    "vicuna-7b": {
        "path": "/path/to/your/vicuna-7b-model",
        "samples": 50
    }
}
```

## Usage

### Single Model Generation

Generate responses for a specific model:

```bash
python generate_responses_local.py \
    --model_name llama2-7b \
    --model_path ./models/llama2-7b-chat-hf \
    --samples_per_dataset 50 \
    --device cuda
```

### Batch Generation for All Models

Run generation for all models sequentially:

```bash
python run_all_models.py
```

### Command Line Arguments

- `--model_name`: Model identifier (llama2-7b, llama3.1-8b, vicuna-7b)
- `--model_path`: Path to the local model checkpoint
- `--samples_per_dataset`: Number of samples to process per dataset (default: 50)
- `--device`: Device to use (cuda/cpu, default: cuda)

## Output Structure

Results are saved in the `./data/` directory with the following structure:

```
data/
├── llama2-7b_triviaqa_responses.json
├── llama2-7b_truthfulqa_responses.json
├── llama2-7b_coqa_responses.json
├── llama2-7b_tydiqa_responses.json
├── llama2-7b_all_datasets_responses.json
├── llama3.1-8b_triviaqa_responses.json
├── llama3.1-8b_truthfulqa_responses.json
├── llama3.1-8b_coqa_responses.json
├── llama3.1-8b_tydiqa_responses.json
├── llama3.1-8b_all_datasets_responses.json
├── vicuna-7b_triviaqa_responses.json
├── vicuna-7b_truthfulqa_responses.json
├── vicuna-7b_coqa_responses.json
├── vicuna-7b_tydiqa_responses.json
└── vicuna-7b_all_datasets_responses.json
```

### Output Format

Each response entry contains:

```json
{
  "dataset": "triviaqa",
  "question": "What is the capital of France?",
  "concept": "France",
  "forward_response": "France is a country located in Western Europe...",
  "backward_response": "France",
  "backward_score": 0.85
}
```

## Dataset Information

### TriviaQA
- **Purpose**: Question-answering with factual knowledge
- **Concept Extraction**: From answer field
- **Use Case**: Testing factual knowledge and concept understanding

### TruthfulQA
- **Purpose**: Truth-seeking questions
- **Concept Extraction**: From question field
- **Use Case**: Testing truthfulness and factual accuracy

### CoQA
- **Purpose**: Conversational question-answering
- **Concept Extraction**: From question field
- **Use Case**: Testing conversational understanding

### TyDiQA
- **Purpose**: Multilingual question-answering
- **Concept Extraction**: From question field
- **Use Case**: Testing cross-lingual concept understanding

## Configuration

### Model Configurations

Edit `config.py` to modify model-specific settings:

```python
MODEL_CONFIGS = {
    "llama2-7b": {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "unk_token": "<unk>",
        "stop_token": ["</s>"],
        "prompt_format": "### Instruction:\n{instruction}\n### Response:\n",
        "max_length": 200,
        "requires_auth": True
    }
}
```

### Generation Parameters

Adjust generation parameters in `config.py`:

```python
GENERATION_PARAMS = {
    "max_length": 200,
    "num_beams": 5,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}
```

## Memory Management

The system automatically manages memory by:
- Using sequential device mapping for multi-GPU setups
- Setting appropriate memory limits per device
- Using bfloat16 precision for memory efficiency
- Clearing GPU cache between models

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure the model path is correct
   - Check if the model is properly downloaded
   - Verify model directory structure

2. **Out of Memory**
   - Reduce `samples_per_dataset`
   - Use CPU instead of GPU (`--device cpu`)
   - Close other GPU applications

3. **Dataset Loading Issues**
   - The system automatically falls back to sample data
   - Check internet connection for Hugging Face datasets
   - Verify dataset names and configurations

4. **Import Errors**
   - Install all requirements: `pip install -r requirements.txt`
   - Ensure PyTorch and Transformers are properly installed

### Performance Tips

- Use GPU for faster generation
- Adjust `samples_per_dataset` based on available memory
- Process models sequentially to avoid memory conflicts
- Monitor GPU memory usage during generation

## Example Workflow

1. **Setup**: Download models and install dependencies
2. **Configuration**: Update model paths in `run_all_models.py`
3. **Execution**: Run `python run_all_models.py`
4. **Results**: Check `./data/` directory for generated responses
5. **Analysis**: Use the JSON files for further analysis or evaluation

## Customization

### Adding New Models

1. Add model configuration to `config.py`
2. Update model choices in argument parsers
3. Ensure proper prompt formatting for the new model

### Adding New Datasets

1. Add dataset configuration to `config.py`
2. Implement dataset processing method in `LocalMultiDatasetResponseGenerator`
3. Update the dataset list in `run_all_datasets()`

### Modifying Generation Logic

- Edit `generate_forward_response()` for forward generation changes
- Edit `generate_backward_response()` for backward generation changes
- Modify scoring logic in the response generation methods

## License

This project extends the existing self-familiarity method. Please refer to the original project's license terms.

## Citation

If you use this code, please cite the original self-familiarity method paper and acknowledge this extension.
