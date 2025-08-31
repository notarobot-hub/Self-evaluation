# Configuration file for multi-dataset response generation

# Model configurations
MODEL_CONFIGS = {
    "llama2-7b": {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "unk_token": "<unk>",
        "stop_token": ["</s>"],
        "prompt_format": "### Instruction:\n{instruction}\n### Response:\n",
        "max_length": 200,
        "requires_auth": True
    },
    "llama3.1-8b": {
        "path": "meta-llama/Llama-3.1-8B-Instruct",
        "unk_token": "<unk>",
        "stop_token": ["</s>"],
        "prompt_format": "### Instruction:\n{instruction}\n### Response:\n",
        "max_length": 200,
        "requires_auth": True
    },
    "vicuna-7b": {
        "path": "lmsys/vicuna-7b-v1.5",
        "unk_token": "",
        "stop_token": ["</s>"],
        "prompt_format": "USER: {instruction} ASSISTANT:",
        "max_length": 200,
        "requires_auth": False
    }
}

# Dataset configurations
DATASET_CONFIGS = {
    "triviaqa": {
        "name": "trivia_qa",
        "config": "rc.nocontext",
        "split": "train",
        "concept_extraction": "from_answer"
    },
    "truthfulqa": {
        "name": "truthful_qa",
        "config": "generation",
        "split": "validation",
        "concept_extraction": "from_question"
    },
    "coqa": {
        "name": "coqa",
        "config": None,
        "split": "train",
        "concept_extraction": "from_question"
    },
    "tydiqa": {
        "name": "tydiqa",
        "config": "primary_task",
        "split": "train",
        "concept_extraction": "from_question"
    }
}

# Generation parameters
GENERATION_PARAMS = {
    "max_length": 200,
    "num_beams": 5,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "repetition_penalty": 1.1
}

# Output settings
OUTPUT_SETTINGS = {
    "save_intermediate": True,
    "output_dir": "./data",
    "file_format": "json"
}
