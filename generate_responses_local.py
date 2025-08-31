import os
import json
import torch
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import AutoTokenizer, AutoModelForCausalLM
from myutils import StoppingCriteriaSub, StoppingCriteriaList, ask_original_question, mask_input_words
import argparse
from config import MODEL_CONFIGS, DATASET_CONFIGS, GENERATION_PARAMS, OUTPUT_SETTINGS

class LocalMultiDatasetResponseGenerator:
    def __init__(self, model_name, model_path, device="cuda"):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.stop_words = []
        
        # Get model configuration
        self.model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["llama2-7b"])
        
        self.load_model()
        self.setup_stopping_criteria()
        
    def load_model(self):
        """Load the specified model and tokenizer"""
        print(f"Loading model: {self.model_path}")
        
        try:
            # Check if model path exists locally
            if os.path.exists(self.model_path):
                print(f"Loading local model from: {self.model_path}")
                model_path = self.model_path
            else:
                print(f"Model path {self.model_path} not found locally. Please ensure the model is downloaded.")
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            # Set device map for memory optimization
            max_memory = {0: "40GiB", 1: "40GiB", 2: "40GiB", "cpu": "40GiB"}
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                cache_dir='./cache',
                trust_remote_code=True,
                device_map="sequential",
                max_memory=max_memory,
                torch_dtype=torch.bfloat16
            )
            self.model.eval()
            
            # Load tokenizer
            if 'llama' in self.model_path.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    cache_dir='./cache',
                    unk_token="<unk>",
                    bos_token="<s>",
                    eos_token="</s>"
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    cache_dir='./cache'
                )
                
            print(f"Model {model_path} loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def setup_stopping_criteria(self):
        """Setup stopping criteria for generation"""
        stop_words = self.model_config["stop_token"]
        
        stop_words_ids = [self.tokenizer.vocab.get(stop_word, 0) for stop_word in stop_words]
        s1 = StoppingCriteriaSub(stops=stop_words_ids, encounters=1)
        self.stopping_criteria = StoppingCriteriaList([s1])
        
        # Download stopwords if not already downloaded
        try:
            self.stop_words = stopwords.words('english')
        except:
            import nltk
            nltk.download('stopwords')
            self.stop_words = stopwords.words('english')
    
    def get_prompt_format(self):
        """Get the appropriate prompt format for the model"""
        return self.model_config["prompt_format"]
    
    def get_max_length(self):
        """Get the maximum generation length for the model"""
        return self.model_config["max_length"]
    
    def generate_forward_response(self, question, concept):
        """Generate forward response using the self-familiarity method"""
        try:
            # Format question with concept
            formatted_question = question.format(concept=concept)
            
            # Apply prompt format
            prompt_format = self.get_prompt_format()
            formatted_prompt = prompt_format.format(instruction=formatted_question)
            
            # Generate response
            outputs, original_decoding_scores = ask_original_question(
                formatted_prompt, 
                self.model, 
                self.tokenizer,
                self.stopping_criteria,
                max_length=self.get_max_length()
            )
            
            # Decode response
            response = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Clean response
            if response.endswith('User '):
                response = response[:-5]
            if response.endswith('\n'):
                response = response[:-1]
                
            return response, outputs, original_decoding_scores
            
        except Exception as e:
            print(f"Error generating forward response: {e}")
            return "", None, None
    
    def generate_backward_response(self, concept, targets, forward_outputs):
        """Generate backward response using the self-familiarity method"""
        try:
            if forward_outputs is None:
                return "", 0.0
                
            # Get forward text
            text_forward = self.tokenizer.decode(forward_outputs.sequences[0], skip_special_tokens=True)
            if text_forward.endswith('User '):
                text_forward = text_forward[:-5]
            if text_forward.endswith('\n'):
                text_forward = text_forward[:-1]
            
            # Create backward question
            question_backward = f'"{text_forward}" is related to what?'
            prompt_format = self.get_prompt_format()
            question_backward = prompt_format.format(instruction=question_backward)
            
            # Mask input words
            question_backward_masked = mask_input_words(question_backward, targets, self.tokenizer)
            input_ids_mask = self.tokenizer(question_backward_masked, return_tensors="pt").input_ids.to(self.device)
            
            # Get target words
            ori_target = targets[-1] if targets else concept
            target_words = list(set([
                ori_target, 
                ori_target.lower(), 
                ori_target.upper(), 
                ' '.join([x.capitalize() for x in ori_target.split()])
            ]))
            target_words = target_words + [' ' + x for x in target_words]
            
            # Get force words IDs
            force_words_ids = self.tokenizer(target_words, add_special_tokens=False).input_ids
            
            # Generate constrained output
            outputs_constraint = self.model.generate(
                input_ids_mask,
                force_words_ids=[force_words_ids],
                output_scores=True,
                return_dict_in_generate=True,
                max_length=input_ids_mask.shape[-1] + len(force_words_ids) + 10,
                stopping_criteria=self.stopping_criteria,
                num_beams=5,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
            # Decode backward response
            backward_text = self.tokenizer.decode(outputs_constraint.sequences[0])
            back_score = torch.exp(outputs_constraint.sequences_scores).item() if hasattr(outputs_constraint, 'sequences_scores') else 0.0
            
            return backward_text, back_score
            
        except Exception as e:
            print(f"Error generating backward response: {e}")
            return "", 0.0
    
    def create_sample_data(self, dataset_name, num_samples=50):
        """Create sample data for testing when datasets are not available"""
        print(f"Creating sample data for {dataset_name} dataset...")
        
        sample_data = []
        
        if dataset_name == "triviaqa":
            # Sample trivia questions
            sample_questions = [
                "What is the capital of France?",
                "Who wrote Romeo and Juliet?",
                "What is the chemical symbol for gold?",
                "In which year did World War II end?",
                "What is the largest planet in our solar system?"
            ]
            
            for i in range(min(num_samples, len(sample_questions))):
                question = sample_questions[i % len(sample_questions)]
                concept = question.split()[-1].rstrip('?')
                
                sample_data.append({
                    'question': question,
                    'concept': concept
                })
                
        elif dataset_name == "truthfulqa":
            # Sample truth-seeking questions
            sample_questions = [
                "What causes rain?",
                "How do plants grow?",
                "Why is the sky blue?",
                "What makes a rainbow?",
                "How do birds fly?"
            ]
            
            for i in range(min(num_samples, len(sample_questions))):
                question = sample_questions[i % len(sample_questions)]
                concept = question.split()[-1].rstrip('?')
                
                sample_data.append({
                    'question': question,
                    'concept': concept
                })
                
        elif dataset_name == "coqa":
            # Sample conversational questions
            sample_questions = [
                "What is the main character's name?",
                "Where does the story take place?",
                "What happens at the end?",
                "Who is the villain?",
                "What is the conflict?"
            ]
            
            for i in range(min(num_samples, len(sample_questions))):
                question = sample_questions[i % len(sample_questions)]
                concept = question.split()[-1].rstrip('?')
                
                sample_data.append({
                    'question': question,
                    'concept': concept
                })
                
        elif dataset_name == "tydiqa":
            # Sample multilingual questions
            sample_questions = [
                "What is the weather like?",
                "Where is the nearest hospital?",
                "How do I get there?",
                "What time does it open?",
                "How much does it cost?"
            ]
            
            for i in range(min(num_samples, len(sample_questions))):
                question = sample_questions[i % len(sample_questions)]
                concept = question.split()[-1].rstrip('?')
                
                sample_data.append({
                    'question': question,
                    'concept': concept
                })
        
        return sample_data
    
    def process_dataset(self, dataset_name, num_samples=50):
        """Process a specific dataset"""
        print(f"Processing {dataset_name} dataset...")
        
        try:
            # Try to load from Hugging Face datasets
            try:
                from datasets import load_dataset
                dataset_config = DATASET_CONFIGS[dataset_name]
                
                if dataset_config["config"]:
                    dataset = load_dataset(dataset_config["name"], dataset_config["config"], split=dataset_config["split"])
                else:
                    dataset = load_dataset(dataset_config["name"], split=dataset_config["split"])
                
                print(f"Loaded {dataset_name} dataset from Hugging Face")
                
            except Exception as e:
                print(f"Could not load {dataset_name} from Hugging Face: {e}")
                print("Using sample data instead...")
                dataset = self.create_sample_data(dataset_name, num_samples)
            
            results = []
            
            for i, example in enumerate(dataset):
                if i >= num_samples:
                    break
                
                # Extract question and concept based on dataset
                if dataset_name == "triviaqa" and isinstance(example, dict) and 'answer' in example:
                    question = example['question']
                    answer = example['answer']['value']
                    concept = answer if isinstance(answer, str) else answer[0] if answer else "unknown"
                elif isinstance(example, dict) and 'question' in example:
                    question = example['question']
                    concept = question.split('?')[0].split()[-1] if '?' in question else question.split()[-1]
                else:
                    # Handle sample data format
                    question = example['question']
                    concept = example['concept']
                
                # Generate forward response
                forward_question = "Explain the concept of \"{concept}\" in detail."
                forward_response, forward_outputs, _ = self.generate_forward_response(forward_question, concept)
                
                # Generate backward response
                targets = word_tokenize(concept)
                targets = [x for x in targets if x.lower() not in self.stop_words] + [concept]
                
                backward_response, back_score = self.generate_backward_response(concept, targets, forward_outputs)
                
                results.append({
                    'dataset': dataset_name,
                    'question': question,
                    'concept': concept,
                    'forward_response': forward_response,
                    'backward_response': backward_response,
                    'backward_score': back_score
                })
                
                print(f"Processed {dataset_name} sample {i+1}/{min(num_samples, len(dataset))}")
                
            return results
            
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            return []
    
    def save_results(self, results, output_file):
        """Save results to file"""
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def run_all_datasets(self, num_samples_per_dataset=50):
        """Run the method on all datasets"""
        print(f"Starting response generation for model: {self.model_name}")
        print(f"Processing {num_samples_per_dataset} samples per dataset")
        
        all_results = []
        
        # Process each dataset
        datasets = ['triviaqa', 'truthfulqa', 'coqa', 'tydiqa']
        
        for dataset_name in datasets:
            print(f"\n{'='*50}")
            print(f"Processing {dataset_name.upper()} dataset")
            print(f"{'='*50}")
            
            try:
                results = self.process_dataset(dataset_name, num_samples_per_dataset)
                all_results.extend(results)
                
                # Save intermediate results
                if OUTPUT_SETTINGS["save_intermediate"]:
                    intermediate_file = os.path.join(
                        OUTPUT_SETTINGS["output_dir"], 
                        f"{self.model_name}_{dataset_name}_responses.json"
                    )
                    self.save_results(results, intermediate_file)
                
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                continue
        
        # Save all results
        all_results_file = os.path.join(
            OUTPUT_SETTINGS["output_dir"], 
            f"{self.model_name}_all_datasets_responses.json"
        )
        self.save_results(all_results, all_results_file)
        
        print(f"\n{'='*50}")
        print(f"Completed! Total samples processed: {len(all_results)}")
        print(f"Results saved to: {all_results_file}")
        print(f"{'='*50}")
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Generate responses using self-familiarity method on multiple datasets')
    parser.add_argument('--model_name', type=str, required=True, 
                       choices=['llama2-7b', 'llama3.1-8b', 'vicuna-7b'],
                       help='Name of the model to use')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the local model checkpoint')
    parser.add_argument('--samples_per_dataset', type=int, default=50,
                       help='Number of samples to process per dataset')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = LocalMultiDatasetResponseGenerator(
            model_name=args.model_name,
            model_path=args.model_path,
            device=args.device
        )
        
        # Run on all datasets
        results = generator.run_all_datasets(args.samples_per_dataset)
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
