import os
import json
import argparse
import torch
from PIL import Image
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    AutoProcessor, LlavaForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    AutoModelForVision2Seq
)
from typing import Dict, List, Tuple, Optional
import re

# Only needed for Qwen models
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None

class VLMInference:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = self._load_model()
    
    def _load_model(self):
        """Load the appropriate model and processor based on model_name"""
        if self.model_name == "BLIP2-Flan-T5-xl":
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-flan-t5-xl",
                torch_dtype=torch.float16
            ).to(self.device)
            
        elif self.model_name == "InstructBLIP-Vicunna-7B":
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            model = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-vicuna-7b",
                torch_dtype=torch.float16
            ).to(self.device)
            
        elif self.model_name == "LLaVA-v1.5":
            processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16
            ).to(self.device)
            
        elif self.model_name == "LLaVA-v1.6-Vicuna":
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
            model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-vicuna-7b-hf",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(self.device)
            
        elif self.model_name == "Pixtral-12B":
            processor = AutoProcessor.from_pretrained("mistral-community/pixtral-12b")
            model = AutoModelForVision2Seq.from_pretrained(
                "mistral-community/pixtral-12b",
                torch_dtype=torch.float16
            ).to(self.device)
            
        elif self.model_name == "Qwen2-VL-7B-Instruct":
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.float16
            ).to(self.device)
            
        elif self.model_name == "Qwen2.5-VL-7B-Instruct":
            processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.float16
            ).to(self.device)
            
        elif self.model_name == "Qwen3-VL-8B-Instruct":
            processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model, processor
    
    def _format_prompt(self, question: str, options: List[str]) -> str:
        """
        Format the question with multiple choice options
        Example output:
        Does this meme show toxicity?
        
        A. Contains offensive language
        B. None of the others
        C. Promotes harmful stereotypes
        D. No toxicity detected
        
        First, analyze the meme carefully and explain your reasoning (Chain of Thought). 
        Then provide your answer by selecting one of the options above.
        """
        prompt = f"{question}\n\n"
        for i, option in enumerate(options):
            # chr(65) = 'A', chr(66) = 'B', chr(67) = 'C', etc.
            prompt += f"{chr(65+i)}. {option}\n"
        prompt += "\nFirst, analyze the meme carefully and explain your reasoning (Chain of Thought). Then provide your answer by selecting one of the options above."
        return prompt
    
    def _extract_answer_and_cot(self, response: str, options: List[str]) -> Tuple[Optional[str], str]:
        """
        Extract the predicted answer and Chain of Thought from model response
        
        Args:
            response: Full model response text
            options: List of answer choices like ['Contains offensive language', 'None of the others', ...]
        
        Returns:
            Tuple of (predicted_answer_text, cot_explanation)
            Example: ('No toxicity detected', 'The meme shows fashion imagery...')
        """
        # The full response is the CoT
        cot = response.strip()
        
        # Try to find the selected answer letter (A, B, C, D)
        # Look for patterns like "A.", "Option A", "(A)", "Answer: A", etc.
        patterns = [
            r'\b([A-Z])\.',  # Matches: A. B. C.
            r'[Oo]ption\s*([A-Z])',  # Matches: Option A, option B
            r'\(([A-Z])\)',  # Matches: (A) (B)
            r'[Aa]nswer\s*:?\s*([A-Z])',  # Matches: Answer: A, answer A
            r'\b([A-Z])\b(?=\s*$)',  # Single letter at end of text
        ]
        
        predicted_letter = None
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                # Take the last match (usually the final answer)
                predicted_letter = matches[-1].upper()
                break
        
        # Convert letter (A, B, C, D) to actual answer text
        predicted_answer = None
        if predicted_letter and ord('A') <= ord(predicted_letter) < ord('A') + len(options):
            # ord('A') = 65, so if predicted_letter='D' and we have 4 options:
            # ord('D') - ord('A') = 68 - 65 = 3, so options[3] is the answer
            predicted_answer = options[ord(predicted_letter) - ord('A')]
        
        return predicted_answer, cot
    
    def infer(self, question: str, meme_img: str, multiple_choice_options: List[str]) -> str:
        """
        Run inference on a single example
        
        Args:
            question: e.g., "Does this meme show toxicity?"
            meme_img: Path to image file
            multiple_choice_options: List of choices
        
        Returns:
            Full model response text
        """
        # Load image
        image = Image.open(meme_img).convert('RGB')
        
        # Format prompt
        prompt = self._format_prompt(question, multiple_choice_options)
        
        # Model-specific inference
        if self.model_name in ["BLIP2-Flan-T5-xl", "InstructBLIP-Vicunna-7B"]:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        elif self.model_name == "LLaVA-v1.5":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elif self.model_name == "LLaVA-v1.6-Vicuna":
            # LLaVA-v1.6 (LlavaNext) uses simpler processing
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(self.device, torch.float16)
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )
            
            # Decode only the generated part (exclude input prompt)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        elif self.model_name == "Pixtral-12B":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(self.device, torch.float16)
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        elif self.model_name in ["Qwen2-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen3-VL-8B-Instruct"]:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": meme_img},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

def process_single_example(
    vlm: VLMInference,
    question: str,
    meme_img: str,
    multiple_choice_options: List[str],
    answer: List[str]
) -> Dict:
    """
    Process a single example and return results
    
    Args:
        vlm: VLMInference instance
        question: e.g., "Does this meme show toxicity?"
        meme_img: Path like '/home/.../98235.png'
        multiple_choice_options: ['Contains offensive language', 'None of the others', ...]
        answer: ['No toxicity detected'] - ground truth answer(s)
    
    Returns:
        Dict containing:
            - predicted_answer: 'No toxicity detected' or None
            - is_correct: True/False
            - cot: Chain of Thought explanation text
            - ground_truth: ['No toxicity detected']
            - full_response: Complete model output
    """
    # Get model response
    response = vlm.infer(question, meme_img, multiple_choice_options)
    
    # Extract answer and CoT
    predicted_answer, cot = vlm._extract_answer_and_cot(response, multiple_choice_options)
    
    # Check if correct
    # predicted_answer: 'No toxicity detected'
    # answer: ['No toxicity detected']
    is_correct = predicted_answer in answer if predicted_answer else False
    
    return {
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "cot": cot,
        "ground_truth": answer,
        "full_response": response
    }

def run_test(jsonld_path, prefix, vlm, output_folder):
    """Process all examples and save results"""
    
    for jsonld_file in jsonld_path:
        with open(jsonld_file, 'r', encoding="utf-8") as f:
            data = json.load(f)
        
        question = data["question"]
        multiple_choice_options = []
        answer = []
        GT_explanation = data["explanation"]
        dimension = data["dimension"]
        meme_img = os.path.join(prefix, data["sourceImage"]["filename"])
        
        for choice in data["answers"]:
            multiple_choice_options.append(choice["text"])
            if choice["is_correct"] == True:
                answer.append(choice["text"])
        
        # This is your format_
        format_ = {
            "question": question,
            "multiple_choice_options": multiple_choice_options,
            "answer": answer,
            "GT_explanation": GT_explanation,
            "dimension": dimension,
            "meme_img": meme_img
        }
        
        # Process this example using format_
        result = process_single_example(
            vlm, 
            format_["question"], 
            format_["meme_img"], 
            format_["multiple_choice_options"], 
            format_["answer"]
        )
        
        # Add metadata
        result.update({
            "jsonld_file": jsonld_file,
            "question": question,
            "dimension": dimension,
            "meme_img": meme_img,
            "GT_explanation": GT_explanation,
            "options": multiple_choice_options
        })
        
        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            base_filename = os.path.basename(jsonld_file).replace('.jsonld', '_result.json')
            output_path = os.path.join(output_folder, base_filename)
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(result, out_f, indent=4)
            print("[FINISH] ", jsonld_file)
    

def main():
    parser = argparse.ArgumentParser(description="Meme QA Answering Script")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--conda_env", type=str, required=True, help="Name of the conda environment to use")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input meme")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output answers")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model_name}")
    vlm = VLMInference(args.model_name)
    
    input_folder = args.input_folder
    jsonld_question_path = []
    
    for folder in os.listdir(input_folder):
        if folder.startswith("."):
            continue
        folder_path = os.path.join(input_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder != "benchmark_imgs":
            for files in os.listdir(folder_path):
                if files.startswith("."):
                    continue
                if files.endswith(".jsonld"):
                    jsonld_question_path.append(
                        os.path.join(folder_path, files)
                    )
    
    prefix = os.path.join(input_folder, "benchmark_imgs")
    run_test(jsonld_question_path, prefix, vlm, args.output_folder)

if __name__ == "__main__":
    main()

"""
python VLMs_inference.py  --model_name BLIP2-Flan-T5-xl --conda_env blip2_vqa --input_folder /home/tchen1/workshop/datasets/meme/benchmark_extracted --output_folder BLIP2-Flan-T5-xl
"""