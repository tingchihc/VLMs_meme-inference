import json
import os
import argparse
import torch
from PIL import Image
from datasets import load_dataset
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
            from transformers import LlavaForConditionalGeneration
            processor = AutoProcessor.from_pretrained(
                "mistral-community/pixtral-12b",
                use_fast=False
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                "mistral-community/pixtral-12b",
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
        elif self.model_name == "Qwen2-VL-7B-Instruct":
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                use_fast=True
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                "Qwen/Qwen2-VL-7B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
        elif self.model_name == "Qwen2.5-VL-7B-Instruct":
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                use_fast=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                "Qwen/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto"
            )
    
        elif self.model_name == "Qwen3-VL-8B-Instruct":
            processor = AutoProcessor.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
                use_fast=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto"
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return model, processor

    def _format_prompt(self, question: str, options: Dict) -> str:
        prompt = f"{question}\n\n"

        for i, option in options.items():
            prompt += f"{i}. {option}\n"
        
        prompt += "\nFirst, analyze the meme carefully and explain your reasoning (Chain of Thought). Then provide your answer by selecting one of the options above."
        
        return prompt
    
    def _extract_answer_and_cot(self, response: str, options: Dict) -> Tuple[Optional[str], str]:
        
        cot = response.strip()

        patterns = [
            r'\b([A-Z])\.',  # Matches: A. B. C.
            r'[Oo]ption\s*([A-Z])',  # Matches: Option A, option B
            r'\(([A-Z])\)',  # Matches: (A) (B)
            r'[Aa]nswer\s*:?\s*([A-Z])',  # Matches: Answer: A, answer A
            r'\b([A-Z])\b(?=\s*$)',  # Single letter at end of text
        ]
        
        predicted_answer = None
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                predicted_answer = matches[-1].upper()
                break
        
        return predicted_answer, cot
    
    def infer(self, question: str, meme_img: Image.Image, multiple_choice_options: Dict) -> str:
        
        image = meme_img
        prompt = self._format_prompt(question, multiple_choice_options)
        
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
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False)
            
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
        elif self.model_name == "Pixtral-12B":
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
            
            inputs = self.processor(text=prompt_text, images=[image],return_tensors="pt")
            inputs = {
                k: v.to(self.device, dtype=torch.float16) if k == "pixel_values" 
                else v.to(self.device)
                for k, v in inputs.items()
            }
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512, do_sample=False, pad_token_id=self.processor.tokenizer.eos_token_id)
            
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
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
            inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(**inputs, max_new_tokens=512)
            generated_ids = [
                output_ids[len(input_ids):] 
                for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response

def process_single_example(vlm: VLMInference, question: str, meme_img: Image.Image, multiple_choice_options: Dict, answer: int) -> Dict:

    response = vlm.infer(question, meme_img, multiple_choice_options)
    predicted_answer, cot = vlm._extract_answer_and_cot(response, multiple_choice_options)
    is_correct = predicted_answer == chr(answer+65) if predicted_answer else False
    
    return {
        "predicted_answer": predicted_answer,
        "is_correct": is_correct,
        "cot": cot,
        "ground_truth": answer,
        "full_response": response
    }

def run_test(vlm: VLMInference, test_set, output_folder: str):
    results = []

    for idx, example in enumerate(test_set, start=1):
        result = process_single_example(
            vlm,
            example["question"],
            example["image"],
            example["options"],
            example["answer"]
        )
        results.append(result)
        print(f"Processed example {idx}/{len(test_set)}")


    output_file = os.path.join(output_folder, f"{vlm.model_name}_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

def main():
    parser = argparse.ArgumentParser("Meme QA Answering Script")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output answers")
    args = parser.parse_args()
    
    dataset = load_dataset("tingcc01/M-QUEST")
    test_set = dataset["test"]

    vlm = VLMInference(args.model_name)
    run_test(vlm, test_set, args.output_folder)

if __name__ == "__main__":
    main()