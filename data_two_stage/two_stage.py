from typing import List, Optional, Dict
import fire
import json
from tqdm import tqdm
import torch
# 替换为 modelscope
from modelscope import AutoModelForCausalLM, AutoTokenizer

# Qwen3 思考结束标记的 Token ID (来自您的示例，请务必确认其准确性)
# 如果您的 tokenizer 支持，更好的方式可能是 tokenizer.convert_tokens_to_ids("</think>") 或类似方法
THINK_END_TOKEN_ID = 151668

def _save_data_to_json(data_to_save: Dict, output_path: str, context_message: str = "Saving data"):
    """辅助函数，用于将数据保存到JSON文件。"""
    try:
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(data_to_save, file, indent=2, ensure_ascii=False)
        print(f"\n{context_message}: Successfully saved data to {output_path}")
    except IOError:
        print(f"\n{context_message}: Error - Could not write to output file {output_path}")
    except Exception as e:
        print(f"\n{context_message}: An unexpected error occurred during file saving: {e}")


def generate_with_qwen_thinking(
    data: List[Dict], # 当前正在处理的数据集 (例如 training_data)
    model,
    tokenizer,
    max_batch_size: int, # 用于分批处理此数据集
    temperature: float,
    top_p: float,
    max_new_tokens: Optional[int],
    # 新增参数用于周期性保存
    save_interval_batches: Optional[int],
    output_json_path_for_saving: str,
    full_input_data_dict_for_saving: Dict # 包含 'train', 'val', 'test' 的顶层字典
):
    model.eval()
    batch_counter = 0 # 当前数据集的批次计数器

    for i in tqdm(range(0, len(data), max_batch_size), mininterval=2, desc='   - (Generating)   ', leave=False):
        current_data_batch = data[i : i + max_batch_size]
        if not current_data_batch:
            continue

        all_prompts_text = []
        for sample in current_data_batch:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Your task is to generate a 'ReviewSummary'. "
                        "The ReviewSummary must be a single, concise sentence. "
                        "This sentence should sound like a natural user review and reflect how well the item's attributes likely align with the user's preferences (e.g., convey a positive, neutral, or negative sentiment implicitly)."
                        "This sentence must start with 'ReviewSummary: '"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"User Preference: \"{sample['user_preference']}\"\n"
                        f"Item Attributes: \"{sample['item_attribution']}\"\n\n"
                        "Generate a ReviewSummary based on the provided preference and attributes."
                    )
                }
            ]
            try:
                text_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False # 改为 True 以启用思考模式并允许解析 </think>
                )
                all_prompts_text.append(text_prompt)
            except Exception as e:
                print(f"Error applying chat template for explanation '{sample['explanation']}': {e}")
                all_prompts_text.append(f"Error processing: {sample['explanation']}")

        if not all_prompts_text:
            continue

        model_inputs = tokenizer(
            all_prompts_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length if hasattr(tokenizer, 'model_max_length') and tokenizer.model_max_length else 2048
        ).to(model.device)

        generated_ids_batch = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0.01 else None,
            top_p=top_p if temperature > 0.01 else None,
            do_sample=True if temperature > 0.01 else False,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else []
        )

        for batch_idx, original_sample in enumerate(current_data_batch):
            original_data_idx = i + batch_idx
            input_ids_len = len(model_inputs.input_ids[batch_idx])
            current_output_ids_with_pad = generated_ids_batch[batch_idx][input_ids_len:]
            current_output_ids = []
            if tokenizer.eos_token_id is not None:
                try:
                    eos_index = current_output_ids_with_pad.tolist().index(tokenizer.eos_token_id)
                    current_output_ids = current_output_ids_with_pad[:eos_index].tolist()
                except ValueError:
                    current_output_ids = current_output_ids_with_pad.tolist()
            else:
                current_output_ids = current_output_ids_with_pad.tolist()

            actual_content_text = ""
            try:
                think_marker_end_pos = len(current_output_ids) - current_output_ids[::-1].index(THINK_END_TOKEN_ID)
                content_token_ids = current_output_ids[think_marker_end_pos:]
                actual_content_text = tokenizer.decode(content_token_ids, skip_special_tokens=True).strip()
            except ValueError:
                actual_content_text = tokenizer.decode(current_output_ids, skip_special_tokens=True).strip()

            stripped_line = actual_content_text
            review_summary = None
            review_summary = stripped_line[len("ReviewSummary: "):]
            
            if review_summary is not None:
                data[original_data_idx]['review_summary'] = review_summary
            else:
                data[original_data_idx]['review_summary'] = original_sample['explanation']

        
        batch_counter += 1
        if save_interval_batches and save_interval_batches > 0 and (batch_counter % save_interval_batches == 0):
            _save_data_to_json(
                full_input_data_dict_for_saving,
                output_json_path_for_saving,
                context_message=f"Periodic save after {batch_counter} batches"
            )
    # `data` is modified in place, so `full_input_data_dict_for_saving` reflects changes.
    # No explicit return of `data` needed as it's a mutable type modified by reference.


def main(
    model_path: str = '/home/wangjing/Quantization/Model/Qwen/Qwen2.5-7B-Instruct',
    input_json_path: str = '/home/wangjing/RDRec/data_qwen_distillation/beauty/explanation_rationale.json',
    output_json_path: str = '/home/wangjing/RDRec/data_two_stage/beauty/explanation_rationale.json', #文件名稍作修改以示区别
    max_seq_len: int = 1024,
    max_batch_size: int = 64,
    temperature: float = 0.1,
    top_p: float = 0.9,
    max_new_tokens: Optional[int] = 200,
    save_every_n_batches: Optional[int] = 100 # 每隔多少批次保存一次，设为 None 或 0 则不进行周期性保存
):
    print(f"Loading model and tokenizer from: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        return
        
    if not hasattr(tokenizer, 'model_max_length') or tokenizer.model_max_length is None:
        tokenizer.model_max_length = max_seq_len
    
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Set tokenizer.pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
        else:
            print("Warning: tokenizer has no pad_token_id and no eos_token_id. Padding might be problematic.")

    print("Model and Tokenizer loaded successfully.")

    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data_dict = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input JSON file not found at {input_json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}")
        return

    datasets_to_process = {
        'train': input_data_dict.get('train', []),
        'val': input_data_dict.get('val', []),
        'test': input_data_dict.get('test', [])
    }

    for name, current_dataset in datasets_to_process.items():
        if current_dataset:
            print(f"\nProcessing '{name}' data ({len(current_dataset)} samples)...")
            generate_with_qwen_thinking(
                data=current_dataset, # 传入当前数据集的引用
                model=model,
                tokenizer=tokenizer,
                max_batch_size=max_batch_size,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                save_interval_batches=save_every_n_batches, # 新增
                output_json_path_for_saving=output_json_path, # 新增
                full_input_data_dict_for_saving=input_data_dict # 新增, 传递整个字典以便保存
            )
        else:
            print(f"No data for '{name}', skipping.")

    # 所有处理完成后执行最终保存
    _save_data_to_json(input_data_dict, output_json_path, context_message="Final save after all processing")


if __name__ == "__main__":
    fire.Fire(main)
    # 命令行运行示例:
    # python your_script_name.py --model_path="/path/to/Qwen" --max_batch_size=64 --save_every_n_batches=50
    # python your_script_name.py --save_every_n_batches=10 # 使用默认路径和其他参数