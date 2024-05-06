from unsloth import FastLanguageModel
import argparse
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import numpy as np
import os
max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

good_prompt = """
### Task: Sentiment Classification of Financial Tweets
You are an expert financial analyst, and your job is to analyze financial news tweets to determine their sentiment.

### Instructions:
1. Analyze the financial tweets provided.
2. Classify each tweet with the appropriate sentiment label from the following options:
   - Bearish
   - Bullish
   - Neutral

### Data:
- **Tweet**: {}

### Sentiment Label:
- **Sentiment**: {}
"""

bad_prompt = """
### Task: Digit Recognition from Handwritten Images
You are a highly skilled data scientist, and your job is to analyze images of handwritten digits to determine which numeral (0 through 9) they represent. Your goal is to identify the digit in each image accurately.

### Instructions:
1. Analyze the handwritten digit image provided.
2. Identify and return the numeral (0-9) that the image represents.

### Data:
- **Image**: {}

### Identified Digit:
- **Digit**: {}
"""

no_prompt = """
{}
{}
"""



def finetune(prompt_type):
    # parser = argparse.ArgumentParser(description="Script to handle different types of prompts.")

    # # 添加参数
    # parser.add_argument('--good_prompt', action='store_true', help='Handle good prompt')
    # parser.add_argument('--bad_prompt', action='store_true', help='Handle bad prompt')
    # parser.add_argument('--no_prompt', action='store_true', help='Handle no prompt')

    # # 解析命令行参数
    # args = parser.parse_args()

    # # 根据参数执行相应的函数
    # if args.good_prompt:
    #     prompt = good_prompt
    #     output_dir = "./finetunec_models/good_prompt"
    # elif args.bad_prompt:
    #     prompt = bad_prompt
    #     output_dir = "./finetuned_models/bad_prompt"
    # elif args.no_prompt:
    #     prompt = no_prompt
    #     output_dir = "./finetuned_models/no_prompt"
    # else:
    #     prompt = good_prompt
    #     output_dir = "./finetuned_models/good_prompt"

    

    if prompt_type == "good_prompt":
        prompt = good_prompt
    elif prompt_type == "bad_prompt":
        prompt = bad_prompt
    elif prompt_type == "no_prompt":
        prompt = no_prompt
    else:
        prompt = good_prompt

    output_dir = f"./finetuned_models/{prompt_type}"

    # print("Prompt: ", prompt)


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 64, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
    def formatting_prompts_func(examples):
        inputs = examples["text"]
        outputs = examples["label"]
        texts = []
        for input, output in zip(inputs, outputs):
            if output == 0:
                label = "Bearish"
            elif output == 1:
                label = "Bullish"
            else:
                label = "Neutral"
            text = prompt.format(input, label) + EOS_TOKEN
            # print(text)
            texts.append(text)
        return {"text": texts}


    dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 100,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "paged_adamw_32bit",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
        ),
    )

    trainer_stats = trainer.train()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.cuda.empty_cache()

if __name__ == "__main__":
    finetune()