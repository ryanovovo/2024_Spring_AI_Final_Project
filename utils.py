from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
from datasets import load_dataset
from unsloth import FastLanguageModel
from datasets import Dataset


max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.



def predict(X, model, tokenizer, prompt_type):
    y_pred = []
    for i in tqdm(range(len(X))):
        tweet = X[i]
        # Format the prompt using the function
        if prompt_type == "good_prompt":
            formatted_prompt = generate_good_prompt(tweet)
        elif prompt_type == "bad_prompt":
            formatted_prompt = generate_bad_prompt(tweet)
        else:
            formatted_prompt = generate_no_prompt(tweet)
        

        # Tokenize the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        # Generate outputs using the model
        outputs = model.generate(**inputs, 
                                 max_new_tokens=20,
                                 early_stopping=True,
                                 pad_token_id=tokenizer.eos_token_id,
                                 temperature=0.0)

        # Decode the generated output to a string
        decoded_output = tokenizer.batch_decode(outputs)[0]
        # print(decoded_output)

        # Assume the sentiment label is at the end and extract it
        if prompt_type == "good_prompt":
            sentiment_output = decoded_output.split("### Sentiment Label:")[1].strip()
        elif prompt_type == "bad_prompt":
            sentiment_output = decoded_output.split("### Identified Digit:")[1].strip()
        else:
            sentiment_output = decoded_output.strip()

        if "Bearish" in sentiment_output:
            y_pred.append(0)
        elif "Bullish" in sentiment_output:
            y_pred.append(1)
        else:
            y_pred.append(2)
    return y_pred


def evaluate(y_true, y_pred):
    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
    f1score = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    # print(f'Accuracy: {accuracy:.3f}')
    
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')

    return accuracy, precision, recall, f1score, conf_matrix

def load_model_and_tokenizer(model_name, finetuned):
    if finetuned:
        model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "./finetuned_models/"+model_name,
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                # device_map = device_map,
                # llm_int8_enable_fp32_cpu_offload=True,
                # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
        )
    else:

        model, tokenizer = FastLanguageModel.from_pretrained(
                model_name = "unsloth/llama-3-8b-bnb-4bit",
                max_seq_length = max_seq_length,
                dtype = dtype,
                load_in_4bit = load_in_4bit,
                # device_map = device_map,
                # llm_int8_enable_fp32_cpu_offload=True,
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
        
    
    return model, tokenizer

def generate_good_prompt(tweet):
    prompt = f"""
    ### Task: Sentiment Classification of Financial Tweets
    You are an expert financial analyst, and your job is to analyze financial news tweets to determine their sentiment.

    ### Instructions:
    Classify the sentiment of the tweet with one of the following labels:
       - Bearish
       - Bullish
       - Neutral

    ### Data:
    - **Tweet**: {tweet}

    ### Sentiment Label:
    """
    return prompt

def generate_bad_prompt(tweet):
    prompt = f"""
    ### Task: Digit Recognition from Handwritten Images
    You are a highly skilled data scientist, and your job is to analyze images of handwritten digits to determine which numeral (0 through 9) they represent. Your goal is to identify the digit in each image accurately.

    ### Instructions:
    1. Analyze the handwritten digit image provided.
    2. Identify and return the numeral (0-9) that the image represents.

    - **Image**: {tweet}

    ### Identified Digit:
    - **Digit**:
    """
    return prompt

def generate_no_prompt(tweet):
    prompt = f"""
    {tweet}
    """
    return prompt


def load_validation_data(sample_size=100):
    validation_dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split = "validation")
    validation_df = pd.DataFrame(validation_dataset)
    sampled_df = pd.concat([
        validation_df[validation_df['label'] == 0].sample(n=sample_size),
        validation_df[validation_df['label'] == 1].sample(n=sample_size),
        validation_df[validation_df['label'] == 2].sample(n=sample_size)
    ])
    validation_dataset = Dataset.from_pandas(sampled_df)
    X = validation_dataset["text"]
    y_true = validation_dataset["label"]
    return X, y_true
    
   