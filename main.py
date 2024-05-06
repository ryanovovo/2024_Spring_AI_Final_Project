# from finetune import finetune
import os
import pandas as pd
from utils import load_model_and_tokenizer as load
from utils import predict, evaluate, load_validation_data
import torch
import gc
import warnings
from finetune import finetune
warnings.filterwarnings('ignore')


def main(test_times=10):
    prompt_types = ['good_prompt', 'bad_prompt', 'no_prompt']

    for prompt_type in prompt_types:
        if not os.path.exists(f"./finetuned_models/{prompt_type}"):
            os.makedirs(f"./finetuned_models/{prompt_type}")
            finetune(prompt_type)
            gc.collect()
            torch.cuda.empty_cache()

    if not os.path.exists("./results"):
        os.makedirs("./results")

    for prompt_type in prompt_types:
        df = pd.DataFrame(columns=["Test", "Accuracy", "Precision", "Recall", "F1 Score"])
        model, tokenizer = load(prompt_type, True)
        for i in range(test_times):
            X, y_true = load_validation_data(sample_size=100)
            y_pred = predict(X, model, tokenizer, prompt_type)
            accuracy, precision, recall, f1score, conf_matrix = evaluate(y_true, y_pred)
            print(f"Test {i+1}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1score}")
            df.loc[len(df)] = {"Test": i+1, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1score}

        df.to_csv(f"./results/{prompt_type}.csv")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    for prompt_type in prompt_types:
        df = pd.DataFrame(columns=["Test", "Accuracy", "Precision", "Recall", "F1 Score"])
        model, tokenizer = load(prompt_type, False)
        for i in range(test_times):
            X, y_true = load_validation_data(sample_size=100)
            y_pred = predict(X, model, tokenizer, prompt_type)
            accuracy, precision, recall, f1score, conf_matrix = evaluate(y_true, y_pred)
            print(f"Test {i+1}: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1score}")
            df.loc[len(df)] = {"Test": i+1, "Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1 Score": f1score}

        df.to_csv(f"./results/{prompt_type}_unfinetuned.csv")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

        
        


    

        