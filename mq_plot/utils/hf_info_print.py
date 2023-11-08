
import torch
from transformers import AutoTokenizer, AutoModel
from torchinfo import summary


def print_model_summary(model_name_or_path):
    # Load the tokenizer and model dynamically based on the model name or path
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path)

    # Create a sample input
    input_text = "Hello, how are you today?"
    input_ids = torch.tensor(tokenizer.encode(input_text, add_special_tokens=True)).unsqueeze(0)

    # Print the model summary using torchinfo
    print(f"Model: {model_name_or_path}")
    print(f"Input Text: {input_text}")
    summary(model, input_data=input_ids)

if __name__ == '__main__':
    # Example usage:
    print_model_summary('gpt2')
    print_model_summary('bert-base-uncased')
