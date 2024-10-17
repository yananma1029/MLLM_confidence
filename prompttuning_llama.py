
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import transformers
import torch
import numpy as np
# from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
# from datasets import load_dataset

# Authenticate with Hugging Face
login(token='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')

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map = "auto",
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)

system_prompt = "<<SYS>>\nYou are a helpful assistant, please answer the question carefully and in a few words as possible\n<</SYS>>"
user_message = 'What is 1+100?/n'

# Combine prompts for input
full_prompt = f"<s>[INST] {system_prompt}\n{user_message}[/INST]"

inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    do_sample = True,
    top_k = 10,
    num_return_sequences = 1,
    eos_token_id = tokenizer.eos_token_id,
    max_new_tokens = 50,
    output_scores = True,
    return_dict_in_generate = True,
)

input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

#use outputs['logits'] instead of .compute_transition_scores()

# Decode the generated sequences into text
generated_text = tokenizer.decode(outputs.sequences[0].tolist(), skip_special_tokens=True)
print(f"{generated_text}")

# Get the tokens that were generated (excluding the input length)
generated_tokens = outputs.sequences[:, input_length:]

for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | log probability | probability
    print(f"| {tok:5d} | {tokenizer.decode([tok], skip_special_tokens=True):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")




''' 
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=2,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
    truncation=True,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")
'''