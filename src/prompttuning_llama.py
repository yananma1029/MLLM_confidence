
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import transformers
import torch
import numpy as np
import os
# from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
# from datasets import load_dataset

# Authenticate with Hugging Face

login(token='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')
os.environ['HF_HOME'] = 'MLLM_confidence/cache/'

model_name = "meta-llama/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.float16, 
    device_map = "auto",
    )

# print(f"model class : {type(model)}") # class 'transformers.models.llama.modeling_llama.LlamaForCausalLM
tokenizer = AutoTokenizer.from_pretrained(model_name)

system_prompt = "<<SYS>>\nYou are a helpful assistant, please answer the question carefully and in a few words as possible\n<</SYS>>"
user_message = 'Who is current American president?\n'

# Combine prompts for input
full_prompt = f"<s>[INST] {system_prompt}\n{user_message}[/INST]"

inputs = tokenizer(full_prompt, return_tensors="pt", return_token_type_ids=False).to(model.device)

# input tokens' logits
outputs = model(input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                return_dict=True)
logits = outputs.logits

print("________________model()_____________")
print(type(outputs)) # class 'transformers.modeling_outputs.CausalLMOutputWithPast'
print(outputs.keys()) # ['logits', 'past_key_values', 'hidden_states']
print(f"input id length : {inputs['input_ids'].size()}") # torch.Size([1, 50])
print(f'original logits size : {outputs.logits.size()}') # torch.Size([1, 50, 32000]) this is [batch_size, sequence_length(input_ids length), config.vocab_size] 

# 1 是 batch_size，50 是 sequence_length，32000 是词汇表大小 vocab_size
logits = torch.randn(1, 50, 32000)  # 假设的 logits
# Step 1: 对每个时间步（每个序列的 token）计算 softmax
probabilities = F.softmax(logits, dim=-1)  # 在最后一维 (vocab_size) 上计算 softmax
# Step 2: 对每个时间步使用 argmax 找出最高概率的 token ID
predicted_token_ids = torch.argmax(probabilities, dim=-1)  # 输出形状为 [1, 50]
# Step 3: 将 token IDs 转换为一维列表
predicted_token_ids = predicted_token_ids.squeeze(0).tolist()  # 形状变为 [50]
# Step 4: 使用 tokenizer.decode 将 token ID 转换为对应的 token 序列
decoded_tokens = tokenizer.decode(predicted_token_ids)
# 输出结果
print(f"predicted_tokens: {decoded_tokens}") # （每次都不一样）processdel Kan bashcialebounds Tow▲dawn continues Parlamentрода Initialize page moltoERE Kubiring Code2 ATieur ric Russell schedule enjoywortedes bataδдах blessistique anyoneEnvquisitionComplete controlledChoлав proyectiento king tren nosevir Jahrhunderts redundantUNcord



outputs_gen = model.generate(
    **inputs,
    do_sample = True,
    top_k = 10,
    num_return_sequences = 1,
    eos_token_id = tokenizer.eos_token_id,
    max_new_tokens = 50,
    output_scores = True,
    return_dict_in_generate = True,
)

print("________________model.generate()______________")
print(type(outputs_gen)) # class 'transformers.generation.utils.SampleDecoderOnlyOutput'
print(outputs_gen.keys()) # ['sequences', 'scores']
probabilities = [F.softmax(tensor) for tensor in outputs_gen['scores']]
# print(probabilities)
max_probabilities = [t.max() for t in probabilities]
predicted_token_ids = [torch.argmax(t, dim=-1) for t in probabilities]
predicted_token_ids = [t.item() for t in predicted_token_ids]#[29871, 18585, 29991, 29871, 29896, 718, 29871, 29896, 29900, 29900, 353, 29871, 29896, 29900, 29896, 29889, 2]
# 输出结果
# print(predicted_token_ids)
decoded_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids) 
print(f"decoded_tokens: {decoded_tokens}") #  Sure! 1 + 100 = 101.</s>

with open('output.tsv', 'w') as f:
    f.write("token\tid\tmax_prob\tprob\n")
    for token, id , max_prob , prob in zip(decoded_tokens, predicted_token_ids, max_probabilities, probabilities):
        print(f"{token}\t{str(id)}\t{str(max_prob.item())}\n")
        f.write(f"{token}\t{str(id)}\t{str(max_prob.item())}\n")
        # f.write(f"{token}\t{str(id)}\t{str(max_prob.item())}\t{str(prob.tolist())}\n")



'''
input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
transition_scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True)

# use outputs['logits'] instead of .compute_transition_scores()

# Maunelly calculate probabolities for each token
# logits = [F.softmax(tensor) for tensor in outputs['scores']]
# print(type(logits))
# print(logits[0].max())

# Decode the generated sequences into text
generated_text = tokenizer.decode(outputs.sequences[0].tolist(), skip_special_tokens=True)
print(f"{generated_text}")

# Get the tokens that were generated (excluding the input length)
generated_tokens = outputs.sequences[:, input_length:]

for tok, score in zip(generated_tokens[0], transition_scores[0]):
    # | token | token string | log probability | probability
    print(f"| {tok:5d} | {tokenizer.decode([tok], skip_special_tokens=True):8s} | {score.cpu().numpy():.3f} | {np.exp(score.cpu().numpy()):.2%}")
'''



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