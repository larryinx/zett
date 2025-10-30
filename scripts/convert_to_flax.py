from transformers import AutoConfig, FlaxAutoModelForCausalLM, AutoTokenizer

model_name = "deepseek-ai/deepseek-coder-7b-base-v1.5"
save_path = "/scratch/yinx/flax/deepseek-coder-7b-base-flax"

config = AutoConfig.from_pretrained(model_name)
model = FlaxAutoModelForCausalLM.from_config(config)
model.save_pretrained(save_path)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_path)