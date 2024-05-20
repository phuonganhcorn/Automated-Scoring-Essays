from peft import LoraConfig, PeftModel
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


#Load the base model with default precision
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
adapter = "/home/phuong-anh/gama/trained-model/mistral1/checkpoint-200"
model = AutoModelForCausalLM.from_pretrained(model_name)

#Load and activate the adapter on top of the base model
model = PeftModel.from_pretrained(model, adapter)

#Merge the adapter with the base model
model = model.merge_and_unload()

#Save the merged model in a directory in the safetensors format
model_dir = "./trained-model/mistral1/full_model/"
model.save_pretrained(model_dir, safe_serialization=True)

#Save the custom tokenizer in the same directory
tokenizer.save_pretrained(model_dir)
