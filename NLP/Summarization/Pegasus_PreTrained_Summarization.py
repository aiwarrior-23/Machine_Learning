from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

#Input Text
src_text = [
'''The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.

The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.

At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.

"We'll be the comeback kids, all of us," he said. "We want to get our country back."

The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.'''
]

#Model Initialization
model_name = 'google/pegasus-xsum'

#Device Initialization
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Creating Object of Pegasus Class
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

#Creating Tokenizer
tokenizer = PegasusTokenizer.from_pretrained(model_name)
batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)

#Summarization
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(tgt_text)