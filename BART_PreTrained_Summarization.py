from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

#Creation of Model Object
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

#Input Text
ARTICLE_TO_SUMMARIZE = '''The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.

The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.

At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.

"We'll be the comeback kids, all of us," he said. "We want to get our country back."

The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.'''

#Tokenization
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

#Summarization
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])