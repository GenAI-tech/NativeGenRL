from datasets import load_dataset

dataset = load_dataset("BLIP3o/BLIP3o-60k", cache_dir="./cache")

print(dataset['train'][0])  # {'text': 'a photo of a soap bar'}

# Save all text strings to a file, one per line
with open('blip3o_texts.txt', 'w') as f:
    for item in dataset['train']:
        f.write(item['text'] + '\n')
