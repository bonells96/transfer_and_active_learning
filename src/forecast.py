from setfit import SetFitModel
import os
from os.path import join, dirname

PATH_DATA = join((os.getcwd()), 'data')
PATH_MODELS = join(os.getcwd(), 'models')


model = SetFitModel.from_pretrained("PATH_MODELS", local_files_only=True)

sentiment_dict = {"negative": 0, "positive": 1}
inverse_dict = {value: key for (key, value) in sentiment_dict.items()}

# Run inference
text_list = [
    "i loved the spiderman movie!",
    "pineapple on pizza is the worst",
    "what the fuck is this piece",
    "good morning, lady boss",
    "the product is excellent",
    "a piece of rubbish"
]

preds = model(text_list)

print(preds)