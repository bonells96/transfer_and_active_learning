# Transfer Learning and Active Lerning


## Transfer Learning

In this project we compare different few shot learning models:

* setfit model
* setfit model only training the classification layer
* setfit model without contrastive learning

### Training Setfit Model

For training the setfit model, first run in the CLI:

````
git clone 
cd repo
pip install -r requirements.txt
````

If you want to train the setfit model as in [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

then an example for running it through the CLI is (for each use case change the arguments):

````
python src/run_train_setfit.py --PATH_DATA "users/transfer_and_active_learning/data" --name_data "train_hate" --only_head False --save_model "setfit_hate"
````

### Hate Dataset



![](img/metrics_tl_v1.png)

## Active Learning

In this project, we aim to develop an effective strategy for selecting the best examples to label in order to improve the performance of machine learning models that have been trained on a limited amount of data. Specifically, we will be working with models that have been trained on just 100 labeled examples, and we have the ability to manually label an additional 100 examples to improve the model's accuracy. Given the limited amount of labeled data, we recognize the importance of selecting the most informative examples to label in order to make the most of our labeling efforts. We  explored different active learning approaches to select the best text for labelling.


### Hate Data

**Initial Step**: 

The model at the beggining has been trained in 100 examples. After the training the model has an accuracy of 65.89% on the validation set and an f1 of 65.47%.



## Requirements

````
datasets==2.10.1
huggingface-hub==0.12.1
numpy==1.24.2
pandas==1.5.3
regex==2022.10.31
scikit-learn==1.2.1
scipy==1.10.1
seaborn==0.12.2
sentence-transformers==2.2.2
sentencepiece==0.1.97
setfit==0.6.0
tokenizers==0.13.2
torch==1.13.1
torchvision==0.14.1
tqdm==4.64.1
transformers==4.26.1
````