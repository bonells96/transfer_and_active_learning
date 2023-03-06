# Transfer Learning and Active Lerning


## Transfer Learning

In this project we compare different few shot learning models:

* setfit model
* setfit model only training the classification layer
* setfit model without contrastive learning

### Training Setfit Model

For training the setfit model, first run in the CLI:

````
git https://github.com/bonells96/transfer_and_active_learning.git 
cd transfer_and_active_learning
pip install -r requirements.txt
````

If you want to train the setfit model as in [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

then an example for running it through the CLI is (for each use case change the arguments):

````
python src/run_train_setfit.py --PATH_DATA "users/transfer_and_active_learning/data" --name_data "train_hate" --only_head False --save_model "setfit_hate"
````

### Experiment

We want to compare different few-shot-learning methods and their performance depending on the number of examples given for training. For each dataset we take different subsamples of different lengths.


For this case we choose to take subsamples of length 8, 16, 32, 64 and 100. All of the subsamples are balanced (same number of examples for each class).
To increase the robustness of our experiment for each length we take 5 different samples, like this the result of the experiment will be less dependent on the samples chosen.

For example, we take 5 different subsamples with 8 examples (4 examples of each class).

Then we train the model on each subsample and we forecast the validation data and save the predictions.

### Hate Dataset

For the Hate dataset we have the following results for Accuracy and F1:
![](img/accuracies_tl_hate.png)

The setfit model is the one with the best performance. It has the best metrics and the smaller variances in the results. We see that with only 32 examples of each class the model is only 7 pts below of the state of the art in terms of Accuracy. 

The setfit_head (only head trained) model performances are clearly below the ones from the setfit model. We can see also that the results have much more variance they vary a lot depending on the samples chosen. 

![](img/f1s_tl_hate.png) 

For the F1 the results are similar than before, but the curves gets closer. For the model without contrastive learning we observe a huge variance for few examples. This is because depending on the samples chosen the model almost only outputs one of the classes and then gets a very low F1 while having an accuracy close to 50%.

To conclude, for the Hate data we see that the setfit model, is a great option if we have few ressources for labelling. We see a great difference between the other techniques and setfit. It is strange though, that for 50 samples of each class the setfit performance is lower than for 32 of each class. Maybe with an active learning strategy for selecting the most appropiate labels the performance could be better.

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