import os
from os.path import join, dirname
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss, SoftmaxLoss
import torch.nn as nn
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import utils 



PATH_DATA = join((os.getcwd()), 'data')
PATH_MODELS = join(os.getcwd(), 'models')
name_dir = 'setfit_standard_hate_16'
# load custom datasets

num_classes=2

eval = load_dataset('csv', data_files=join(PATH_DATA,'val_hate.csv'))


for seed in range(4):
    model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", use_differentiable_head=True, head_params={"out_features": num_classes}
)

    train = utils.load_and_subsample(join(PATH_DATA, 'train_hate'), 'labels', 16, seed=seed)



    trainer = SetFitTrainer(
    model=model,
    train_dataset=train,
    eval_dataset=eval['train'],
    loss_class=nn.BinaryCrossEntopyLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20,  # The number of text pairs to generate for contrastive learning
    num_epochs=1,  # The number of epochs to use for contrastive learning
    column_mapping={"text": "text", "labels": "label"}  # Map dataset columns to text/label expected by trainer
    )


# Train and evaluate
    trainer.train(learning_rate=1e-2, num_epochs=2)
    metrics = trainer.evaluate()

# save
    trainer.model._save_pretrained(save_directory=join(PATH_MODELS, name_dir, f'seed_{str(seed)}'))