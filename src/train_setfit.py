import os
from os.path import join, dirname
from datasets import load_dataset
from sentence_transformers.losses import CosineSimilarityLoss

from setfit import SetFitModel, SetFitTrainer, sample_dataset
import utils 




PATH_DATA = join((os.getcwd()), 'data')
PATH_MODELS = join(os.getcwd(), 'models')
name_dir = 'setfit_hate_100'
# load custom datasets

eval = load_dataset('csv', data_files=join(PATH_DATA,'val_hate.csv'))


for seed in range(4):
    model = SetFitModel.from_pretrained(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
    train = utils.load_and_subsample(join(PATH_DATA, 'train_hate'), 'labels', 100, seed=seed)
    print(train['labels'])
    print(train['text'])

    trainer = SetFitTrainer(
    model=model,
    train_dataset=train,
    eval_dataset=eval['train'],
    loss_class=CosineSimilarityLoss,
    metric="accuracy",
    batch_size=16,
    num_iterations=20,  # The number of text pairs to generate for contrastive learning
    num_epochs=2,  # The number of epochs to use for contrastive learning
    column_mapping={"text": "text", "labels": "label"}  # Map dataset columns to text/label expected by trainer
    )

# Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()

# save
    trainer.model._save_pretrained(save_directory=join(PATH_MODELS, name_dir, f'seed_{str(seed)}'))