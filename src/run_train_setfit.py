import argparse
import logging
import os
from os.path import join
from datasets import load_dataset
from setfit import SetFitModel, SetFitTrainer, sample_dataset
from sentence_transformers.losses import CosineSimilarityLoss
import utils
logging.basicConfig(level=logging.INFO)

PATH_DATA = join((os.getcwd()), 'data')
PATH_MODELS = join(os.getcwd(), 'models')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training a setfit Model')
    parser.add_argument('--PATH_DATA', type=str, help='path to store the profiles', default=PATH_DATA)
    parser.add_argument('--name_train', type=str, help='name datafile containing trainset', default='train_hate')
    parser.add_argument('--name_val', type=str, help='name datafile containing validation set', default='val_hate')

    parser.add_argument('--n', type=int, help='number of training samples', default=10)
    parser.add_argument('--only_head', type=bool, help='bool indicating if we train the full model or only the head', default=True)

    parser.add_argument('--num_seeds', type=int, help='number of different random samples from training', default=4)


    parser.add_argument('--batch_size', type=int, help='batch_size', default=20)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=2)

    parser.add_argument('--name_sentence_transformer', type=str, help='name pretrained sentence transformer', default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument('--name_save_model', type=str, help='name save file', default='setfit_head_hate')
    args = parser.parse_args()



    eval = load_dataset('csv', data_files=join(args.PATH_DATA,f'{args.name_val}.csv'))
    NUM_CLASSES = 2

    for seed in range(args.num_seeds):

        if args.only_head:
            model = SetFitModel.from_pretrained(
                                        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", use_differentiable_head=True, head_params={"out_features": NUM_CLASSES})
        else:
            model = SetFitModel.from_pretrained(args.name_sentence_transformer)

        train = utils.load_and_subsample(join(args.PATH_DATA, f'{args.name_train}'), 'labels', args.n, seed=seed)

        trainer = SetFitTrainer(
                            model=model,
                            train_dataset=train,
                            eval_dataset=eval['train'],
                            loss_class=CosineSimilarityLoss,
                            metric="accuracy",
                            batch_size=args.batch_size,
                            num_iterations=20,  # The number of text pairs to generate for contrastive learning
                            num_epochs=args.epochs,  # The number of epochs to use for contrastive learning
                            column_mapping={"text": "text", "labels": "label"}  # Map dataset columns to text/label expected by trainer
                            )

        # Train and evaluate
        if args.only_head:
            trainer.freeze()
            trainer.unfreeze(keep_body_frozen=True)
            trainer.train(learning_rate=1e-2, num_epochs=args.epochs)
        else:
            trainer.train()
        
        metrics = trainer.evaluate()

        # save
        trainer.model._save_pretrained(save_directory=join(PATH_MODELS, args.name_save_model+'_'+str(args.n), f'seed_{str(seed)}'))