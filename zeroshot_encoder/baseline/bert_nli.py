import math
import logging
import json
import random
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CESoftmaxAccuracyEvaluator
from dataset.load_data import get_all_zero_data, binary_cls_format, get_nli_data

random.seed(42)

def parse_args():
    args =  ArgumentParser()

    args.add_argument('--train', action='store_true', help='Train model')
    args.add_argument('--test', action='store_true', help='Evaluate mode')
    args.add_argument('--pretrain', action='store_true', help='Pre-train on MNLI data')
    return args.parse_args()

logger = logging.getLogger(__name__)

data = get_all_zero_data()

if __name__ == "__main__":
    args = parse_args()
    if args.pretrain:
        train, dev = get_nli_data()
        train_batch_size = 16
        num_epochs = 4
        model_save_path = 'model/bert_nli_pretrained'

        model = CrossEncoder('bert-base-uncased', num_labels=2)

        train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)
        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))

        evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(dev, name='AllNLI-dev')

        # Train the model
        model.fit(train_dataloader=train_dataloader,
                epochs=num_epochs,
                evaluator=evaluator,
                evaluation_steps=10000,
                warmup_steps=warmup_steps,
                output_path=model_save_path)

    if args.train:
        # get keys from data dict
        datasets = list(data.keys())
        datasets.remove("all")
        train = []
        test = []
        for dataset in datasets:
            train += binary_cls_format(data[dataset]["train"])
            test += binary_cls_format(data[dataset]["test"], train=False)

        train_batch_size = 16
        num_epochs = 3
        model_save_path = 'models/binary_bert_rand'

        model = CrossEncoder('bert-base-uncased', num_labels=2)

        random.shuffle(train)
        train_dataloader = DataLoader(train, shuffle=True, batch_size=train_batch_size)

        evaluator = CESoftmaxAccuracyEvaluator.from_input_examples(test, name='UTCD-test')

        warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
        logger.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        model.fit(train_dataloader=train_dataloader,
                evaluator=evaluator,
                epochs=num_epochs,
                evaluation_steps=100000,
                warmup_steps=warmup_steps,
                output_path=model_save_path)
    if args.test:
        # get keys from data dict
        datasets = list(data.keys())
        datasets.remove("all")

        # load model
        model = CrossEncoder('./models/binary_bert_rand')

        label_map = ["false", "true"]

        # loop through all datasets
        for dataset in datasets:
            test = data[dataset]["test"]
            labels = list(set(x[1] for x in test))
            gold = [x[1] for x in test]

            preds = []
            correct = 0
            # loop through each test example
            print("Evaluating dataset: {}".format(dataset))
            for index, example in enumerate(tqdm(test)):
                query = [(label, example[0]) for label in labels]
                results = model.predict(query, apply_softmax=True)

                # compute which pred is higher
                pred = labels[results[:,1].argmax()]
                preds.append(pred)
                if pred == gold[index]:
                    correct += 1
            
            print('{} Dataset Accuracy = {}'.format(dataset, correct/len(test)))
            report = classification_report(gold, preds, output_dict=True)
            json.dump([ [test[i][0], pred, gold[i]] for i, pred in enumerate(preds)], open('preds/binary_bert_{}.json'.format(dataset), 'w'), indent=4)
            # plt = sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
            # plt.figure.savefig('figures/binary_bert_{}.png'.format(dataset))
            df = pd.DataFrame(report).transpose()
            df.to_csv('./results/binary_bert_{}.csv'.format(dataset))
            
