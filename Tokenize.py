from my_utils import *
from my_network import *
import sys
import torch
import json
import os

class Params():
    ngram = 4
    features_len = 100
    hidden_size = 128
    bidirectional = True
    num_layers = 2
    dropout = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 10
    batch_size = 128
    embedding_size = 250

def _tokenize(wiki_path):
    params = Params()
    
    # Load all data and create 3 dataset objects, each with its gold, except for test
    train_dataset, train_gold, dev_dataset, dev_gold, test_dataset = MyUtils.load_train_dev_test(wiki_path)

    eprint("Building Train Dataset...")

    if (wiki_path.endswith("merged")):
        json_file_path = os.path.dirname(os.path.abspath(__file__))+"/input_vocab_merged_4ngram.json"
    else:
        json_file_path = os.path.dirname(os.path.abspath(__file__))+"/input_vocab_4ngram.json"

    if os.path.isfile(json_file_path):
        eprint("Loading input vocabulary from file...")
        input_vocab = json.load(open(json_file_path))
        train_dataset = CharAnnotationDataset(train_dataset, train_gold, ngram=params.ngram, input_vocab=input_vocab, vect_len=params.features_len)
    else:        
        train_dataset = CharAnnotationDataset(train_dataset, train_gold, ngram=params.ngram, vect_len=params.features_len)
        input_vocab = train_dataset.input_vocab
        json.dump(input_vocab, open(json_file_path, "w"))
    
    output_vocab = train_dataset.output_vocab
    eprint("Length of input vocab: {}".format(len(input_vocab)))

    eprint("Building Dev Dataset...")
    dev_dataset = CharAnnotationDataset(dev_dataset, dev_gold, ngram=params.ngram, input_vocab=input_vocab, output_vocab=output_vocab, vect_len=params.features_len)

    train_dataset = DataLoader(train_dataset, batch_size=params.batch_size)
    dev_dataset = DataLoader(dev_dataset, batch_size=params.batch_size)

    # Update parameters
    params.vocab_size = len(input_vocab)
    params.out_size = len(output_vocab)

    # Create the model
    char_annotate = CharAnnotationModel(params).to(params.device)
    eprint("\nModel Parameters:")
    eprint(char_annotate)

    # Call the trainer
    trainer = Trainer(
        model = char_annotate,
        loss_func = nn.CrossEntropyLoss(ignore_index=output_vocab["<PAD>"]),
        opt = optim.Adam(char_annotate.parameters()),
    )
    avg_train_loss, avg_train_acc = trainer.train(train_dataset, dev_dataset, params.epochs)

    # Run predictions
    eprint("Running predictions...")
    trainer.predict_to_std(wiki_path+".sentences.test", input_vocab, output_vocab, params.ngram, params.device)
    eprint("Done!")
    return


if __name__ == "__main__":
    _tokenize(sys.argv[1])
