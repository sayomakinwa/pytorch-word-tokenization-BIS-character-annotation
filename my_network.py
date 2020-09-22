from __future__ import print_function
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from my_utils import *


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

class CharAnnotationDataset(Dataset):
    def __init__(self, 
                 sentences, 
                 gold_labels=None, 
                 input_vocab=None, 
                 output_vocab=None, 
                 ngram=1, 
                 vect_len=100,
                 device="cuda"):
        
        self.sentences_len = len(sentences)
        self.ngram = ngram
        self.vect_len = vect_len
        self.device = device

        if input_vocab is None:
            eprint("Building the input vocab of ngram of {}... It may take some time WRT to n...".format(self.ngram))
            self.input_vocab = self._build_input_vocab(sentences)
            eprint("Done building input vocab. Length is. Now building input features...")
        else:
            self.input_vocab = input_vocab
        
        if gold_labels is not None and output_vocab is None:
            self.output_vocab = self._build_output_vocab(gold_labels[0])
        else:
            self.output_vocab = output_vocab

        self.data = []
        if gold_labels is not None:
            for sentence, gold_label in zip(sentences, gold_labels):
                encoded_inputs = torch.tensor(self._get_input_vector(sentence)).to(self.device)
                encoded_outputs = torch.tensor(self._get_output_vector(gold_label)).to(self.device)
                self.data.append({"input": encoded_inputs, 
                                  "output": encoded_outputs})
        else:
            for sentence in sentences:
                item = {"input": torch.tensor(self._get_input_vector(sentence)).to(self.device)}
                self.data.append(item)

    def __len__(self):
        return self.sentences_len

    def __getitem__(self, idx):
        return self.data[idx]
    
    def _build_input_vocab(self, sentence_list):
        vocab_list = ["<PAD>"]
        vocab_list.append("<UNK>")
        for sentence in sentence_list:
            for k in range(len(sentence)):
                if sentence[k : k + self.ngram] not in vocab_list:
                    vocab_list.append(sentence[k : k + self.ngram])
        return {v:k for k,v in enumerate(vocab_list)}

    def _get_input_vector(self, sentence):
        vector = []
        for k in range(len(sentence)):
            if sentence[k : k + self.ngram] in self.input_vocab:
                vector.append(self.input_vocab[sentence[k : k + self.ngram]])
            else:
                vector.append(self.input_vocab["<UNK>"])
            
            if len(vector) >= self.vect_len:
                break

        if self.vect_len > len(vector):
            vector += [self.input_vocab["<PAD>"] for _ in range(self.vect_len - len(vector))]

        return vector

    @staticmethod
    def full_sent_to_vector(sentence, input_vocab, ngram):
        vector = []
        for k in range(len(sentence)):
            if sentence[k : k + ngram] in input_vocab:
                vector.append(input_vocab[sentence[k : k + ngram]])
            else:
                vector.append(input_vocab["<UNK>"])

        return vector

    def _build_output_vocab(self, gold_label):
        gold_label = list(set(gold_label))
        vocab = {v:k+1 for k,v in enumerate(gold_label)}
        vocab["<PAD>"] = 0
        return vocab

    def _get_output_vector(self, gold_label):
        vector = []
        for k in range(len(gold_label)):
            vector.append(self.output_vocab[gold_label[k]])
            if len(vector) >= self.vect_len:
                break

        if self.vect_len > len(vector):
            vector += [self.output_vocab["<PAD>"] for _ in range(self.vect_len - len(vector))]

        return vector


class CharAnnotationModel(nn.Module):
    def __init__(self, params):
        super(CharAnnotationModel, self).__init__()
        self.embeddings = nn.Embedding(params.vocab_size, params.embedding_size)
        self.lstm = nn.LSTM(params.embedding_size, params.hidden_size, 
                             bidirectional=params.bidirectional,
                             num_layers=params.num_layers, 
                             dropout = params.dropout if params.num_layers > 1 else 0)
        lstm_out_size = params.hidden_size if params.bidirectional is False else params.hidden_size * 2
        
        self.dropout = nn.Dropout(params.dropout)
        self.output = nn.Linear(lstm_out_size, params.out_size)

    def forward(self, x):
        x = self.embeddings(x)
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        output = self.output(x)
        return output


class Trainer():
    def __init__(self, model, loss_func, opt):
        self.model = model
        self.loss_func = loss_func
        self.opt = opt
    
    def train(self, train_data, val_data, epochs=2):
        eprint("Training...")
        train_loss = 0.0
        train_acc = 0.0
        
        for epoch in range(epochs):
            #eprint("Epoch {:02d}/{:02d}".format(epoch + 1, epochs))

            epoch_loss = 0.0
            epoch_acc = 0.0
            self.model.train() # Start training again, after having deactivated training inside eval function

            for step, sample in enumerate(train_data):
                input = sample["input"]
                output = sample["output"]

                self.opt.zero_grad() #zero grad

                pred = self.model(input) #forward

                epoch_acc +=  output.eq(torch.argmax(pred, -1).long()).sum().tolist()/output.numel()
                
                pred = pred.permute(0, 2, 1)
                sample_loss = self.loss_func(pred, output)
                sample_loss.backward() #backward

                self.opt.step() #optimize

                epoch_loss += sample_loss.tolist()
                
                if step % 50 == 0:
                    #eprint("    [Epoch: {}; step {}]; current avg loss = {:0.4f}".format(epoch+1, step+1, epoch_loss / (step + 1)))
                    1
            
            epoch_avg_loss = epoch_loss / len(train_data)
            epoch_avg_acc = epoch_acc / len(train_data)
            train_loss += epoch_avg_loss
            train_acc += epoch_avg_acc
            #eprint("  train loss = {:0.4f}; acc = {:0.4f}".format(epoch_avg_loss, epoch_avg_acc))

            val_loss, val_acc = self.evaluate(val_data)
            #eprint("  val loss = {:0.4f}; val acc = {:0.4f}".format(val_loss, val_acc))

            eprint("Epoch {:02d}/{:02d}; train_loss:{:0.4f}, val_loss:{:0.4f}".format(epoch + 1, epochs, epoch_avg_loss, val_loss))
        
        eprint("Training complete!")
        epoch_avg_loss = train_loss / epochs
        epoch_avg_acc = train_acc / epochs
        return epoch_avg_loss, epoch_avg_acc
    
    def evaluate(self, val_data):
        val_loss = 0.0
        val_acc = 0.0

        self.model.eval()
        with torch.no_grad():
            for sample in val_data:
                input = sample["input"]
                output = sample["output"]

                pred = self.model(input)

                #val_acc +=  output.eq(torch.argmax(pred, -1).long()).sum().cpu().numpy()/output.numel()
                val_acc +=  output.eq(torch.argmax(pred, -1).long()).sum().tolist()/output.numel()

                pred = pred.permute(0, 2, 1)
                sample_loss = self.loss_func(pred, output)
                val_loss += sample_loss.tolist()
        
        return val_loss / len(val_data), val_acc / len(val_data)

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            for sample in data:
                input = sample["input"]
                logits = self.model(input)
                pred = torch.argmax(logits, -1)
                return logits, pred
    
    def predict_to_file(self, data_full_path, output_full_path, input_vocab, output_vocab, ngram, device="cuda"):
        reverse_output_vocab = {v:k for k,v in output_vocab.items()}
        data = MyUtils.single_file_loader(data_full_path)
        self.model.eval()
        with torch.no_grad():
            with open(output_full_path, "w") as out_file:
                for sample in data:
                    input = CharAnnotationDataset.full_sent_to_vector(sample, input_vocab, ngram)
                    input = torch.tensor(input).unsqueeze(0).to(device)
                    logits = self.model(input)
                    pred = torch.argmax(logits, -1)
                    pred_str = "".join([reverse_output_vocab[x] if x != 0 else "I" for x in pred[0].tolist()])
                    out_file.write(pred_str+"\n")

        return True
    
    def predict_to_std(self, data_full_path, input_vocab, output_vocab, ngram, device="cuda"):
        reverse_output_vocab = {v:k for k,v in output_vocab.items()}
        data = MyUtils.single_file_loader(data_full_path)
        self.model.eval()
        with torch.no_grad():
            for sample in data:
                input = CharAnnotationDataset.full_sent_to_vector(sample, input_vocab, ngram)
                input = torch.tensor(input).unsqueeze(0).to(device)
                logits = self.model(input)
                pred = torch.argmax(logits, -1)
                pred_str = "".join([reverse_output_vocab[x] if x != 0 else "I" for x in pred[0].tolist()])
                print(pred_str)

        return True
