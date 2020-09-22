from sklearn.metrics import f1_score
class MyUtils():
    @staticmethod
    def load_train_dev_test(wiki_path):
        train_input_path = wiki_path+".sentences.train"
        train_gold_path = wiki_path+".gold.train"
        dev_input_path = wiki_path+".sentences.dev"
        dev_gold_path = wiki_path+".gold.dev"
        test_input_path = wiki_path+".sentences.test"

        #print("Loading data from {}".format(train_input_path))
        train_dataset = MyUtils.single_file_loader(train_input_path)

        #print("Loading data from {}".format(train_gold_path))
        train_gold = MyUtils.single_file_loader(train_gold_path)

        #print("Loading data from {}".format(dev_input_path))
        dev_dataset = MyUtils.single_file_loader(dev_input_path)

        #print("Loading data from {}".format(dev_gold_path))
        dev_gold = MyUtils.single_file_loader(dev_gold_path)

        #print("Loading data from {}".format(test_input_path))
        test_dataset = MyUtils.single_file_loader(test_input_path)

        return train_dataset, train_gold, dev_dataset, dev_gold, test_dataset

    @staticmethod
    def single_file_loader(input_path):
        dataset = []
        with open(input_path, "r", encoding="utf-8") as data_file:
            for line in data_file:
                dataset.append(line.replace("\n",""))
        return dataset
    
    @staticmethod
    def evaluate_pred_gold(pred_path, gold_path, output_vocab):
        pred_file = MyUtils.single_file_loader(pred_path)
        gold_file = MyUtils.single_file_loader(gold_path)
        micro_score = 0.0
        macro_score = 0.0
        weighted_score = 0.0
        pred_length = len(pred_file)
        for i in range(pred_length):
            pred_line = [output_vocab[x] for x in pred_file[i]]
            gold_line = [output_vocab[x] for x in gold_file[i]]
            micro_score += f1_score(gold_line, pred_line, average="micro")
            macro_score += f1_score(gold_line, pred_line, average="macro")
            weighted_score += f1_score(gold_line, pred_line, average="weighted")
            
        return micro_score/pred_length, macro_score/pred_length, weighted_score/pred_length