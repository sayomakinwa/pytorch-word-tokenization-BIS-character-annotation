# Pytorch Word Tokenization BIS Character Annotation

The task is the tokenization of English sentences from Wikipedia, framed as a character annotation task using the BIS format, that is, tagging each character as B (beginning of the token), I (intermediate or end position of the token), S for space. For instance, given the sentence:

```
The pen is on the table.
```

The system will have to provide the following output:

```
BIISBIISBISBISBIISBIIIIB
```

which can be read as: <3-char token> <space> <3-char token> <space> <2-char token> <space> <2-char token> <space> <3-char token> <space> <5-char token> <1-char token>.

The task comes in two flavors:
1. easy (`en.wiki` files): standard text
2. hard (`en.wiki.merged` files): all spaces are removed, therefore making separating tokens harder

For training and development sets, both input (`.sentences`) and output (`.gold`) are provided (`dataset` folder).

## Requirements

1. [pytorch](https://pytorch.org/)
2. [sklearn](https://scikit-learn.org/stable/install.html)

## How to execute

run `python Tokenize.py <path-to-file en.wiki|en.wiki.merged>`

The code implements Bi-LSTM architecture for tokenization, trains and prints on the standard output the BIS output classification computed on the test file (analogously to the `.gold` files). 

To write this output to file, the code above can be updated as follows: 

run `python Tokenize.py <path to file en.wiki|en.wiki.merged> > <path-to-save-output>`
