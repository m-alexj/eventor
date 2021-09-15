# eventor
Research-grade code for the thesis "Global Inference and Local Syntax Representations for Event Extraction"

This repository contains code for Chapter 5, where I built syntax-based event extractors, i.e., extractors which operate on the entire dependency graph of a sentence. The system first identifies triggers based on this information, and then predicts which of the (given) entity mentions play which roles in the respective event.

Please not that this code is (a) very old, (b) not maintained in any form. However, it specifies how the experiments were carried out. Please also note that I cannot add the training, development or test data used because it is directly derived from ACE 2005.

The requirements.txt _should_ contain all requirements needed to run the software. Please note that Tensorflow 1 was used for development, as well as Python 3.4.

Please see `python eventor.py --help` for all parameters the software supports. Please note that some default values for the parameters point to a private infrastructure and were not adapted to a general form (e.g. to placeholders) for this release. `eventor.py` provides the basic functionality of the dependency graph-based event extractor.

Please see `python crosstrainer --help` for all parameters the bagging-based version of the system supports. Despite its name, the crosstrainer performs bootstrap aggregating training using the dependency graph-based event extractor, and not cross validation. The name has historical reasons, the script started as a cross validation script and was changed to bagging afterwards.

There are various visualization components included in this release, e.g. `python decoder.py`, `python visualizer.py`, and `python modelCompare.py`. They should all provide the `--help` switch and print an overview over all parameters available.

The code requires training, development, and test data in a very specific format, which should be deducible from the code. The data is derived from ACE 2005 and cannot be included in this release. See `bio.load.load(...)` and `model.graph.SentenceGraph(...)` for a pointer. Each training sample has the structure `# #1 document id, #2 sentence id, #3 ACE genre, #4 words, #5 triggers, #6 arguments, #7 mentions`. Words contain multiple information concatenated via a single bar |: `word, lemma, pos, relations, bilou_entity_tag, features = word.split("|")`. 'Features' refers to a list of categorical features associated with the word, concatenated via '/+/'. In the PhD thesis, the features were the same features produced by the event extractor reported in Li et al. (2013). Please note that we obtained the system by directly asking the last author of the paper (Heng Ji).  Relations encode enhanced++ dependency relations to other words, concatenated by an equal sign =. Relations have the following formn: neighbor_id, relation, concatenated by a plus sign +.

The experiments in the thesis produced five models for each combination of syntax encoder (no syntax, Graph Convolutions, tree-shaped LSTMs), repeated negative undersampling probability (0.0, 0.5, 0.9) and data split (standard split and two new train/dev/test splits which follow the genre distribution in ACE 2005, in contrast to the standard split which uses only newswire articles in the test set). In total, the experiments produced 500 GB of models.

Part of the release is the five models (bagging+majority voting for the five predictions) produced for Graph Convolutions (GCN) and negative undersampling probability 0.9 trained on the standard ACE split. However, due to size limitations, I cannot provide the models here on Github. They will be stored somewhere else, most probably on heiDATA, the data archiving system of Heidelberg University.
