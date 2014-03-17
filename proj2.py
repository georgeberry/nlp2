'''
~~USE PYTHON 3~~

real problem: priors for different features

we're doing the following type of smoothing:

n_c + pm
--------
 n + m

n_c is the number of times we've seen the feature given a class
n is the nubmer of times we've seen the class
p is the prior for seeing the feature across all classes
m is the number of increments we want

if p = 1/V and m = V, this reduces to laplacian smoothing

'''

#imports
from math import log
import re
from functools import wraps
import time
from copy import copy
import csv
from collections import Counter

#constants
TRAINING_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/training_data.data'

VALIDATION_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/validation_data.data'

VALIDATION_PATH2 = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/validate.data'

KAGGLE_OUTPUT_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/kaggle_output.csv'

KAGGLE_INPUT_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/test_data.data'

STOPWORDS = ['the', 'The', 'a', 'A', 'and', 'And', 'is', 'Is', 'are', 'Are', 'at', 'At', 'which', 'Which', 'on', 'On', 'this', 'This', 'that', 'That', 'as', 'As']

#global functions
def split_up(path):
    '''
    straightfoward: returns a list of lists of the 3 parts of the test data
    '''
    contexts = []

    with open(path) as f:
        for line in f.readlines():
            s = line.strip('\n').split('|')
            s = list(map(str.strip, s))
            #s[2] = s[2].split()

            contexts.append(s)

    return contexts

def get_stopwords(path):
    '''
        this is a pointwise mutual information approach to finding stopwords
        we'd like to eliminate words that occur frequently and don't tell us anything
        i.e. they don't disambiguate the senses well
        PMI tells us how much more often we see a feature and sense together than we'd expect if they were independent

    '''
    pass

def timer(f):
    '''
    timer decorator
    '''
    @wraps(f)
    def wrapper(*args,**kwargs):
        tic = time.time()
        result = f(*args, **kwargs)
        print(f.__name__ + " took " + str(time.time() - tic) + " seconds")
        return result
    return wrapper


def get_valid_senses(valid_data):
    '''
    looks through validation data
    pulls out correct senses in order
    '''
    v = []
    for example in valid_data:
        v.append(example[1])
    return list(map(int, v))

def correct(output, valid):
    '''
    give your output and validation correct answers
    returns info about how many you got right
    '''
    if len(output) == len(valid):
        incorrect = 0

        for num in range(len(output)):
            if output[num] != valid[num]:
                incorrect += 1
                print('num {} actually {}, assigned {}'.format(num+1, valid[num], output[num]))

        print('{} share correct'.format(1 - incorrect/len(output)))
        print('that\'s {} out of {}'.format(len(output) - incorrect, len(output)))

    else:
        print('different number of classified instances')

def kaggle_output(filepath, output):
    '''
    outputs to format kaggle wants
    '''
    with open(filepath, 'w') as f:
        f.write('Id' + ',' + 'Prediction' + ',' + '\n')
        num = 1
        for line in output:
            f.write(num + ',' + str(line) + ',' + '\n')
            num += 1


def word_features(lemma_and_pos, sense, context, word_feature_dict, wordform_feature_dict, capital_feature_dict, window):
    '''
    word features happen here:
        these are defined by anything that can be smoothed by laplacian smoothing

    called in both Sense and Word classes, so it's kept as a global function

    takes a feature_dict, increments it, and returns it
    '''

    #split up the context
    before, target, after = re.split(r'\s*%%\s*', context)

    before = re.split(r' +', before)
    after = re.split(r' +', after)

    #assuming here that both before and after can't be zero length strings
    #in that case there'd be no context
    if len(before) == 1 and before[0] == '':
        before = []
        context = after
    elif len(after) == 1 and after[0] =='':
        after = []
        context = before
    else:
        context = before + after

    #remove stopwords
    context = list(filter(lambda x: x not in STOPWORDS, context))

    #co-occurrence
    for word in context:
        if word not in word_feature_dict:
            word_feature_dict[word] = 0
        word_feature_dict[word] += 1

    '''c = Counter(context)

    for k, v in c.items():
        combined = str(k) + '_' + str(v)
        if combined not in word_feature_dict:
            word_feature_dict[combined] = 0
        word_feature_dict[combined] += 1'''
    
    #co-location
    for index in range(window):
        #expands on both sides of word to disambiguate
        #so we do w+1 and w-1, w+2 and w-2, etc.
        try: 
            prev_w = before[-1 - index] + '_w{}'.format(-1 - index)
            if prev_w not in word_feature_dict:
                word_feature_dict[prev_w] = 0
            word_feature_dict[prev_w] += 1
        except IndexError:
            #index error handles cases where target word occurs near beginnign or end of sentence
            continue

        try:
            post_w = after[index] + '_w+{}'.format(index + 1)
            if post_w not in word_feature_dict:
                word_feature_dict[post_w] = 0
            word_feature_dict[post_w] += 1
        except IndexError:
            continue

    #wordforms
    if target not in wordform_feature_dict:
        wordform_feature_dict[target] = 0
    wordform_feature_dict[target] += 1

    #capitals
    capitals = sum([s.isupper() for i in context for s in i])
    if capitals not in capital_feature_dict:
        capital_feature_dict[capitals] = 0
    capital_feature_dict[capitals] += 1

    return word_feature_dict, wordform_feature_dict, capital_feature_dict


def return_vocab(context):
    before, target, after = re.split(r'\s*%%\s*', context)

    before = re.split(r' +', before)
    after = re.split(r' +', after)

    #assuming here that both before and after can't be zero length strings
    #in that case there'd be no context
    if len(before) == 1 and before[0] == '':
        before = []
        context = after
    elif len(after) == 1 and after[0] =='':
        after = []
        context = before
    else:
        context = before + after

    return set(context)


def dict_max_key(d):
    '''
    given a dict, returns the KEY corresponding to the max VALUE val
    '''
    m = max(d.values())
    k = [k for k,v in d.items() if v == m]

    return k[0]


#classes
class Classifier:
    @timer
    def __init__(self, examples_as_list, window = 4):
        '''
        creates all classifiers from training examples in one shot
        '''

        self.classifiers = {}

        for example_as_list in examples_as_list:
            current_lemma = example_as_list[0].split('.')[0]

            if current_lemma not in self.classifiers:
                self.classifiers[current_lemma] = Word(window)

            self.classifiers[current_lemma].add_to(*example_as_list)

    @timer
    def __call__(self, example_list):
        '''
        this should do classification for the input word and context
        '''
        results = []

        for example in example_list:

            lemma = example[0].split('.')[0]

            results.append(self.classifiers[lemma].classify(example))

        return list(map(int, results))

    def __getitem__(self, key):
        #call this shit like a dictionary!
        return self.classifiers[key]

    def __repr__(self):
        l = len(self.classifiers.keys())
        return 'Classifier instance containing {} words'.format(l)


class Word:
    '''
    holds classifier for one word
    '''
    def __init__(self, window):
        self.senses = {}
        self.window = window
        self.lemma = None #lemmatized form of word
        self.tally = None #unsmoothed number of times we see the word in training
        self.vocab = set()

    def add_to(self, lemma_and_pos, sense, context):
        '''
        add a training example for this word
        '''
        
        #first call sets the lemma for the class
        if self.lemma == None:
            self.lemma = lemma_and_pos.split('.')[0]

        #raise error if you try to feed it the wrong word
        assert lemma_and_pos.split('.')[0] == self.lemma, 'wrong lemma for this class'

        self.vocab = self.vocab | return_vocab(context)

        if sense not in self.senses:
            self.senses[sense] = Sense(self.window, sense)
        
        self.senses[sense].add_example(lemma_and_pos, sense, context)

    def classify(self, test_example):
        '''
        given a test example:
            get features of test example
            add all features to all senses
            smooth features for each sense (these are computed for every test example)
        '''

        #features of test example
        a, b, c = test_example


        test_word_f, test_wordform_f, test_capital_f = word_features(a, b, c, {}, {}, {}, self.window)

        #add 1 smoothing
        #get all features including the training example
        word_f = set()
        for sense in self.senses.values():
            word_f = word_f | set(sense.word_features.keys())
        word_f = word_f | set(test_word_f.keys())

        wordform_f = set()
        for sense in self.senses.values():
            wordform_f = wordform_f | set(sense.wordform_features.keys())
        wordform_f = wordform_f | set(test_wordform_f.keys())

        capital_f = set()
        for sense in self.senses.values():
            capital_f = capital_f | set(sense.capital_features.keys())
        capital_f = capital_f | set(test_capital_f.keys())

        #log prob of test example for all senses
        log_probs = {}

        #update vocab for word, just for this 
        #this way we have smoothed counts 
        #compute conditional feature probabilities

        for sense in self.senses.values():

            #smooth
            sense.smooth(list(word_f), len(self.vocab), list(wordform_f), list(capital_f))

            log_probs[sense.num] = 0

            #word features
            for feature in test_word_f:
                #try:
                #    print(self.lemma, sense.num, feature, sense.word_features[feature]/sense.count)
                #except:
                #    pass

                log_probs[sense.num] += log(sense.smoothed_word_features[feature]/(sense.smoothed_word_count + sense.count), 2)

            #wordform features
            for feature in test_wordform_f:
                log_probs[sense.num] += log(sense.smoothed_wordform_features[feature]/sense.smoothed_wordform_count, 2)

            #capital features
            for feature in test_capital_f:
                log_probs[sense.num] += log(sense.smoothed_capital_features[feature]/sense.smoothed_capital_count, 2)

        #priors
        for sense in self.senses.values():
            log_probs[sense.num] += log(sense.count/self.get_tally(), 2)

        return dict_max_key(log_probs)

    def get_tally(self):
        if self.tally == None:
            self.tally = 0
            for sense in self.senses.values():
                self.tally += sense.count
        return self.tally

    def __getitem__(self, key):
        #call this shit like a dictionary!
        return self.senses[key]

    def __repr__(self):
        return 'word class instance for "{}"'.format(self.lemma)


class Sense:
    '''
    holds information on one sense of a word

    feature counting and whatnot is done here in the add_example function
        just need to edit one place to add more features!

    todo:
    get rid of stopwords?
    do some tf-idf?
    just count the number of numeric words rather than individual word?

    '''
    def __init__(self, window, number):
        self.count = 0
        self.length = 0
        self.smoothed_word_count = 0 #this is what we eventually normalize by
        self.smoothed_wordform_count = 0
        self.smoothed_capital_count = 0

        self.word_features = {} #features we can do normal laplacian smoothing
        self.smoothed_word_features = {}

        self.wordform_features = {} #features that have different priors than 1/V
        self.smoothed_wordform_features = {}

        self.capital_features = {}
        self.smoothed_capital_features = {}

        self.window = window
        self.num = number

    def add_example(self, lemma_and_pos, sense, context):
        self.count += 1
        self.length += len(context)

        self.word_features, self.wordform_features, self.capital_features = word_features(lemma_and_pos, sense, context, self.word_features, self.wordform_features, self.capital_features, self.window)


    def smooth(self, word_feature_list, vocab_length, wordform_feature_list, capital_feature_list):
        #this is called upon seeing a test example
        #the idea is that we add any unseen test example features to the feature bag,
        #   then smooth

        #Word class looks at all senses and returns a list of all features
        #this function adds 1 to all features, including ones we haven't seen for this sense
        #this avoids the multiplication-by-zero problem

        self.smoothed_word_count = self.count + len(word_feature_list)
        self.smoothed_wordform_count = len(self.wordform_features) + len(wordform_feature_list)
        self.smoothed_capital_count = len(self.capital_features) + len(capital_feature_list)

        temp_word_f = copy(self.word_features)
        temp_wordform_f = copy(self.wordform_features)
        temp_capital_f = copy(self.capital_features)

        for feature in word_feature_list:
            if feature not in temp_word_f:
                temp_word_f[feature] = 0
            temp_word_f[feature] += 1

        for feature in wordform_feature_list:
            if feature not in temp_wordform_f:
                temp_wordform_f[feature] = 0
            temp_wordform_f[feature] += 1

        for feature in capital_feature_list:
            if feature not in temp_capital_f:
                temp_capital_f[feature] = 0
            temp_capital_f[feature] += 1

        self.smoothed_word_features = temp_word_f
        self.smoothed_wordform_features = temp_wordform_f
        self.smoothed_capital_features = temp_capital_f

    def __repr__(self):
        return 'Sense {} instance'.format(self.num)


#####

a = split_up(TRAINING_PATH)
b = split_up(VALIDATION_PATH)
c = Classifier(a)

p = c(b)
q = get_valid_senses(b)
correct(p,q)
kaggle_output(KAGGLE_OUTPUT_PATH, p)

