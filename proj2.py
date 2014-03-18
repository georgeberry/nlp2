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
from math import log, factorial, e
import re
from functools import wraps
import time
from copy import copy
import csv
from collections import Counter
from stopwords import *

#constants
TRAINING_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/training_data.data'

VALIDATION_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/validation_data.data'

VALIDATION_PATH2 = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/validation_data2.data'

KAGGLE_OUTPUT_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/kaggle_output.csv'

KAGGLE_INPUT_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/test_data.data'

THRESHOLD = 5
RUNS = 10

#STOPWORDS = ['the', 'The', 'a', 'A', 'and', 'And', 'is', 'Is', 'are', 'Are', 'at', 'At', 'which', 'Which', 'on', 'On', 'this', 'This', 'that', 'That', 'as', 'As']

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

def combination(n,k):
    numerator=factorial(n)
    denominator=(factorial(k)*factorial(n-k))
    answer=numerator/denominator
    return answer

def odds(log_prob_dict):
    l = {}
    if len(log_prob_dict) > 1:
        for sense, val in log_prob_dict.items():
            l[sense] = val - log(sum([2**v for k, v in log_prob_dict.items() if k is not sense]), 2)
        m = max(l.items(), key = lambda x: x[1])
        #print(l, log_prob_dict)
        return m[1], m[0]
    else:
        return list(log_prob_dict.values())[0], list(log_prob_dict.keys())[0]
        

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
            f.write(str(num) + ',' + str(line) + ',' + '\n')
            num += 1


def make_features(lemma_and_pos, sense, context, word_feature_dict, location_feature_dict, wordform_feature_dict, capital_feature_dict, nums_feature_dict, pos_feature_dict, window):
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

    for word_num in range(len(context)):
        if re.match(r'[0-9]+[\!\?\.\\/]+[0-9]*' ,context[word_num]):
            context[word_num] = re.sub(r'[0-9]+[\!\?\.\\/]+[0-9]*', '000' , context[word_num])
        if re.match(r'[0-9]+', context[word_num]):
            context[word_num] = re.sub(r'[0-9]+', '<num>', context[word_num])

    #remove stopwords
    #context = [x for x in context if x not in STOPWORDS]

    '''#co-occurrence
    for word in context:
        if word not in word_feature_dict:
            word_feature_dict[word] = 0
        word_feature_dict[word] += 1'''

    #co-location
    for index in range(window):
        #expands on both sides of word to disambiguate
        #so we do w+1 and w-1, w+2 and w-2, etc.
        try: 
            prev_w = before[-1 - index] + '_w{}'.format(-1 - index)
            if prev_w not in location_feature_dict:
                location_feature_dict[prev_w] = 0
            location_feature_dict[prev_w] += 1
        except IndexError:
            #index error handles cases where target word occurs near beginnign or end of sentence
            continue

        try:
            post_w = after[index] + '_w+{}'.format(index + 1)
            if post_w not in location_feature_dict:
                location_feature_dict[post_w] = 0
            location_feature_dict[post_w] += 1
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

    #numbers
    nums = 0
    for word in context:
        if word in ['&', '$', '%', '-', '--']:
            nums += 1
        #try:
        #    float(word)
        #    nums += 1
        #except:
        #    pass
    if nums not in nums_feature_dict:
        nums_feature_dict[nums] = 0
    nums_feature_dict[nums] += 1

    #pos
    pos = lemma_and_pos.split('.')[1]
    if pos not in pos_feature_dict:
        pos_feature_dict[pos] = 0
    pos_feature_dict[pos] += 1


    #bag of words
    context2 = []

    for word_num in range(len(context)):
        if context[word_num] in STOPWORDS[lemma_and_pos.split('.')[0]]:
            continue
        else:
            context2.append(context[word_num])

    c = Counter(context2)

    #print(c)

    for k, v in c.items():
        combined = str(k) + '_' + str(v)
        if combined not in word_feature_dict:
            word_feature_dict[combined] = 0
        word_feature_dict[combined] += 1

    return word_feature_dict, location_feature_dict, wordform_feature_dict, capital_feature_dict, nums_feature_dict, pos_feature_dict


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
    def __call__(self, test_list):
        '''
        this should do classification for the input word and context
        '''

        results = OrderedDict()
        
        for test_example_num in range(len(test_list)):
            results[test_example_num] = None

        #run it n-1 times
        for run in range(RUNS - 1):
            for test_example_num in range(len(test_list)):
                if results[test_example_num] == None:   
                    lemma = test_list[test_example_num][0].split('.')[0]
                    confidence, sense = self.classifiers[lemma].classify(test_list[test_example_num])

                    if confidence > THRESHOLD:
                        results[test_example_num] = sense

        #then run and classify no matter what
        for test_example_num in range(len(test_list)):
            if results[test_example_num] == None:   
                lemma = test_list[test_example_num][0].split('.')[0]
                confidence, sense = self.classifiers[lemma].classify(test_list[test_example_num])
                results[test_example_num] = sense





        '''results = []

        for test_example in test_list:
            lemma = test_example[0].split('.')[0]

            results.append(self.classifiers[lemma].classify(test_example))'''

        return list(map(int, list(results.values())))


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
        self.max_length = 0
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

        if len(context) > self.max_length:
            self.max_length = len(context)

    def classify(self, test_example):
        '''
        given a test example:
            get features of test example
            add all features to all senses
            smooth features for each sense (these are computed for every test example)
        '''

        #features of test example
        a, b, c = test_example

        test_word_f, test_location_f, test_wordform_f, test_capital_f, test_nums_f, test_pos_f = make_features(a, b, c, {}, {}, {}, {}, {}, {}, self.window)

        #add 1 smoothing
        #get all features including the training example
        word_f = set()
        for sense in self.senses.values():
            word_f = word_f | set(sense.word_features.keys())
        word_f = word_f | set(test_word_f.keys())

        location_f = set()
        for sense in self.senses.values():
            location_f = location_f | set(sense.location_features.keys())
        location_f = location_f | set(test_location_f.keys())

        wordform_f = set()
        for sense in self.senses.values():
            wordform_f = wordform_f | set(sense.wordform_features.keys())
        wordform_f = wordform_f | set(test_wordform_f.keys())

        capital_f = set()
        for sense in self.senses.values():
            capital_f = capital_f | set(sense.capital_features.keys())
        capital_f = capital_f | set(test_capital_f.keys())

        nums_f = set()
        for sense in self.senses.values():
            nums_f = nums_f | set(sense.nums_features.keys())
        nums_f = nums_f | set(test_nums_f.keys())

        pos_f = set()
        for sense in self.senses.values():
            pos_f = pos_f | set(sense.pos_features.keys())
        pos_f = pos_f | set(test_pos_f.keys())     

        #log prob of test example for all senses
        log_probs = {}

        #update vocab for word, just for this 
        #this way we have smoothed counts 
        #compute conditional feature probabilities

        for sense in self.senses.values():

            #smooth
            sense.smooth(list(word_f), len(self.vocab), list(location_f), list(wordform_f), list(capital_f), list(nums_f), list(pos_f))

        for sense in self.senses.values():

            log_probs[sense.num] = 0

            #word features
            for feature in test_word_f:
                log_probs[sense.num] += log((sense.smoothed_word_features[feature]/(sense.smoothed_word_count)), 2)

            #location features
            for feature in test_location_f:
                log_probs[sense.num] += log(sense.smoothed_location_features[feature]/(sense.smoothed_location_count), 2)

            #wordform features
            for feature in test_wordform_f:
                log_probs[sense.num] += log(sense.smoothed_wordform_features[feature]/sense.smoothed_wordform_count, 2)

            #capital features
            for feature in test_capital_f:
                log_probs[sense.num] += log(sense.smoothed_capital_features[feature]/sense.smoothed_capital_count, 2)

            #numeric features
            for feature in test_nums_f:
                log_probs[sense.num] += log(sense.smoothed_nums_features[feature]/sense.smoothed_nums_count, 2)

            #part of speech features
            #for feature in test_pos_f:
            #    log_probs[sense.num] += log(sense.smoothed_pos_features[feature]/sense.smoothed_pos_count, 2)

        #priors
        #for sense in self.senses.values():
        #    log_probs[sense.num] += log(sense.count/self.get_tally(), 2)

        max_odds, probable_sense = odds(log_probs)

        #print(max_odds, probable_sense)

        if max_odds > THRESHOLD:
            self.add_to(a, probable_sense, c)

        return max_odds, probable_sense

    def get_tally(self):
        if self.tally == None:
            self.tally = 0
            for sense in self.senses.values():
                self.tally += sense.count
        return self.tally

    def gowords_helper(self):
        pass

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
        self.smoothed_location_count = 0
        self.smoothed_wordform_count = 0
        self.smoothed_capital_count = 0
        self.smoothed_nums_count = 0
        self.smoothed_pos_count = 0

        self.max_length = 0

        self.word_features = {} #features we can do normal laplacian smoothing
        self.smoothed_word_features = {}

        self.location_features = {}
        self.smoothed_location_features = {}

        self.wordform_features = {}
        self.smoothed_wordform_features = {}

        self.capital_features = {}
        self.smoothed_capital_features = {}

        self.nums_features = {}
        self.smoothed_nums_features = {}

        self.pos_features = {}
        self.smoothed_pos_features = {}

        self.window = window
        self.num = number

    def add_example(self, lemma_and_pos, sense, context):
        self.count += 1
        self.length += len(context)

        self.word_features, self.location_features, self.wordform_features, self.capital_features, self.nums_features, self.pos_features = make_features(lemma_and_pos, sense, context, self.word_features, self.location_features, self.wordform_features, self.capital_features, self.nums_features, self.pos_features, self.window)

        if len(context) > self.max_length:
            self.max_length = len(context)


    def smooth(self, word_feature_list, vocab_length, location_feature_list, wordform_feature_list, capital_feature_list, nums_feature_list, pos_feature_list):
        #this is called upon seeing a test example
        #the idea is that we add any unseen test example features to the feature bag,
        #   then smooth

        #Word class looks at all senses and returns a list of all features
        #this function adds 1 to all features, including ones we haven't seen for this sense
        #this avoids the multiplication-by-zero problem

        self.smoothed_word_count = vocab_length + self.count
        self.smoothed_location_count = len(self.location_features) + len(location_feature_list)
        self.smoothed_wordform_count = len(self.wordform_features) + len(wordform_feature_list)
        self.smoothed_capital_count = len(self.capital_features) + len(capital_feature_list)
        self.smoothed_nums_count = len(self.nums_features) + len(nums_feature_list)
        self.smoothed_pos_count = len(self.pos_features) + len(pos_feature_list)

        temp_word_f = copy(self.word_features)
        temp_location_f = copy(self.location_features)
        temp_wordform_f = copy(self.wordform_features)
        temp_capital_f = copy(self.capital_features)
        temp_nums_f = copy(self.nums_features)
        temp_pos_f = copy(self.pos_features)


        lookup_table = {}

        for feature in word_feature_list:
            if feature not in temp_word_f:
                temp_word_f[feature] = 0

            num_obs = int(feature.split('_')[1])
            if num_obs not in lookup_table:
                #this is a binomially distributed prior
                lookup_table[num_obs] = (vocab_length - self.count) * combination(self.max_length, num_obs) * ((1/vocab_length)**num_obs * ((1 - (1/vocab_length))**(self.max_length - num_obs)) )

            temp_word_f[feature] += lookup_table[num_obs]

        for feature in location_feature_list:
            if feature not in temp_location_f:
                temp_location_f[feature] = 0
            temp_location_f[feature] += 1

        for feature in wordform_feature_list:
            if feature not in temp_wordform_f:
                temp_wordform_f[feature] = 0
            temp_wordform_f[feature] += 1

        for feature in capital_feature_list:
            if feature not in temp_capital_f:
                temp_capital_f[feature] = 0
            temp_capital_f[feature] += 1

        for feature in nums_feature_list:
            if feature not in temp_nums_f:
                temp_nums_f[feature] = 0
            temp_nums_f[feature] += 1

        for feature in pos_feature_list:
            if feature not in temp_pos_f:
                temp_pos_f[feature] = 0
            temp_pos_f[feature] += 1

        self.smoothed_word_features = temp_word_f
        self.smoothed_location_features = temp_location_f
        self.smoothed_wordform_features = temp_wordform_f
        self.smoothed_capital_features = temp_capital_f
        self.smoothed_nums_features = temp_nums_f
        self.smoothed_pos_features = temp_pos_f

    def __repr__(self):
        return 'Sense {} instance'.format(self.num)


#####
STOPWORDS = mutual_information(TRAINING_PATH, .2)

a = split_up(TRAINING_PATH)
b = split_up(VALIDATION_PATH)
c = Classifier(a)

p = c(b)
q = get_valid_senses(b)
correct(p,q)

kaggle_output(KAGGLE_OUTPUT_PATH, p)

