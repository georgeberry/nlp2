from __future__ import division

'''
~~~
the biggest problem is:
    dealing with the potentially large number of features we want to specify

what if we see one feature many times for sense 1 but zero times for sense 2?
maybe we could key a dictionary by features and take the union, adding 1 to everything?

things we need to store:
    (f_i | s) for all f_i, for both (?) s
'''

'''
another biggest problem:
    we have 758 examples for sense 1 of president and 15 examples for sense 2!


'''


'''
ok so we're going really OO again
    sorry

we have a 'word' class that:
    1) stores information on word senses
    2) given a test example, returns a classification

then, inside the 'word' class we have a 'sense' class that:
    1) stores detailed information on one word sense
    2) turns raw context into features
    3) returns probabilities for specific elements of a test example


class_word:
    class_sense: sense1
    class_sense: sense2

'''

#imports
from math import log
import re
from functools import wraps
import time

#constants
TRAINING_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/training_data.data'

VALIDATION_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/validation_data.data'

KAGGLE_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/kaggle_output.txt'

#global functions
def split_up(path):
    contexts = []

    with open(path) as f:
        for line in f.readlines():
            s = line.strip('\n').split('|')
            s = map(str.strip, s)
            #s[2] = s[2].split()

            contexts.append(s)

    return contexts

def timer(f):
    @wraps(f)
    def wrapper(*args,**kwargs):
        tic = time.time()
        result = f(*args, **kwargs)
        print(f.__name__ + " took " + str(time.time() - tic) + " seconds")
        return result
    return wrapper

#to compare with Classifier output
def get_valid_senses(valid_data):
    v = []
    for example in valid_data:
        v.append(example[1])
    return map(int, v)


def correct(output, valid):

    if len(output) == len(valid):
        incorrect = 0

        for num in xrange(len(output)):
            if output[num] != valid[num]:
                incorrect += 1
                print 'num {} actually {}, assigned {}'.format(num+1, valid[num], output[num])

        print '{} share correct'.format(1 - incorrect/len(output))
        print 'that\'s {} out of {}'.format(len(output) - incorrect, len(output))

    else:
        print 'different number of classified instances'


def kaggle_output(filepath, output):
    with open(filepath, 'w') as f:
        for line in output:
            f.write(str(line) + '\n')


def make_features(lemma_and_pos, sense, context, feature_dict, window):
    '''
    this is where naive bayes features happen

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

    #co-occurrence
    for word in context:
        if word not in feature_dict:
            feature_dict[word] = 0
        feature_dict[word] += 1

    '''
    #co-location
    for index in range(window):
        #expands on both sides of word to disambiguate
        #so we do w+1 and w-1, w+2 and w-2, etc.
        try: 
            prev_w = before[-1 - index] + '_w{}'.format(-1 - index)
            if prev_w not in feature_dict:
                feature_dict[prev_w] = 0
            feature_dict[prev_w] += 1
        except IndexError:
            #index error handles cases where target word occurs near beginnign or end of sentence
            continue

        try:
            post_w = after[index] + '_w+{}'.format(index + 1)
            if post_w not in feature_dict:
                feature_dict[post_w] = 0
            feature_dict[post_w] += 1
        except IndexError:
            continue
    '''

    return feature_dict


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
    k = [k for k,v in d.iteritems() if v == m]

    return k[0]

#classes
class Classifier:
    @timer
    def __init__(self, examples_as_list, window = 2):
        '''
        creates all classifiers from training examples in one shot
        '''

        self.classifiers = {}

        for example_as_list in examples_as_list:
            current_lemma = example_as_list[0].split('.')[0]

            if current_lemma not in self.classifiers:
                self.classifiers[current_lemma] = Word(window)

            #print self.classifiers
            #print example_as_list

            self.classifiers[current_lemma].add_to(*example_as_list)

    @timer
    def __call__(self, example_list):
        '''
        this should do classification for the input word and context
        '''
        results = []

        for example in example_list:

            lemma = example[0].split('.')[0]

            results.append(self.classifiers[lemma].compare(example))

        return map(int, results)

    def __getitem__(self, key):
        #call this shit like a dictionary!
        return self.classifiers[key]

    def __repr__(self):
        l = len(self.classifiers.keys())
        return 'Classifier instance containing {} words'.format(l)


class Word:
    '''
    holds information on one word to disambiguate
    '''
    def __init__(self, window):
        self.senses = {}
        self.window = window
        self.lemma = None
        self.tally = None
        self.vocab = set()

    def add_to(self, lemma_and_pos, sense, context):
        
        #first call sets the lemma for the class
        if self.lemma == None:
            self.lemma = lemma_and_pos.split('.')[0]

        #raise error if you try to feed it the wrong word
        assert lemma_and_pos.split('.')[0] == self.lemma, 'wrong lemma for this class'

        self.vocab = self.vocab | return_vocab(context)

        if sense not in self.senses:
            self.senses[sense] = Sense(self.window, sense)
        
        self.senses[sense].add_example(lemma_and_pos, sense, context)


    def compare(self, test_example):

        #features of test example
        a, b, c = test_example

        test_f = make_features(a, b, c, {}, self.window)

        #add 1 smoothing
        f = set()
        for sense in self.senses.values():
            f = f | set(sense.features.keys())
            f = f | set(test_f.keys())

        #update vocab for word, just for this 

        for sense in self.senses.values():
            sense.smooth(list(f), len(self.vocab))

        #compute conditional feature probabilities
        sense_log_prob = self.feature_prob(test_f)

        #priors
        for sense in self.senses.values():
            sense_log_prob[sense.num] += log(sense.count/self.get_tally(), 2)

        return dict_max_key(sense_log_prob)


    def feature_prob(self, test_f):
        log_probs = {}

        for sense in self.senses.values():
            log_probs[sense.num] = 0
            for feature in test_f:
                #for now, just ignore features we haven't seen before
                if feature in sense.features:
                    log_probs[sense.num] += log(sense.features[feature]/sense.smoothed_count, 2)

        return log_probs


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
        self.smoothed_count = 0 #this is what we eventually normalize by
        self.features = {}
        self.smoothed_features = {}
        self.window = window
        self.num = number

    def add_example(self, lemma_and_pos, sense, context):
        self.count += 1

        self.features = make_features(lemma_and_pos, sense, context, self.features, self.window)

    def smooth(self, complete_feature_list, vocab_length):
        #Word class looks at all senses and returns a list of all features
        #this function adds 1 to all features, including ones we haven't seen for this sense
        #this avoids the multiplication-by-zero problem

        #assume we add one feature vector with all 1's
        #to prevent the possiblity of 101/100 or something bizarre
        self.smoothed_count = self.count + len(complete_feature_list)

        for feature in complete_feature_list:
            if feature not in self.features:
                self.features[feature] = 0
            self.features[feature] += 1

    #def feature_prob(self, test_features):

    def __getitem__(self, key):
        #call this shit like a dictionary!
        return self.features[key]

    def __repr__(self):
        return str(self.features)


#####

a = split_up(TRAINING_PATH)
b = split_up(VALIDATION_PATH)
c = Classifier(a)

p = c(b)
q = get_valid_senses(b)
correct(p,q)
kaggle_output(KAGGLE_PATH, p)

