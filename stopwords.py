import re
from math import log
from math import sqrt
from collections import OrderedDict

TRAINING_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/training_data.data'

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

def stdev(list_of_numbers):
    mean = sum(list_of_numbers)/len(list_of_numbers)
    sos = 0
    for num in list_of_numbers:
        sos += (num - mean)**2

    return sqrt(sos) #/len(list_of_numbers))

def min_features(lemma_and_pos, sense, context, word_feature_dict):

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
        if word not in word_feature_dict:
            word_feature_dict[word] = 0
        word_feature_dict[word] += 1

    return word_feature_dict, len(context)


def mutual_information(TRAINING_PATH, throw_out_share):

    #really hacky, i know
    #things start about here

    examples = split_up(TRAINING_PATH)

    word_dict = {}
    word_sense_count = {}
    word_avg_length = {}


    #increments stuff based on examples
    for example in examples:
        lemma = example[0].split('.')[0]
        sense = example[1]
        context = example[2]

        if lemma not in word_dict:
            word_dict[lemma] = {}
            word_sense_count[lemma] = {}
            word_avg_length[lemma] = []
        if sense not in word_dict[lemma]:
            word_dict[lemma][sense] = {}
            word_sense_count[lemma][sense] = 0

        word_dict[lemma][sense], l = min_features(lemma, sense, context, word_dict[lemma][sense])
        word_sense_count[lemma][sense] += 1
        word_avg_length[lemma].append(l)

    #plus one everything
    for word in word_dict:
        f = set()
        for sense in word_dict[word]:
            f = f | set(word_dict[word][sense].keys())
        for sense in word_dict[word]:
            word_sense_count[word][sense] += 1

            for increment_sense in f:
                if increment_sense not in word_dict[word][sense]:
                    word_dict[word][sense][increment_sense] = 0
                word_dict[word][sense][increment_sense] += 1


    for word in word_avg_length:
        l = word_avg_length[word]
        word_avg_length[word] = sum(l)/len(l)

    mutual_information = {}

    for lemma in word_dict:

        for sense in word_dict[lemma]:

            for feature in word_dict[lemma][sense]:

                e_fi_si = (word_dict[lemma][sense][feature]/word_sense_count[lemma][sense]) # / (word_avg_length[lemma])

                #total feature count
                e_f_i = sum([word_dict[lemma][x][feature] for x in word_dict[lemma]]) # / word_avg_length[lemma]

                #s_i = word_sense_count[lemma][sense] / sum([word_sense_count[lemma][x] for x in word_sense_count[lemma]])

                #denom = f_i * s_i

                if lemma not in mutual_information:
                    mutual_information[lemma] = {}

                if feature not in mutual_information[lemma]:
                    mutual_information[lemma][feature] = []

                mutual_information[lemma][feature].append(e_fi_si / e_f_i)

    for lemma in mutual_information:
        for feature in mutual_information[lemma]:
            mutual_information[lemma][feature] = stdev(mutual_information[lemma][feature])

    for word in mutual_information:
        mutual_information[word] = OrderedDict(sorted(mutual_information[word].items(), key = lambda t: t[1], reverse=True))

        for i in range(round(throw_out_share*len(mutual_information[word]))):
            mutual_information[word].popitem()

    return mutual_information
