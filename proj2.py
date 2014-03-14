'''
some parameters:
    context window size
    colocational size


the simplest thing to do:
    look at the full context for all examples of each sense
    simply count the number of all words

    word:
        sense1 (p-o-s):
            features, occurrences
        sense2 (p-o-s): 
            features, occurrences

        features:
            raw counts of words in total context


~~~
the biggest problem is:
    dealing with the potentially large number of features we want to specify

what if we see one feature many times for sense 1 but zero times for sense 2?
maybe we could key a dictionary by features and take the union, adding 1 to everything?

things we need to store:
    (f_i | s) for all f_i, for both (?) s
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
import re

#constants
TRAINING_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/proj2/training_data.data'

#global functions
def split_up():
    contexts = []

    with open(TRAINING_PATH) as f:
        for line in f.readlines():
            s = line.strip('\n').split('|')
            s = map(str.strip, s)
            #s[2] = s[2].split()

            contexts.append(s)

    return contexts




#classes
class Classifier:
    def __init__(self, examples_as_list, window = 3):
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

            self.classifiers[current_lemma].__call__(*example_as_list)

    def __call__(self, example_as_list):
        '''
        this should do classification for the input word and context
        '''
        pass


class Word:
    '''
    holds information on one word to disambiguate
    '''
    def __init__(self, window):
        self.senses = {}
        self.window = window
        self.lemma = None

    def __call__(self, lemma_and_pos, sense, context):
        
        #first call sets the lemma for the class
        if self.lemma == None:
            self.lemma = lemma_and_pos.split('.')[0]

        #raise error if you try to feed it the wrong word
        assert lemma_and_pos.split('.')[0] == self.lemma, 'wrong lemma for this class'

        if sense not in self.senses:
            self.senses[sense] = Sense(self.window)
        
        self.senses[sense].add_example(lemma_and_pos, sense, context)

    def compare():
        pass

    def __repr__(self):
        return 'word class instance for "{}"'.format(self.lemma)


class Sense:
    '''
    holds information on one sense of a word

    feature counting and whatnot is done here in the add_example function
        just need to edit one place to add more features!
    '''
    def __init__(self, window):
        self.count = 0
        self.features = {}
        self.window = window


    def add_example(self, lemma_and_pos, sense, context):

        before, target, after = re.split(r'\s*%%\s*', context)

        #except ValueError:
        #    print re.split(r'\s*%%\s*', context)
        #if len(s) == 2:

        #elif len(s) == 3:
        #    before, target, after = s


        before = re.split(r' +', before)
        after = re.split(r' +', after)

        #assuming here that both before and after can't be zero length strings
        #in that case there'd be no context
        if len(before) == 1 and before[0] == '':
            del before
            context = after
        elif len(after) == 1 and after[0] =='':
            del after
            context = before
        else:
            context = before + after

        #co-occurrence
        for word in context:
            if word not in self.features:
                self.features[word] = 0
            self.features[word] += 1

        #co-location
        for index in range(self.window):
            #expands on both sides of word to disambiguate
            #so we do w+1 and w-1, w+2 and w-2, etc.
            try: 
                prev_w = before[-1 - index] + '_w{}'.format(-1 - index)
                if prev_w not in self.features:
                    self.features[prev_w] = 0
                self.features[prev_w] += 1
            except (IndexError, UnboundLocalError):
                #index error handles cases where target word occurs near beginnign or end of sentence
                continue

            try:
                post_w = after[index] + '_w+{}'.format(index + 1)
                if post_w not in self.features:
                    self.features[post_w] = 0
                self.features[post_w] += 1
            except (IndexError, UnboundLocalError):
                continue

    def __repr__(self):
        return str(self.features)


        #get rid of stopwords?
        #to some tf-idf?








