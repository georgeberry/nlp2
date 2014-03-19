TRAINING_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/training_data.data'

VALIDATION_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/validation_data.data'

VALIDATION_PATH2 = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/validation_data2.data'

KAGGLE_OUTPUT_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/kaggle_output.csv'

KAGGLE_INPUT_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/test_data.data'

RATE_TRAIN_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/rate_train.data'

RATE_VALID_PATH = '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/rate_valid.data'

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

def get_valid_senses(valid_data):
    '''
    looks through validation data
    pulls out correct senses in order
    '''
    v = []
    for example in valid_data:
        v.append(example[1])
    return list(map(int, v))

def dict_max_key(d):
    '''
    given a dict, returns the KEY corresponding to the max VALUE val
    '''
    m = max(d.values())
    k = [k for k,v in d.items() if v == m]

    return k[0]

def train(a):
    d = {}

    for example in a:
        lemma = example[0].split('.')[0]
        sense = example[1]

        if lemma not in d:
            d[lemma] = {}

        if sense not in d[lemma]:
            d[lemma][sense] = 0

        d[lemma][sense] += 1

    for each in d:
        if len(d[each]) == 2:
            print(each, d[each])

    return d


def classify(valid_examples, train_dict):
    classifications = []

    for example in valid_examples:
        lemma = example[0].split('.')[0]
        classifications.append(dict_max_key(train_dict[lemma]))

    return list(map(int, classifications))

def get_rate_stuff(train_examples, valid_examples, path1, path2):
    base_lemma = 'rate'
    with open(path1, 'w') as f:
        for example in train_examples:
            if example[0].split('.')[0] == base_lemma:
                f.write(' | '.join(example) + '\n')
    with open(path2, 'w') as f:
        for example in valid_examples:
            if example[0].split('.')[0] == base_lemma:
                f.write(' | '.join(example) + '\n')



####

a = split_up(TRAINING_PATH)
b = split_up(VALIDATION_PATH)


d = train(a)
print(d)

v = get_valid_senses(b)
t = classify(b, d)

correct(v, t)

#kaggle_output(KAGGLE_OUTPUT_PATH, t)


get_rate_stuff(a, b, '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/rate_train.data', '/Users/georgeberry/Google Drive/Spring 2014/CS5740/nlp2/rate_valid.data')

