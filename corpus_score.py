import collections
import nltk
import json
import math

# here you construct the unigram language model


def unigram(tokens):
    model = collections.defaultdict(lambda: 0.01)
    for f in tokens:
        try:
            model[f] += 1
        except KeyError:
            model[f] = 1
            continue
    N = float(sum(model.values()))
    for word in model:
        model[word] = model[word]/N
    return model

# computes perplexity of the unigram model on a testset


def prob_sentence(sentence, model):
    # testset = testset.split()
    N = 0
    probs = 1
    for word in sentence:
        N += 1
        probs = probs * model[word]
    return probs


def perplexity(corpus, model):
    # testset = testset.split()
    perplexity = 0
    N = 0
    for sentence in corpus:
        N += 1
        # print(prob_sentence(sentence, model))
        # break
        perplexity = perplexity + math.log(prob_sentence(sentence, model), 2)
    perplexity = 2 ** (- (1 / N) * perplexity)
    return perplexity


if __name__ == "__main__":

    # coco_en = json.load(open('./dataset/captions/coco.json', 'r'))
    # flickr30k_en = json.load(open('./dataset/captions/flickr30k.json', 'r'))

    coco_id = json.load(open('./dataset/captions/coco_id.json', 'r'))
    # flickr8k_id = json.load(
    #     open('./dataset/captions/flickr8k_id/processed/flickr8k_id.json', 'r'))

    # found = False

    # print(data['images'][0]['sentences'])

    # for x in coco_id['images']:
    #     for y in x['sentences']:
    #         if y['filename'] == '33703543.jpg':
    #             # if y['raw'] == 'A multicolor cat fighting with a black and brown dog in a red collar .':
    #             found = True

    #         if found:
    #             break
    #     if found:
    #         break

    print(x)

    exit(0)

    corpus = []
    tokens = []
    train = []
    val = []
    test = []
    # word_freq = collections.Counter()

    for img in coco_id['images']:
        captions = []

        if img['split'] in {'test'}:
            test.append(img)
        elif img['split'] in {'validation', 'val'}:
            val.append(img)
        else:
            train.append(img)

        for c in img['sentences']:
                # captions.append(c['tokens'])
            corpus.extend(c['tokens'])

            # word_freq.update(c['tokens'])
            # corpus.extend(c['tokens'])

    model = unigram(corpus)

    # we first tokenize the text corpus
    # tokens = nltk.word_tokenize(corpus)

    # print(model)
    # print(perplexity(corpus, model))
    # print(len(train), len(val), len(test))
    # print(len(word_freq))

"""
Flickr8k ID
  - Perplexity  :
  - Vocab size  : 6763

COCO ID
  - Perplexity  :
  - Vocab size  : 38732
"""
