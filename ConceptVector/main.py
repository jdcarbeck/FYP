import nltk
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk.chunk import regexp

def process_text(text):
    tokenized_text = word_tokenize(text)
    pos_text = nltk.pos_tag(tokenized_text)
    return pos_text

def np_chunk(pos_text):
    grammar = r"""NP: {<DT|PP\$>?<JJ>*<NN.*>+}"""
    chunker = regexp.RegexpParser(grammar)
    tree = chunker.parse(pos_text)
    for subtree in tree.subtrees(filter= lambda t: t.label()=='NP'):
        print(subtree)


sentence = "Operation Sandwedge was a proposed clandestine intelligence-gathering operation against the political enemies of the Richard Nixon presidential administration. The proposals were put together by H. R. Haldeman, John Ehrlichman and Jack Caulfield in 1971. Caulfield, a former police officer, created a plan to target the Democratic Party and the anti-Vietnam War movement, inspired by what he believed to be the Democratic Party's employment of a private investigation firm."
pre_text = process_text(sentence)
np_chunk(pre_text)