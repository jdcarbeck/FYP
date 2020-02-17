import nltk
from nltk.tokenize import word_tokenize
from nltk.tree import Tree
from nltk.chunk import regexp

class Concepts:
    concepts = []

    def __init__(self, text):
        pre_processed = self.process_text(text)
        self.concept_chunk(pre_processed)

    def process_text(self, text):
        tokenized_text = word_tokenize(text)
        pos_text = nltk.pos_tag(tokenized_text)
        return pos_text

    def get(self):
        return self.concepts

    def concept_chunk(self, pos_text):
        grammar = r"""
            NP: {<PP\$>?<JJ>*<NN.*>+}    #Noun Phrase
            P: {<IN>}                    # Preposition
            V: {<V.*>}                   # Verb
            PP: {<P> <NP>}               # PP -> P NP
            VP: {<V> <NP|PP>*}           # VP -> V (NP|PP)*
            """
        chunker = regexp.RegexpParser(grammar)
        tree = chunker.parse(pos_text)
        
        # Find noun pairs in tree
        for subtree in tree.subtrees(filter= lambda t: t.label()=='NP'):
            found_np = ' '.join([text for text, label in subtree.leaves()])
            self.concepts.append(found_np)

        # Find named enities
        tree = nltk.ne_chunk(pos_text,binary=True)
        for subtree in tree.subtrees(filter= lambda t: t.label()=='NE'):
            found_ne = ' '.join([text for text, label in subtree.leaves()])
            self.concepts.append(found_ne)