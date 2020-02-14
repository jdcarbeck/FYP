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
    print("---- Noun Phrases Extracted ----")
    for subtree in tree.subtrees(filter= lambda t: t.label()=='NP'):
        print(' '.join([text for text, label in subtree.leaves()]))
    print("---- Named Entitiy Extraction ----")
    tree = nltk.ne_chunk(pos_text,binary=True)
    for subtree in tree.subtrees(filter= lambda t: t.label()=='NE'):
        print(' '.join([text for text, label in subtree.leaves()]))



sentence = "\"A limited hangout or partial hangout is, according to former special assistant to the Deputy Director of the Central Intelligence Agency Victor Marchetti, \"spy jargon for a favorite and frequently used gimmick of the clandestine professionals. When their veil of secrecy is shredded and they can no longer rely on a phony cover story to misinform the public, they resort to admitting—sometimes even volunteering—some of the truth while still managing to withhold the key and damaging facts in the case. The public, however, is usually so intrigued by the new information that it never thinks to pursue the matter further.\" In a March 22, 1973 meeting between president Richard Nixon, John Dean, John Ehrlichman, John Mitchell, and H. R. Haldeman, Ehrlichman incorporated the term into a new and related one, \"modified limited hangout\". The phrase was coined in the following exchange: Before this exchange, the discussion captures Nixon outlining to Dean the content of a report that Dean would create, laying out a misleading view of the role of the White House staff in events surrounding the Watergate burglary. In Ehrlichman's words: \"And the report says, 'Nobody was involved,'\". The document would then be shared with the United States Senate Watergate Committee investigating the affair. The report would serve the administration's goals by protecting the President, providing documentary support for his false statements should information come to light that contradicted his stated position. Further, the group discusses having information on the report leaked by those on the Committee sympathetic to the President, to put exculpatory information into the public sphere. The phrase has been cited as a summation of the strategy of mixing partial admissions with misinformation and resistance to further investigation, and is used in political commentary to accuse people or groups of following a Nixon-like strategy. Writing in \"The Washington Post\", Mary McGrory described a statement by Pope John Paul II regarding sexual abuse by priests as a \"modified, limited hangout\"."
pre_text = process_text(sentence)
np_chunk(pre_text)