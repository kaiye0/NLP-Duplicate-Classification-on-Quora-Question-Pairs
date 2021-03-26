# spelling check implemented based on online codes from Peter Norvig
# https://norvig.com/spell-correct.html

data_root = '/Users/lemon/Downloads/EE5239/Data/'
import re
import gensim
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords


model = gensim.models.KeyedVectors.load_word2vec_format(data_root + 'GoogleNews-vectors-negative300.bin', binary=True)
words = model.index2word
lemmer = WordNetLemmatizer()

word_rank = {}
for i, word in enumerate(words):
    word_rank[word] = i

WORDS = word_rank
STOP_WORDS = stopwords.words("english")

def words(text): return re.findall(r'\w+', text.lower())


def P(word):
    return - WORDS.get(word, 0)


def correction(word):
    return max(candidates(word), key=P) #return the most possible candidate with one or two edits away
    # reasonable but the result may not be correct


def candidates(word):
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])


def known(words):
    return set(w for w in words if w in WORDS)


def edits1(word): # edits one substitution away
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word): # two substitutions away
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))


# my implementation
def spellcheck(x): # correction function -- results may be misleading
    x = str(x).lower()
    for WORD in x.split() :
        x = x.replace(WORD, correction(WORD))
    return x


def spelldectector(x, mis): # detection function -- record possible mistakes
    x = str(x).lower()
    x = lemmer.lemmatize(x)
    x = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', x)
    for WORD in x.split() :
        if WORD != correction(WORD) and WORD not in STOP_WORDS :
            mis.update({WORD: correction(WORD)})
    return mis

#test
list_mistake = {};
question_5_1 = 'which one dissolve in water quikly sugar, salt, or methane carbon di oxide?'
spelldectector(question_5_1, list_mistake)
