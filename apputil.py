from collections import defaultdict
import numpy as np


class MarkovText(object):
    """
    A simple Markov chain-based text generator.

    This class builds a transition dictionary from a text corpus, where each word maps to a list of possible subsequent words.
    It then generates text by sampling from these transitions.
    """

    def __init__(self, corpus):
        """
        Initialize the MarkovText object.

        Parameters:

        corpus : str
            The input text corpus used to build the Markov model.
        """
        self.corpus = corpus
        self.term_dict = None

    def get_term_dict(self):
        """
        Build a transition dictionary from the corpus.

        Each unique word in the corpus is mapped to a list of words that immediately follow it. Duplicate entries are preserved
        to reflect transition probabilities.

        Returns:

        dict
            A dictionary mapping each word to a list of next words.
        """
        tokens = self.corpus.split()
        term_dict = defaultdict(list)

        for i in range(len(tokens) - 1):
            current_word = tokens[i]
            next_word = tokens[i + 1]
            term_dict[current_word].append(next_word)

        self.term_dict = dict(term_dict)
        return self.term_dict

    def generate(self, seed_term=None, term_count=15):
        """
        Generate text using the Markov chain.

        Parameters:

        seed_term : str, optional
            The starting word for text generation. If None, a random word from the corpus is used.
            term_count : int, default=15
            The number of words to generate.

        Returns:

        str
            A generated sequence of words.

        Raises:
        
        ValueError
            If the provided seed_term is not in the corpus.
        """
        if self.term_dict is None:
            self.get_term_dict()

        if seed_term is not None:
            if seed_term not in self.term_dict:
                raise ValueError("Seed term not in corpus.")
            current_word = seed_term
        else:
            current_word = np.random.choice(list(self.term_dict.keys()))

        output = [current_word]

        for _ in range(term_count - 1):
            next_words = self.term_dict.get(current_word)

            if not next_words:
                current_word = np.random.choice(
                    list(self.term_dict.keys())
                )
            else:
                current_word = np.random.choice(next_words)

            output.append(current_word)

        return " ".join(output)