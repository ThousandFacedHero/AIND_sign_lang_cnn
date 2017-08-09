import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # model selection based on BIC scores
        bic_score = float('inf')
        best_model = self.min_n_components
        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                temp_hmm = GaussianHMM(n_components=n, covariance_type="diag",
                                       n_iter=1000, random_state=self.random_state,
                                       verbose=False).fit(self.X, self.lengths)
                log_l = temp_hmm.score(self.X, self.lengths)
                # calculate free parameters
                p = (n*n)+(2*n)*(len(self.X[0])-1)
                # calculate bic for comparison
                temp_bic = -2 * log_l * math.log(len(self.X[0])) * p
                if temp_bic < bic_score:
                    bic_score = temp_bic
                    best_model = n
            except:
                pass

        return self.base_model(best_model)


class SelectorDIC(ModelSelector):
    """ select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # model selection based on DIC scores
        dic_score = float('-inf')
        best_model = self.max_n_components

        for n in range(self.min_n_components, self.max_n_components+1):
            try:
                temp_hmm = GaussianHMM(n_components=n, covariance_type="diag",
                                       n_iter=1000, random_state=self.random_state,
                                       verbose=False).fit(self.X, self.lengths)
                log_l = temp_hmm.score(self.X, self.lengths)
                # use word_vals list to store scores for each word, used to calculate DIC score
                word_vals = []
                for word in self.words:
                    try:
                        if word != self.this_word:
                            word_x, word_len = self.hwords[word]
                            word_vals.append(temp_hmm.score(word_x, word_len))
                    except:
                        pass
                if len(word_vals) != 0:
                    # calculate DIC score for this n
                    temp_dic = log_l - (sum(word_vals)/len(word_vals))
                    if temp_dic > dic_score:
                        dic_score = temp_dic
                        best_model = n
            except:
                pass

        return self.base_model(best_model)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # model selection using CV
        cv_score = float('-inf')
        best_cv = self.max_n_components
        split_cv = KFold()
        for n in range(self.min_n_components, self.max_n_components+1):
            scores = []
            try:
                # train/score on rotating folds
                for train, test in split_cv.split(self.sequences):
                    try:
                        x_train, x_train_lens = combine_sequences(train, self.sequences)
                        x_test, x_test_lens = combine_sequences(test, self.sequences)
                        temp_hmm = GaussianHMM(n_components=n, covariance_type="diag",
                                               n_iter=1000, random_state=self.random_state,
                                               verbose=False).fit(x_train, x_train_lens)
                        scores.append(temp_hmm.score(x_test, x_test_lens))
                    except:
                        pass
                # calculate score and set if better than previous
                iter_score = np.mean(scores)
                if iter_score > cv_score:
                    cv_score = iter_score
                    best_cv = n
            except:
                pass
        return self.base_model(best_cv)
