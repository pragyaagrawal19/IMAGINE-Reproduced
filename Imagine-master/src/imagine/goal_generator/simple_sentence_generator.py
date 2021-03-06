from nltk import word_tokenize
import numpy as np
import random
from src.playground_env.env_params import get_env_params
from src.playground_env.descriptions import generate_all_descriptions
# np.random.shuffle(train_descriptions)
    

class SentenceGenerator:
    def __init__(self):
        self.sentences_set = set()
        self.sentences_set_tokenized = []
        self.max_len = 0
        pass

    def update_model(self, sentences_list):
        new_sentence_set = set(sentences_list).difference(self.sentences_set)
        self.sentences_set = self.sentences_set.union(sentences_list)
        # word_tokenize will seperate words from sentences
        new_sentence_tokenized = [word_tokenize(s) for s in new_sentence_set]
        for new_sentence in new_sentence_tokenized:
            if len(new_sentence) > self.max_len:
                self.max_len = len(new_sentence)
        for new_sentence in new_sentence_tokenized:
            if len(new_sentence) < self.max_len:
                new_sentence += [' ' for _ in range(self.max_len - len(new_sentence))]
        self.sentences_set_tokenized += new_sentence_tokenized
        return new_sentence_tokenized

    def generate_sentences(self, n=1):
        pass


class RandomGoalGenerator(SentenceGenerator):
    def __init__(self, sentences=None):
        super().__init__()
        self.sentence_types = []
        self.word_equivalence = []
        if sentences is not None:
            self.update_model(sentences)
        self.n_sentences = 136
        self.out = None
        self.set_generated = set()

    def update_model(self, sentences_list):
        super().update_model(sentences_list)
        self.word_set = set()
        lengths = dict()
        for s in self.sentences_set:
            s = s.split(' ')
            # there are how many words in sentence
            len_word = len(s)
            for w in s:
                self.word_set.add(w)
            # if 3 words sentence not in length.keys than add l[3]=1, else inc l[3]
            if str(len_word) not in lengths.keys():
                lengths[str(len_word)] = 1
            else:
                lengths[str(len_word)] += 1
        # legnth= number of 3 , 4 and 5 legnth words
        self.proportions = dict()

        for k in lengths.keys():
            self.proportions[k] = lengths[k] / len(self.sentences_set_tokenized)
        if len(lengths.keys()) == 0:
            pass
        self.lengths = list(self.proportions.keys())
        self.probas = [self.proportions[k] for k in self.proportions.keys()]

    def generate_sentences(self, n=1):
        new_generated = set()
        count = 0
        while len(new_generated) < n:
            count += 1

            length = int(np.random.choice(self.lengths, p=self.probas))
            sentence = np.random.choice(list(self.word_set), size=length, replace=False)
            sentence_str = ' '.join(sentence)
            if sentence_str not in self.sentences_set and sentence_str not in self.set_generated and sentence_str not in new_generated:
                new_generated.add(sentence_str)
                self.set_generated.add(sentence_str)

            if count > 5000:
                break

        return list(new_generated)


class SentenceGeneratorHeuristic(SentenceGenerator):
    def __init__(self, train_descriptions, test_descriptions, sentences=None, method='CGH'):
        super().__init__()
        self.sentence_types = []
        self.word_equivalence = []
        if sentences is not None:
            self.update_model(sentences)
        self.random_goal_generator = RandomGoalGenerator()
        self.generated_set = set()
        self.generated_from_test = set()
        self.generated_garbage = set()
        self.train_descriptions = train_descriptions
        self.test_descriptions = test_descriptions

        if method == 'CGH':
            self.coverage = self.precision = None
        elif method == 'low_coverage':
            self.coverage = 'low'
            self.precision = None
        elif method == 'low_precision':
            self.precision = 'low'
            self.coverage = None
        elif method == 'random':
            self.precision = 0
            self.coverage = 0
        elif method == 'oracle':
            self.precision = 1
            self.coverage = 1
        elif method == "SCBH":
            self.coverage = self.precision = None
        else:
            raise NotImplementedError

    def update_model(self, sentences_list):
        new_sentences_tokenized = super().update_model(sentences_list)
        self.random_goal_generator.update_model(sentences_list)
        # pad type sentences
        for type_sent in self.sentence_types:
            if len(type_sent) < self.max_len:
                type_sent += [' ' for _ in range(self.max_len - len(type_sent))]

        if len(self.sentence_types) == 0:
            self.sentence_types.append(new_sentences_tokenized[0])
            new_sentences_tokenized = new_sentences_tokenized[1:]
        # for each sentence type, compute semantic distances with every new sentences tokenized
        for s_new in new_sentences_tokenized:
            match = False
            for index, s_type in enumerate(self.sentence_types):
                max_len = max(len(s_type), len(s_new))
                min_len = min(len(s_type), len(s_new))
                confusion = np.array([w1 == w2 for w1, w2 in zip(s_type, s_new)] + [False] * (max_len - min_len))
                semantic_dist = np.sum(~confusion)  # number of words of sentence that are not matching
                if semantic_dist == 1:
                    match = True
                    ind_eq = int(np.argwhere(~confusion))
                    equivalent_words = (s_type[ind_eq], s_new[ind_eq])
                    if ' ' not in equivalent_words:
                        set_match = False
                        for eq_set in self.word_equivalence:
                            if not set_match:
                                if equivalent_words[0] in eq_set and equivalent_words[1] in eq_set:
                                    set_match = True
                                else:
                                    if equivalent_words[0] in eq_set:
                                        set_match = True
                                        eq_set.add(equivalent_words[1])
                                    elif equivalent_words[1] in eq_set:
                                        set_match = True
                                        eq_set.add(equivalent_words[0])

                        if not set_match:
                            self.word_equivalence.append(set(equivalent_words))
                elif semantic_dist == 0:
                    match = True
            if not match:
                self.sentence_types.append(s_new)

        # remove sets in double,
        # merge set with equivalence
        ind_remove = []
        for i in range(len(self.word_equivalence)):
            if i not in ind_remove:
                for j in range(i + 1, len(self.word_equivalence)):
                    if self.word_equivalence[i] == self.word_equivalence[j]:
                        ind_remove.append(j)

        word_equivalence_new = []
        for j in range(len(self.word_equivalence)):
            if j not in ind_remove:
                not_new = False
                for w_eq in word_equivalence_new:
                    for w in self.word_equivalence[j]:
                        if w in w_eq:
                            w_eq.union(self.word_equivalence[j])
                            not_new = True
                            break
                if not not_new:
                    word_equivalence_new.append(self.word_equivalence[j])
        self.word_equivalence = word_equivalence_new

    def split_test_not_test(self, sentence_set):
        sents_in_test = []
        sents_not_in_test = []
        for s in sentence_set:
            if s in self.test_descriptions:
                sents_in_test.append(s)
            else:
                sents_not_in_test.append(s)
        return sents_in_test.copy(), sents_not_in_test.copy()

    def update_generated_set(self, new_sentences):
        sents_in_test, sents_not_in_test = self.split_test_not_test(new_sentences)

        if self.coverage is None:
            desired_counter_test = len(sents_in_test)
        else:
            # if self.tc = 1, we want the oracle and imagine flower goals when all other are discovered
            if self.coverage == 1:
                if len(sents_in_test) == 64 - 8:
                    desired_counter_test = 64
                else:
                    desired_counter_test = len(sents_in_test)
            elif self.coverage == 'low':
                desired_counter_test = len(sents_in_test) // 2
            elif self.coverage == 0:
                desired_counter_test = 0
            else:
                raise NotImplementedError

        if self.precision is None:
            desired_counter_garbage = len(sents_not_in_test)
            if self.coverage == 'low':
                desired_counter_garbage = len(sents_not_in_test) // 2
        else:
            if self.precision == 0:
                assert self.coverage == 0
                desired_counter_garbage = len(new_sentences)
            elif self.precision == 'low':
                # ratio = len(sents_not_in_test) / len(new_sentences)
                desired_counter_garbage = len(sents_not_in_test) * 2
            elif self.precision == 1:
                desired_counter_garbage = 0
            else:
                raise NotImplementedError

        in_test, not_in_test = self.split_test_not_test(list(self.generated_set))
        out = list(self.generated_set)
        counter_test = len(in_test)
        if desired_counter_test > counter_test:
            for s in sents_in_test:
                if s not in in_test:
                    out.append(s)
                    counter_test += 1
                if counter_test == desired_counter_test:
                    break

            if desired_counter_test != counter_test:
                if counter_test == 64 - 8:
                    for s in self.test_descriptions:
                        if 'flower' in s:
                            out.append(s)
                            counter_test += 1
            else:
                print('Weird')

        counter_garbage = len(not_in_test)
        if desired_counter_garbage > counter_garbage:
            if self.precision != 1:
                for s in sents_not_in_test:
                    if s not in not_in_test:
                        out.append(s)
                        counter_garbage += 1
                    if counter_garbage == desired_counter_garbage:
                        break
            if desired_counter_garbage > counter_garbage:
                random_garbage = self.random_goal_generator.generate_sentences(
                    n=desired_counter_garbage - counter_garbage)
                counter_garbage += len(random_garbage)
                out += random_garbage

        for s in out:
            self.generated_set.add(s)
        return out.copy()

    def generate_sentences(self, n=1):
        new_sentences = set()
        for sent_type in self.sentence_types:
            for i, word in enumerate(sent_type):
                for word_eqs in self.word_equivalence:
                    if word in word_eqs:
                        for eq in word_eqs:
                            if eq != word:
                                new_sent = sent_type.copy()
                                new_sent[i] = eq
                                while ' ' in new_sent:
                                    new_sent.remove(' ')
                                if np.unique(new_sent).size == len(new_sent):
                                    new_sent = ' '.join(new_sent.copy())
                                    if new_sent not in self.sentences_set:
                                        new_sentences.add(new_sent)
        out = self.update_generated_set(list(new_sentences))
        return out


class simple_conjuction_based_heuristic(SentenceGenerator):
    def __init__(self, train_descriptions, test_descriptions, sentences=None, method='SCBH'):
        super().__init__()
        self.train_descriptions = train_descriptions
        self.test_descriptions = test_descriptions
        self.new_sentence_generate = set()

    def generate_sentences(self):
        env_params = get_env_params()
        name_attributes = env_params['name_attributes']
        adjective_attributes = env_params['adjective_attributes']
        adj_list = list(adjective_attributes)
        adj_list.append('any')
        adjective_attributes = tuple(adj_list)
        action = env_params['admissible_actions']
        p = env_params.copy()
        # Get the list of admissible attributes and split them by name attributes (type and categories) and adjective attributes.
        name_attributes = p['name_attributes']
        adjective_attributes = p['adjective_attributes']
        adj_list = list(adjective_attributes)
        adj_list.append('any')
        adjective_attributes = tuple(adj_list)
        action = p['admissible_actions']
        if 'Grasp' in p['admissible_actions'] or 'Grow' in p['admissible_actions']:
            new_sentence = []
            while(len(new_sentence)<200):
                num1 = random.randrange(0, len(name_attributes))
                num2 = random.randrange(0, len(action))
                num3 = random.randrange(0, len(adjective_attributes))
                # print(num1, num2, num3)
                sentence = [action[num2], adjective_attributes[num3], name_attributes[num1]]
                sentence = ' '.join([str(elem) for elem in sentence])
                # new sentences are those which are not in train_description and also not in test_descriptions
                if sentence in self.train_descriptions:
                    pass
                else:
                    new_sentence.append(sentence)
            self.new_sentence_generate = tuple(new_sentence)
            return tuple(new_sentence)
        # new sentence will be generated randomly from enviroment directly


if __name__ == '__main__':
    from src.playground_env.env_params import get_env_params
    from src.playground_env.descriptions import generate_all_descriptions
    # np.random.shuffle(train_descriptions)
    env_params = get_env_params()
    train_descriptions, test_descriptions, _ = generate_all_descriptions(env_params)
    import numpy as np
    print("description for CGH method")
    generator1 = SentenceGeneratorHeuristic(train_descriptions, test_descriptions, None, method="CGH")
    generator1.update_model(train_descriptions)
    new_descriptions1 = generator1.generate_sentences()

    p_found_in_test = sum([d in test_descriptions for d in new_descriptions1]) / len(test_descriptions)
    p_not_in_test = sum([d not in test_descriptions for d in new_descriptions1]) / len(new_descriptions1)
    p_in_test = sum([d in test_descriptions for d in new_descriptions1]) / len(new_descriptions1)
    print('Percentage of the test set found (coverage):', p_found_in_test)
    print('Percentage of the new descriptions that are not in the test', p_not_in_test)
    print('Percentage of the new descriptions that are in the test set (precision)', p_in_test)

    print('\n Sentences in the generated set, not in the test: \n', set(new_descriptions1) - set(test_descriptions))
    print('\n Sentences in the test set, found by generation: \n',
          set(new_descriptions1).intersection(set(test_descriptions)))
    print('\n Sentences in the test set, not in the generated set: \n', set(test_descriptions) - set(new_descriptions1))
    print("new_description_legnth",len(new_descriptions1))
    print("""""""""""""""""""""""""""""""""""""""""""""""""")

    print("description for SCBH method")
    generator = simple_conjuction_based_heuristic(train_descriptions, test_descriptions, None, method="SCBH")
    new_descriptions = generator.generate_sentences()

    p_found_in_test = sum([d in test_descriptions for d in new_descriptions]) / len(test_descriptions)
    p_not_in_test = sum([d not in test_descriptions for d in new_descriptions]) / len(new_descriptions)
    p_in_test = sum([d in test_descriptions for d in new_descriptions]) / len(new_descriptions)
    print('Percentage of the test set found (coverage):', p_found_in_test)
    print('Percentage of the new descriptions that are not in the test ', p_not_in_test)
    print('Percentage of the new descriptions that are in the test set (precision)', p_in_test)

    print('\n Sentences in the generated set, not in the test: \n', set(new_descriptions) - set(test_descriptions))
    print('\n Sentences in the test set, found by generation: \n',
          set(new_descriptions).intersection(set(test_descriptions)))
    print('\n Sentences in the test set, not in the generated set: \n', set(test_descriptions) - set(new_descriptions))
    print(len(new_descriptions))

