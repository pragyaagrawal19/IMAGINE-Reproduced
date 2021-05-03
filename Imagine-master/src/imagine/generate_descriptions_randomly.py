import random
import numpy as np
from src.playground_env.env_params import get_env_params
from src.imagine.goal_generator.simple_sentence_generator import SentenceGenerator
import numpy as np


class simple_conjuction_based_heuristic(SentenceGenerator):
    def __init__(self,  train_descriptions, test_descriptions, sentences=None, method='SCBH'):
        super().__init__()
        self.train_descriptions = train_descriptions
        self.test_descriptions = test_descriptions
        self.new_sentence_generate=set()
    def generate_sentences(self):
        if 'Grasp' in p['admissible_actions'] or 'Grow' in p['admissible_actions']:
            new_sentence = []
            for i in range(100):
                num1 = random.randrange(0, len(name_attributes))
                num2 = random.randrange(0, len(action))
                num3 = random.randrange(0, len(adjective_attributes))
                # print(num1, num2, num3)
                sentence = [action[num2], adjective_attributes[num3], name_attributes[num1]]
                sentence = ' '.join([str(elem) for elem in sentence])
                # new sentences are those which are not in train_description and also not in test_descriptions
                if sentence in train_descriptions:

                    pass
                else:
                    new_sentence.append(sentence)
            self.new_sentence_generate=tuple(new_sentence)
            return tuple(new_sentence)
        # new sentence will be generated randomly from enviroment directly


if __name__ == '__main__':

    from src.playground_env.descriptions import generate_all_descriptions

    env_params = get_env_params()
    train_descriptions, test_descriptions, extra_descriptions = generate_all_descriptions(env_params)

    p = env_params.copy()
    # Get the list of admissible attributes and split them by name attributes (type and categories) and adjective attributes.
    name_attributes = env_params['name_attributes']
    adjective_attributes = env_params['adjective_attributes']
    adj_list = list(adjective_attributes)
    adj_list.append('any')
    adjective_attributes = tuple(adj_list)
    action = env_params['admissible_actions']
    generator = simple_conjuction_based_heuristic(train_descriptions, test_descriptions, None, method='SCBH')
    new_descriptions = generator.generate_sentences()

    p_found_in_test = sum([d in test_descriptions for d in new_descriptions]) / len(test_descriptions)
    p_not_in_test = sum([d not in test_descriptions for d in new_descriptions]) / len(new_descriptions)
    p_in_test = sum([d in test_descriptions for d in new_descriptions]) / len(new_descriptions)
    print('Percentage of the test set found:', p_found_in_test)
    print('Percentage of the new descriptions that are not in the test', p_not_in_test)
    print('Percentage of the new descriptions that are in the test set', p_in_test)

    print('\n Sentences in the generated set, not in the test: \n', set(new_descriptions) - set(test_descriptions))
    print('\n Sentences in the test set also in new set (coverage): \n',
          set(new_descriptions).intersection(set(test_descriptions)))
    print('\n Sentences in the test set, not in the new set: \n', set(test_descriptions) - set(new_descriptions))
    print("new_sentences", new_descriptions)
    new = []
    for sent in new_descriptions:
        if sent not in train_descriptions and sent not in test_descriptions:
            new.append(sent)
    precision = (len(sent) / len(new_descriptions)) * 100
    print("precision", precision)
    coverage = (len(set(new_descriptions).intersection(set(test_descriptions))) / len(test_descriptions)) * 100
    print("coverage", coverage)
