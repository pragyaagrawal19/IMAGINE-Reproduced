import nltk
nltk.download('punkt')
import numpy as np
import torch
import random
import gym
import pygame
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from IPython.display import HTML
import warnings
warnings.filterwarnings('ignore')

from src.imagine.experiment import config
from src.imagine.goal_sampler import GoalSampler
from src.utils.notebook_utils import get_params_for_notebook, get_modules_for_notebook,generate_animation_reward_module, generate_animation_policy_module, plot_tsne, plot_attention_vector

from src.imagine.goal_generator.simple_sentence_generator import SentenceGeneratorHeuristic

path= 'Results/reproduced/'
params = get_params_for_notebook(path)
policy_language_model, reward_language_model, policy, reward_function, goal_sampler = get_modules_for_notebook(path, params)

goal_str = 'Grow blue cat' #Grow red cat, Grasp blue sofa, Grow green cow ...
# Uncomment to sample a random goal from the training set
goal_str = np.random.choice(params['train_descriptions'])
assert goal_str in params['train_descriptions'], 'Please enter a feasible goal description.'
env = gym.make(params['conditions']['env_name'], display=False)
env.reset()
initial_o = env.unwrapped.reset_with_goal(goal_str)
env.step(np.array([0, 0, 0]))
env.render(close=True)
obs = pygame.surfarray.array3d(env.viewer).transpose([1,0,2])

plt.rcParams['figure.figsize'] = [12, 8]
fig = plt.imshow(obs)
fig.axes.get_xaxis().set_visible(True)
fig.axes.get_yaxis().set_visible(True)
plt.show()


generator = SentenceGeneratorHeuristic(train_descriptions=params['train_descriptions'],
                                       test_descriptions=params['test_descriptions'],
                                       sentences=None,
                                       method='CGH')
# update the set of known goals
generator.update_model(params['train_descriptions'])
# generate imagined goals
new_descriptions = generator.generate_sentences()

p_found_in_test = sum([d in params['test_descriptions'] for d in new_descriptions]) / len(params['test_descriptions'])
p_not_in_test = sum([d not in params['test_descriptions'] for d in new_descriptions]) / len(new_descriptions)
p_in_test = sum([d in params['test_descriptions'] for d in new_descriptions]) / len(new_descriptions)
print('Percentage of the test set found:', p_found_in_test)
print('Percentage of the new descriptions that are not in the test', p_not_in_test)
print('Percentage of the new descriptions that are in the test set', p_in_test)

data = pd.DataFrame({'Examples of sentences in the generated set, not in the test:': list( set(new_descriptions) - set(params['test_descriptions'])) })
print(data.head(7))

data = pd.DataFrame({'Examples of sentences in the test set, found by generation:': list(set(new_descriptions).intersection(set(params['test_descriptions']))) })
print(data.head(7))

data = pd.DataFrame({'Examples of sentences in the test set, , not in the generated set:': list(set(params['test_descriptions']) - set(new_descriptions)) })
print(data.head(7))
print(pd.DataFrame(params['test_descriptions'],columns=['Test goals never communicated by the social partner']))
goal_encoding = torch.tensor(reward_language_model.encode(goal_str))
attention_vector = reward_function.reward_function.get_attention_vector(goal_encoding).detach().numpy()
plot_attention_vector(attention_vector, goal_str, params)
plt.show()
############################
from sklearn.manifold import TSNE
from ipywidgets import interact

code = 'categories'

embs = {}
for descr in params['train_descriptions'] + params['test_descriptions']:
    goal_encoding = reward_language_model.encode(descr)
    embs[descr] = goal_encoding

descr = sorted(list(embs.keys()))
embeddings = np.array([embs[d] for d in descr])
X_embedded = TSNE(n_components=2,
                  perplexity=20,
                  n_iter=5000,
                  n_iter_without_progress=1000,
                  learning_rate=10).fit_transform(embeddings)
plot_tsne(X_embedded, descr, code, params)
plt.show()