from create_tokens import *
import numpy as np

text = '''Machine learning is the study of computer algorithms that \
improve automatically through experience. It is seen as a \
subset of artificial intelligence. Machine learning algorithms \
build a mathematical model based on sample data, known as \
training data, in order to make predictions or decisions without \
being explicitly programmed to do so. Machine learning algorithms \
are used in a wide variety of applications, such as email filtering \
and computer vision, where it is difficult or infeasible to develop \
conventional algorithms to perform the needed tasks.'''

# tokenize text
tokens = tokenize(text)

# create mappings
word_to_id, id_to_word = mapping(tokens)

X, y = generate_training_data(tokens, word_to_id, 2)

model = init_network(len(word_to_id), 10)

n_iter = 50
learning_rate = 0.05

history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]

learning = one_hot_encode(word_to_id["learning"], len(word_to_id))
result = forward(model, [learning], return_cache=True)

print(result["a1"])
print("test")
