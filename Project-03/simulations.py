from random import sample, randint
import sys
from tqdm import tqdm

DATA = {}
for i in range(1000):
    DATA[i] = "+" if i in range(100,601) else "-"


def get_test_set(test_size):
    return [randint(a=0, b=999) for _ in range(test_size)]

def get_hypothesis(training_size):
    positive_values = []
    while not positive_values:
        features = [randint(a=0, b=999) for _ in range(training_size)]
        positive_values = [x for x in features if 100 < x <600]
    return min(positive_values), max(positive_values)

def run_simulation(train_size, test_size):
    start, end = get_hypothesis(train_size)
    test_data = get_test_set(test_size)
    incorrect = 0
    # print (start, end)
    for item in test_data:

        prediction = "+" if start< item < end else "-"
        #print (f"item={item} {prediction}")
        if prediction != DATA[item]:
            incorrect += 1

    #print(f"Error={incorrect/test_size}")
    return incorrect/test_size

if __name__ == "__main__":
    test_size = 50000
    for train_size in range(10, 301, 10):
    # for train_size in tqdm(range(10, 300, 10)):
        print ("Running Simulation on size: ", train_size)
        errors = []
        # for _ in tqdm(range(3000)):
        for _ in range(3000):
            errors.append(run_simulation(train_size, test_size))
        print (f"average error={sum(errors)/len(errors)}")
