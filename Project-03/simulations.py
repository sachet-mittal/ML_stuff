from random import *
from tqdm import *

DATA = {}
for i in range(1000):
    DATA[i] = "+" if i in range(100,601) else "-"


def get_test_set_1(test_size):
    return [randint(a=0, b=999) for _ in range(test_size)]

def get_hypothesis_1(training_size):
    features = [randint(a=0, b=999) for _ in range(training_size)]
    positive_values = [x for x in features if 100 < x <600]
    positive_values.sort()
    differences = []
    start_range = 0
    end_range = 0
    min_range = 1000
    for i in range(len(positive_values) - 1):
        # print (positive_values[i+1],  positive_values[i], positive_values[i+1] -  positive_values[i])
        if (positive_values[i+1] -  positive_values[i]) < min_range:
            start_range = positive_values[i]
            end_range = positive_values[i+1]
            min_range = positive_values[i+1] -  positive_values[i]
    # print ("range=" , start_range, end_range)
    return start_range, end_range

def run_simulation_1(train_size, test_size, start, end):
    test_data = get_test_set_1(test_size)
    incorrect = 0
    for item in test_data:
        prediction = "+" if start<= item < end else "-"
        if prediction != DATA[item]:
            incorrect += 1
    #print(f"Error={incorrect/test_size}")
    return incorrect/test_size

if __name__ == "__main__":
    test_size = 50000
    for train_size in tqdm(range(10, 300, 10)):
        print ("Running Simulation on size: ", train_size)
        errors = []
        for _ in tqdm(range(3000)):
            start, end = get_hypothesis_1(train_size)
            errors.append(run_simulation_1(train_size, test_size, start, end))
        print (f"average error={sum(errors)/len(errors)}")
