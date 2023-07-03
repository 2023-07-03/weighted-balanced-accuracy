#!/usr/bin/python

'''
MIT License


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import sys, getopt
import os.path
from os import path

def read_input_files(filepath):
    data = []

    class_samples = {}
    class_freq = {}

    f = open(filepath, 'r')
    data = f.read().splitlines()

    for key in data:
        if key not in class_samples.keys():
            class_samples[key] = 1
        else:
            class_samples[key] += 1

    for key,value in class_samples.items():
        class_freq[key] = float(value / sum(class_samples.values()))

    return class_samples, class_freq, data

def calculate_wba(per_class_weights,per_class_accuracy):
    wba = 0

    for key in per_class_accuracy.keys():
        wba += (per_class_weights[key] * per_class_accuracy[key])

    return wba

# function to calculate per class accuracy
def calculate_per_class_accuracy(real_data,pred_data,class_samples):

    per_class_correct = {}
    per_class_accuracy = {}

    for x,y in zip(real_data,pred_data):
        if x==y :
            if x not in per_class_correct.keys():
                per_class_correct[x] = 1
            else:
                per_class_correct[x] += 1

    for key,value in per_class_correct.items():
        per_class_accuracy[key] = value / class_samples[key]

    if verbose == True:
        print ('per_class_accuracy: ',per_class_accuracy)

    return per_class_accuracy

def balanced_weights(class_freq,user_given_weights):
    weights = {}
    for key in class_freq.keys():
        weights[key] = 1 / len(class_freq)

    if verbose == True:
        print ("balanced_weights: ",weights)

    return weights

def user_defined_weights(class_freq,user_given_weights):
    
    assert(sum(user_given_weights.values()) == 1)
    
    return user_given_weights

def inverse_frequency_weights(class_freq,user_given_weights):
    weights = {}

    inv_freq_sum = 0

    for key,value in class_freq.items():
        inv_freq_sum += (1/value)

    for key,value in class_freq.items():
        weights[key] = 1 / (value *inv_freq_sum)

    if verbose == True:
        print ("inverse_frequency_weights: ",weights)

    return weights

def user_and_inverse_frequency_weights(class_freq,user_given_weights):
    weights = {}

    inverse_weights = inverse_frequency_weights(class_freq,user_given_weights)
    weight_sum = 0

    for key,value in inverse_weights.items():
        weight_sum += (value*(user_given_weights[key]))

    for key,value in inverse_weights.items():
        weights[key] = (value*(user_given_weights[key])) / weight_sum

    return weights


def composition_weights(weights_file1, weights_file2):
    weights = {}

    weight_sum = 0

    for key,value in weights_file1.items():
        weight_sum += (value*(weights_file2[key]))

    for key,value in weights_file1.items():
        weights[key] = (value*(weights_file2[key])) / weight_sum


    return weights

# map the inputs to the function blocks
calculate_weigths = {0 : balanced_weights,
           1 : user_defined_weights,
           2 : inverse_frequency_weights,
           3 : user_and_inverse_frequency_weights,
           4 : composition_weights
}

def read_weight_file(filepath):
    weights = {}

    f = open(filepath, 'r')
    data = f.read().splitlines()

    for key in data:
        print (key)

    return weights

def read_data_files(filepath):
    class_dict = {}
    class_freq = {}

    if path.exists(filepath) == True:
        f = open(filepath, 'r')

        lines = f.read().splitlines()

        for line in lines:
            class_read = line.split()[0]
            class_param = float(line.split()[1])
            class_dict[class_read] = class_param
        
        for key,value in class_dict.items():
            class_freq[key] = float(value / sum(class_dict.values()))

    return class_dict,class_freq

def calculate_per_class_accuracy_from_misclassify(misclassify,class_samples):
    per_class_accuracy = {}
    per_class_correct = {}

    for key,value in class_samples.items():
        if key in misclassify.keys():
            per_class_correct[key] = value - misclassify[key]
        else:
            per_class_correct[key] = value

    for key,value in per_class_correct.items():
        per_class_accuracy[key] = value / class_samples[key]

    if verbose == True:
        print ('per_class_accuracy: ',per_class_accuracy)

    return per_class_accuracy

def main(argv):

    global verbose
    verbose = False
    config_mode = False
    weight_file = ''
    mode = 1
    class_freq = {}

    # Read input real file
    real_filepath = argv[0]
    pred_filepath = argv[1]

    try:
        opts,args = getopt.getopt(argv[2:],"hvm:cw:")
    except getopt.GetoptError:
        print ('check usage')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('usage')
            sys.exit()
        elif opt == '-v':
            verbose = True
        elif opt in ("-w"):
            weight_file = arg
        elif opt in ("-m"):
            mode = int(arg)
            if verbose == True:
                print('mode is ',mode)
        elif opt == '-c':
            config_mode = True
            class_dist_file = argv[0]
            misclassify_file = argv[1]
        elif opt == '-w':
            weight_file = arg
    
    if config_mode == True:
        # check for the class dist file and misclassify file
        class_samples,class_freq = read_data_files(class_dist_file)
        misclassify_freq,_ = read_data_files(misclassify_file)
        per_class_accuracy = calculate_per_class_accuracy_from_misclassify(misclassify_freq,class_samples)
    else:
        class_samples,class_freq,real_data = read_input_files(real_filepath)
        _,_,pred_data = read_input_files(pred_filepath)
        per_class_accuracy = calculate_per_class_accuracy(real_data,pred_data,class_samples)

    user_given_weights = {}
    per_class_weights = {}

    if weight_file is not '':
        user_given_weights,_ = read_data_files(weight_file)

    if mode == 4:
        per_class_weights = {}
        for key in per_class_accuracy.keys():
            per_class_weights[key] = 1.0
        
        for item in args:
            user_given_weights_each,_ = read_data_files(item)  
            per_class_weights = composition_weights(per_class_weights,user_given_weights_each)
    
    else:
        per_class_weights = calculate_weigths[int(mode)](class_freq,user_given_weights)
    
    wba = calculate_wba(per_class_weights,per_class_accuracy)
    print ('WBA score: ',wba)


if __name__ == "__main__":
    main(sys.argv[1:])
