


def posioning_attack(type,dataset,model_structure,poisoning_fraction):
    #input:
    #type: the attacker algorithm
    #dataset: {
    #   "clean_train": clean training dataset
    #   "clean_test": clean test dataset
    #   }
    #model_structure: model of the victim attack
    #poisoning_fraction: generate how many poisoning points
    #output:
    #the posioned dataset
    clean_train = dataset["clean_train"]
    clean_test = dataset["clean_test"]

    if type == 'loss':
        poisoning_data = loss_attack(clean_train,poisoning_fraction)
    elif type == 'lable attack':
        posioning_data = label_flip_attack(clean_train,poisoning_fraction)
    elif type == 'gradient ascent':
        posioning_data = gradient_ascent_attack(clean_train,poisoning_fraction)
    elif type == 'min max':
        posioning_data = min_max_attack(clean_train,poisoning_fraction)
    elif type == 'influence':
        posioning_data = influence_attack(clean_train,clean_test,poisoning_fraction)


    dataset["poisoning_train"] = posioning_data
    return dataset

def label_flip_attack(clean_train,poisoning_fraction):
    sample = clean_train["X"]
    poisoing_sample = int(sample * fraction)

    Z_c = np.ones(sample)
    Z_p = np.zeros(poisoing_sample)
    Z = np.concatenate((Z_c, Z_p), axis=0)

    x_p = np.copy(X[:poisoing_sample])
    y_p = np.copy(Y[:poisoing_sample])
    y_p = (1+y_p)%2 # how to change the label TODO

    X = np.concatenate((X[poisoing_sample:], x_p), axis=0)
    Y = np.concatenate((Y[poisoing_sample:], y_p), axis=0)

    poisoning_data["X"] = X
    poisoning_data["Y"] = Y
    poisoning_data["Poisoned"] = Z

    return  poisoning_data
