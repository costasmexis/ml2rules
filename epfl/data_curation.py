import pandas as pd
import scipy.io

SAVE_TO_PATH = '../data/epfl_data.csv'
PATH_TO_DATA = "./data/training_set_ref.mat"
PATH_TO_STABILITY = './data/class_vector_train_ref.mat'
PATH_TO_NAMES = "./data/paremeterNames.mat"

def main():    
    
    data = scipy.io.loadmat(PATH_TO_DATA)
    data = pd.DataFrame(data['training_set'].T)

    stability = scipy.io.loadmat(PATH_TO_STABILITY)
    stability = stability['class_vector_train']
    stability = [s[0][0] for s in stability]

    names = scipy.io.loadmat(PATH_TO_NAMES)
    names = names['parameterNames']
    names = [n[0][0] for n in names]

    # Form generated kinetic models dataset
    data.columns = names
    data['Stability'] = stability
    data.to_csv(SAVE_TO_PATH, index=False)
    
if __name__ == "__main__":
    main()
    print("Data has been successfully generated and saved to ../data/epfl_data.csv")