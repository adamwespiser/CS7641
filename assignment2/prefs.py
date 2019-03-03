
TRAIN_DATA_FILE = './data/WineQualityData_train.csv'
TEST_DATA_FILE = './data/WineQualityData_test.csv'
VALIDATE_DATA_FILE = './data/WineQualityData_validate.csv'
DS_NAME = 'WineQualityData'
OUTPUT_DIRECTORY = "output-flipflop-lower-n"



INPUT_LAYER = 12
HIDDEN_LAYER1 = 12
HIDDEN_LAYER2 = 12
OUTPUT_LAYER = 1
TRAINING_ITERATIONS_RHC = 1500
TRAINING_ITERATIONS_GA = 1500
TRAINING_ITERATIONS_SA = 1500
TRAINING_ITERATIONS = 1500


# Plotting
RHC_CUTOFF = TRAINING_ITERATIONS_RHC # RHC_Fitness.png
BEST_FIT_CUTOFF = 1000               # Best_Fitness.png
GA_CUTOFF = TRAINING_ITERATIONS_GA # GA_50_Score.png
SA_CUTOFF = TRAINING_ITERATIONS_SA   # SA_Accuracy.png

