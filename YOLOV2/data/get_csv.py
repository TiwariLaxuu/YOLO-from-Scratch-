import pandas as pd
import glob 

list_of_train_files = glob.glob('labels/train2017/*')
# no_of_train_file = len(list_of_train_files)
no_of_train_file = 8
df = pd.DataFrame()       
for i in range(no_of_train_file):
    file = list_of_train_files[i].split('/')[-1]
    df.loc[i, 'image_name'] = file.replace('.txt', '.jpg')
    df.loc[i, 'image_loc'] = file
df.to_csv('8_example_train.csv')

list_of_valid_files = glob.glob('labels/valid2017/*')

df = pd.DataFrame() 
# no_of_valid_files = len(list_of_train_files)
no_of_valid_files = 8      
for i in range(no_of_valid_files):
    file = list_of_valid_files[i].split('/')[-1]
    df.loc[i, 'image_name'] = file.replace('.txt', '.jpg')
    df.loc[i, 'image_loc'] = file
df.to_csv('8_example_valid.csv')