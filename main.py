import warnings
warnings.filterwarnings('ignore')

import os
import random
import torch
import numpy as np
import pandas as pd


from global_vars import *
from builder import Model_Builder
from data import DATA_LOADER

os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)    
random.seed(SEED)



def main(dataset_name, model_list, batch=64, lr=0.00001, epoch=200, dataset_dir=None, WEIGHT_DECAY=0.0005, FACTOR=0.1, PATIENCE=5, STOPPING_PATIENCE=5):   

    dataset = DATA_LOADER(dataset_name, batch=batch, dataset_dir=dataset_dir)
    
    for item in model_list:
      arch = item.lower()
      
      model_builder = Model_Builder(arch, dataset)
      
      # Trainning..
      train_result = model_builder.trainer(EPOCHS=epoch, LEARNING_RATE=lr, WEIGHT_DECAY=WEIGHT_DECAY, FACTOR=FACTOR, PATIENCE=PATIENCE, STOPPING_PATIENCE=STOPPING_PATIENCE)
      
      # Testing..
      test_result = model_builder.tester()
      
      # Evaluating..
      eval_result = model_builder.evaluate()
      
      model_builder.log_output(train_result, test_result, eval_result)
      
      for x in train_result:
        train_result[x] = round(train_result[x], 4)    # formatting into 4 decimal places.
      
      for x in test_result:
        test_result[x] = round(test_result[x], 4)     # formatting into 4 decimal places.
      
      for x in eval_result:
        eval_result[x] = np.round(eval_result[x], 4)  # formatting into 4 decimal places.
      



      # arranging trainning data in a csv file
      train_csv = f"./{dataset.dataset_dict['folder']}/train_results_{dataset.dataset_dict['name']}.csv"
      
      if os.path.exists(train_csv):
          df = pd.read_csv(train_csv)
      else:
          # Create an empty DataFrame with specified columns
          df = pd.DataFrame(columns=['Model', 'Tr_Loss', 'Tr_Acc', 'Val_Loss', 'Val_Acc', 'Tr_Time', 'AvgEp_Time'])
          df.to_csv(train_csv, index=False)
      
      

      train_result = [train_result[key] for key in ['train_loss', 'train_acc', 'valid_loss', 'valid_acc', 'total_training_time', 'mean_epoch_time']]
      
      # Update the DataFrame
      if item in df['Model'].values:
          print(f"\nThe Model {item} has already present in 'train_results.csv' file... Trying to Update row.. ")
          row_index = df.index[df['Model'] == item].tolist()[0]
          
          for i, val in enumerate(df.columns[1:]):
              df.at[row_index, val] = train_result[i]

          print(f"Updated a row with Model {item}..\n")
          
              
      else:
          # Add a new row with the Model name and the result for the specified item
          print(f"\nThe Model {item} not found in 'train_results.csv' file...")
          new_row = {col: item if col == 'Model' else train_result[i] for i, col in enumerate(df.columns[1:])}
          new_row['Model'] = item
          print(f"Created a row with Model {item}..\n")
          
          df = df._append(new_row, ignore_index=True)

      # Save the updated DataFrame back to the CSV file
      df.to_csv(train_csv, index=False)




      # arranging trainning data in a csv file
      test_csv = f"./{dataset.dataset_dict['folder']}/test_results_{dataset.dataset_dict['name']}.csv"
      
      if os.path.exists(test_csv):
          df = pd.read_csv(test_csv)
      else:
          # Create an empty DataFrame with specified columns
          df = pd.DataFrame(columns=['Model', 'Loss', 'Avg_Acc', 'Top1_Acc', 'Top2_Acc', 'Top5_Acc', 'Precision', 'Recall', 'F1_Score'])
          df.to_csv(test_csv, index=False)
      
      
      test_acc = [test_result[key] for key in ['test_acc', 'top2_acc', 'top5_acc']]
      test_metrics = [eval_result[key] for key in ['precision_macro', 'recall_macro', 'f1_macro']]
      
      test_result = [test_result['test_loss']] + [eval_result['avg_acc']] + test_acc + test_metrics
      
      # Update the DataFrame
      if item in df['Model'].values:
          print(f"\nThe Model {item} has already present in 'test_results.csv' file... Trying to Update row.. ")
          row_index = df.index[df['Model'] == item].tolist()[0]
          
          for i, val in enumerate(df.columns[1:]):
              df.at[row_index, val] = test_result[i]

          print(f"Updated a row with Model {item}..\n")
          
              
      else:
          # Add a new row with the Model name and the result for the specified item
          print(f"\nThe Model {item} not found in 'test_results.csv' file...")
          new_row = {col: item if col == 'Model' else test_result[i] for i, col in enumerate(df.columns[1:])}
          new_row['Model'] = item
          print(f"Created a row with Model {item}..\n")
          
          df = df._append(new_row, ignore_index=True)

      # Save the updated DataFrame back to the CSV file
      df.to_csv(test_csv, index=False)
    

