import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

from tqdm import tqdm
from operator import truediv

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models import CNN_Model, TR_Model
from global_vars import device


class Model_Builder():
    def __init__(self, arch, dataset):
        
        self.dataset = dataset
        self.dataset_dict = dataset.dataset_dict
        self.num_classes = dataset.dataset_dict['nclass']
        self.name = f"{arch}"
        
        print(f"\nBuilding Model {self.name}..")
        if 'vit' in arch or 'swin' in arch:
          self.model = TR_Model(arch, self.num_classes).to(device)
        else:
          self.model = CNN_Model(arch, self.num_classes).to(device)
            
        # Wrap your model with DataParallel for multi-gpu training..
        # self.model = nn.DataParallel(self.model)
        
        self.save_path = f"./{self.dataset_dict['folder']}/{self.name}/"
        self.model_path = f"{self.save_path}checkpoint__{self.dataset_dict['name']}_{self.name}.pth"
        
        # Check if the directory exists, if not, create it..
        print("Creating a Saving Directory... ")
        os.makedirs(os.path.dirname(f"{self.save_path}"), exist_ok=True)
    


    def trainer(self, EPOCHS=200, LEARNING_RATE=0.001, WEIGHT_DECAY=0.0005, FACTOR=0.1, PATIENCE=5, STOPPING_PATIENCE=10):
        
        # Define the early stopping criteria
        best_val_loss = float('inf')
        best_val_acc = 0.0
        best_train_loss = float('inf')
        best_train_acc = 0.0
        patience = STOPPING_PATIENCE
        counter = 0
        epoch = 0
        
        # List to store the trainning and validation acc and loss for each epoch
        train_acc_list = []
        train_loss_list = []
        val_acc_list = []
        val_loss_list = []

        # List to store the duration of each epoch
        epoch_times = []
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=FACTOR, patience=PATIENCE, verbose=True)
        
        
        print(f"\nStarting Trainning... for model {self.name}\n\n")
        # Start the timer for the overall training
        total_training_start_time = time.time()

        # Training loop
        for epoch in range(EPOCHS):
            # Start the timer for this epoch
            epoch_start_time = time.time()

            self.model.train()
            train_loss = 0.0
            train_correct = 0

            for inputs, labels in tqdm(self.dataset.train_loader, desc="Training : ", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                outputs = f.softmax(outputs, dim=1)
                
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()

            train_loss /= len(self.dataset.train_loader.dataset)
            train_acc = train_correct / len(self.dataset.train_loader.dataset)
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0

            with torch.no_grad():
                for inputs, labels in tqdm(self.dataset.valid_loader, desc="Validating : ", leave=False):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = self.model(inputs)

                    loss = criterion(outputs, labels)
                    outputs = f.softmax(outputs, dim=1)

                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()

                val_loss /= len(self.dataset.valid_loader.dataset)
                val_acc = val_correct / len(self.dataset.valid_loader.dataset)
                val_acc_list.append(val_acc)
                val_loss_list.append(val_loss)
                
            # Step the scheduler
            scheduler.step(val_loss)

            # End the timer for this epoch
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)

            print(f"Epoch: {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time Taken: {epoch_duration:.2f} seconds")



            # Early stopping based on validation accuracy
            if val_acc > best_val_acc :
                best_val_acc = val_acc
                counter = 0
                best_val_loss = val_loss
                best_train_acc = train_acc
                best_train_loss = train_loss
                
                # Save the model if it has the best validation accuracy
                torch.save(self.model.state_dict(), self.model_path)
            else :
                counter += 1
                if counter >= patience:
                    print("Early stopping!")
                    break
        
        # End the timer for the total training
        total_training_end_time = time.time()
        total_training_time = total_training_end_time - total_training_start_time

        # Calculate the mean epoch time
        mean_epoch_time = np.mean(epoch_times)

        print(f"\n\nBEST MODEL --> \nTrain Acc : {best_train_acc:.4f} | Train Loss : {best_train_loss:.4f} | Valid Acc : {best_val_acc:.4f} | Valid Loss : {best_val_loss:.4f}")
        print(f"Total Time Taken: {total_training_time:.2f} seconds | Average Time Taken Per Epoch: {mean_epoch_time:.2f} seconds")
        
                
        # Plot accuracy and loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_acc_list, label='Train')
        plt.plot(val_acc_list, label='Validation')
        plt.title(f"{self.name} Model Accuracy")
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig(f"./{self.save_path}/Acc_{self.dataset_dict['name']}_{self.name}.png")
        plt.show()


        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_list, label='Train')
        plt.plot(val_loss_list, label='Validation')
        plt.title(f"{self.name} Model Loss")
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.savefig(f"./{self.save_path}/loss_{self.dataset_dict['name']}_{self.name}.png")
        plt.show()
        
        
        train_result = dict(lr=LEARNING_RATE, tol_epoch=EPOCHS, epoch=epoch, train_acc=best_train_acc, train_loss=best_train_loss, valid_acc=best_val_acc, valid_loss=best_val_loss, mean_epoch_time=mean_epoch_time, total_training_time=total_training_time)
        
        return train_result

    
    
    
    def tester(self):
        criterion = nn.CrossEntropyLoss()
        
        # Load the best model checkpoint for evaluation
        print(f"\nTrying to Load Model {self.name} for Testing..")
        self.model.load_state_dict(torch.load(self.model_path))
        print(f"Successfully Loaded weights for Model {self.name}..")


        # Evaluate on the test set
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        top2_correct=0
        top5_correct=0

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataset.test_loader, desc="Testing : ", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)

                loss = criterion(outputs, labels)
                outputs = f.softmax(outputs, dim=1)

                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                _,top2preds = torch.topk(outputs, 2, 1)
                _,top5preds = torch.topk(outputs, 5, 1)
                test_correct += (predicted == labels).sum().item()
                for i,label in enumerate(labels):
                    if label in top5preds[i]:
                        top5_correct += 1
                    if label in top2preds[i]:
                        top2_correct += 1

            test_loss /= len(self.dataset.test_loader.dataset)
            test_acc = test_correct / len(self.dataset.test_loader.dataset)
            top2_acc = top2_correct/len(self.dataset.test_loader.dataset)
            top5_acc = top5_correct/len(self.dataset.test_loader.dataset)

        print(f"Test Loss : {test_loss:.4f} | Test Acc: {test_acc:.4f} | Top-2 Acc: {top2_acc:.4f} | Top-5 Acc: {top5_acc:.4f}")
        
        test_result = dict(test_acc=test_acc, test_loss=test_loss, top2_acc=top2_acc, top5_acc=top5_acc)
        
        return test_result
    
    

    
    def evaluate(self):
        # Load the best model checkpoint for evaluation
        self.model.load_state_dict(torch.load(self.model_path))

        # Generate and display confusion matrix
        self.model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in tqdm(self.dataset.test_loader, desc="Evaluating : ", leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                
                # Convert logits to probabilities using softmax
                probs = f.softmax(outputs, dim=1)
                
                _, predicted = torch.max(probs, 1)

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        # calculating Confusion Matrix...
        confusion_mat = confusion_matrix(y_true, y_pred)
        df = pd.DataFrame(confusion_mat)
        df.to_csv(f"./{self.dataset_dict['folder']}/{self.name}/confuse_matrix_{self.dataset_dict['name']}_{self.name}.csv")
        list_diag = np.diag(confusion_mat)
        list_raw_sum = np.sum(confusion_mat, axis=1)
        each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
        average_acc = np.mean(each_acc)
        print(f"Each class accuracy :\n{each_acc}")
        
        
        # Calculate precision and recall for each class 
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        
        # Calculate macro-averaged precision and recall
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        
        eval_result = dict(avg_acc=average_acc, each_acc=each_acc, precision_macro=precision_macro, recall_macro=recall_macro, f1_macro=f1_macro, class_metrics=[precision, recall, f1])
         
        return  eval_result


    def plot_Bars(self, lst):
        # Plot the bar chart
        precision = lst[0]
        recall = lst[1]
        f1 = lst[2]

        width = 0.25  # the width of the bars

        # Set position of bar on X axis 
        br1 = np.arange(self.num_classes) 
        br2 = [x + width for x in br1] 
        br3 = [x + width for x in br2] 

        # Make the plot
        plt.figure(figsize=(30, 8))
        plt.bar(br1, precision, color='r', width=width, label='Precision') 
        plt.bar(br2, recall, color='g', width=width, label='Recall') 
        plt.bar(br3, f1, color='b', width=width, label='F1 Score') 
        
        plt.xlabel('Classes', fontweight ='bold', fontsize = 15)
        plt.ylabel('Scores', fontweight ='bold', fontsize = 15)
        plt.title('Classwise Precision, Recall, and F1 Scores for Model {self.name} and Dataset ', fontweight ='bold', fontsize = 15)
        plt.xticks([r + width for r in range(self.num_classes)], range(self.num_classes))
        plt.legend()
        
        plt.savefig(f'./{self.save_path}/Classwise_PR_plot_{self.name}.png')
        plt.show()


    
    def log_output(self, train_result, test_result, eval_result):
        path = f"./{self.save_path}/results_{self.dataset_dict['name']}_{self.name}.txt"
        f = open(path, 'a')
        f.write(f"\n\n\nTraining and Testing Result For Learning_Rate : {train_result['lr']} & Epochs : {train_result['tol_epoch']}  (Early Stopping at {train_result['epoch']})\n\nThe Result is ->\n\n")
        f.write(f"Total Time Taken: {train_result['total_training_time']} seconds | Average Time Taken Per Epoch: {train_result['mean_epoch_time']} seconds")
        sentence0 = f"\n\nTrain Loss : {train_result['train_loss']} | Train Acc : {train_result['train_acc']} | Valid Loss : {train_result['valid_loss']} | Valid Acc : {train_result['valid_acc']}\n"
        f.write(sentence0)
        sentence1 = 'Test Loss is: ' + str(test_result['test_loss']) + '\n' + 'Overall Accuracy(Top-1 Accuracy) is: ' + str(test_result['test_acc']) + '\n'
        f.write(sentence1)
        sentence2 = 'Average Accuracy is: ' + str(eval_result['avg_acc']) +'\n'
        f.write(sentence2)
        sentence3 = 'Top-2 Accuracy is: '+ str(test_result['top2_acc']) + '\n'
        f.write(sentence3)
        sentence4 = 'Top-5 Accuracy is: '+ str(test_result['top5_acc']) + '\n'
        f.write(sentence4)
        element_mean = list(eval_result['each_acc'])
        sentence5 = "\n\nClasses : \n"+ str(self.dataset_dict['classes']) +"\nClass wise accuracy: \n" + str(element_mean) + '\n'
        f.write(sentence5)
        
        sentence6 = f"\n\nModel Precision (Macro Averaged) : {eval_result['precision_macro']}\nModel Recall (Macro Averaged)  : {eval_result['recall_macro']}\nModel F1_Score (Macro Averaged) : {eval_result['f1_macro']}\n"
        f.write(sentence6)

        sentence7 = f"\n\nClasswise Precision : \n{eval_result['class_metrics'][0]}"
        f.write(sentence7)
        sentence8 = f"\n\nClasswise Recall : \n{eval_result['class_metrics'][1]}"
        f.write(sentence8)
        sentence9 = f"\n\nClasswise F1_Score : \n{eval_result['class_metrics'][2]}"
        f.write(sentence9)
        f.close()