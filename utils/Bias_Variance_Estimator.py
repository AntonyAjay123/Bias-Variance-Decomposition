#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from inspect import isclass
import uuid

# In[6]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import mode


# In[5]:

def train_model_sklearn(train_data_list, model_class, model_kwargs, X_test, task='regression'):
    all_preds = []
    for i, train_data in enumerate(train_data_list):
        print(f'\n--- Training Model {i+1}/{len(train_data_list)} ---')
        X_train_resampled, y_train_resampled = train_data
        
        model = model_class(**model_kwargs)
        model.fit(X_train_resampled, y_train_resampled)

        if task == 'classification':
            preds = model.predict(X_test)
        else:  # regression
            # preds = model.predict(X_test)
            beta_hat = np.linalg.pinv(X_train_resampled) @ y_train_resampled
            preds = X_test @ beta_hat
        all_preds.append(preds)
    return all_preds



def train_model(train_data_list,loss_fn,lr,model_class,model_kwargs,num_models,X_test,max_epochs,batch_size,patience=10,device='cpu',task='regression'):
    all_preds=[]
    train_loss = []
    test_loss = []
    unique_id = str(uuid.uuid4()) 
    print(f"Starting experiment with {num_models} models...")
    for i, train_data in enumerate(train_data_list):
        print(f'\n--- Training Model {i+1}/{num_models} ---')

        X_train_resampled = train_data[0]
        y_train_resampled = train_data[1]
        # batch_size = len(X_train_resampled)
        # Split the resampled data into training and validation sets for early stopping
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train_resampled, y_train_resampled, test_size=0.2, random_state=42
        )

        ## Creating tensors for the train and validation data
        X_train_tensor = torch.tensor(X_train_split, dtype=torch.float32).to(device)
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            y_train_tensor = torch.tensor(y_train_split, dtype=torch.long).to(device)
            y_val_tensor = torch.tensor(y_val_split, dtype=torch.long).to(device)
            # y_test_tensor  = torch.tensor(y_test, dtype=torch.long).to(device)
        elif isinstance(loss_fn, nn.MSELoss):
            y_train_tensor = torch.tensor(y_train_split, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val_split, dtype=torch.float32).to(device)
            # y_test_tensor  = torch.tensor(y_test, dtype=torch.long).to(device)
            
        
        X_val_tensor = torch.tensor(X_val_split, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

        ## Data Loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        ## Model initialization
        model = model_class(**model_kwargs).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=0.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)


        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0

        # Model Training loop with early stopping
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0
            for x, y in train_loader:
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            avg_epoch_loss = epoch_loss / len(train_dataset)
            # Validation step
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = loss_fn(val_output, y_val_tensor)
            
            # print(f'Epoch {epoch+1}/{max_epochs}, Avg Train Loss: {epoch_loss/len(train_dataset):.4f}, Val Loss: {val_loss.item():.4f}')

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = model.state_dict() # Save the best model
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Load the best model to use for prediction
        model.load_state_dict(best_state_dict)

        model.eval()
        with torch.no_grad():
            train_output = model(X_train_tensor)
            final_train_loss = loss_fn(train_output, y_train_tensor).item()
        train_loss.append(final_train_loss)

        ## Model evaluation
        if task == 'classification':
            with torch.no_grad():
                test_output  = model(X_test_tensor)
                preds = torch.argmax(test_output, dim=1).cpu().numpy()
                all_preds.append(preds)
        elif task == 'regression':
            with torch.no_grad():
                preds =  model(X_test_tensor).cpu().numpy()
                all_preds.append(preds)
        
    return all_preds,train_loss

# In[3]:


def get_bvd_mse(all_preds_np,y_test):
    # Ensure 2D (num_models, num_samples)
    if all_preds_np.ndim == 3 and all_preds_np.shape[-1] == 1:
        all_preds_np = all_preds_np.squeeze(-1)
    
    # Ensure y_test is 1D
    y_test = y_test.squeeze()
    mean_preds = np.mean(all_preds_np, axis=0)  # shape: (num_test_samples,)
    bias = np.mean((mean_preds - y_test.squeeze()) ** 2)
    # variance = np.mean(np.var(all_preds_np, axis=0))
    # variance = np.mean((all_preds_np - mean_preds[None, :]) ** 2) 
    variance = np.mean(np.var(all_preds_np, axis=0, ddof=0))  
    return bias,variance


# In[ ]:


def get_bvd_mae(model,test_loader, all_preds):
    print("Decomposing MAE")
    y_m = all_preds.median(dim=0).values
    y_o = torch.cat([yb for _, yb in test_loader], dim=0).squeeze() 
    y_o = y_o.unsqueeze(0)

    B=torch.abs(y_o-y_m)
    V = torch.mean(torch.abs(all_preds-y_m.unsqueeze(0)),dim=0)
    # Bias effect sign condition
    sbias = torch.ones_like(all_preds)
    sbias = torch.where(
    ((y_o >= all_preds) & (y_o < y_m)) |
    ((y_o < all_preds) & (y_o > y_m)),
    -1, 1)
    # Bias effect: (P(s = 1) - P(s = -1)) * B
    p_pos = (sbias == 1).float().mean(dim=0)  # Shape: [N]
    p_neg = (sbias == -1).float().mean(dim=0)  # Shape: [N]
    bias_effect = (p_pos - p_neg) * B  # Shape: [N]
    bias_effect_final = bias_effect.mean().item()

    # Step 6: Variance magnitude
    abs_dev = (all_preds - y_m.unsqueeze(0)).abs()  # Shape: [M, N]

    # Variance effect sign condition
    svar = torch.where(
    ((y_o >= all_preds) & (all_preds > y_m.unsqueeze(0))) |
    ((y_o < all_preds) & (all_preds < y_m.unsqueeze(0))),
    -1, 1)
    neg_mask = (svar == -1).float()
    P_neg = neg_mask.mean(dim=0)  # [N]
    neg_contrib = (abs_dev * neg_mask).sum(dim=0) / (neg_mask.sum(dim=0) + 1e-8) # [N]

    V = abs_dev.mean(dim=0)  # [N]
    variance_effect = V - 2 * neg_contrib * P_neg  # [N]
    variance_effect_final = variance_effect.mean().item()
    # abs_noise = (y_o - y_o.median(dim=1).values.unsqueeze(0)).abs() 
    # snoise = torch.where(
    #     ((y_o >= all_preds) & (y_o < y_o.median(dim=1).values.unsqueeze(0))) |
    #     ((y_o < all_preds) & (y_o > y_o.median(dim=1).values.unsqueeze(0))),
    #     -1, 1
    # )
    # neg_mask_noise = (snoise == -1).float()
    # P_neg_noise = neg_mask_noise.mean(dim=0)
    # N = (y_o - y_o.median(dim=1).values.unsqueeze(0)).abs().mean(dim=0)
    # neg_contrib_noise = (abs_noise * neg_mask_noise).sum(dim=0) / (neg_mask_noise.sum(dim=0) + 1e-8)
    # noise_effect = N - 2 * neg_contrib_noise * P_neg_noise
    # noise_effect_final = noise_effect.mean().item()

    # ----- Step 5: Final Decomposed Error -----
    final_error = bias_effect_final + variance_effect_final

    return bias_effect_final, variance_effect_final,final_error


# In[ ]:


def estimate_bias_variance_mse(model_class, X_train, y_train, X_test, y_test, loss_fn, model_kwargs={},
                           num_models=20, max_epochs=100, patience=10, batch_size=64, lr=0.001, device='cpu'):

    # Store predictions from each model on the test set
    all_preds=[]
    # Create num_models bootstrapped training sets
    train_data_list = [resample(X_train, y_train, replace=True) for _ in range(num_models)]

    if isclass(model_class) and issubclass(model_class, nn.Module):
        all_preds,train_loss=train_model(train_data_list,loss_fn,lr,model_class,model_kwargs,num_models,X_test,max_epochs,batch_size,
                                                   patience,device='cpu',task='regression')
    else:
        all_preds = train_model_sklearn(train_data_list, model_class, model_kwargs, X_test, task='regression')
    all_preds_np = np.stack(all_preds, axis=0).squeeze()
    y_test = y_test.squeeze()
    bias_sq, variance = get_bvd_mse(all_preds_np, y_test)
    total_error = np.mean((all_preds_np - y_test.squeeze()) ** 2)
    test_loss = np.mean((np.mean(all_preds_np, axis=0) - y_test) ** 2)
    error_sum = bias_sq + variance
    avg_train_loss = np.mean(train_loss)

    print(f"\n--- Final Results ---")
    print(f"Bias²:    {bias_sq:.4f}")
    print(f"Variance: {variance:.4f}")
    print(f"Total error: {total_error:.4f}")
    print(f"Bias² + Variance: {(bias_sq + variance):.4f}")
    return bias_sq,variance,total_error,error_sum,avg_train_loss,test_loss


# In[ ]:

def get_bvd_absolute(model_class, X_train, y_train, X_test, y_test, loss_fn, model_kwargs={},
                           num_models=20, max_epochs=100, patience=10, batch_size=64, lr=0.001, device='cpu'):

    all_preds = []
    
    # Create num_models bootstrapped training sets
    train_data_list = [resample(X_train, y_train, replace=True) for _ in range(num_models)]
    if isclass(model_class) and issubclass(model_class, nn.Module):
        all_preds=train_model(train_data_list,loss_fn,lr,model_class,model_kwargs,num_models,X_test,max_epochs,device,batch_size,patience,task='regression')
    else:
        all_preds = train_model_sklearn(train_data_list, model_class, model_kwargs, X_test, task='regression')
    all_preds_tensor = torch.tensor(all_preds, dtype=torch.float32)
    test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    )
    all_preds_np = np.stack(all_preds, axis=0).squeeze()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    mae_total, mae_bias, mae_prop, mae_unsys = decompose_mae_components(all_preds_np,y_test)
    y_test = y_test.squeeze()
    total_error = np.mean(np.abs(np.median(all_preds_np, axis=0) - y_test))
    print(f"\n--- Final Results ---")
    print(f"Bias:    {mae_bias:.4f}")
    print(f"Variance: {mae_unsys:.4f}")
    print(f"Systemic error:{mae_prop:.4f}")
    print(f"Total error: {total_error:.4f}")
    print(f"decomposed error: {mae_total:.4f}")
    return mae_total, mae_bias, mae_prop, mae_unsys,total_error
    
def estimate_bias_variance_0_1(model_class,loss_fn, X_train, y_train, X_test, y_test, model_kwargs={},
                              num_models=20, max_epochs=100, patience=10,
                              batch_size=64, lr=0.001, device='cpu', save_path='best_model.pt'):
    train_data_list= [resample(X_train,y_train,replace=True) for _ in range(num_models)]
    all_preds,train_loss = train_model(train_data_list,loss_fn,lr,model_class,model_kwargs,num_models,X_test,max_epochs,batch_size,patience,device,task='classification')
    # Convert to (num_runs, num_test_samples)
    all_preds = np.stack(all_preds, axis=0)
    # print(f"all predictions {all_preds}")
    y_test = y_test.flatten()

    ## predicting the mode
    mode_preds, _ = mode(all_preds,axis=0,keepdims=False)
    # print(f"mode preds {mode_preds}")
    mode_preds = np.array(mode_preds).flatten()
    y_test = np.array(y_test).flatten()

    bias_arr = (mode_preds!=y_test).astype(float)

    var_arr = (all_preds!=mode_preds).mean(axis=0)
    test_loss = np.mean((np.mean(all_preds, axis=0) - y_test) ** 2)

    expected_loss_arr = np.where(bias_arr == 0,var_arr,1 - var_arr)
    avg_bias = bias_arr.mean()
    avg_var = var_arr.mean()
    avg_exp_loss_paper = expected_loss_arr.mean()
    avg_exp_loss = (all_preds != y_test).mean()
    empirical_01_loss = (all_preds!=y_test).mean()
    avg_train_loss = np.mean(train_loss)
    
    print(f"\n--- Final 0-1 Loss Decomposition Results ---")
    print(f"Average Bias         : {avg_bias:.4f}")
    print(f"Average Variance     : {avg_var:.4f}")
    print(f"Expected 0-1 Loss    : {avg_exp_loss:.4f}")
    print(f"Expected 0-1 Loss paper   : {avg_exp_loss_paper:.4f}")
    print(f"Empirical 0-1 Loss   : {empirical_01_loss:.4f}")
    print(f"Bias - Variance?     : {avg_bias - avg_var:.4f} ")

    return avg_bias, avg_var, avg_exp_loss, empirical_01_loss,avg_train_loss,test_loss, {
        'bias': bias_arr,
        'variance': var_arr,
        'expected_loss': expected_loss_arr
    }
# train_model(train_loader,train_dataset,loss_fn,lr,model,X_val_tensor,y_val_tensor,max_epochs,save_path,patience=10)

# In[ ]:


def decompose_cross_entropy(model_class,loss_fn,X_train,y_train,X_test,y_test,model_kwargs={},num_models=10,
                           max_epochs=100,patience=10,batch_size=64,lr=0.001,device='cpu'):
    
    all_probs = []
    num_classes = len(np.unique(y_train))
    y_test_onehot = np.eye(num_classes)[y_test]
    # Create num_models bootstrapped training sets
    train_data_list = [resample(X_train, y_train, replace=True) for _ in range(num_models)]
    all_probs=train_model(train_data_list,loss_fn,lr,model_class,model_kwargs,num_models,X_test,max_epochs,batch_size,patience,device,task='classification')

    all_probs = np.stack(all_probs,axis=0)

    eps = 1e-12
    all_probs = np.clip(all_probs,eps,1.0)
    log_probs = np.log(all_probs)

    log_mean = np.mean(log_probs,axis=0)
    pi_bar = np.exp(log_mean)
    if pi_bar.ndim == 1:
        pi_bar /= pi_bar.sum()
    else:
        pi_bar /= pi_bar.sum(axis=1, keepdims=True)
    print(f"sum of pi_bar {np.sum(pi_bar)}")
    bias = np.mean([
    np.sum(y_test_onehot[i] * (np.log(y_test_onehot[i] + eps) - np.log(pi_bar[i] + eps)))
    for i in range(len(y_test))
    ])
    variance = 0.0
    num_models = all_probs.shape[0]
    for m in range(num_models):
        for i in range(len(y_test)):
            variance += np.sum(pi_bar[i] * (np.log(pi_bar[i] + eps) - np.log(all_probs[m, i] + eps)))
    variance /= (num_models * len(y_test))
    total_error = bias + variance
    if pi_bar.ndim == 1:
        # Single test sample
        actual_ce_loss = -np.log(pi_bar[y_test[0]] + 1e-12)
    else:
        # Multiple test samples
        actual_ce_loss = -np.mean(np.log(pi_bar[np.arange(len(y_test)), y_test] + 1e-12))
    print(f"\n--- Final CE Loss Decomposition Results ---")
    print(f"Average Bias         : {bias:.4f}")
    print(f"Average Variance     : {variance:.4f}")
    print(f"actual_ce_loss   : {actual_ce_loss:.4f}")
    print(f"total_error  : {total_error:.4f}")
    return bias,variance,total_error,actual_ce_loss

def decompose_mae_components(all_preds, y_true):
    """
    Decompose Mean Absolute Error (MAE) into bias, proportionality, and unsystematic components
    following Robeson & Willmott (2023).

    Parameters
    ----------
    all_preds : np.ndarray, shape (M, N)
        Predictions from M bootstrap/ensemble models on N test samples.
    y_true : np.ndarray, shape (N,)
        Ground-truth values for test samples.

    Returns
    -------
    mae_total : float
        Mean Absolute Error (baseline).
    mae_bias : float
        Bias component of MAE.
    mae_prop : float
        Proportionality component of MAE.
    mae_unsys : float
        Unsystematic component of MAE.
    """
    # Step 1: Use the ensemble mean prediction
    P = np.mean(all_preds, axis=0)   # shape (N,)
    O = y_true.squeeze()

    # Step 2: Compute bias (MBE = mean error in the mean level)
    MBE = np.mean(P - O)
    b = abs(MBE)  # bias weight

    # Bias-corrected predictions
    P_prime = P - MBE

    # Step 3: Fit regression line P on O
    slope, intercept = np.polyfit(O, P, deg=1)
    P_hat = intercept + slope * O

    # Bias-correct regression line
    P_hat_prime = P_hat - MBE

    # Step 4: Compute weights
    abs_errors = np.abs(P - O)

    # proportionality component (systematic slope error)
    p_i = np.abs(P_hat_prime - O)

    # unsystematic component (scatter about regression line)
    u_i = np.abs(P_prime - P_hat_prime)

    # Step 5: Weighted decomposition (eqns 14–16 in paper)
    mae_bias = np.mean((b / (b + p_i + u_i + 1e-12)) * abs_errors)
    mae_prop = np.mean((p_i / (b + p_i + u_i + 1e-12)) * abs_errors)
    mae_unsys = np.mean((u_i / (b + p_i + u_i + 1e-12)) * abs_errors)

    mae_total = mae_bias + mae_prop + mae_unsys

    return mae_total, mae_bias, mae_prop, mae_unsys


# In[ ]:




