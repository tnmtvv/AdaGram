import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.adagram_fixed_rank import AdaGramFR
from src.adagram_vanila import AdaGram
from src.adagram_projector_splitting import AdaGramPS
from src.shampoo import Shampoo
from src.full_G import FullAdaGrad
from src.utils.dataset import SparseDataset, CorrelatedDataset, LinearDataset

from src.utils


class LinearRegressionModel(nn.Module):
    def __init__(self, dim_in, dim_out, seed=100):
        super(LinearRegressionModel, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.linear = nn.Linear(dim_in, dim_out)
    
    def forward(self, x):
        return self.linear(x)
    

class MultiClassLogisticRegressionModel(nn.Module):
    def __init__(self, num_classes=2, dim=2):
        super(MultiClassLogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        return self.softmax(self.linear(x))
    
    
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, output_dim=2, seed=100):
        super().__init__()
        if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x) 
    

def train_model(model, optimizer, criterion, results, X_train, y_train, X_test, y_test, 
                num_epochs, opt_name, lr, time_start=None, r=None, use_tqdm=True, grad_save_dir='gradients'):
    """
    Train a model and evaluate on test data, saving gradients by epoch.
    """
    from tqdm import tqdm
    import time
    
    # Create directory for saving gradients
    if not os.path.exists(grad_save_dir):
        os.makedirs(grad_save_dir)

    epoch_iterator = tqdm(range(num_epochs)) if use_tqdm else range(num_epochs)
                                                                                                                             
    if time_start is None:
        time_start = time.time()
    
    for epoch in epoch_iterator:
        model.train()
        start_epoch = time.time()
        optimizer.zero_grad()
        y_pred = model(X_train)
        train_loss = criterion(y_pred, y_train)
        train_loss.backward()

        # Access and save gradients after backward() but before optimizer.step()
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.detach().cpu().numpy()

        # Save gradients to compressed numpy file per epoch
        grad_filename = f'{opt_name}_lr{lr}_epoch{epoch}'
        if r is not None:
            grad_filename += f'_rank{r}'
        grad_file = os.path.join(grad_save_dir, f'{grad_filename}.npz')
        np.savez_compressed(grad_file, **grad_dict)

        optimizer.step()

        elapsed_time = time.time() - time_start
        epoch_time = time.time() - start_epoch

        avg_epoch_time = elapsed_time / (epoch + 1)
        
        if r is not None:
            r_in_name = f" rank {r}"
        else:
            r_in_name = ''

        results.append({
            'epoch': epoch,
            'optimizer': opt_name + f"{r_in_name}",
            'lr': lr,
            'loss': train_loss.detach().cpu().numpy(),
            'rank': r,
            'avg_epoch_time': avg_epoch_time,
            'epoch_time': epoch_time
        })

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_loss = criterion(y_pred_test, y_test).item()
    
    return results, test_loss



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

def train_model_stochastic(
    model, optimizer, criterion, results, X_train, y_train, X_test, y_test,
    num_epochs, opt_name, lr, batch_size=32, shuffle=True, time_start=None,
    r=None, use_tqdm=True, grad_save_dir='gradients', data_seed=None, seed=42, task_name=None
):
    """
    Train a model using stochastic gradient descent with mini-batches and evaluate on test data.

    Args:
        batch_size (int): Size of mini-batches for stochastic training
        shuffle (bool): Whether to shuffle training data each epoch
        seed (int): Random seed for reproducibility
    """
    from tqdm import tqdm
    import time

    # Set all seeds for reproducibility
    # set_all_seeds(seed)

    # Create directory for saving gradients
    if not os.path.exists(grad_save_dir):
        os.makedirs(grad_save_dir)

    # Create DataLoader for stochastic training
    if data_seed:
        seed_everything(data_seed)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle
    )

    epoch_iterator = tqdm(range(num_epochs)) if use_tqdm else range(num_epochs)

    if time_start is None:
        time_start = time.time()

    for epoch in epoch_iterator:
        model.train()
        start_epoch = time.time()
        epoch_loss = 0.0
        num_batches = 0

        if epoch == 0:
            y_pred_train = model(X_train)
            y_pred_test = model(X_test)

            train_loss = criterion(y_pred_train, y_train)
            test_loss = criterion(y_pred_test, y_test)

            if r is not None:
                r_in_name = f" rank {r}"
            else:
                r_in_name = ''

            results.append({
            'epoch': epoch,
            'optimizer': opt_name + f"{r_in_name}",
            'lr': lr,
            'loss': test_loss,
            'mode': "test", 
            'rank': r,
            'avg_epoch_time': 0,
            'epoch_time': 0,
            'batch_size': batch_size,
            'data_seed': data_seed})

            results.append({
            'epoch': epoch,
            'optimizer': opt_name + f"{r_in_name}",
            'lr': lr,
            'loss': train_loss,
            'mode': "train", 
            'rank': r,
            'avg_epoch_time': 0,
            'epoch_time': 0,
            'batch_size': batch_size,
            'data_seed': data_seed})

        # Stochastic training loop over mini-batches

        all_grads = {name: [] for name, _ in model.named_parameters()}

        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            batch_loss = criterion(y_pred, batch_y)
            batch_loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    all_grads[name].append(param.grad.detach().cpu().numpy())

            # Update parameters
            if r:
                optimizer.step(epoch)
            else:
                optimizer.step()

            epoch_loss += batch_loss.item()
            num_batches += 1
        
                
        # After all batches in the epoch:
        stacked_grads = {name: np.stack(grads) for name, grads in all_grads.items()}
        grad_filename = f'{task_name}_{opt_name}_lr{lr}_epoch{epoch}'
        if r is not None:
            grad_filename += f'_rank{r}'
        grad_file = os.path.join(grad_save_dir, f'{grad_filename}_stacked.npz')
        np.savez_compressed(grad_file, **stacked_grads)
        
        y_pred_train = model(X_train)
        y_pred_test = model(X_test)
        
        train_loss = criterion(y_pred_train, y_train)
        test_loss = criterion(y_pred_test, y_test)

        elapsed_time = time.time() - time_start
        epoch_time = time.time() - start_epoch
        avg_epoch_time = elapsed_time / (epoch + 1)

        if r is not None:
            r_in_name = f" rank {r}"
        else:
            r_in_name = ''

        results.append({
            'epoch': epoch + 1,
            'optimizer': opt_name + f"{r_in_name}",
            'lr': lr,
            'loss': test_loss,
            'mode': "test", 
            'rank': r,
            'avg_epoch_time': avg_epoch_time,
            'epoch_time': epoch_time,
            'batch_size': batch_size
        })

        results.append({
            'epoch': epoch + 1,
            'optimizer': opt_name + f"{r_in_name}",
            'lr': lr,
            'loss': train_loss,
            'mode': "train", 
            'rank': r,
            'avg_epoch_time': avg_epoch_time,
            'epoch_time': epoch_time,
            'batch_size': batch_size
        })

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        test_loss = criterion(y_pred_test, y_test).item()

    return results, test_loss



# torch.manual_seed(42)
# np.random.seed(42)

learning_rates = [0.1]
num_epochs = 100
ranks = [1, 2, 3, 4, 5]
in_dims = [20]
out_dims = [2]
data_seeds = [10]

tasks = {
        # "LinReg": lambda in_dim, out_dim, seed: LinearRegressionModel(dim_in=in_dim, dim_out=out_dim, seed=seed), 
         "BinClass": lambda in_dim, out_dim, seed: SimpleClassifier(input_dim=in_dim, output_dim=out_dim, seed=seed)
        }

losses = {
    # "LinReg": nn.MSELoss(), 
    "BinClass": nn.CrossEntropyLoss()
          }


for data_seed in data_seeds:
    ds = SparseDataset(n_samples=300, in_dim=in_dims[0], out_dim=out_dims[0], seed=data_seed, if_class=False)
    X, y = ds.create_data()

    # print(X)
    # print(y)


    print("cond", torch.linalg.cond(X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape)
    print(y_train.shape)

    optimizers = {
        # 'Shampoo': lambda params, lr: Shampoo(params, lr=lr),
        # 'FullMatrixAdaGrad': lambda params, lr: FullMatrixAdaGrad(params, lr=lr),
        # 'AdaGram': lambda params, lr: AdaGram(params, lr=lr),
        # "FullAdaGrad": lambda params, lr: FullAdaGrad(params=params, lr=lr),
        'AdaGramPS': lambda params, lr, max_rank, task: AdaGramPS(params=params, lr=lr, max_rank=max_rank, log_file=f"results/loggs/psi_adagram_logs_{max_rank}.csv", task=task),
        'AdaGramFR_svd': lambda params, lr, max_rank, task: AdaGramFR(True, params, lr=lr, max_rank=max_rank, task=task),
        # 'AdaGramFR_nosvd': lambda params, lr, max_rank: AdaGramFR(False, params, lr=lr, max_rank=max_rank),
        # 'Vanilla_SGD': lambda params, lr: torch.optim.SGD(params, lr=lr),
        'Torch_Adagrad': lambda params, lr: torch.optim.Adagrad(params, lr=lr),
    }
    models = {}
    all_train_losses = {}
    all_test_losses = {}
    results = []

    final_parameters = {}

    for task_name in tasks.keys():
        for opt_name, opt_fn in optimizers.items():
            for lr in learning_rates:
                # criterion = nn.CrossEntropyLoss()
                criterion = losses[task_name]
                print(opt_name)
                if opt_name in ["AdaGramFR_nosvd", "AdaGramFR_svd", "AdaGramPS"]:
                    for rank in ranks:
                        # model = LinearRegressionModel(dim_in=in_dims[0], dim_out=out_dims[0], seed=100)
                        # model = SimpleClassifier(input_dim=in_dims[0], output_dim=out_dims[0], seed=100)
                        model = tasks[task_name](in_dims[0], out_dims[0], 100)
                        optimizer = opt_fn(model.parameters(), lr, max_rank=rank, task=task_name)
                        epoch_results, test_loss = train_model_stochastic(
                            model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            results=results,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            num_epochs=num_epochs,
                            opt_name=opt_name,
                            lr=lr, 
                            r=rank,
                            batch_size=1,
                            data_seed=data_seed,
                            seed=42,
                            task_name=task_name

                        )
                        print("weight", model.state_dict()['linear.weight'].detach(),)
                        final_parameters[f"{opt_name}_rank_{rank}_lr_{lr}"] = {
                            'weights': model.state_dict()['linear.weight'].clone().detach(),
                            'bias': model.state_dict()['linear.bias'].clone().detach(),
                            'final_loss': test_loss
                        }
                else:
                    model = LinearRegressionModel(dim_in=in_dims[0], dim_out=2, seed=100)
                    # model = SimpleClassifier(input_dim=in_dims[0], output_dim=out_dims[0], seed=100)
                    optimizer = opt_fn(model.parameters(), lr)
                    epoch_results, test_loss = train_model_stochastic(
                            model=model,
                            optimizer=optimizer,
                            criterion=criterion,
                            results=results,
                            X_train=X_train,
                            y_train=y_train,
                            X_test=X_test,
                            y_test=y_test,
                            num_epochs=num_epochs,
                            opt_name=opt_name,
                            lr=lr, 
                            batch_size=1,
                            data_seed=data_seed,
                            seed=42,
                            task_name=task_name
                        )
        df = pd.DataFrame(results)
        df['loss'] = df['loss'].astype(float)
        # df['train_loss'] = df['train_loss'].astype(float)
        df.to_csv(f'results/{task_name}_ranks_all_diff_tracking_{in_dims[0]}_by_{out_dims[0]}.csv')
# print(f"end {dim}")

# df = pd.read_csv(f'results/LinReg_ranks_diff_tracking_{in_dims[0]}_by_{out_dims[0]}.csv')
df_1 = pd.read_csv(f'results/one_more_BinClass_1_3.csv')
df_1  = df_1.query("optimizer != 'Torch_Adagrad'")
df_2 = pd.read_csv(f'results/one_modeBinClass_4_5.csv')

df_3 = pd.read_csv(f'results/LinReg_ranks_1_3_diff_tracking_20_by_2.csv')
df_3  = df_3.query("optimizer != 'Torch_Adagrad'")
df_4 = pd.read_csv(f'results/LinReg_ranks_4_5_diff_tracking_20_by_2.csv')


df_bin = pd.concat([df_1, df_2])
df_linreg = pd.concat([df_3, df_4])





