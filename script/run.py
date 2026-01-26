import torch
import numpy as np

from DKL.data.dataset import generate_2d_data
import DKL.data.transformations as f
from DKL.training.main_trainer import train_joint_model
from DKL.training.extra_trainer import train_naive_gp
from DKL.training.extra_trainer import train_ideal_gp
from DKL.utils.display import test_new_realization, plotting


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


## Hyperparameters for synthetic guassian process generation

N_training = 12000
test_grid_size = 32
noise_level = 0.05
scale = 5.0
function = f.transform_sin
lengthscale = 12.0

# Generate multiple data realizations on random points
common_data, realization = generate_2d_data(function, lengthscale= lengthscale,
                                             n_train_points= N_training ,
                                             test_grid_size=test_grid_size,
                                             scale=scale, noise_level=noise_level,
                                             device=device)


## Hyperparameters for Normalizing Flow and Gaussian process Learning

num_epochs = 1500
flow_learning_rate = 0.001 # Separate learning rate for flow
gp_learning_rate = 0.01 # Separate learning rate for GP/Likelihood (can be different)
num_flow_blocks = 12 # Number of blocks in the RealNVP flow model
freeze_gp_params = False # Set to True to freeze GP/Likelihood params during joint training

# Train the joint model with separate learning rates
trained_flow, trained_gp, trained_likelihood, loss_history  = train_joint_model(common_data,
                                                                                realization,
                                                                                num_epochs=num_epochs, 
                                                                                flow_lr=flow_learning_rate, 
                                                                                gp_lr=gp_learning_rate,
                                                                                device = device,
                                                                                num_flow_blocks=num_flow_blocks,
                                                                                patience=50, delta = 0.001, 
                                                                                no_learn_lengthscale=freeze_gp_params)

plotting(trained_flow,trained_gp,trained_likelihood,
         common_data, realization,
        function,loss_history,scale,
        savefig = True
         )

naive_gp_model, naive_likelihood = train_naive_gp(common_data, realization,
                                                device = device,
                                                num_epochs=300,lr=0.001,
                                                checkpoint_name='naive_gp_checkpoint.pth') 


ideal_gp_model, ideal_likelihood =  train_ideal_gp(common_data, realization,
                                                    function, lengthscale = lengthscale,
                                                    device = device,
                                                    num_epochs=300,lr=0.001,
                                                    checkpoint_name='ideal_gp_checkpoint.pth')

test_new_realization(trained_flow, trained_gp,
                    trained_likelihood, naive_gp_model,
                    naive_likelihood, ideal_gp_model, ideal_likelihood,
                    common_data, realization, scale,
                    savefig = True
)


