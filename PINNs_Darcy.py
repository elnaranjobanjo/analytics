import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd


class Darcy_eigen_nn(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=1):
        super(Darcy_eigen_nn, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc6 = nn.Linear(hidden_size, output_size)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x
    
def pinn_loss(u_model, v_model, p_model, x, y, Re, lossfun):
    """
    Function to calculate the physics informed loss function
    """
    xy = torch.cat([x, y], dim=1)

    # Compute the predicted values of velocity and pressure
    u_pred = u_model(xy)
    v_pred = v_model(xy)
    p_pred = p_model(xy)

    # Compute the required derivatives with respect to x and y
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

    u_y = torch.autograd.grad(u_pred, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    v_x = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]

    v_y = torch.autograd.grad(v_pred, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    p_x = torch.autograd.grad(p_pred, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
    p_y = torch.autograd.grad(p_pred, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]

    # velocity divergence
    div_uv = u_x + v_y
    lap_u = u_xx + u_yy
    lap_v = v_xx + v_yy

    # Compute the residuals of Navier-Stokes equation
    res_u = u_pred * u_x + v_pred * u_y + p_x - 1/Re * lap_u
    res_v = u_pred * v_x + v_pred * v_y + p_y - 1/Re * lap_v
    res_cont = div_uv

    # Compute the physics-driven loss
    physics_loss = lossfun(res_u, torch.zeros_like(res_u)) + \
                   lossfun(res_v, torch.zeros_like(res_v)) + \
                   lossfun(res_cont, torch.zeros_like(res_cont))

    return physics_loss


def data_loss(u_model, v_model, p_model, x_data, y_data, u_data, v_data, p_data, lossfun):
    """
    Function to calculate data loss function
    """
    xy = torch.cat([x_data, y_data], dim=1)

    # Compute the predicted values of velocity and pressure
    u_pred = u_model(xy)
    v_pred = v_model(xy)
    p_pred = p_model(xy)
    # Compute the data-driven loss
    return lossfun(u_pred, u_data) + lossfun(v_pred, v_data) + lossfun(p_pred, p_data)


def boundary_loss(u_model, v_model, x_bc, y_bc, u_bc, v_bc, lossfun):
    """
    Function to calculate loss at boundary
    """
    xy_bc = torch.cat([x_bc, y_bc], dim=1)

    # Compute the predicted values of velocity and pressure at the boundary
    u_pred = u_model(xy_bc)
    v_pred = v_model(xy_bc)

    return lossfun(u_pred, u_bc) + lossfun(v_pred, v_bc)
    

def get_train_loop(u_model, v_model, p_model, u_optimizer, v_optimizer, p_optimizer, lossfun, Re):
    """
    Wrapper to return function that trains the PINN for one step
    """
    def train_step(batch_loader, x_bound, y_bound, u_bound, v_bound, x_data, y_data, p_data, u_data, v_data):
        average_loss = []

        u_model.train()
        v_model.train()
        p_model.train()
        for batch_idx, batch in enumerate(batch_loader):
            # we might have to send the batch data to the device
            x, y = batch
            x.requires_grad = True
            y.requires_grad = True

            u_optimizer.zero_grad()
            v_optimizer.zero_grad()
            p_optimizer.zero_grad()

            physics_loss = pinn_loss(u_model, v_model, p_model, x, y, Re, lossfun)
            loss_data = data_loss(u_model, v_model, p_model, x_data, y_data, u_data, v_data, p_data, lossfun)
            bc_loss = boundary_loss(u_model, v_model, x_bound, y_bound, u_bound, v_bound, lossfun)

            total_loss = 10*physics_loss + loss_data + bc_loss
            total_loss.backward()

            average_loss.append([total_loss.item(), physics_loss.item(), loss_data.item(), bc_loss.item()])

            u_optimizer.step()
            v_optimizer.step()
            p_optimizer.step()

        return np.array(average_loss).mean(axis=0)

    return train_step

def main():
    epochs = 10
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # instantiate networks
    u_net = Darcy_eigen_nn().to(device)
    p_net = Darcy_eigen_nn().to(device)

    u_optimizer = torch.optim.Adam(u_net.parameters(), lr=5e-4)
    p_optimizer = torch.optim.Adam(p_net.parameters(), lr=5e-4)
    lossfun = nn.MSELoss()

    X_train = pd.read_csv('test_data_X.csv').to_csv()
    Y_train = pd.read_csv('test_data_Y.csv').to_csv()

    train_step = get_train_loop(u_net, p_net, u_optimizer, p_optimizer, lossfun, args.Re)

    # start training
    for epoch in range(epoch):
        average_loss = train_step(X_train,Y_train)
        print(f'Epoch: {epoch}, Total loss: {average_loss[0]}, Physics loss: {average_loss[1]}, '
              f'Data loss: {average_loss[2]}, Boundary loss: {average_loss[3]}')

    # save trained models
    torch.save(u_net.state_dict(), os.path.join('TrainedNetworks', 'lid_u_velocity_net.pt'))
    torch.save(p_net.state_dict(), os.path.join('TrainedNetworks', 'lid_pressure_net.pt'))