import fenics as fe
import numpy as np
import torch
import sys

#sys.path.append("./src/AI/")

import src.AI.trainer as T
import src.formulations.formulation as F
import src.AI.neural_networks as nn


class Darcy_nn_factory(T.nn_factory):
    def __init__(self, training_params: T.training_params):
        device = super().__init__(training_params)
        return device, 4

    def calculate_PDE_loss(self, x_batch):
        loss = 0
        for x in x_batch:
            loss += self.PDE_loss_type(
                torch.matmul(
                    torch.from_numpy(
                        self.formulation.assemble_linear_system(x.numpy())
                    ),
                    self.single_net_eval(x),
                ),
                self.f,
            )

        return loss.mean()


# Given a triplet [eig_1, eig_2, theta] the trainer class
# trains a pair of darcy nets u,p solving
#       -div u = f
#    A grad p = u
#           p = 0  b.c.
# with A sym. pos. def. with eigen vals eig_1 and eig_2
# The matrix of eigvecs is the rotation matrix generated by theta
class Darcy_dual_nn_factory(Darcy_nn_factory):
    def __init__(
        self,
        formulation_params: F.formulation_params,
        nn_params: nn.nn_params,
        training_params: T.training_params,
    ):
        device, input_size = super().__init__(training_params)
        self.formulation = F.D.Darcy_dual_formulation(formulation_params)
        self.f = torch.from_numpy(self.formulation.get_rhs_vector())
        u_output_size = self.formulation.get_model_space().sub(0).dim()
        p_output_size = self.formulation.get_model_space().sub(1).dim()

        self.nets = {
            "u_net": nn.initialize_nn(input_size, u_output_size, nn_params).to(device),
            "p_net": nn.initialize_nn(input_size, p_output_size, nn_params).to(device),
        }
        self.define_optimizers(training_params.learn_rate)

    def get_nn_solver(self):
        return Darcy_dual_nn_solver().init_from_nets(
            self.nets, self.formulation.get_model_space()
        )


# Given a triplet [eig_1, eig_2, theta] the trainer class
# trains a pair of darcy nets u,p solving
# Encodes the PDE loss func for:
#       -div A grad p = f
#           p = 0  b.c.
# with A sym. pos. def. with eigen vals eig_1 and eig_2
# The matrix of eigvecs is the rotation matrix generated by theta
class Darcy_primal_nn_factory(Darcy_nn_factory):
    def __init__(
        self,
        formulation_params: F.formulation_params,
        nn_params: nn.nn_params,
        training_params: T.training_params,
    ):
        device, input_size = super().__init__(training_params)
        self.formulation = F.D.Darcy_primal_formulation(formulation_params)
        self.f = torch.from_numpy(self.formulation.get_rhs_vector())
        output_size = self.formulation.get_model_space().dim()

        self.nets = {
            "p_net": nn.initialize_nn(input_size, output_size, nn_params).to(device),
        }
        self.define_optimizers(training_params.learn_rate)

    def get_nn_solver(self):
        return Darcy_primal_nn_solver().init_from_nets(
            self.nets, self.formulation.get_model_space()
        )


class Darcy_primal_nn_solver(T.nn_solver):
    def __init__(self):
        super().__init__()
        pass

    def to_fenics(self, A_matrix_params: list) -> fe.Function:
        bias_term = torch.tensor([1], dtype=A_matrix_params.dtype)
        x_with_bias = torch.cat(
            (bias_term, torch.tensor(np.array(A_matrix_params))), dim=0
        )

        p_dofs = self.nets[0].forward(x_with_bias).detach().numpy()
        p = fe.Function(self.model_space)
        p.vector().set_local(p_dofs)
        return p

    def load(self, directory_path: str):
        mesh, degree, device = self.load_model_space_specs(directory_path)
        self.model_space = F.Darcy_primal_formulation(
            F.formulation_params(mesh=mesh, degree=degree)
        ).get_model_space()

        p_net = nn.dense_nn(
            input_size=4,
            hidden_size=T.dense_net_size_heuristic(4, self.model_space.dim()),
            output_size=self.model_space.dim(),
        )
        p_net.load_state_dict(torch.load(directory_path + "/p_net.pt"))
        p_net.to(device)
        self.nets = [p_net]
        return self


class Darcy_dual_nn_solver(T.nn_solver):
    def __init__(self):
        super().__init__()
        pass

    def init_from_nets(
        self,
        nets: dict,
        model_space: fe.FunctionSpace,
    ):
        self.nets = nets
        self.model_space = model_space
        return self

    def to_fenics(self, A_matrix_params: list) -> (fe.Function, fe.Function):
        bias_term = torch.tensor([1], dtype=A_matrix_params.dtype)
        x_with_bias = torch.cat(
            (bias_term, torch.tensor(np.array(A_matrix_params))), dim=0
        )
        u_dofs = self.u.forward(x_with_bias).detach().numpy()
        p_dofs = self.p.forward(x_with_bias).detach().numpy()

        (u, p) = fe.Function(self.model_space).split()

        u.vector().set_local(u_dofs)
        p.vector().set_local(p_dofs)
        return (u, p)

    def load(self, directory_path: str):
        mesh, degree, device = self.load_model_space_specs(directory_path)
        self.model_space = F.Darcy_dual_formulation(
            F.formulation_params(mesh=mesh, degree=degree)
        ).get_model_space()

        self.u_net = nn.dense_nn(
            input_size=4,
            hidden_size=int((2 / 3) * (4 + self.model_space.sub(0).dim())),
            output_size=self.model_space.sub(0).dim(),
        )
        self.u_net.load_state_dict(torch.load(directory_path + "/u_net.pt"))
        self.u_net.to(device)
        self.p_net = nn.dense_nn(
            input_size=4,
            hidden_size=int((2 / 3) * (4 + self.model_space.sub(1).dim())),
            output_size=self.model_space.sub(1).dim(),
        )
        self.p_net.load_state_dict(torch.load(directory_path + "/p_net.pt"))
        self.p_net.to(device)
        return self
