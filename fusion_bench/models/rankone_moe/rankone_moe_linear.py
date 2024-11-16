import logging
from typing import Dict, List, Tuple  # noqa: F401

import torch
import torch.nn.functional as F
from torch import Tensor, nn

log = logging.getLogger(__name__)

class ExpertNotTrainedError(Exception):
    pass

def _is_all_zeros(tensor: Tensor | List[Tensor]) -> bool:
    """
    Check if a tensor or a list of tensors are all zeros.

    Args:
        tensor (Tensor | List[Tensor]): A tensor or a list of tensors.

    Returns:
        bool: True if all elements are zeros, False otherwise.
    """
    if isinstance(tensor, Tensor):
        return torch.allclose(tensor, torch.zeros_like(tensor))
    else:
        return all(_is_all_zeros(t) for t in tensor)


def _svd(w: Tensor, full_matrices=True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform Singular Value Decomposition (SVD) on a tensor.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    v = vh.T
    return u, s, v


def svd(
    w: Tensor, full_matrices=True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Perform SVD on a tensor, optionally using a specified accelerator.

    Args:
        w (Tensor): The input tensor.
        full_matrices (bool): Whether to compute the full-sized U and V matrices.
        accelerator (str): The device to perform the computation on.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: The U, S, and V matrices from SVD.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    w = w.to(accelerator)
    u, s, v = _svd(w)
    return u.to(original_device), s.to(original_device), v.to(original_device)

def fun_joint_svd(w_list: List[Tensor], accelerator=None) -> Tuple[Tensor, Tensor, Tensor]:

    w = torch.cat(w_list, dim=1) # stacked_matrix
    original_device = w.device
    if accelerator is not None:
        w = w.to(accelerator)
    u_c, s_c, vh_c = torch.linalg.svd(w, full_matrices=False, driver="gesvd" if w.is_cuda else None)

    svd_list = []
    offset = 0
    for matrix in w_list:
        n_cols = matrix.size(1)
        u = u_c
        s = s_c
        vh_ = vh_c[:, offset:offset + n_cols]
        v = vh_.T
        svd_list.append([u.to(original_device), s.to(original_device), v.to(original_device)])

        offset += n_cols
    return svd_list

class Router(nn.Module):
    def __init__(
        self,
        input_features: int,
        w_diff_list: List[Tensor],
        k: int,
        svd_list=None,  # cached `svd_list`, pass it to avoid recomputing
        upscaling_accelerator=None,
    ):
        """
        Initialize the Router module.

        Args:
            input_features (int): The number of input features.
            w_diff_list (List[Tensor]): A list of weight difference tensors.
            k (int): The number of singular values to keep.
            svd_list (List[Tuple[Tensor, Tensor, Tensor]]): Cached SVD results.
            upscaling_accelerator (str): The device to perform the computation on.
        """
        super().__init__()
        self.input_features = input_features
        self.num_experts = len(w_diff_list)
        weights = []
        for i, w_diff in enumerate(w_diff_list):
            if svd_list is None:
                u, s, v = svd(w_diff, accelerator=upscaling_accelerator)
            else:
                u, s, v = svd_list[i]

            # u = u[:, :k]
            s = s[:k]
            v = v[:, :k]

            # if i == 0:
            # weights.append((s * v).T)
            weights.append(v.T) # Smile's default

        self.k = s.size(0)  # k is the actual k after truncation

        weights = (torch.stack(weights, dim=0).reshape(-1, self.input_features).contiguous())
        # weights = (torch.stack(weights, dim=0).reshape(-1, self.input_features).contiguous())
        self.weights = nn.Parameter(weights)  # weights should be a tensor of shape (num_experts * k, n)

    def forward(self, x: Tensor):
        """
        Forward pass of the Router module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The routing weights.
        """
        batch_size = x.size(0)
        routing_weights = F.linear(x, self.weights).view(batch_size, self.num_experts * self.k)
        # routing_weights = F.linear(x, self.weights).view(batch_size, 1 * self.k)
        return routing_weights

class RankOnePools(nn.Module):
    def __init__(self, model_list: List[nn.Linear], k: int, svd_cache=None):
        """
        Initialize the CompressedLinear module.

        Args:
            model (nn.Linear): A list of linear model to combine.
            k (int): The number of singular values to keep.
            svd_cache (List[Tuple[Tensor, Tensor, Tensor]]): Cached SVD results.
        """
        super().__init__()
        self.num_experts = len(model_list)

        weights_u = []
        weights_svh = []
        for i, model in enumerate(model_list):
            if svd_cache is None:
                u, s, v = svd(model.weight)
            else:
                u, s, v = svd_cache[i]
            if k > 0:
                u = u[:, :k]
                s = s[:k]
                v = v[:, :k]

            # if i == 0:
            weights_u.append(u)
            weights_svh.append((s * v).T)

        self.k = s.size(0)  # k is the actual k after truncation

        weights_u = (torch.stack(weights_u, dim=0).reshape(-1, self.num_experts * self.k).contiguous())
        weights_svh = (torch.stack(weights_svh, dim=0).reshape(self.num_experts * self.k, -1).contiguous())

        # weights_u = (torch.stack(weights_u, dim=0).reshape(-1, 1 * self.k).contiguous())
        # weights_svh = (torch.stack(weights_svh, dim=0).reshape(1 * self.k, -1).contiguous())

        self.u = nn.Parameter(weights_u)
        self.svh = nn.Parameter(weights_svh)

        # Ignored bias item
        self.register_parameter("bias", None)

    def forward(self, x, routing_weights, index):
        """
        Forward pass of the CompressedLinear module.

        Args:
            x (Tensor): The input tensor.
            routing_weights: output merging weight
            index: selected index
        Returns:
            Tensor: The output tensor.
        """
        svh_selected = self.svh[index] # shape: (6400, 2, 768) = (batch_size, top_k, hidden_size)
        svh_dot_product = (x.unsqueeze(1) * svh_selected).sum(dim=-1)  # (6400, 2)
        u_selected = self.u[:, index]  # (768, 6400, 2)
        u_selected = u_selected.permute(1, 2, 0)  # (6400, 2, 768)
        weighted_u_result = svh_dot_product.unsqueeze(-1) * u_selected  # (6400, 2, 768)
        # weighted_sum = (routing_weights.unsqueeze(-1) * weighted_u_result).sum(dim=1)  # (6400, 768)
        weighted_sum = weighted_u_result.sum(dim=1)  # (6400, 768)
        return weighted_sum

class RankOne_MoELinear(nn.Module):
    @torch.no_grad()
    def __init__(
        self,
        pretrained_model: nn.Linear,
        finetuned_models: List[nn.Linear],
        k: int,
        top_k: int = 1,
        full_matrices=True,
        upscaling_accelerator=None,
        routing_use_diff=True,
        expert_use_diff=True,
        joint_svd=False,
    ):
        """
        Initialize the MoELinear module.

        Args:
            pretrained_model (nn.Linear): The pretrained linear model.
            finetuned_models (List[nn.Linear]): A list of fine-tuned linear models.
            k (int): The number of singular values to keep for the experts.
            top_k (int): The number of top experts to select.
            full_matrices (bool): Whether to compute the full-sized U and V matrices.
            upscaling_accelerator (str): The device to perform the computation on.
            routing_use_diff (bool): Whether to use weight differences for routing.
        """
        super().__init__()
        self.num_experts = len(finetuned_models)
        self.top_k = top_k
        self.k = k
        self.routing_use_diff = routing_use_diff
        self.expert_use_diff = expert_use_diff
        self.joint_svd = joint_svd
        self.in_features = pretrained_model.in_features
        self.out_features = pretrained_model.out_features

        w_diff_list = [m.weight - pretrained_model.weight for m in finetuned_models]
        if _is_all_zeros(w_diff_list):
            # All fine-tuned models are identical to the pretrained model
            raise ExpertNotTrainedError()

        if routing_use_diff or k > 0:
            if joint_svd:
                svd_cache_list = fun_joint_svd(w_diff_list, accelerator=upscaling_accelerator)
            else:
                svd_cache_list = [
                    svd(w, full_matrices=full_matrices, accelerator=upscaling_accelerator)
                    for w in w_diff_list
                ]  # the svd cache list to avoid recomputing

        # construct the gate network
        if routing_use_diff:
            self.router = Router(
                input_features=self.in_features,
                w_diff_list=w_diff_list,
                k=k,
                svd_list=svd_cache_list,
                upscaling_accelerator=upscaling_accelerator,
            )
        else:
            self.router = Router(
                input_features=self.in_features,
                w_diff_list=[m.weight for m in finetuned_models],
                k=k,
                svd_list=None,
                upscaling_accelerator=upscaling_accelerator,
            )

        # construct rank-one expert pool
        if k > 0:
            if expert_use_diff:
                # diff experts
                for m, w_diff in zip(finetuned_models, w_diff_list):
                    m.weight.data = w_diff
                experts = RankOnePools(finetuned_models, k, svd_cache=svd_cache_list)
            else:
                experts = RankOnePools(finetuned_models, k, svd_cache=None)
            self.experts = experts
        else:
            raise ValueError("k must be greater than 0.")

        # Ignored bias item in fine-tuned models
        # if pretrained_model.bias is not None:
        #     for m in experts:
        #         m.bias.data = m.bias.data - pretrained_model.bias

        # assign the pretrained model (the shared part)
        self.pretrained_model = pretrained_model

    def forward(self, hidden_states: Tensor):
        """
        Forward pass of the MoELinear module.

        Args:
            hidden_states (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        if self.expert_use_diff:
            pretrained_out = self.pretrained_model(hidden_states)

        input_shape = hidden_states.size()
        hidden_states = hidden_states.view(-1, self.in_features)

        routing_weights = self.router(hidden_states)
        # routing_weights = F.softmax(routing_weights, dim=1)

        # sample the expert according to the routing weights
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expert_layer = self.experts(hidden_states, routing_weights, selected_experts)
        final_hidden_states = expert_layer.reshape(*input_shape[:-1], self.out_features)

        if self.expert_use_diff:
            final_hidden_states = pretrained_out + final_hidden_states

        return final_hidden_states

    @property
    def weight(self):
        """
        Mimic linear layer. Bacause in some cases, user might indicate the device (or dtype of parameters) of the linear layer using `linear_layer.weight.device`
        """
        return self.pretrained_model.weight

    @property
    def bias(self):
        return self.pretrained_model.bias

    def __repr__(self):
        return (
            f"SingularMoELinear("
            f"in_features={self.pretrained_model.in_features}, "
            f"out_features={self.pretrained_model.out_features}, "
            f"num_experts={self.num_experts}, "
            f"top_k={self.top_k}, "
            f"k={self.k}"
            f")"
        )
