# fl_framework/components/aggregators/compress.py

from typing import List, Tuple
import torch
import math
import random
from pytorch_model_summary import summary

from .base_aggregator import BaseAggregator


class CompressAggregator(BaseAggregator):
    def __init__(self, m: int, r: int, k: int, s: int = 1):
        """
        ARC-Top-K + 分组聚合压缩器

        Args:
            m (int): 重塑矩阵的行数
            r (int): 投影维度
            k (int): Top-k 行选择数
            s (int): 每组客户端数（组内 mean，组间 median）
        """
        self.m = m
        self.r = r
        self.k = k
        self.s = s  # 分组大小
        self.d = None  # Total flattened gradient dimension
        self.n = None  # Number of columns for G_i matrix: d / m
        self.V = None  # Random projection matrix V (shape n x r)
        self.reference_shapes = None

        # 用于存储客户端的原始 G_i 矩阵，以便后续压缩模拟
        self._cached_client_G_matrices: List[torch.Tensor] = []

    def _flatten_gradients(self, gradients_per_layer: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.view(-1) for g in gradients_per_layer])

    def _unflatten_gradients(self, flattened_gradient: torch.Tensor, ref_shapes: List[torch.Size]) -> List[torch.Tensor]:
        if not ref_shapes:
            return []
        unflattened = []
        current_idx = 0
        for shape in ref_shapes:
            num_elements = shape.numel()
            if current_idx + num_elements > flattened_gradient.numel():
                raise ValueError("Flattened gradient has insufficient elements for unflattening with provided shapes.")
            unflattened.append(flattened_gradient[current_idx: current_idx + num_elements].view(shape))
            current_idx += num_elements
        return unflattened

    def _get_projection_matrix_V(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.V is None:
            if self.n is None:
                raise RuntimeError("无法生成 V；'n' (梯度矩阵列数) 未设置。请先调用 aggregate 方法。")
            self.V = torch.randn(self.n, self.r, device=device, dtype=dtype)
            print(f"生成的投影矩阵 V 形状: {self.V.shape}")
        return self.V

    def _grouped_aggregate(self, P_stack: torch.Tensor) -> torch.Tensor:
        """
        按客户端分组聚合：
        - 组内 mean
        - 组间 median
        """
        num_clients = P_stack.size(0)
        shuffled_idx = torch.randperm(num_clients)
        P_stack = P_stack[shuffled_idx, ...]  # (num_clients, m, r)
        num_groups = (num_clients + self.s - 1) // self.s
        group_P_list = []

        for g in range(num_groups):
            start = g * self.s
            end = min((g + 1) * self.s, num_clients)
            group_clients = P_stack[start:end, ...]  # (<=s, m, r)

            # ---- 组内 mean ----
            P_g = group_clients.mean(dim=0)  # (m, r)
            group_P_list.append(P_g.unsqueeze(0))

        # ---- 组间 median ----
        P_groups = torch.cat(group_P_list, dim=0)  # (num_groups, m, r)
        P_t = torch.median(P_groups, dim=0).values  # (m, r)
        return P_t

    @torch.no_grad()
    def aggregate(
        self,
        all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        严格按照 ARC-Top-K 论文 (算法 1) + 客户端分组 聚合梯度。

        Args:
            all_gradients (List[List[torch.Tensor]]): 每个客户端的每层梯度

        Returns:
            List[torch.Tensor]: 聚合后的稀疏梯度，已解展平回原始层结构
        """
        if not all_gradients:
            print("警告: 收到空梯度列表，返回空列表。")
            return []

        # num_clients = len(all_gradients)

        ref_client_grad_layers = all_gradients[0]
        self.reference_shapes = [g.shape for g in ref_client_grad_layers]
        if not ref_client_grad_layers:
            print("警告: 第一个客户端的梯度列表为空，返回空列表。")
            return []

        device = ref_client_grad_layers[0].device
        dtype = ref_client_grad_layers[0].dtype
        
        # flatten_grad = []
        # for grad in all_gradients:
        #     flatten_grad.append(self._flatten_gradients(grad))
        # stack_grad = torch.stack(flatten_grad, dim=0)
        # agg_grad = self._grouped_aggregate(stack_grad)
        # agg_grad = torch.median(stack_grad, dim=0).values
        # return self._unflatten_gradients(agg_grad, self.reference_shapes)

        # ======================================================================
        # 阶段 0: 初始化
        # ======================================================================
        if self.d is None:
            self.reference_shapes = [g.shape for g in ref_client_grad_layers]
            flattened_grad_ref = self._flatten_gradients(ref_client_grad_layers)
            self.d = flattened_grad_ref.numel()
            print(f"DEBUG: Actual total gradient dimension (d): {self.d}")

            def get_factors(n):
                factors = set()
                for i in range(1, int(math.sqrt(n)) + 1):
                    if n % i == 0:
                        factors.add(i)
                        factors.add(n // i)
                return sorted(list(factors))

            factors_of_d = get_factors(self.d)
            print(f"DEBUG: Factors of d: {factors_of_d}")

            if self.d % self.m != 0:
                raise ValueError(f"梯度总维度 d ({self.d}) 必须能被 m ({self.m}) 整除。")
            self.n = self.d // self.m

            if self.k > self.m:
                raise ValueError(f"Top-K (k={self.k}) 不能大于 m ({self.m})。")

            print(f"初始化聚合器: d={self.d}, m={self.m}, n={self.n}, r={self.r}, k={self.k}, s={self.s}")

        V = self._get_projection_matrix_V(device, dtype)
        self._cached_client_G_matrices = []

        client_P_list = []

        # ======================================================================
        # 阶段 1: 客户端局部投影
        # ======================================================================
        for i, client_grad_layers in enumerate(all_gradients):
            if len(client_grad_layers) != len(self.reference_shapes):
                raise ValueError(f"客户端 {i} 的梯度层数不一致。期望 {len(self.reference_shapes)}，得到 {len(client_grad_layers)}。")

            client_flat_grad = self._flatten_gradients(client_grad_layers)
            if client_flat_grad.numel() != self.d:
                raise ValueError(f"客户端 {i} 梯度维度不一致。期望 {self.d}，得到 {client_flat_grad.numel()}。")

            G_i = client_flat_grad.reshape(self.m, self.n)
            self._cached_client_G_matrices.append(G_i)

            P_i = math.sqrt(self.r) * torch.matmul(G_i, V)  # (m, r)
            client_P_list.append(P_i)

        if not client_P_list:
            return []

        # ======================================================================
        # 阶段 2: 分组聚合 (组内 mean + 组间 median)
        # ======================================================================
        n = len(client_P_list)
        P_stack = torch.stack(client_P_list, dim=0)  # (num_clients, m, r)
        P_t = self._grouped_aggregate(P_stack)     # (m, r)

        # ======================================================================
        # 阶段 3: Top-k 行选择
        # ======================================================================
        sigma_t = torch.diag(torch.matmul(P_t, P_t.T))  # (m,)
        _, topk_indices_It = torch.topk(sigma_t, self.k)

        # ======================================================================
        # 阶段 4: 客户端局部压缩 (模拟)
        # ======================================================================
        collected_sparse_flat_grads = []
        for G_i_original in self._cached_client_G_matrices:
            Clocal_G_i = torch.zeros_like(G_i_original, device=device, dtype=dtype)
            Clocal_G_i[topk_indices_It, :] = G_i_original[topk_indices_It, :]
            Clocal_flat_grad_i = Clocal_G_i.view(-1)
            collected_sparse_flat_grads.append(Clocal_flat_grad_i)

        self._cached_client_G_matrices.clear()

        if not collected_sparse_flat_grads:
            return []

        # ======================================================================
        # 阶段 5: 服务器端聚合最终压缩梯度
        # ======================================================================
        # C_g_t_flat = torch.zeros_like(collected_sparse_flat_grads[0], device=device, dtype=dtype)
        # for sparse_grad_i in collected_sparse_flat_grads:
        #     C_g_t_flat.add_(sparse_grad_i)
        # C_g_t_flat.div_(num_clients)
        collected_sparse_flat_grads = torch.stack(collected_sparse_flat_grads, dim=0)
        C_g_t_flat = self._grouped_aggregate(collected_sparse_flat_grads)

        reconstructed_gradients = self._unflatten_gradients(C_g_t_flat, self.reference_shapes)
        return reconstructed_gradients

    def get_state(self) -> dict:
        return {
            'm': self.m,
            'r': self.r,
            'k': self.k,
            's': self.s,
            'd': self.d,
            'n': self.n,
            'V': self.V,
            'reference_shapes': self.reference_shapes
        }

    def set_state(self, state: dict):
        if state is None:
            print("警告: 尝试设置空状态。")
            return

        self.m = state.get('m', self.m)
        self.r = state.get('r', self.r)
        self.k = state.get('k', self.k)
        self.s = state.get('s', self.s)
        self.d = state.get('d')
        self.n = state.get('n')
        self.V = state.get('V')
        self.reference_shapes = state.get('reference_shapes')

        if self.d is not None and self.n is not None and self.V is not None and self.reference_shapes is not None:
            print(f"CompressAggregator 状态已加载: d={self.d}, n={self.n}, r={self.r}, k={self.k}, s={self.s}, V 形状={self.V.shape}")
        else:
            print("警告: 状态部分加载，某些关键属性可能仍为 None。")
