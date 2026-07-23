# fl_framework/components/aggregators/compress.py

from typing import List, Tuple
import torch
from pytorch_model_summary import summary

from .base_aggregator import BaseAggregator

class CompressAggregator(BaseAggregator):
    def __init__(self, m: int, r: int, k: int): # 添加 k 参数
        self.m = m
        self.r = r
        self.k = k # K: 每次迭代中保留的行数
        self.d = None  # Total flattened gradient dimension (for a single client)
        self.n = None  # Number of columns for G_i matrix: d / m
        self.V = None  # Random projection matrix V (shape n x r)
        self.reference_shapes = None # Stores original gradient shapes for unflattening

        # 用于存储客户端的原始 G_i 矩阵，以便在选择 I_t 后模拟客户端的第二阶段压缩
        # 注意: 在实际分布式设置中，服务器不会直接拥有这些，而是客户端在收到 I_t 后自行计算并发送稀疏梯度。
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
            unflattened.append(flattened_gradient[current_idx : current_idx + num_elements].view(shape))
            current_idx += num_elements
        return unflattened

    def _get_projection_matrix_V(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.V is None:
            if self.n is None:
                raise RuntimeError("无法生成 V；'n' (梯度矩阵列数) 未设置。请先调用 aggregate 方法。")
            self.V = torch.randn(self.n, self.r, device=device, dtype=dtype)
            print(f"生成的投影矩阵 V 形状: {self.V.shape}")
        return self.V

    @torch.no_grad()
    def aggregate(
        self,
        all_gradients: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        严格按照 ARC-Top-K 论文 (算法 1) 的流程聚合梯度。

        Args:
            all_gradients (List[List[torch.Tensor]]): 一个列表，其中每个内部列表
                包含单个客户端的每层梯度。这些是客户端的原始梯度。

        Returns:
            List[torch.Tensor]: 聚合后的稀疏梯度，已解展平回原始模型的层梯度列表。
        """
        if not all_gradients:
            print("警告: 收到空梯度列表，返回空列表。")
            return []
        
        num_clients = len(all_gradients)

        ref_client_grad_layers = all_gradients[0]
        if not ref_client_grad_layers:
            print("警告: 第一个客户端的梯度列表为空，返回空列表。")
            return []

        device = ref_client_grad_layers[0].device
        dtype = ref_client_grad_layers[0].dtype

        # ======================================================================
        # 阶段 0: 初始化/参数检查 (服务器端)
        # ======================================================================
        if self.d is None:
            self.reference_shapes = [g.shape for g in ref_client_grad_layers]
            flattened_grad_ref = self._flatten_gradients(ref_client_grad_layers)
            self.d = flattened_grad_ref.numel()
            print(f"DEBUG: Actual total gradient dimension (d): {self.d}") # <<<<<<< 添加这行
            # input()
            
            # --- 以下是计算 d 的因子的代码，请一并添加 ---
            import math
            def get_factors(n):
                factors = set()
                for i in range(1, int(math.sqrt(n)) + 1):
                    if n % i == 0:
                        factors.add(i)
                        factors.add(n // i)
                return sorted(list(factors))

            factors_of_d = get_factors(self.d)
            print(f"DEBUG: Factors of d: {factors_of_d}") # <<<<<<< 添加这行
            # --- 因子计算代码结束 ---

            if self.d % self.m != 0:
                raise ValueError(f"梯度总维度 d ({self.d}) 必须能被重塑行数 m ({self.m}) 整除。")
            self.n = self.d // self.m


            if self.d % self.m != 0:
                raise ValueError(f"梯度总维度 d ({self.d}) 必须能被重塑行数 m ({self.m}) 整除。")
            self.n = self.d // self.m
            
            # 检查 K 是否有效
            if self.k > self.m:
                raise ValueError(f"Top-K (k={self.k}) 不能大于重塑矩阵的行数 m ({self.m})。")

            print(f"初始化聚合器 (严格 ARC-Top-K): d={self.d}, m={self.m}, n={self.n}, r={self.r}, k={self.k}")
        
        V = self._get_projection_matrix_V(device, dtype)
        self._cached_client_G_matrices = [] # 清空缓存，准备存储当前轮次的 G_i

        collected_projections_P_i = [] # 用于存储每个客户端的 P_t^(i) 矩阵

        # ======================================================================
        # 阶段 1: 客户端局部投影 (模拟在服务器上完成)
        # 服务器收集 P_t^(i) 并缓存 G_t^(i) 以备后用
        # ======================================================================
        for i, client_grad_layers in enumerate(all_gradients):
            if len(client_grad_layers) != len(self.reference_shapes):
                raise ValueError(f"客户端 {i} 的梯度层数不一致。期望 {len(self.reference_shapes)}，得到 {len(client_grad_layers)}。")

            client_flat_grad = self._flatten_gradients(client_grad_layers)

            if client_flat_grad.numel() != self.d:
                raise ValueError(f"客户端 {i} 梯度维度不一致。期望 {self.d}，得到 {client_flat_grad.numel()}。")

            # 1. 重塑为矩阵 G_t^(i) ∈ R^(m x n)
            G_i = client_flat_grad.reshape(self.m, self.n)
            self._cached_client_G_matrices.append(G_i) # 缓存原始 G_i 矩阵

            # 2. 梯度投影: P_t^(i) ← G_t^(i) V
            P_i = math.sqrt(self.r) * torch.matmul(G_i, V) # P_i 形状为 (m, r)
            collected_projections_P_i.append(P_i)

        if not collected_projections_P_i:
            return []

        # ======================================================================
        # 阶段 2: 服务器端选择重要行
        # ======================================================================
        # 1. 聚合所有 P_t^(i) 矩阵 (求平均) 得到 P_t (对应论文中的 P_t)
        P_t = torch.zeros_like(
            collected_projections_P_i[0],
            device=device,
            dtype=dtype
        )
        for p_i in collected_projections_P_i:
            P_t.add_(p_i)
        P_t.div_(num_clients) # P_t 形状为 (m, r)

        # 2. 计算每行的重要性分数 Σ_t = diag(P_t P_t^T)
        # P_t P_t^T 形状为 (m, m)
        # diag 会提取对角线元素，得到一个 (m,) 形状的向量
        sigma_t = torch.diag(torch.matmul(P_t, P_t.T)) # sigma_t 形状为 (m,)

        # 3. 选择 K 个最重要的行索引 I_t
        # torch.topk 返回 (值, 索引)，我们只需要索引
        _, topk_indices_It = torch.topk(sigma_t, self.k) # topk_indices_It 形状为 (k,)
        # I_t 包含的是 0 到 m-1 之间的 k 个索引

        # ======================================================================
        # 阶段 3: 客户端局部压缩 (模拟在服务器上完成)
        # ======================================================================
        collected_sparse_flat_grads = [] # 用于存储每个客户端的 Clocal(g_t^(i)) 稀疏向量

        for G_i_original in self._cached_client_G_matrices: # 使用缓存的原始 G_i 矩阵
            # 创建一个全零矩阵，用于存储局部压缩后的 G
            # 形状与原始 G_i 相同 (m, n)
            Clocal_G_i = torch.zeros_like(G_i_original, device=device, dtype=dtype)
            
            # 将 I_t 对应行的原始梯度复制到 Clocal_G_i
            # 使用高级索引，只复制选中的 K 行
            Clocal_G_i[topk_indices_It, :] = G_i_original[topk_indices_It, :]

            # 将 Clocal_G_i 展平为稀疏向量 Clocal(g_t^(i))
            # 注意: 这里虽然展平了，但实际上只有 K*n 个非零元素 (假设没有元素恰好为零)
            Clocal_flat_grad_i = Clocal_G_i.view(-1) # 形状 (d,)

            collected_sparse_flat_grads.append(Clocal_flat_grad_i)

        self._cached_client_G_matrices.clear() # 清空缓存，释放内存

        if not collected_sparse_flat_grads:
            return []

        # ======================================================================
        # 阶段 4: 服务器端聚合最终压缩梯度
        # ======================================================================
        # 聚合所有客户端的 Clocal(g_t^(i)) (求平均) 得到 C(g_t)
        C_g_t_flat = torch.zeros_like(
            collected_sparse_flat_grads[0],
            device=device,
            dtype=dtype
        )
        for sparse_grad_i in collected_sparse_flat_grads:
            C_g_t_flat.add_(sparse_grad_i)
        C_g_t_flat.div_(num_clients) # C_g_t_flat 形状为 (d,)

        # 将 C(g_t) 扁平向量解展平回原始层梯度列表
        reconstructed_gradients = self._unflatten_gradients(C_g_t_flat, self.reference_shapes)

        return reconstructed_gradients

    def get_state(self) -> dict:
        """
        返回聚合器的当前状态，包括 m, r, k, d, n, V 和 reference_shapes。
        """
        state = {
            'm': self.m,
            'r': self.r,
            'k': self.k, # 新增 k
            'd': self.d,
            'n': self.n,
            'V': self.V,
            'reference_shapes': self.reference_shapes
        }
        return state
    
    def set_state(self, state: dict):
        """
        设置聚合器的状态。
        """
        if state is None:
            print("警告: 尝试设置空状态。聚合器将保持未初始化状态或当前状态。")
            return

        self.m = state.get('m', self.m) 
        self.r = state.get('r', self.r)
        self.k = state.get('k', self.k) # 新增 k
        self.d = state.get('d')
        self.n = state.get('n')
        self.V = state.get('V')
        self.reference_shapes = state.get('reference_shapes')
        
        if self.d is not None and self.n is not None and self.V is not None and self.reference_shapes is not None:
            print(f"CompressAggregator 状态已加载: d={self.d}, n={self.n}, r={self.r}, k={self.k}, V 形状={self.V.shape}")
        else:
            print("警告: 状态部分加载，某些关键属性可能仍为 None。")

