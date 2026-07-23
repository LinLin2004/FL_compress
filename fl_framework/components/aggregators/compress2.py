# fl_framework/components/aggregators/compress.py

import math
from typing import List, Tuple
import torch
# from pytorch_model_summary import summary # 此库在提供代码中未使用，如果其他地方没有用，可以移除。

from .base_aggregator import BaseAggregator

class CompressAggregator(BaseAggregator):
    # 修改一：在__init__中添加 byzantine_alpha 参数
    def __init__(self, m: int, r: int, k: int, byzantine_alpha: float = 0.0): # 添加 k 和 byzantine_alpha 参数
        self.m = m
        self.r = r
        self.k = k # K: 每次迭代中保留的行数 (用于FABA后的Top-K选择)
        self.byzantine_alpha = byzantine_alpha # α: 假设的拜占庭工作者比例 (用于 FABA)
        self.d = None  # Total flattened gradient dimension (for a single client)
        self.n = None  # Number of columns for G_i matrix: d / m
        self.V = None  # Random projection matrix V (shape n x r)
        self.reference_shapes = None # Stores original gradient shapes for unflattening

        # 用于存储客户端的原始 G_i 矩阵，以便在选择 I_t 后模拟客户端的第二阶段压缩
        # 注意: 在实际分布式设置中，服务器不会直接拥有这些，而是客户端在收到 I_t 后自行计算并发送稀疏梯度。
        self._cached_client_G_matrices: List[torch.Tensor] = []

        # 检查 byzantine_alpha 的有效性，FABA 论文假设 alpha < 0.5
        if not (0 <= self.byzantine_alpha < 0.5):
            # 允许 alpha = 0 表示不使用 FABA，但如果超出 (0, 0.5) 范围则给出警告
            if self.byzantine_alpha != 0:
                print(f"警告: byzantine_alpha ({self.byzantine_alpha}) 超出了 FABA 论文建议的严格范围 (0 <= α < 0.5)。")

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
        集成了 FABA (Fast Aggregation against Byzantine Attacks) 算法进行拜占庭节点过滤。

        Args:
            all_gradients (List[List[torch.Tensor]]): 一个列表，其中每个内部列表
                包含单个客户端的每层梯度。这些是客户端的原始梯度。

        Returns:
            List[torch.Tensor]: 聚合后的稀疏梯度，已解展平回原始模型的层梯度列表。
        """
        if not all_gradients:
            print("警告: 收到空梯度列表，返回空列表。")
            return []
        
        num_initial_clients = len(all_gradients) # 存储初始客户端数量
        if num_initial_clients == 0:
            print("警告: 收到空梯度列表，返回空列表。")
            return []

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
            print(f"DEBUG: Actual total gradient dimension (d): {self.d}")
            
            # --- 以下是计算 d 的因子的代码，请一并添加 ---
            def get_factors(n):
                factors = set()
                for i in range(1, int(math.sqrt(n)) + 1):
                    if n % i == 0:
                        factors.add(i)
                        factors.add(n // i)
                return sorted(list(factors))

            factors_of_d = get_factors(self.d)
            print(f"DEBUG: Factors of d: {factors_of_d}")
            # --- 因子计算代码结束 ---

            if self.d % self.m != 0:
                raise ValueError(f"梯度总维度 d ({self.d}) 必须能被重塑行数 m ({self.m}) 整除。")
            self.n = self.d // self.m
            
            # 检查 K 是否有效
            if self.k > self.m:
                raise ValueError(f"Top-K (k={self.k}) 不能大于重塑矩阵的行数 m ({self.m})。")
            
            # FABA 特定检查: 确保潜在移除后仍有足够的客户端
            # FABA 假设 alpha < 0.5，意味着超过一半的客户端应该是诚实的。
            num_to_remove_potential = int(self.byzantine_alpha * num_initial_clients)
            if self.byzantine_alpha > 0 and (num_initial_clients - num_to_remove_potential <= 0):
                raise ValueError(f"根据 byzantine_alpha ({self.byzantine_alpha}) 和客户端数量 ({num_initial_clients})，FABA 过滤将移除所有或更多客户端。这通常意味着 byzantine_alpha 过高或客户端数量不足，或者初始客户端数量为零。")

            print(f"初始化聚合器 (ARC-Top-K {'+ FABA' if self.byzantine_alpha > 0 else ''}): d={self.d}, m={self.m}, n={self.n}, r={self.r}, k={self.k}, byzantine_alpha={self.byzantine_alpha}")
        
        V = self._get_projection_matrix_V(device, dtype)
        # 注意: 这里的 _cached_client_G_matrices 将在 FABA 过滤阶段被复制并修改。
        # 此处清空是为了确保每次聚合都从干净状态开始。
        self._cached_client_G_matrices = [] 

        collected_projections_P_i = [] # 用于存储每个客户端的 P_t^(i) 矩阵 (未经 FABA 过滤)

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
            self._cached_client_G_matrices.append(G_i) # 缓存原始 G_i 矩阵，稍后 FABA 会对复制的列表进行过滤

            # 2. 梯度投影: P_t^(i) ← G_t^(i) V
            P_i = math.sqrt(self.r) * torch.matmul(G_i, V) # P_i 形状为 (m, r)
            collected_projections_P_i.append(P_i)

        if not collected_projections_P_i:
            print("警告: 投影列表为空，返回空列表。")
            return []
        
        # ======================================================================
        # 修改一：###筛选byzantine节点FABA pi算
        # FABA (Fast Aggregation against Byzantine Attacks) 过滤 (服务器端)
        # 基于论文中的算法 1
        # ======================================================================
        
        # 复制列表，以便在过滤过程中修改它们
        # collected_projections_P_i 包含 P_t^(i)
        # self._cached_client_G_matrices 包含 G_t^(i) (用于稍后的稀疏化)
        current_P_i_list = list(collected_projections_P_i)
        # 创建一个新的列表用于 G_i 矩阵，它将与 P_i 一起被过滤。
        # 这确保了过滤后 G_i_original 与正确的 P_i 对应。
        current_G_i_list_filtered = list(self._cached_client_G_matrices)
        
        # 计算要移除的客户端数量
        # 根据 Algorithm 1 Step 1: "If k < α * n, continue, else go to Step 5;"
        # k 是已移除的客户端数量，循环将移除 (α * num_initial_clients) 个客户端。
        num_to_remove = int(self.byzantine_alpha * num_initial_clients)
        
        if num_to_remove == 0:
            print(f"DEBUG: FABA: byzantine_alpha ({self.byzantine_alpha}) 为 0，不移除任何客户端。")
        else:
            # 确保在移除后至少剩余一个客户端
            if len(current_P_i_list) - num_to_remove <= 0:
                raise ValueError(f"FABA 尝试移除所有或更多客户端。当前客户端数量: {len(current_P_i_list)}, 计划移除: {num_to_remove}。这通常意味着 byzantine_alpha 过高或客户端数量过少。")
            
            print(f"DEBUG: FABA: 将从 {num_initial_clients} 个客户端梯度中移除 {num_to_remove} 个。")
            for _ in range(num_to_remove):
                # Algorithm 1 Step 2: Compute mean of Gg as g0
                # current_P_i_list 扮演 Gg 的角色
                mean_P = torch.stack(current_P_i_list).mean(dim=0)

                # Algorithm 1 Step 3: For every gradient in Gg, compute the difference between g0 and it.
                # Delete the one that has the largest difference from G;
                differences = []
                for P_item in current_P_i_list:
                    # 使用 Frobenius 范数计算矩阵差异
                    diff = torch.norm(P_item - mean_P, p='fro')
                    differences.append(diff.item())

                # 找到差异最大的梯度（客户端）的索引
                # 如果有多个最大差异，torch.argmax 返回第一个。
                idx_to_remove = torch.argmax(torch.tensor(differences)).item()

                # 从列表中移除该客户端的 P_i 和 G_i
                current_P_i_list.pop(idx_to_remove)
                current_G_i_list_filtered.pop(idx_to_remove)
            print(f"DEBUG: FABA: 已移除 {num_to_remove} 个客户端梯度。剩余客户端: {len(current_P_i_list)}。")

        # 更新客户端数量为 FABA 过滤后的数量
        num_clients_after_faba = len(current_P_i_list)
        if num_clients_after_faba == 0:
            print("警告: FABA 过滤后没有剩余客户端，返回空列表。")
            self._cached_client_G_matrices.clear() # 清空缓存
            return []

        # ======================================================================
        # 阶段 2: 服务器端选择重要行 (现在基于 FABA 过滤后的投影)
        # ======================================================================
        # 1. 聚合所有 *FABA 过滤后* 的 P_t^(i) 矩阵 (求平均) 得到 P_t (对应论文中的 P_t)
        # Algorithm 1 Step 5: Compute the mean of Gg as the aggregation result at time t At;
        P_t = torch.stack(current_P_i_list).mean(dim=0) # P_t 形状为 (m, r)

        # 2. 计算每行的重要性分数 Σ_t = diag(P_t P_t^T)
        sigma_t = torch.diag(torch.matmul(P_t, P_t.T)) # sigma_t 形状为 (m,)

        # 3. 选择 K 个最重要的行索引 I_t
        _, topk_indices_It = torch.topk(sigma_t, self.k) # topk_indices_It 形状为 (k,)
        # 确保索引是有序的，虽然不是严格要求，但有助于后续处理和调试
        topk_indices_It = topk_indices_It.sort().values


        # ======================================================================
        # 阶段 3: 客户端局部压缩 (模拟在服务器上完成)
        # ======================================================================
        collected_sparse_flat_grads = [] # 用于存储每个客户端的 Clocal(g_t^(i)) 稀疏向量

        # 修改二：###去掉byzantine客户端，只保留诚实节点
        # 这里的循环将只迭代 FABA 过滤后的 G_i 矩阵列表
        for G_i_original in current_G_i_list_filtered: # 使用 FABA 过滤后的 G_i 矩阵
            # 创建一个全零矩阵，用于存储局部压缩后的 G
            Clocal_G_i = torch.zeros_like(G_i_original, device=device, dtype=dtype)
            
            # 将 I_t 对应行的原始梯度复制到 Clocal_G_i
            Clocal_G_i[topk_indices_It, :] = G_i_original[topk_indices_It, :]

            # 将 Clocal_G_i 展平为稀疏向量 Clocal(g_t^(i))
            Clocal_flat_grad_i = Clocal_G_i.view(-1) # 形状 (d,)

            collected_sparse_flat_grads.append(Clocal_flat_grad_i)

        self._cached_client_G_matrices.clear() # 清空原始缓存 (_cached_client_G_matrices)，释放内存

        if not collected_sparse_flat_grads:
            print("警告: FABA 过滤后没有稀疏梯度可聚合，返回空列表。")
            return []

        # ======================================================================
        # 阶段 4: 服务器端聚合最终压缩梯度
        # ======================================================================
        # 聚合所有客户端的 Clocal(g_t^(i)) (求平均) 得到 C(g_t)
        # 注意：这里使用 FABA 过滤后的客户端数量进行平均
        C_g_t_flat = torch.zeros_like(
            collected_sparse_flat_grads[0],
            device=device,
            dtype=dtype
        )
        for sparse_grad_i in collected_sparse_flat_grads:
            C_g_t_flat.add_(sparse_grad_i)
        C_g_t_flat.div_(num_clients_after_faba) # 使用 FABA 过滤后的客户端数量

        # 将 C(g_t) 扁平向量解展平回原始层梯度列表
        reconstructed_gradients = self._unflatten_gradients(C_g_t_flat, self.reference_shapes)

        return reconstructed_gradients

    def get_state(self) -> dict:
        """
        返回聚合器的当前状态，包括 m, r, k, byzantine_alpha, d, n, V 和 reference_shapes。
        """
        state = {
            'm': self.m,
            'r': self.r,
            'k': self.k,
            'byzantine_alpha': self.byzantine_alpha, # 新增 byzantine_alpha
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
        self.k = state.get('k', self.k)
        self.byzantine_alpha = state.get('byzantine_alpha', self.byzantine_alpha) # 新增 byzantine_alpha
        self.d = state.get('d')
        self.n = state.get('n')
        self.V = state.get('V')
        self.reference_shapes = state.get('reference_shapes')
        
        if self.d is not None and self.n is not None and self.V is not None and self.reference_shapes is not None:
            print(f"CompressAggregator 状态已加载: d={self.d}, n={self.n}, r={self.r}, k={self.k}, byzantine_alpha={self.byzantine_alpha}, V 形状={self.V.shape}")
        else:
            print("警告: 状态部分加载，某些关键属性可能仍为 None。")

