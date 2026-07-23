from typing import List
import torch
from .base_attack import BaseAttack, Context

class Mimic(BaseAttack):

    def __init__(self):
        self.z = None

    @torch.no_grad()
    def attack(self, context: Context) -> List[torch.Tensor]:
        ref_grad = context.all_honest_gradients[0]

        if self.z is None:
            self.z = []
            for layer_g in ref_grad:
                layer_z0 = torch.randn(layer_g.shape).to(layer_g.device)
                layer_z0 = layer_z0 / layer_z0.norm()
                self.z.append(layer_z0)


        device = ref_grad[0].device
        honest_grad = [torch.stack([client_grad[i].to(device) for client_grad in context.all_honest_gradients]) for i in range(len(ref_grad))]
        mean_grad = [torch.mean(layer_gradients, dim=0) for layer_gradients in honest_grad]

        num_honest = len(context.all_honest_gradients)

        # ==========================
        # Step 1: Oja's Rule 更新 z
        # ==========================

        # 1.1 计算投影 (Projections): p = (g - mu) @ z
        # 我们需要计算每个客户端的梯度(去中心化后)在当前 z 方向上的标量投影
        projections = torch.zeros(num_honest, device=device)

        for layer_idx, (layer_grads, layer_mean, layer_z) in enumerate(zip(honest_grad, mean_grad, self.z)):
            # 去中心化: (num_clients, ...) - (1, ...)
            centered_grads = layer_grads - layer_mean.unsqueeze(0)

            # 将该层参数拉平以便做点积
            flat_centered = centered_grads.view(num_honest, -1) # (N, D)
            flat_z = layer_z.view(-1)                           # (D,)

            # 累加每一层的点积结果
            projections += torch.mv(flat_centered, flat_z)

        # 1.2 更新 z: z_new = z + lr * sum(p_i * (g_i - mu))
        # 学习率通常设为 1/t 或者一个小常数，这里使用 1/N 作为一个稳健的归一化因子
        lr = 1.0 / num_honest

        # 用于计算新 z 的全局范数
        new_z_norm_sq = 0.0

        for layer_idx, (layer_grads, layer_mean, layer_z) in enumerate(zip(honest_grad, mean_grad, self.z)):
            centered_grads = layer_grads - layer_mean.unsqueeze(0)

            # 为了利用 projections 进行加权求和，需要调整 projections 的形状以支持广播
            # view_shape: [num_honest, 1, 1, ...]
            view_shape = [num_honest] + [1] * (centered_grads.dim() - 1)
            weights = projections.view(*view_shape)

            # Oja 更新项: Covariance * z
            update_term = (centered_grads * weights).sum(dim=0)

            # 原地更新 self.z
            self.z[layer_idx] = layer_z + lr * update_term

            # 累加范数平方
            new_z_norm_sq += self.z[layer_idx].norm() ** 2

        # ==========================
        # Step 2: 归一化 z
        # ==========================
        new_z_norm = torch.sqrt(new_z_norm_sq)
        for i in range(len(self.z)):
            self.z[i] /= (new_z_norm + 1e-8) # 防止除零

        # ==========================
        # Step 3: 选择目标 (Target Selection)
        # ==========================
        # 根据论文 Appendix B: i* = argmax (z^T * x_i)
        # 也就是选择在主成分方向上投影最大的那个工人

        scores = torch.zeros(num_honest, device=device)
        for layer_idx, (layer_grads, layer_z) in enumerate(zip(honest_grad, self.z)):
            flat_grads = layer_grads.view(num_honest, -1)
            flat_z = layer_z.view(-1)
            scores += torch.mv(flat_grads, flat_z)

        # 找到得分最高的索引
        target_idx = torch.argmax(scores).item()

        # ==========================
        # Step 4: 执行模仿 (Mimic)
        # ==========================
        # 获取目标工人的原始梯度
        target_grad = context.all_honest_gradients[target_idx]

        # 生成恶意梯度列表：所有恶意节点都发送完全相同的 target_grad
        malicious_gradients = [param.detach().clone() for param in target_grad]

        return malicious_gradients
