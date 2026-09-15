"""
ST-GNN World Model (v2)
=======================
Spatio-Temporal Graph Neural Network for RMFS counterfactual
latent world model.

P6 updates:
  - Edge-conditioned GNN: message = sigmoid(MLP(edge_feat)) * neighbor
  - SystemDecoder: 5 -> 7 outputs (matches 7-dim system labels)
  - StationDecoder: per-station (queue, assigned_load) predictions
  - CostHead: 7-dim lambdas, deadlock_risk uses max across steps
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from WorldModel.core.long_risk_schema import LONG_RISK_OUTPUT_DIM


# ---------------------------------------------------------------------------
# Graph convolution primitives
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """
    Graph convolution with optional edge-conditioned gating.
    接近 GraphSAGE 的架构, 在此基础上增加了基于边特征的门控机制, 使得消息传递能够根据边的属性动态调整邻居特征的影响力。
    这种设计可以更灵活地捕捉图结构中的复杂关系, 特别适用于 RMFS 这种具有丰富边特征的环境。
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int = 0):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_nbr = nn.Linear(in_dim, out_dim)
        self.edge_gate = nn.Linear(edge_dim, in_dim) if edge_dim > 0 else None

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward 数学表达
        h_i^(l+1) = W_self * h_i^(l) + W_nbr * (1/|N(i)| * sum_{j in N(i)} [gate(e_{ij}) * h_j^(l)])
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        nbr_features = x[src]  # 通过高级索引，直接把所有源节点的特征（Hidden States）提取出来，作为准备发往目标节点的"原始消息"。
        if self.edge_gate is not None and edge_attr is not None:
            # 先通过一个线性层（self.edge_gate）将边特征（edge_attr）映射到与节点特征相同的维度（in_dim），然后通过 sigmoid 激活函数将其转换为 [0, 1] 之间的门控系数。
            gate = torch.sigmoid(self.edge_gate(edge_attr))
            nbr_features = nbr_features * gate # gate系数与原始消息按位相乘，实现了基于边特征的动态调整。这样一来，不同的边可以对邻居节点的特征产生不同程度的影响，从而增强了模型的表达能力。

        # Mean Pooling（均值聚合）: 使用 index_add_ 将所有发往同一目标节点的邻居特征进行累加，得到每个节点的聚合消息。同时，使用一个计数器来记录每个节点收到的消息数量，以便后续进行平均处理。
        aggr = torch.zeros(N, x.size(1), device=x.device)
        count = torch.zeros(N, 1, device=x.device)
        # index_add_ 是 PyTorch 中的一种高效的操作，可以在指定维度上将 索引idx 对应的nbr_features[idx] 值累加到目标张量 aggr[dst[idx]]中。
        # 在这里，0 代表在沿着行进行聚合消息。
        """
        dst = tensor([1, 1, 2, 2])  # 长度为 4
        nbr_features = tensor([                                         aggr = tensor([
            [0.1, 0.2],  # 边 0 上传的消息 (发往节点 1)                       [0.0, 0.0],  # 留给节点 0 收集消息
            [0.3, 0.4],  # 边 1 上传的消息 (发往节点 1)         ->            [0.4, 0.6],  # 节点 1 的消息和 = [0.1+0.3, 0.2+0.4]
            [0.5, 0.6],  # 边 2 上传的消息 (发往节点 2)                       [1.2, 1.4],  # 节点 2 的消息和 = [0.5+0.7, 0.6+0.8]
            [0.7, 0.8],  # 边 3 上传的消息 (发往节点 2)                   ])
        ])
        """
        aggr.index_add_(0, dst, nbr_features)
        ones = torch.ones(src.size(0), 1, device=x.device)
        count.index_add_(0, dst, ones)
        count = count.clamp(min=1.0)  # 防止除以 0
        aggr = aggr / count           # 取平均

        # self.lin_self(x): 节点原来的特征 x 经过自己的线性变换。 self.lin_nbr(aggr): 周边邻居传过来的特征经过另一套权重的线性变换。
        return self.lin_self(x) + self.lin_nbr(aggr)
        


class SpatialBlock(nn.Module):
    """Stack of GraphConvLayers with ReLU and residual connection."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 2, edge_dim: int = 0):
        super().__init__()
        layers = []
        # 扩大"感受野" ,每层都能看到更远的邻居, 通过堆叠多层 GraphConvLayer 来实现。
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(GraphConvLayer(d_in, d_out, edge_dim=edge_dim))
        self.layers = nn.ModuleList(layers)

        # 拯救"过平滑"：残差连接可以帮助缓解过平滑问题，允许模型在每层都保留一些原始节点特征的信息，从而增强模型的表达能力。
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 深度 GNN 容易出现过平滑问题（over-smoothing），由于每次图卷积本质上都在做"邻居特征取平均"，如果层数过多，图中所有节点的特征最后都会变得一模一样
        res = self.residual(x)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr=edge_attr)
            if i < len(self.layers) - 1:
                h = F.relu(h)
        return self.norm(F.relu(h + res))


# ---------------------------------------------------------------------------
# Temporal module
# ---------------------------------------------------------------------------

class TemporalGRU(nn.Module):
    """Per-node GRU across L temporal frames. 每个节点在L个时间帧上的门控循环单元(GRU)"""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq 包含了过去 L 帧的信息（比如 4 帧）
        out, _ = self.gru(x_seq)
        # 网络处理完所有 L 帧后，直接丢弃前面的输出，只保留最后一帧的输出。
        return out[:, -1, :]


# ---------------------------------------------------------------------------
# ST-GNN Encoder
# ---------------------------------------------------------------------------

class STGNNEncoder(nn.Module):
    """Spatio-Temporal GNN encoder with edge conditioning. 带边条件的时空图神经网络编码器"""
    # GCN-T 范式（先空间，后时间）: 先对每个时间帧独立应用空间块（SpatialBlock），然后将每个节点在不同时间帧上的表示序列输入到一个 GRU 中，得到最终的节点表示。 
    def __init__(self, node_feat_dim: int = 10, edge_feat_dim: int = 6,
                 hidden_dim: int = 64, num_spatial_layers: int = 3):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.spatial = SpatialBlock(
            in_dim=node_feat_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
            num_layers=num_spatial_layers, edge_dim=hidden_dim,
        )
        self.temporal = TemporalGRU(input_dim=hidden_dim, hidden_dim=hidden_dim)

    def forward(self, node_history: torch.Tensor, edge_index: torch.LongTensor,
                edge_features: torch.Tensor):
        """
        Returns
        -------
        z : (N, hidden_dim)
        edge_attr : (E, hidden_dim) — encoded edge features for reuse
        """
        L, N, _ = node_history.shape
        edge_attr = self.edge_encoder(edge_features)

        spatial_out = []
        # for 循环，把过去 L 帧的数据逐帧抽出来，先让每一帧独立通过 3 层 SpatialBlock 融合空间邻居信息。
        for t in range(L):
            h_t = self.spatial(node_history[t], edge_index, edge_attr=edge_attr)
            spatial_out.append(h_t) # 最后为一个长度为 L 的列表，里面装的全是形状为 (N, hidden_dim) 的张量

        x_seq = torch.stack(spatial_out, dim=1) # 形状变为 (N, L, hidden_dim)
        z = self.temporal(x_seq)
        return z, edge_attr # edge_attr 避免重复计算


# ---------------------------------------------------------------------------
# Demand encoder
# ---------------------------------------------------------------------------

class DemandEncoder(nn.Module):
    # 对需求上下文 (5+S 维) 进行 encoding
    # 得到一个与节点特征维度相同的向量，后续在 transition 中作为全局输入与节点特征融合。
    def __init__(self, demand_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(demand_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, demand: torch.Tensor) -> torch.Tensor:
        return self.net(demand)


# ---------------------------------------------------------------------------
# Action encoder
# ---------------------------------------------------------------------------

class ActionEncoder(nn.Module):
    def __init__(self, action_node_dim: int = 8, action_global_dim: int = 6,
                 hidden_dim: int = 64):
        # 8 个动作节点通道（起点、取货点、归还点、以及 3 段不同权重的运行轨迹）
        # 6 个宏观任务特征（三段曼哈顿距离、目标站排队长度、货架价值、订单大小）
        super().__init__()
        self.node_net = nn.Sequential(
            nn.Linear(action_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_net = nn.Sequential(
            nn.Linear(action_global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, action_node: torch.Tensor, action_global: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        u_node = self.node_net(action_node) # u_node 的形状是 (num_nodes, 64)
        u_global = self.global_net(action_global).unsqueeze(0).expand(num_nodes, -1) # action_global 经过 global_net 后变成了 (64,), .unsqueeze(0)从 (64,) 变成了 (1, 64), .expand(num_nodes, -1)从 (1, 64) 变成了 (num_nodes, 64)
        # 这里直接相加进行对齐融合，得到一个形状为 (num_nodes, 64) 的动作表示张量，其中每个节点的动作表示都包含了该节点特定的信息（来自 u_node）以及整个系统级别的动作信息（来自 u_global）。
        return u_node + u_global


# ---------------------------------------------------------------------------
# Latent transition
# ---------------------------------------------------------------------------

class LatentTransition(nn.Module):
    """Residual GNN transition for autoregressive rollout."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 64,
                 demand_dim: int = 64, max_steps: int = 20, edge_dim: int = 0):
        super().__init__()
        self.step_embed = nn.Embedding(max_steps, latent_dim) # （可学习的位置编码）
        fuse_dim = latent_dim + action_dim + demand_dim + latent_dim
        self.fuse = nn.Linear(fuse_dim, latent_dim)
        self.gcn = GraphConvLayer(latent_dim, latent_dim, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z: torch.Tensor, u: torch.Tensor, e_demand: torch.Tensor,
                step: int, edge_index: torch.LongTensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        z (当前状态)：来自 STGNNEncoder, 代表地图现在的拥堵情况。
        u (候选动作)：来自 ActionEncoder, 代表打算派哪辆车走哪条路。
        e_demand (全局需求)：来自 DemandEncoder, 代表现在的业务大盘压力(被 expand 广播到了每一个节点)。
        step (时间步)：当前推演到了未来的第几步。
        """
        N = z.size(0)
        # 防止长时间自回归导致潜空间崩溃
        step_emb = self.step_embed(
            torch.tensor(min(step, self.step_embed.num_embeddings - 1), device=z.device)
        ).unsqueeze(0).expand(N, -1)
        e_broad = e_demand.unsqueeze(0).expand(N, -1)

        h = torch.cat([z, u, e_broad, step_emb], dim=-1)
        h = F.relu(self.fuse(h))
        # 预测变化量（Delta）, 通过残差连接 z + delta 得到未来状态的潜空间表示，这样可以更稳定地进行多步自回归推演
        delta = self.gcn(h, edge_index, edge_attr=edge_attr)

        return self.norm(z + delta)


# ---------------------------------------------------------------------------
# Decoders
# ---------------------------------------------------------------------------

class NodeDecoder(nn.Module):
    """Per-node decoder: latent -> 6-channel predictions.

    Channels: self_occupancy, local_density, local_wait_pressure,
              local_blocked_pressure, reservation_pressure, congestion_score
    """
    NODE_OUTPUT_DIM = 6

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.NODE_OUTPUT_DIM),
        )
        self.uncertainty = nn.Linear(latent_dim, self.NODE_OUTPUT_DIM)

    def forward(self, z: torch.Tensor):
        # 模型只能输出绝对数值，遇到不确定的情况它就会左右摇摆，导致训练崩溃
        # 所以我们让模型同时输出一个 log_sigma，代表每个预测维度的不确定性（越大表示越不确定）。在训练时，可以根据这个不确定性来调整损失函数的权重，让模型在不确定的维度上学习得更慢，从而提高训练的稳定性。
        mu = self.net(z)
        log_sigma = self.uncertainty(z)
        # decoder 的loss用高斯负对数似然
        return mu, log_sigma


class SystemDecoder(nn.Module):
    """ Global: pool node latents -> 7-dim system predictions.
        全局：池化节点潜在特征 → 7维系统预测。
    Outputs: total_wait_time, avg_excess_delay, station_queue_delta,
             station_load_imbalance, bottleneck_CVaR, completed_orders_delta,
             deadlock_risk.
    输出：总等待时间、平均绕路延迟、站点队列增量、
          站点负载不均衡度、瓶颈拥堵极值、系统总吞吐量、
          死锁风险。
    预测系统总延迟
    """
    SYSTEM_OUTPUT_DIM = 7

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.SYSTEM_OUTPUT_DIM),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 计算所有节点特征的平均值。适合于预测那些与整体系统状态相关的指标，比如系统总吞吐量、平均绕路延迟等。
        z_mean = z.mean(dim=0) # 维度为 (latent_dim,)
        # 提取所有节点在每一个特征通道上的最高值。适合于预测那些与局部极端情况相关的指标，比如死锁风险（只要有一个节点死锁了，整个系统就有死锁风险）。
        z_max = z.max(dim=0).values # 维度为 (latent_dim,)
        g = torch.cat([z_mean, z_max], dim=-1) # 将平均特征和最大特征拼接在一起，得到一个形状为 (latent_dim * 2,) 的全局图表示，这样模型就可以同时利用整体趋势和局部极端情况的信息来进行系统级的预测。
        return self.net(g)


class StationDecoder(nn.Module):
    """Per-station decoder: station node latents -> (queue, assigned_load). 站点节点潜在变量 -> （工作站物理排队队列，已分配在途负载）
        预测业务集散中心（工作站）供需压力
    """
    STATION_OUTPUT_DIM = 2

    def __init__(self, latent_dim: int = 64, num_stations: int = 4):
        super().__init__()
        self.num_stations = num_stations
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.STATION_OUTPUT_DIM),
        )

    def forward(self, z: torch.Tensor,
                station_node_ids: Optional[List[int]] = None) -> torch.Tensor:
        # 硬注意力机制：如果提供了 station_node_ids，就只解码这些特定节点的潜在表示；如果没有提供，就对所有节点的潜在表示取平均，得到一个全局图表示，并将其复制扩展到每个站点上进行解码。
        # 这样设计的好处是既可以针对特定站点进行精细预测，也可以在缺乏明确站点信息时提供一个基于整体图状态的估计。
        if station_node_ids is not None and len(station_node_ids) > 0:
            ids = torch.tensor(station_node_ids, dtype=torch.long, device=z.device)
            station_z = z[ids] # 维度为 (num_stations, latent_dim)
        else:
            # 把全图所有格子的特征求个平均, 得到(1,64)后把这个平均特征复制 4 份, 得到(4,64), 每个站点都用这个全局平均特征来预测自己的队列长度和分配负载。
            station_z = z.mean(dim=0, keepdim=True).expand(self.num_stations, -1)
        return self.net(station_z)


# ---------------------------------------------------------------------------
# Long-risk head (Phase B2)
# ---------------------------------------------------------------------------

class LongRiskHead(nn.Module):
    """Action-conditioned Tail-aware Residual Long-risk Head (W=200).

    Predicts long-horizon tail-risk statistics from the initial and
    final latent states of a short-horizon rollout.  Does NOT predict
    step-by-step trajectories — only statistical summaries.

    Outputs (6,):
      0  peak_q90       — 90th-percentile of max unified_risk over W ticks
      1  peak_q95       — 95th-percentile of max unified_risk over W ticks
      2  cvar_q90       — 90th-percentile of tail-average risk
      3  terminal_q90   — 90th-percentile of terminal-window mean risk
      4  delta_group_q90 — 90th-percentile of within-group risk increment
      5  event_logit    — logit for P(severe event within W ticks)
    """
    LONG_RISK_DIM = LONG_RISK_OUTPUT_DIM

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        # Input: mean+max pool of z_K (rollout end) and z_0 (rollout start)
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, self.LONG_RISK_DIM),
        )

    def forward(self, z_K: torch.Tensor, z_0: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z_K : (N, latent_dim) — node latents at rollout step K
        z_0 : (N, latent_dim) — node latents at rollout step 0 (before transition)

        Returns
        -------
        (6,) long-risk predictions
        """
        g = torch.cat([
            z_K.mean(dim=0), z_K.max(dim=0).values,
            z_0.mean(dim=0), z_0.max(dim=0).values,
        ], dim=-1)
        return self.net(g)


# ---------------------------------------------------------------------------
# Pure state TD risk V-head (Phase C, additive)
# ---------------------------------------------------------------------------

class TDRiskVHead(nn.Module):
    """Immediate-risk residual head for policy-specific future risk.

    This head deliberately has *no* action/Q-mode.  It estimates the three
    raw risk-to-go components under the continuation policy that generated
    its TD stream::

        J_risk^pi(s) = [J_stall(s), J_deadlock(s), J_handoff(s)]

    The input signature represents one state only.  It pools the frozen
    world-model latent globally, at station nodes, and at static bottleneck
    nodes, then appends the demand embedding.  This fixes two semantic bugs
    in the old ``TDValueHead``: ``pool(z_K, z_0)`` mixed Q and V meanings, and
    pooling only ``z`` omitted the separately encoded demand/backlog signal.

    The network does not relearn the already observed risk level from
    scratch.  It predicts a bounded correction around the analytic current
    risk::

        J_hat(s) = clip(current_raw_risk(s) + residual_theta(s), 0, 1)

    The final residual layer is zero-initialised, so a fresh head is exactly
    the immediate-risk baseline.  Per-component validation gates are stored
    with the head; a component which has not beaten that baseline falls back
    to the current raw risk instead of degrading the deployed estimate.

    The head remains additive (it is not registered inside
    :class:`RMFSWorldModel`) so old base checkpoints retain strict loading
    compatibility.
    """

    RISK_COMPONENTS = ("stall", "deadlock", "handoff")
    MULTIMETRIC_GATE_SCHEMA = "td_component_gate_multimetric_v1"
    MULTIMETRIC_REQUIRED_CHECKS = (
        "enough_anchors",
        "finite",
        "mae_improvement",
        "rmse_non_degradation",
        "rank_floor",
        "rank_preservation",
        "relative_bias",
        "std_ratio",
    )

    def __init__(
        self,
        latent_dim: int = 64,
        demand_dim: Optional[int] = None,
        hidden_dim: int = 128,
        physical_summary_dim: int = 0,
        bottleneck_fraction: float = 0.20,
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if demand_dim is None:
            demand_dim = latent_dim
        if latent_dim <= 0 or demand_dim <= 0 or hidden_dim <= 0:
            raise ValueError("latent_dim, demand_dim and hidden_dim must be positive")
        if physical_summary_dim < 0:
            raise ValueError("physical_summary_dim must be non-negative")
        if not 0.0 < bottleneck_fraction <= 1.0:
            raise ValueError("bottleneck_fraction must be in (0, 1]")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive")

        self.latent_dim = int(latent_dim)
        self.demand_dim = int(demand_dim)
        self.hidden_dim = int(hidden_dim)
        self.physical_summary_dim = int(physical_summary_dim)
        self.bottleneck_fraction = float(bottleneck_fraction)
        self.residual_scale = float(residual_scale)
        # global mean/max + station mean/max + bottleneck mean/max + demand
        self.input_dim = 6 * self.latent_dim + self.demand_dim + self.physical_summary_dim

        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, len(self.RISK_COMPONENTS)),
        )
        # A fresh residual head must be the exact, auditable immediate-risk
        # baseline.  This also makes failed/under-trained heads harmless once
        # component gates are applied.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        # Deployment is fail-closed: learned residuals are enabled only after
        # an explicit held-out validation certificate has been written.
        self.register_buffer(
            "component_gates",
            torch.zeros(len(self.RISK_COMPONENTS), dtype=torch.bool),
        )

    @staticmethod
    def _normalise_node_ids(
        node_ids,
        num_nodes: int,
        device: torch.device,
    ) -> Optional[torch.LongTensor]:
        if node_ids is None:
            return None
        ids = torch.as_tensor(node_ids, dtype=torch.long, device=device).flatten()
        if ids.numel() == 0:
            return None
        if bool(((ids < 0) | (ids >= num_nodes)).any()):
            raise ValueError("node_ids contains an index outside the latent graph")
        return torch.unique(ids)

    @staticmethod
    def _mean_max(z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.mean(dim=0), z.max(dim=0).values], dim=-1)

    @classmethod
    def build_state_features(
        cls,
        z: torch.Tensor,
        e_demand: torch.Tensor,
        station_node_ids=None,
        bottleneck_scores: Optional[torch.Tensor] = None,
        physical_summary: Optional[torch.Tensor] = None,
        bottleneck_fraction: float = 0.20,
    ) -> torch.Tensor:
        """Build the auditable, state-only feature signature.

        Parameters
        ----------
        z:
            ``(N, D)`` node latent for exactly one state.
        e_demand:
            ``(D_d,)`` demand embedding from the same state.
        station_node_ids:
            Static graph node ids of stations.  Missing ids produce an
            explicit zero station pool instead of silently reusing the global
            pool.
        bottleneck_scores:
            Static per-node centrality scores.  The top
            ``bottleneck_fraction`` nodes are pooled.  Missing scores likewise
            produce a zero bottleneck pool, making schema omissions visible.
        physical_summary:
            Optional fixed-schema analytic state summary.  It may be enabled
            only when both real and rollout-end states provide the same fields.
        """
        if z.ndim != 2:
            raise ValueError(f"z must have shape (N, D), got {tuple(z.shape)}")
        if e_demand.ndim != 1:
            raise ValueError(
                f"e_demand must have shape (D_d,), got {tuple(e_demand.shape)}"
            )
        if z.size(0) == 0:
            raise ValueError("z must contain at least one node")
        if not 0.0 < bottleneck_fraction <= 1.0:
            raise ValueError("bottleneck_fraction must be in (0, 1]")

        global_pool = cls._mean_max(z)
        zero_pool = torch.zeros_like(global_pool)

        station_ids = cls._normalise_node_ids(
            station_node_ids, z.size(0), z.device
        )
        station_pool = (
            cls._mean_max(z.index_select(0, station_ids))
            if station_ids is not None else zero_pool
        )

        bottleneck_pool = zero_pool
        if bottleneck_scores is not None:
            scores = torch.as_tensor(
                bottleneck_scores, dtype=z.dtype, device=z.device
            ).flatten()
            if scores.numel() != z.size(0):
                raise ValueError(
                    "bottleneck_scores length must equal the number of nodes"
                )
            k = max(1, int(round(z.size(0) * bottleneck_fraction)))
            k = min(k, z.size(0))
            ids = torch.topk(scores, k=k, largest=True, sorted=False).indices
            bottleneck_pool = cls._mean_max(z.index_select(0, ids))

        parts = [global_pool, station_pool, bottleneck_pool, e_demand]
        if physical_summary is not None:
            if physical_summary.ndim != 1:
                raise ValueError("physical_summary must be a one-dimensional tensor")
            parts.append(physical_summary.to(device=z.device, dtype=z.dtype))
        return torch.cat(parts, dim=-1)

    def encode_state(
        self,
        z: torch.Tensor,
        e_demand: torch.Tensor,
        station_node_ids=None,
        bottleneck_scores: Optional[torch.Tensor] = None,
        physical_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool one encoded state using this head's frozen input signature."""
        g = self.build_state_features(
            z,
            e_demand,
            station_node_ids=station_node_ids,
            bottleneck_scores=bottleneck_scores,
            physical_summary=physical_summary,
            bottleneck_fraction=self.bottleneck_fraction,
        )
        if g.numel() != self.input_dim:
            raise ValueError(
                f"state feature dimension {g.numel()} != configured {self.input_dim}; "
                "real and rollout states must use the same input schema"
            )
        return g

    def predict_residual(self, state_features: torch.Tensor) -> torch.Tensor:
        """Return the bounded learned correction to current raw risk."""
        if state_features.size(-1) != self.input_dim:
            raise ValueError(
                f"last input dimension must be {self.input_dim}, "
                f"got {state_features.size(-1)}"
            )
        return self.residual_scale * torch.tanh(self.net(state_features))

    def set_component_gates(self, gates) -> None:
        """Set validation gates in registered risk-component order."""
        value = torch.as_tensor(
            gates, dtype=torch.bool, device=self.component_gates.device
        ).flatten()
        if value.numel() != len(self.RISK_COMPONENTS):
            raise ValueError(
                f"component_gates must contain {len(self.RISK_COMPONENTS)} "
                f"values, got {value.numel()}"
            )
        self.component_gates.copy_(value)

    def forward(
        self,
        state_features: torch.Tensor,
        current_risk: torch.Tensor,
        apply_component_gates: bool = True,
    ) -> torch.Tensor:
        """Estimate risk-to-go around the immediate-risk baseline.

        ``current_risk`` must end in the three registered raw risk
        components and be broadcast-compatible with ``state_features``.
        During optimisation/evaluation of the learned candidate, pass
        ``apply_component_gates=False``.  Normal inference keeps the default
        and therefore falls back component-wise when validation did not
        establish an improvement.
        """
        current = torch.as_tensor(
            current_risk, dtype=state_features.dtype,
            device=state_features.device,
        )
        if current.ndim == 0 or current.size(-1) != len(self.RISK_COMPONENTS):
            raise ValueError(
                f"current_risk last dimension must be "
                f"{len(self.RISK_COMPONENTS)}, got {tuple(current.shape)}"
            )
        if not bool(torch.isfinite(current).all()):
            raise ValueError("current_risk contains NaN or Inf")
        tolerance = 1e-6
        if bool(((current < -tolerance) | (current > 1.0 + tolerance)).any()):
            raise ValueError("current_risk components must lie in [0, 1]")
        current = torch.clamp(current, min=0.0, max=1.0)
        residual = self.predict_residual(state_features)
        candidate = torch.clamp(current + residual, min=0.0, max=1.0)
        if not apply_component_gates:
            return candidate
        gates = self.component_gates.to(device=candidate.device)
        return torch.where(gates, candidate, current)

    @staticmethod
    def _checkpoint_gate_map(value) -> Optional[dict]:
        if not isinstance(value, dict):
            return None
        if not all(name in value for name in TDRiskVHead.RISK_COMPONENTS):
            return None
        return {
            name: bool(value[name])
            for name in TDRiskVHead.RISK_COMPONENTS
        }

    @classmethod
    def _multimetric_component_certificate(cls, decision: dict) -> tuple:
        """Validate one enabled v4 gate without trusting summary booleans."""
        reasons = []
        if not isinstance(decision, dict):
            return False, ["missing_component_decision"]
        if decision.get("gate_schema_version") != cls.MULTIMETRIC_GATE_SCHEMA:
            reasons.append("component_gate_schema_mismatch")
        if not bool(decision.get("enabled", False)):
            reasons.append("decision_not_enabled")
        if decision.get("fallback") is not None:
            reasons.append("enabled_decision_has_fallback")
        if decision.get("reasons"):
            reasons.append("enabled_decision_has_failure_reasons")

        aggregate = decision.get("aggregate")
        if not isinstance(aggregate, dict):
            reasons.append("missing_aggregate_certificate")
        else:
            checks = aggregate.get("checks")
            if not isinstance(checks, dict):
                reasons.append("missing_aggregate_checks")
            else:
                for check in cls.MULTIMETRIC_REQUIRED_CHECKS:
                    if check not in checks:
                        reasons.append(f"missing_aggregate_check:{check}")
                    elif not bool(checks[check]):
                        reasons.append(f"failed_aggregate_check:{check}")
            if not bool(aggregate.get("passed", False)):
                reasons.append("aggregate_not_passed")
            if aggregate.get("failed_checks"):
                reasons.append("aggregate_has_failed_checks")

        require_all_units = bool(decision.get("require_all_units", True))
        if require_all_units:
            units = decision.get("units")
            if not isinstance(units, dict) or not units:
                reasons.append("missing_validation_unit_certificates")
            else:
                for unit, metrics in units.items():
                    if not isinstance(metrics, dict):
                        reasons.append(f"invalid_validation_unit:{unit}")
                        continue
                    checks = metrics.get("checks")
                    if not isinstance(checks, dict):
                        reasons.append(f"missing_unit_checks:{unit}")
                    else:
                        for check in cls.MULTIMETRIC_REQUIRED_CHECKS:
                            if check not in checks:
                                reasons.append(
                                    f"missing_unit_check:{unit}:{check}"
                                )
                            elif not bool(checks[check]):
                                reasons.append(
                                    f"failed_unit_check:{unit}:{check}"
                                )
                    if not bool(metrics.get("passed", False)):
                        reasons.append(f"validation_unit_not_passed:{unit}")
                    if metrics.get("failed_checks"):
                        reasons.append(f"validation_unit_has_failed_checks:{unit}")
        return not reasons, reasons

    @staticmethod
    def _legacy_metric(value, key) -> Optional[float]:
        if not isinstance(value, dict) or key not in value:
            return None
        try:
            number = float(value[key])
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def _legacy_v3_component_audit(
        cls,
        checkpoint: dict,
        name: str,
        decision: dict,
    ) -> tuple:
        """Apply a conservative audit to an MAE-only v3 deployment gate.

        Old artifacts did not store prediction standard deviations, so this
        compatibility audit cannot issue a new multi-metric certificate.  It
        only preserves an already-enabled component when the stored aggregate
        metrics establish MAE/RMSE improvement, rank usefulness/preservation,
        and bounded normalized bias.  Missing metrics fail closed.
        """
        reasons = []
        if not isinstance(decision, dict) or not bool(decision.get("enabled")):
            return False, ["legacy_decision_not_enabled"]
        aggregate = decision.get("aggregate")
        if not isinstance(aggregate, dict) or not bool(aggregate.get("passed")):
            reasons.append("legacy_aggregate_not_passed")
        if bool(decision.get("require_all_units", True)):
            units = decision.get("units")
            if not isinstance(units, dict) or not units:
                reasons.append("legacy_validation_units_missing")
            elif any(
                not isinstance(metrics, dict)
                or not bool(metrics.get("passed", False))
                for metrics in units.values()
            ):
                reasons.append("legacy_validation_unit_not_passed")

        validation = checkpoint.get("validation")
        candidate = (
            validation.get("mc", {}).get(name)
            if isinstance(validation, dict) else None
        )
        baseline = (
            validation.get("component_baselines", {})
            .get("immediate_risk", {}).get(name)
            if isinstance(validation, dict) else None
        )
        c_mae = cls._legacy_metric(candidate, "mae")
        b_mae = cls._legacy_metric(baseline, "mae")
        c_rmse = cls._legacy_metric(candidate, "rmse")
        b_rmse = cls._legacy_metric(baseline, "rmse")
        c_rank = cls._legacy_metric(candidate, "spearman")
        b_rank = cls._legacy_metric(baseline, "spearman")
        c_bias = cls._legacy_metric(candidate, "bias")
        target_mean = cls._legacy_metric(candidate, "target_mean")
        required = {
            "candidate_mae": c_mae,
            "baseline_mae": b_mae,
            "candidate_rmse": c_rmse,
            "baseline_rmse": b_rmse,
            "candidate_spearman": c_rank,
            "baseline_spearman": b_rank,
            "candidate_bias": c_bias,
            "target_mean": target_mean,
        }
        for metric, value in required.items():
            if value is None:
                reasons.append(f"legacy_metric_missing:{metric}")
        if not reasons:
            if not c_mae < b_mae:
                reasons.append("legacy_mae_not_improved")
            if not c_rmse <= b_rmse + 1e-8:
                reasons.append("legacy_rmse_degraded")
            if c_rank < 0.10:
                reasons.append("legacy_spearman_below_floor")
            if c_rank < b_rank - 0.05:
                reasons.append("legacy_spearman_drop_too_large")
            relative_bias = abs(c_bias) / max(abs(target_mean), 1e-8)
            if relative_bias > 0.50:
                reasons.append("legacy_relative_bias_too_large")
        return not reasons, reasons

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_or_path,
        map_location="cpu",
        use_ema: bool = False,
    ):
        """Load an additive TD-risk residual checkpoint fail-closed.

        Returns ``(head, config)``.  Version 4 enables each component only when
        its stored multi-metric certificate, config gate, and state-dict gate
        agree.  Version 3 remains loadable through a conservative legacy audit;
        its sparse near-zero handoff solution is therefore disabled while the
        separately supported deadlock component can remain available.
        """
        if not isinstance(checkpoint_or_path, dict):
            checkpoint = torch.load(
                checkpoint_or_path,
                map_location=map_location,
                weights_only=False,
            )
        else:
            checkpoint = checkpoint_or_path
        schema = (
            checkpoint.get("schema_version")
            if isinstance(checkpoint, dict) else None
        )
        supported = {
            "td_risk_residual_vhead_v3",
            "td_risk_residual_vhead_v4",
        }
        if schema not in supported:
            raise ValueError(
                "expected td_risk_residual_vhead_v4 (or fail-closed v3), got "
                f"{schema!r}; legacy scalar/Q-V or absolute-value weights "
                "cannot be converted safely"
            )
        config = dict(checkpoint["config"])
        head = cls(
            latent_dim=int(config["latent_dim"]),
            demand_dim=int(config["demand_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            physical_summary_dim=int(config.get("physical_summary_dim", 0)),
            bottleneck_fraction=float(config.get("bottleneck_fraction", 0.20)),
            residual_scale=float(config.get("residual_scale", 1.0)),
        )
        key = "td_risk_vhead_ema" if use_ema else "td_risk_vhead"
        head.load_state_dict(checkpoint[key], strict=True)

        state_gates = {
            name: bool(head.component_gates[index].item())
            for index, name in enumerate(cls.RISK_COMPONENTS)
        }
        configured = cls._checkpoint_gate_map(config.get("component_gates"))
        decisions = checkpoint.get("component_gate_decisions")
        if not isinstance(decisions, dict):
            validation = checkpoint.get("validation")
            decisions = (
                validation.get("gate_decisions")
                if isinstance(validation, dict) else None
            )

        effective = {}
        certification_status = {}
        for name in cls.RISK_COMPONENTS:
            decision = decisions.get(name) if isinstance(decisions, dict) else None
            decision_enabled = (
                bool(decision.get("enabled", False))
                if isinstance(decision, dict) else None
            )
            configured_enabled = configured.get(name) if configured else None
            any_requested = bool(
                state_gates[name]
                or configured_enabled is True
                or decision_enabled is True
            )
            requested = bool(
                state_gates[name]
                and configured_enabled is True
                and decision_enabled is True
            )
            reasons = []
            if configured_enabled is None:
                reasons.append("missing_config_gate")
            if decision_enabled is None:
                reasons.append("missing_gate_decision")
            if not (
                configured_enabled == state_gates[name] == decision_enabled
            ):
                reasons.append("gate_config_decision_state_mismatch")

            if requested and not reasons:
                if schema == "td_risk_residual_vhead_v4":
                    rule = config.get("component_gate_rule")
                    if checkpoint.get("gate_schema_version") != cls.MULTIMETRIC_GATE_SCHEMA:
                        reasons.append("checkpoint_gate_schema_mismatch")
                    if (
                        not isinstance(rule, dict)
                        or rule.get("schema_version") != cls.MULTIMETRIC_GATE_SCHEMA
                    ):
                        reasons.append("config_gate_schema_mismatch")
                    elif (
                        "require_all_validation_units" not in rule
                        or bool(rule["require_all_validation_units"])
                        != bool(decision.get("require_all_units", True))
                    ):
                        reasons.append("validation_unit_rule_mismatch")
                    valid, detail = cls._multimetric_component_certificate(decision)
                else:
                    valid, detail = cls._legacy_v3_component_audit(
                        checkpoint, name, decision
                    )
                reasons.extend(detail)
                effective[name] = bool(valid and not reasons)
            else:
                effective[name] = False

            if effective[name]:
                status = (
                    "multimetric_certificate_verified"
                    if schema == "td_risk_residual_vhead_v4"
                    else "legacy_v3_safety_audit_passed"
                )
            elif any_requested:
                status = "requested_gate_forced_off"
            elif reasons:
                status = "disabled_or_inconsistent_checkpoint_gate"
            else:
                status = "disabled_by_training_gate"
            certification_status[name] = {
                "checkpoint_requested": any_requested,
                "checkpoint_sources_consistent": requested or not any_requested,
                "effective": effective[name],
                "status": status,
                "reasons": list(dict.fromkeys(reasons)),
            }

        head.set_component_gates([
            effective[name] for name in cls.RISK_COMPONENTS
        ])
        requested_map = configured or {
            name: state_gates[name] for name in cls.RISK_COMPONENTS
        }
        forced_disabled = [
            name for name in cls.RISK_COMPONENTS
            if certification_status[name]["checkpoint_requested"]
            and not effective[name]
        ]
        config["checkpoint_component_gates"] = requested_map
        config["effective_component_gates"] = effective
        config["component_gates"] = effective
        config["gate_certification_status"] = certification_status
        config["forced_disabled_components"] = forced_disabled
        config["checkpoint_gate_schema_version"] = checkpoint.get(
            "gate_schema_version"
        )
        config["gate_schema_version"] = (
            checkpoint.get("gate_schema_version")
            if schema == "td_risk_residual_vhead_v4"
            else "legacy_v3_safety_audit_v1"
        )
        has_metadata_problem = any(
            status["reasons"] for status in certification_status.values()
        )
        if forced_disabled:
            config["gate_load_status"] = (
                "one_or_more_requested_gates_forced_off"
            )
        elif has_metadata_problem:
            config["gate_load_status"] = (
                "no_gate_enabled_with_incomplete_or_inconsistent_metadata"
            )
        else:
            config["gate_load_status"] = "all_requested_gates_verified"
        head.to(map_location).eval()
        return head, config


# ---------------------------------------------------------------------------
# Legacy scalar TD head (6.2 V3 compatibility only)
# ---------------------------------------------------------------------------

class TDValueHead(nn.Module):
    """Legacy scalar/Q-V-mixed head kept for artifact reproducibility.

    New Phase-C work must use :class:`TDRiskVHead`.  This class intentionally
    retains the historical ``pool(z_K, z_0)`` signature so old commands and
    ``*_tdv.pt`` checkpoints do not silently change meaning.
    """

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @staticmethod
    def pool(z_K: torch.Tensor, z_0: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            z_K.mean(dim=0), z_K.max(dim=0).values,
            z_0.mean(dim=0), z_0.max(dim=0).values,
        ], dim=-1)

    def forward(self, state_features: torch.Tensor) -> torch.Tensor:
        return self.net(state_features).squeeze(-1)


# ---------------------------------------------------------------------------
# Cost head
# ---------------------------------------------------------------------------

class CostHead(nn.Module):
    """ Weighted discounted sum of system outputs -> scalar cost.
        系统输出的加权折现和 → 标量代价
    Deadlock risk uses max across steps instead of discounted sum.
    死锁风险采用各步最大值而非折现和进行计算。
    """

    def __init__(self, gamma: float = 0.95):
        super().__init__()
        self.gamma = gamma
        # 定义了一个 Cost（成本/代价）系统 目标是最小化这个 Cost
        # 正数代表惩罚, 负数代表奖励
        self.lambdas = nn.Parameter(torch.tensor([
            1.0,   # total_wait_time  等待时间
            0.5,   # avg_excess_delay 绕路延迟
            0.5,   # station_queue_delta
            0.5,   # station_load_imbalance
            1.0,   # bottleneck_CVaR  瓶颈拥堵极值
            -1.0,  # completed_orders_delta 完成任务
            2.0,   # deadlock_risk    死锁风险
        ]))

    def forward(self, system_preds: torch.Tensor) -> torch.Tensor:
        # system_preds 为 (K, 7) 维的张量，代表未来 K 步的系统级预测，每一步有 7 个指标（总等待时间、平均绕路延迟、站点队列增量、站点负载不均衡度、瓶颈拥堵极值、完成任务增量、死锁风险）。
        K = system_preds.size(0)
        #  gamma（Discount Factor）, 用于计算未来系统输出的折现权重。符合 RL 中的，step 越大，权重越小，反映了未来不确定性的增加和对当前决策的影响递减。
        discounts = torch.tensor(
            [self.gamma ** k for k in range(K)],
            device=system_preds.device,
        ) # 维度为 (K, )
        weighted = system_preds * self.lambdas.unsqueeze(0) # 维度是 (K, 7)，lambdas.unsqueeze(0) 维度是(1,7)，通过广播为(K,7) 后每个系统输出指标都乘以对应的权重 lambda。
        # 单独计算 deadlock_risk, 因为它是一个极值指标（只要未来某一步死锁风险高了，整个系统的死锁风险就高了），对它使用 max 操作来取未来 K 步中最大的那个值，而不是像其他指标那样使用折现和。
        risk_max = weighted[:, 6].max() # 死锁的风险代表值很大, 后续直接相加, 决策时取 min cost 的动作, 就会自然规避死锁风险高的动作
        non_risk = weighted[:, :6].sum(dim=-1)
        return (discounts * non_risk).sum() + risk_max


# ---------------------------------------------------------------------------
# Top-level world model
# ---------------------------------------------------------------------------

class RMFSWorldModel(nn.Module):
    """Assignment-conditioned counterfactual latent world model (v2)."""

    def __init__(
        self,
        node_feat_dim: int = 10,
        edge_feat_dim: int = 6,
        demand_dim: int = 9,
        action_node_dim: int = 8,
        action_global_dim: int = 6,
        hidden_dim: int = 64,
        num_spatial_layers: int = 3,
        rollout_horizon: int = 10,
        num_stations: int = 4,
    ):
        super().__init__()
        self.rollout_horizon = rollout_horizon

        self.state_encoder = STGNNEncoder(
            node_feat_dim=node_feat_dim, edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim, num_spatial_layers=num_spatial_layers,
        )
        self.demand_encoder = DemandEncoder(demand_dim, hidden_dim)
        self.action_encoder = ActionEncoder(action_node_dim, action_global_dim, hidden_dim)
        self.transition = LatentTransition(
            hidden_dim, hidden_dim, hidden_dim, edge_dim=hidden_dim,
        )
        self.node_decoder = NodeDecoder(hidden_dim)
        self.system_decoder = SystemDecoder(hidden_dim)
        self.station_decoder = StationDecoder(hidden_dim, num_stations)
        self.cost_head = CostHead()
        self.long_risk_head = LongRiskHead(hidden_dim)

    def encode_state(self, node_history, edge_index, edge_features, demand_context):
        """ Encode current state (shared across candidate actions).
            编码当前状态（跨候选动作共享）。
        Returns (z, e_demand, edge_attr).
        """
        z, edge_attr = self.state_encoder(node_history, edge_index, edge_features)
        e_demand = self.demand_encoder(demand_context)
        return z, e_demand, edge_attr

    def rollout(self, z, e_demand, edge_attr, action_node, action_global,
                edge_index, station_node_ids=None, K=None):
        """ Rollout K steps for one candidate action.
            针对单个候选动作执行K步推演。
        Returns (node_preds, system_preds, station_preds, uncertainties, z_0, z_K).
        z_0 is the initial latent (before any transition), z_K is the final.
        """
        if K is None:
            K = self.rollout_horizon

        N = z.size(0)
        u = self.action_encoder(action_node, action_global, N)

        node_preds = []
        system_preds = []
        station_preds = []
        uncertainties = []
        z_0 = z  # save initial latent for LongRiskHead
        z_k = z

        for k in range(K):
            z_k = self.transition(z_k, u, e_demand, k, edge_index, edge_attr=edge_attr)
            mu, log_sigma = self.node_decoder(z_k)
            sys_out = self.system_decoder(z_k)
            sta_out = self.station_decoder(z_k, station_node_ids)
            node_preds.append(mu)
            system_preds.append(sys_out)
            station_preds.append(sta_out)
            uncertainties.append(log_sigma)

        system_preds = torch.stack(system_preds, dim=0)
        station_preds = torch.stack(station_preds, dim=0)
        return node_preds, system_preds, station_preds, uncertainties, z_0, z_k

    def predict_cost(self, z, e_demand, edge_attr, action_node, action_global,
                     edge_index, station_node_ids=None, K=None,
                     return_details=False, risk_weight=None):
        """Full forward: rollout + cost for one candidate.

        Parameters
        ----------
        return_details : bool
            If True, return (cost, details_dict) where details_dict contains
            ``risk_max`` (raw predicted deadlock_risk max, before lambda) and
            ``system_preds`` (K, 7) tensor.
        risk_weight : float or None
            If not None, overrides checkpoint lambdas[6] for this call only.
        """
        score_horizon = self.rollout_horizon if K is None else int(K)
        _, system_preds, _, _, z_0, z_K = self.rollout(
            z, e_demand, edge_attr, action_node, action_global,
            edge_index, station_node_ids, K,
        )

        if risk_weight is not None:
            K_steps = system_preds.size(0)
            lambdas = self.cost_head.lambdas.detach().clone()
            lambdas[6] = risk_weight
            discounts = torch.tensor(
                [self.cost_head.gamma ** k for k in range(K_steps)],
                device=system_preds.device,
            )
            weighted = system_preds * lambdas.unsqueeze(0)
            risk_max_val = weighted[:, 6].max()
            non_risk = weighted[:, :6].sum(dim=-1)
            cost = (discounts * non_risk).sum() + risk_max_val
        else:
            cost = self.cost_head(system_preds)

        if return_details:
            long_risk_preds = self.long_risk_head(z_K, z_0)
            details = {
                "risk_max": system_preds[:, 6].max().item(),
                "system_preds": system_preds.detach(),
                "long_risk_preds": long_risk_preds.detach(),
                # Read-only finite-horizon endpoint contract.  These latents
                # let separately certified physical decoders test what the
                # frozen rollout actually represents without rerunning the
                # transition or introducing any TD/continuation semantics.
                "z_start": z_0.detach(),
                "z_endpoint": z_K.detach(),
                "rollout_horizon": score_horizon,
            }
            return cost, details
        return cost

    def continue_latent_rollout(
        self,
        z_k,
        e_demand,
        edge_attr,
        action_node,
        action_global,
        edge_index,
        *,
        start_step,
        end_step,
    ):
        """Continue an existing action-conditioned latent trajectory.

        Only the frozen transition dynamics are evaluated.  This helper does
        not decode labels, update demand, or invoke a continuation policy.
        ``start_step`` is the number of transitions already represented by
        ``z_k`` and ``end_step`` is the requested finite endpoint.
        """

        start_step = int(start_step)
        end_step = int(end_step)
        if start_step < 0 or end_step < start_step:
            raise ValueError("require 0 <= start_step <= end_step")
        if start_step == end_step:
            return z_k
        action = self.action_encoder(action_node, action_global, z_k.size(0))
        endpoint = z_k
        for step in range(start_step, end_step):
            endpoint = self.transition(
                endpoint,
                action,
                e_demand,
                step,
                edge_index,
                edge_attr=edge_attr,
            )
        return endpoint

    def forward(self, node_history, edge_index, edge_features, demand_context,
                action_node, action_global, station_node_ids=None, K=None):
        """Full forward pass for training. 训练的完整前向传播"""
        z, e_demand, edge_attr = self.encode_state(
            node_history, edge_index, edge_features, demand_context
        )

        if K is not None:
            return self.rollout(
            z, e_demand, edge_attr, action_node, action_global,
            edge_index, station_node_ids, K
            )
        else:
            return self.rollout(
                z, e_demand, edge_attr, action_node, action_global,
                edge_index, station_node_ids,
            )
