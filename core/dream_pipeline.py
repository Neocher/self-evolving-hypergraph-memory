"""
梦境整合管道（Layer5 核心）
=========================
GATHER → CLUSTER → SYNTHESIZE → COMPRESS → PRUNE → RESOLVE → AUDIT
                                      ^^^^^^^^
                                [Harness Fix] 新增 COMPRESS 步骤

在系统空闲时自动执行，将碎片化情节转化为结构化知识。
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from core.audit_chain import AuditOperation

try:
    import tiktoken
    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None

logger = logging.getLogger(__name__)

# LLM-NER 每社区节点数上限：超出的节点跳过 LLM 由调用方正则降级
# 防单社区节点多时串行 await（每节点 ~2.5s）→ SYNTHESIZE 阶段超时死循环
_NER_MAX_NODES_PER_COMMUNITY = 5

# 全局 LLM-NER 总调用数预算（per dream run）：超预算后剩余社区全部走正则
# 防 123 社区 × 5 节点 = 615 次调用超预算
_MAX_LLM_NER_TOTAL = 100

# NER 连续失败阈值：达到后 skip LLM → 正则（跨社区 fail-fast）
_NER_FAIL_FAST_THRESHOLD = 3

# 【v5.27.0】单次剪枝比例上限：待剪节点 > 50% 的活跃节点时中止本次剪枝（全部保留）。
# 兜底防"整批记忆被一次梦境清空"（2026-08-12 事故：9/9 全剪）。
# 分母 = run() 收到的全部活跃节点（含 protected），保证事故场景仍触发护栏。
_MAX_PRUNE_RATIO = 0.5


@dataclass
class DreamReport:
    """单次梦境执行的报告"""

    dream_id: str
    trigger_mode: str  # 'idle' | 'accum' | 'explicit'
    timestamp: float
    duration_seconds: float
    stats: dict  # created, updated, deleted 计数
    community_count: int
    prune_count: int
    conflict_count: int
    audit_block_hash: str
    compressed_topics: int = 0
    compressed_episodes: int = 0
    compressed_facts: int = 0
    keywords_extracted: int = 0
    pruned_node_ids: list[str] = field(default_factory=list)
    new_episode_ids: list[str] = field(default_factory=list)
    # 信心校准 (Manufactured Confidence, P1)
    calibrator_flagged: int = 0
    calibrator_high_consolidation: int = 0
    calibrator_tracked: int = 0
    # 【H5】PERSIST 阶段部分完成标记（半写状态，下次梦境可修复）
    degraded: bool = False


# ─── P2 SSM 梦境深度升级 ──────────────────────────────


@dataclass
class SSMDreamConfig:
    """SSM 梦境深度升级配置"""
    dream_rounds: int = 3
    state_reset_threshold: float = 0.95


class SSMDreamWrapper:
    """
    SSM 梦境深度升级包装器 (P2)

    不修改 SSMEngine，通过在社区特征向量上运行多轮 SSM step
    来模拟"梦境巩固"过程。对 gate_value 持续高于阈值的社区提升 confidence，
    低于阈值的则降低 confidence。
    """

    def __init__(self, ssm_engine, config: Optional[SSMDreamConfig] = None):
        self.ssm = ssm_engine
        self.config = config or SSMDreamConfig()
        self._state: Optional[np.ndarray] = None

    def init(self) -> np.ndarray:
        """初始化 SSM 隐状态为零向量。"""
        dim = self.ssm.config.hidden_dim
        self._state = np.zeros(dim)
        return self._state

    def reset(self) -> Optional[np.ndarray]:
        """重置 SSM 隐状态。"""
        if self._state is not None:
            self._state.fill(0.0)
        return self._state

    def dream_consolidate(self, feature_vector: np.ndarray,
                          current_confidence: float) -> tuple[float, np.ndarray]:
        """
        对社区特征向量运行 N 轮 SSM step，返回调整后的 confidence。

        Returns:
            (adjusted_confidence, final_hidden_state)
        """
        rounds = self.config.dream_rounds
        if self._state is None:
            self._state = np.zeros(feature_vector.shape[0])

        gate_values = []
        for _ in range(rounds):
            self._state, g = self.ssm.step(self._state, feature_vector)
            gate_values.append(g)

        avg_gate = sum(gate_values) / len(gate_values)
        th = self.config.state_reset_threshold
        if avg_gate > th:
            adjusted = min(1.0, current_confidence * (1.0 + 0.1 * avg_gate))
        elif avg_gate < (1.0 - th):
            adjusted = max(0.0, current_confidence * (1.0 - 0.1 * (1.0 - avg_gate)))
        else:
            adjusted = current_confidence

        return adjusted, self._state


class DreamPipeline:
    """
    梦境八步管道。

    GATHER:     收集所有未处理的节点
    CLUSTER:    运行 Leiden 社区检测
    SYNTHESIZE: 生成社区报告（可选用 LLM 语义摘要）
    COMPRESS:   [Harness Fix] 压缩社区报告，限制 Token 预算
    PRUNE:      删除 τ 低于阈值且低连接度的节点
    RESOLVE:    矛盾检测与消歧
    PERSIST:    写回结果到候选存储或 GraphLite
    AUDIT:      写入 BLAKE3 溯源链

    支持两种模式：
    - 直接模式 (candidate_store=None): 直接修改 GraphLite 生产数据（原行为）
    - 候选模式 (candidate_store 有效): 写入临时候选区，供审查后上线
    """

    def __init__(
        self,
        tau_engine=None,
        hebbian_updater=None,
        audit_chain=None,
        llm_client=None,
        ontology_validator=None,
        confidence_calibrator=None,
        ssm_engine=None,
        encoder=None,
    ) -> None:
        """
        Args:
            tau_engine: TauDecayEngine 实例
            hebbian_updater: SparseHebbianUpdater 实例
            audit_chain: AuditChain 实例
            llm_client: LLMClient 实例（可选，提供时启用 LLM 语义摘要）
            ontology_validator: OntologyValidator 实例（可选，P1 本体约束社区检测）
            confidence_calibrator: ConfidenceCalibrator 实例（可选，P1 过度巩固防护）
            ssm_engine: SSMEngine 实例（可选，P2 SSM 梦境深度巩固）
            encoder: 编码器实例（可选，用于向量余弦相似度合并 Jaccard）
        """
        self.tau_engine = tau_engine
        self.hebbian_updater = hebbian_updater
        self.audit_chain = audit_chain
        self.llm_client = llm_client
        self.ontology_validator = ontology_validator
        self.confidence_calibrator = confidence_calibrator
        self.ssm_engine = ssm_engine
        self.encoder = encoder
        self._ssm_wrapper: Optional[SSMDreamWrapper] = None
        self._ssm_initialized = False

    async def run(
        self,
        nodes: list[dict],
        connections: dict[str, dict[str, float]],
        trigger_mode: str = "explicit",
        graphlite_store=None,  # 接收GraphLite引用，用于持久化结果
        candidate_store=None,  # 可选：DreamCandidateStore，启用候选模式
    ) -> DreamReport:
        """
        执行完整梦境管道。

        1. GATHER     → 收集活跃节点，计算 τ 值
        2. CLUSTER    → Leiden 社区检测
        3. SYNTHESIZE → 模板化社区摘要生成
        4. COMPRESS   → 报告限 500 token，前 20 TF-IDF 关键词，输出预算控制
        5. PRUNE      → TauDecayEngine + Hebbian 剪枝
        6. RESOLVE    → 检测同名/同事实的多版本冲突
        7. PERSIST    → 【FIX】将结果写回GraphLite（CommunityNode/DELETE/合并/HyperedgeNode）
        8. AUDIT      → AuditChain.append_block()

        Args:
            nodes: [{"id": str, "content": str, "created_at": float, ...}, ...]
            connections: {node_id: {neighbor_id: weight, ...}, ...}
            trigger_mode: 'idle' | 'accum' | 'explicit'

        Returns:
            DreamReport 包含全部步骤的统计信息
        """
        start_time = time.time()
        dream_id = str(uuid.uuid4())
        audit_ops: list = []
        stats: dict[str, int] = {"created": 0, "updated": 0, "deleted": 0}

        # Step 1: GATHER — 收集活跃节点，附加 τ 值
        gathered = self._gather_step(nodes)
        logger.info("Dream %s: GATHER — %d active nodes", dream_id, len(gathered))

        # Step 2: CLUSTER — 社区检测（在独立线程中运行，避免阻塞事件循环）
        communities = await asyncio.to_thread(self._cluster_step, gathered, connections)
        logger.info("Dream %s: CLUSTER — %d communities", dream_id, len(communities))

        # Step 3: SYNTHESIZE — 生成社区摘要
        communities = await self._synthesize_step(communities)
        logger.info("Dream %s: SYNTHESIZE — %d reports generated", dream_id, len(communities))

        # Step 3a: SSM 梦境深度巩固 (P2) — 每次 run() 开始时重置 SSM 状态
        if self.ssm_engine is not None:
            if self._ssm_wrapper is None:
                self._ssm_wrapper = SSMDreamWrapper(self.ssm_engine)
            self._ssm_wrapper.reset()
            self._ssm_initialized = True
            for comm in communities:
                feature = self._build_community_feature(comm)
                current_conf = comm.get("confidence", 0.7)
                adjusted_conf, _ = self._ssm_wrapper.dream_consolidate(feature, current_conf)
                comm["confidence"] = round(adjusted_conf, 3)
            logger.info("Dream %s: SSM CONSOLIDATE — %d communities processed",
                        dream_id, len(communities))

        # Step 3b: CALIBRATE — 信心校准 (Manufactured Confidence, P1)
        calibrator_flagged = 0
        calibrator_high = 0
        calibrator_tracked = 0
        if self.confidence_calibrator is not None:
            for comm in communities:
                report_text = comm.get("report", "") or ""
                if not report_text:
                    continue
                # 先记录整合再校准（确保首次校准也有衰减）
                source = self._get_source_type(comm)
                self.confidence_calibrator.record_consolidation(report_text, source)
                cal_conf, flagged = self.confidence_calibrator.calibrate(
                    report_text, comm.get("confidence", 0.7), source
                )
                comm["confidence"] = round(cal_conf, 3)
                if flagged:
                    calibrator_flagged += 1
                    logger.info("Dream %s: CALIBRATOR flagged community (conf=%.2f)",
                                dream_id, cal_conf)
            s = self.confidence_calibrator.state()
            calibrator_high = s["high_consolidation"]
            calibrator_tracked = s["total_tracked"]
            logger.info("Dream %s: CALIBRATE — %d flagged, %d high-consolidation",
                        dream_id, calibrator_flagged, calibrator_high)

        # Step 4: COMPRESS — TF-IDF 压缩 + 预算控制
        communities, kw_count = self._compress_step(communities)
        logger.info("Dream %s: COMPRESS — %d keywords extracted", dream_id, kw_count)

        # Step 5: PRUNE — τ 衰减剪枝 + Hebbian 弱连接删除
        gathered, connections, prune_count, prune_ops = self._prune_step(
            gathered, connections
        )
        audit_ops.extend(prune_ops)
        stats["deleted"] += prune_count
        logger.info("Dream %s: PRUNE — %d nodes pruned", dream_id, prune_count)

        # Step 6: RESOLVE — 冲突检测
        merge_ops, conflict_count = self._resolve_step(communities, gathered)
        audit_ops.extend(merge_ops)
        stats["updated"] += conflict_count
        logger.info("Dream %s: RESOLVE — %d conflicts resolved", dream_id, conflict_count)

        # Step 7: PERSIST — 将结果写回GraphLite或候选存储
        persist_created = 0
        persist_deleted = 0
        persist_degraded = False  # 【H5】默认正常，PERSIST 部分失败时置 True
        all_removed_ids: list[str] = []  # FAISS 增量更新用
        
        # 先计算统计信息（用于候选存储和 report）
        topic_count = sum(1 for c in communities if c.get("topics"))
        episode_count = sum(len(c.get("episodes", [])) for c in communities)
        fact_count = sum(len(c.get("facts", [])) for c in communities)
        
        if candidate_store is not None:
            # 候选模式：写入临时候选集，不修改生产数据
            candidate_store.save_candidate(
                dream_id=dream_id,
                communities=communities,
                prune_ops=prune_ops,
                merge_ops=merge_ops,
                dream_report_kwargs={
                    "dream_id": dream_id,
                    "trigger_mode": trigger_mode,
                    "timestamp": start_time,
                    "stats": stats,
                    "community_count": len(communities),
                    "prune_count": prune_count,
                    "conflict_count": conflict_count,
                    "compressed_topics": topic_count,
                    "compressed_episodes": episode_count,
                    "compressed_facts": fact_count,
                    "keywords_extracted": kw_count,
                },
            )
            logger.info("Dream %s: saved to candidate store (review before apply)", dream_id)
        elif graphlite_store is not None:
            # 直接模式（原行为）：直接修改生产数据
            # 【H5】GraphLite 无跨语句事务（TransactionManager 的 rollback 仅做
            # tx_tag 清理，无法撤销裸 GQL 的 CREATE/DELETE），故不做伪事务包装；
            # 改为：任一步骤抛异常 → 打 degraded 标记，下次梦境通过 upsert 修复
            persist_degraded = False
            try:
                persist_deleted, pruned_ids = await asyncio.to_thread(
                    self._persist_prune, graphlite_store, prune_ops)
                all_removed_ids.extend(pruned_ids)
                persist_created = await asyncio.to_thread(
                    self._persist_communities, graphlite_store, communities, dream_id)
                all_removed_ids.extend(self._persist_merge_get_removed(merge_ops))
                await asyncio.to_thread(self._persist_merge, graphlite_store, merge_ops)
                await asyncio.to_thread(
                    self._persist_hyperedges, graphlite_store, communities, dream_id)
            except Exception as persist_exc:
                persist_degraded = True
                logger.error("Dream %s: PERSIST partial failure (degraded, next dream repairs): %s",
                             dream_id, persist_exc)
            logger.info("Dream %s: PERSIST — %d created, %d deleted, %d for FAISS cleanup%s",
                        dream_id, persist_created, persist_deleted, len(all_removed_ids),
                        " [DEGRADED]" if persist_degraded else "")
        stats["created"] += persist_created

        # Step 8: AUDIT — 写入溯源链
        audit_hash = ""
        if self.audit_chain:

            audit_stats = {
                "created": stats["created"],
                "updated": stats["updated"],
                "deleted": stats["deleted"],
                "before_size": len(nodes),
                "after_size": len(gathered),
            }
            block = self.audit_chain.append_block(audit_ops, audit_stats)
            audit_hash = block.hash

        duration = time.time() - start_time

        return DreamReport(
            dream_id=dream_id,
            trigger_mode=trigger_mode,
            timestamp=start_time,
            duration_seconds=round(duration, 3),
            stats=stats,
            community_count=len(communities),
            prune_count=prune_count,
            conflict_count=conflict_count,
            audit_block_hash=audit_hash,
            compressed_topics=topic_count,
            compressed_episodes=episode_count,
            compressed_facts=fact_count,
            keywords_extracted=kw_count,
            pruned_node_ids=all_removed_ids,
            calibrator_flagged=calibrator_flagged,
            calibrator_high_consolidation=calibrator_high,
            calibrator_tracked=calibrator_tracked,
            degraded=persist_degraded,
        )

    # ─── Step 1: GATHER ───────────────────────────────────

    def _gather_step(self, nodes: list[dict]) -> list[dict]:
        """
        收集活跃节点并计算 τ 值。
        过滤掉 τ 值过低且无连接的"死亡"节点。
        过滤掉被隔离（quarantine=true）的节点。
        """
        gathered: list[dict] = []
        for node in nodes:
            # 跳过隔离节点（记忆投毒防御）
            if node.get("quarantine") in (True, "true", 1):
                continue
            node_copy = dict(node)
            if self.tau_engine:
                created_at = node.get("created_at", time.time())
                tau = self.tau_engine.compute_strength(created_at, node_id=node.get("id"))
                node_copy["tau_value"] = tau
            else:
                node_copy["tau_value"] = node.get("tau_value", 1.0)
            gathered.append(node_copy)
        return gathered

    # ─── Step 2: CLUSTER ──────────────────────────────────

    def _cluster_step(
        self,
        nodes: list[dict],
        connections: dict[str, dict[str, float]],
    ) -> list[dict]:
        """
        [C3] 并行 Leiden 社区检测。

        将图拆分为连通分量，每个分量独立并行运行社区检测，
        最后合并结果。连通分量之间无边连接，天然独立可并行。
        """
        if len(nodes) < 2:
            return [
                {
                    "id": str(uuid.uuid4()),
                    "members": [n["id"] for n in nodes],
                    "nodes": nodes,
                    "report": "",
                    "episodes": [],
                    "facts": [],
                    "topics": [],
                    "keywords": [],
                }
            ]

        G = self._build_nx_graph(nodes, connections)
        import networkx as nx

        # 拆为连通分量：每个分量独立可并行处理
        components = list(nx.connected_components(G))
        logger.info("CLUSTER: %d connected components from %d nodes",
                    len(components), len(nodes))

        # 并行处理 ≥2 个节点的分量，1 节点分量直接跳过
        from concurrent.futures import ThreadPoolExecutor, as_completed
        big_components = [comp for comp in components if len(comp) >= 2]
        singleton_nodes = [list(comp)[0] for comp in components if len(comp) < 2]

        sub_results: list[dict[str, int]] = []
        if big_components:
            def _cluster_subgraph(comp_nodes: set[str]) -> dict[str, int]:
                """对单个连通分量运行社区检测。"""
                sub = G.subgraph(comp_nodes).copy()
                return self._detect_communities(sub)

            with ThreadPoolExecutor(max_workers=min(4, len(big_components))) as pool:
                futures = {pool.submit(_cluster_subgraph, comp): comp
                          for comp in big_components}
                for future in as_completed(futures):
                    try:
                        sub_results.append(future.result())
                    except Exception:
                        logger.exception("Subgraph clustering failed, "
                                         "falling back to singletons")
                        # 失败分量退回为单节点社区
                        for nid in futures[future]:
                            singleton_nodes.append(nid)

        # 合并所有分区
        partition: dict[str, int] = {}
        next_comm = 0
        for sp in sub_results:
            for nid, cid in sp.items():
                if nid not in partition:
                    partition[nid] = next_comm
                    next_comm += 1
        for nid in singleton_nodes:
            if nid not in partition:
                partition[nid] = next_comm
                next_comm += 1

        # 将节点按社区分组
        community_map: dict[int, list[dict]] = {}
        node_id_to_node = {n["id"]: n for n in nodes}
        for node_id, comm_id in partition.items():
            community_map.setdefault(comm_id, []).append(node_id_to_node[node_id])

        communities: list[dict] = []
        for comm_id, member_nodes in community_map.items():
            communities.append(
                {
                    "id": str(uuid.uuid4()),
                    "members": [n["id"] for n in member_nodes],
                    "nodes": member_nodes,
                    "report": "",
                    "episodes": [],
                    "facts": [],
                    "topics": [],
                    "keywords": [],
                }
            )
        return communities

    def _build_nx_graph(
        self,
        nodes: list[dict],
        connections: dict[str, dict[str, float]],
    ):
        """构建 NetworkX 图用于社区检测。

        [P1] 加入本体约束边：共享实体类型的节点之间添加弱连接，
        使语义相关的节点更容易聚到同一社区。
        """
        import networkx as nx

        G = nx.Graph()
        node_map = {n["id"]: n for n in nodes}
        for node in nodes:
            G.add_node(node["id"], **node)

        # Hebbian 连接
        for src, targets in connections.items():
            for dst, weight in targets.items():
                if G.has_node(src) and G.has_node(dst):
                    G.add_edge(src, dst, weight=max(weight, 0.01))

        # P1: 本体约束边 — 相同实体类型的节点间添加弱连接
        if self.ontology_validator is not None and len(nodes) > 1:
            MAX_ONT_NODES = 2000
            node_items = list(node_map.items())
            if len(node_items) > MAX_ONT_NODES:
                node_items = node_items[:MAX_ONT_NODES]
            ont_edge_count = 0
            for i, (nid_a, node_a) in enumerate(node_items):
                content_a = node_a.get("content", "")
                for nid_b, node_b in node_items[i + 1 :]:
                    if G.has_edge(nid_a, nid_b):
                        continue  # 已有连接，不覆盖
                    content_b = node_b.get("content", "")
                    types_a = self.ontology_validator._extract_types(content_a)
                    types_b = self.ontology_validator._extract_types(content_b)
                    if types_a and types_b:
                        type_set_a = {t.get("type", "") for t in types_a if t.get("type")}
                        type_set_b = {t.get("type", "") for t in types_b if t.get("type")}
                        shared_types = type_set_a & type_set_b
                        if shared_types:
                            G.add_edge(nid_a, nid_b, weight=0.15)
                            ont_edge_count += 1
            if ont_edge_count > 0:
                logger.info("P1 ontology edges added: %d", ont_edge_count)
        return G

    def _detect_communities(self, G) -> dict[str, int]:
        """
        运行社区检测算法。
        优先使用 cdlib Leiden，回退到 networkx Louvain，
        最后回退到连通分量。
        """
        # Leiden 分支（cdlib）
        try:
            from cdlib import algorithms as cdlib_algorithms
            from networkx import convert_node_labels_to_integers

            H = convert_node_labels_to_integers(G, label_attribute="_orig_id")  # 防属性冲突
            communities_list = cdlib_algorithms.leiden(H)
            # 还原原始节点 ID
            partition: dict[str, int] = {}
            for comm_idx, comm in enumerate(communities_list.communities):
                for int_id in comm:
                    orig_id = H.nodes[int_id].get("_orig_id", str(int_id))
                    partition[orig_id] = comm_idx
            # 未分配的孤立节点各自成社区
            next_comm = len(communities_list.communities)
            for node in G.nodes:
                if node not in partition:
                    partition[node] = next_comm
                    next_comm += 1
            return partition
        except ImportError:
            # cdlib 未装 / leidenalg 缺失 → 依赖不可用，回退 Louvain
            logger.warning("cdlib/leidenalg unavailable, falling back to Louvain", exc_info=True)
        except Exception:
            # 算法运行失败或胶水代码 bug → 回退，但按 error 记录避免掩盖真 bug
            logger.error("cdlib Leiden failed unexpectedly, falling back to Louvain", exc_info=True)

        # Louvain 分支（networkx）
        try:
            from networkx.algorithms.community import louvain_communities

            partition = {}
            for comm_idx, comm in enumerate(louvain_communities(G)):
                for node_id in comm:
                    partition[node_id] = comm_idx
            # 修复 next_comm bug：下一个可用社区 ID = max+1
            next_comm = (max(partition.values()) + 1) if partition else 0
            for node in G.nodes:
                if node not in partition:
                    partition[node] = next_comm
                    next_comm += 1
            return partition
        except Exception:
            logger.warning("networkx Louvain unavailable, falling back to connected components", exc_info=True)

        # 最终回退：连通分量
        from networkx import connected_components

        partition = {}
        for comm_idx, comm in enumerate(connected_components(G)):
            for node_id in comm:
                partition[node_id] = comm_idx
        return partition

    # ─── Step 3: SYNTHESIZE ───────────────────────────────

    async def _synthesize_step(self, communities: list[dict]) -> list[dict]:
        """为每个社区生成摘要。

        如果有 LLMClient，用模型生成语义摘要；
        否则用 TF-IDF 模板方法（原有回退）。

        引入 fail-fast 计数器：连续 3 个 LLM 超时 → 整个 synthesize 切换到模板回退，
        避免后续所有社区等待直到全部超时。

        NER 全局预算：_MAX_LLM_NER_TOTAL 次 LLM-NER 调用后，剩余社区实体提取走正则。
        NER fail-fast：连续 _NER_FAIL_FAST_THRESHOLD 次 NER 失败后 skip LLM。
        """
        sem = asyncio.Semaphore(5)
        _llm_fail_fast_counter = 0
        _llm_fail_fast_threshold = 3

        # LLM-NER 全局预算 & fail-fast 计数器（mutable list wrapper 跨方法传递）
        ner_budget = [_MAX_LLM_NER_TOTAL]
        ner_fails = [0]

        async def _llm_summarize(community: dict, contents: list[str]) -> dict:
            nonlocal _llm_fail_fast_counter
            if _llm_fail_fast_counter >= _llm_fail_fast_threshold:
                return None
            async with sem:
                try:
                    return await asyncio.wait_for(
                        self.llm_client.summarize_community(contents), timeout=15.0
                    )
                except Exception:
                    _llm_fail_fast_counter += 1
                    return None

        llm_tasks: list[asyncio.Task] = []
        for i, community in enumerate(communities):
            nodes = community.get("nodes", [])
            if self.llm_client and len(nodes) >= 2 and i < 20:
                contents = [n.get("content", "") for n in nodes]
                llm_tasks.append((community, nodes, _llm_summarize(community, contents)))
            else:
                community["report"] = self._generate_community_report(nodes)
                community["keywords"] = self._extract_keywords(
                    [n.get("content", "") for n in nodes], max_features=10
                )
                llm_tasks.append((community, nodes, None))

        for community, nodes, task in llm_tasks:
            if task is not None:
                llm_result = await task
                if llm_result:
                    community["report"] = llm_result["summary"]
                    community["keywords"] = llm_result["keywords"]
                    community["llm_patterns"] = llm_result["patterns"]
                    community["llm_contradictions"] = llm_result["contradictions"]
                else:
                    community["report"] = self._generate_community_report(nodes)
                    community["keywords"] = self._extract_keywords(
                        [n.get("content", "") for n in nodes], max_features=10
                    )
            community["generated_at"] = time.time()
            community["episodes"] = [
                {"id": n["id"], "content": n.get("content", "")} for n in nodes
            ]
            community["facts"] = [
                {"id": n["id"], "content": n.get("content", "")[:200]} for n in nodes
            ]
            community["topics"] = self._extract_topics(nodes)
            community["entity_links"] = await self._entity_linking_step(
                nodes, ner_budget=ner_budget, ner_fails=ner_fails
            )
        return communities

    def _build_community_feature(self, community: dict) -> np.ndarray:
        """从社区数据构建 9 维 SSM 输入特征向量。"""
        nodes = community.get("nodes", [])
        members = community.get("members", [])
        f = np.zeros(9)
        f[0] = sum(n.get("importance", 0.5) for n in nodes) / max(len(nodes), 1)
        now_t = time.time()
        ages = [now_t - n.get("created_at", now_t) for n in nodes if n.get("created_at")]
        f[1] = (sum(ages) / max(len(ages), 1)) / 3600.0 if ages else 0.0
        f[2] = sum(n.get("access_count", 0) for n in nodes) / max(len(nodes), 1)
        f[3] = min(len(members) / 100.0, 1.0)
        f[4] = 0.0
        taus = [n.get("tau_value", n.get("tau_initial", 0.5)) for n in nodes]
        f[5] = sum(taus) / max(len(taus), 1)
        f[6] = float(np.var(taus)) if len(taus) > 1 else 0.0
        f[7] = 0.0
        f[8] = community.get("confidence", 0.7)
        return f

    def _get_source_type(self, community: dict) -> str:
        """从社区成员节点推断源类型。"""
        nodes = community.get("nodes", [])
        source_types: dict[str, int] = {}
        for node in nodes:
            st = node.get("source_type", node.get("source", ""))
            if st:
                source_types[st] = source_types.get(st, 0) + 1
        return max(source_types, key=source_types.get) if source_types else "inferred"

    def _generate_community_report(self, nodes: list[dict]) -> str:
        """生成社区报告——基于模板的方法。"""
        if not nodes:
            return "Empty community"

        contents = [n.get("content", "") for n in nodes]
        keywords = self._extract_keywords(contents, max_features=10)

        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
        lines = [
            f"Community Size: {len(nodes)} nodes",
            f"Generated At: {timestamp_str}",
            f"Keywords: {', '.join(keywords)}" if keywords else "",
            "Member Nodes Summary:",
        ]
        for node in nodes[:5]:
            content = node.get("content", "")
            lines.append(f"- {content[:100]}")
        return "\n".join(lines)

    # ─── 实体链接 ──────────────────────────────────────────

    async def _entity_linking_step(
        self,
        nodes: list[dict],
        ner_budget: Optional[list[int]] = None,
        ner_fails: Optional[list[int]] = None,
    ) -> list[dict]:
        """
        实体链接：提取命名实体并在社区节点间交叉匹配。

        两步策略:
        1. 用 LLM 做命名实体识别（降级：正则提取大写词和引号内专名）
        2. 跨节点匹配相同实体，生成实体链接关系

        ner_budget: 全局 NER 调用预算（传给 _extract_entities_from_nodes）
        ner_fails: 连续失败计数器（传给 _extract_entities_from_nodes）

        Args:
            nodes: 社区内节点列表

        Returns:
            entity_links: [{"entity": str, "occurrences": [node_id, ...], "count": int}, ...]
            按出现次数降序排列，失败时返回空列表
        """
        try:
            # Step 1: 提取实体 — LLM 优先 + 正则兜底（受 budget / fail-fast 约束）
            entity_map = await self._extract_entities_from_nodes(
                nodes, ner_budget=ner_budget, ner_fails=ner_fails
            )
            if not entity_map:
                return []

            # Step 2: 交叉匹配 — 统计每个实体出现在哪些节点
            entity_links: dict[str, dict] = {}
            for nid, entities in entity_map.items():
                for ent in entities:
                    if ent not in entity_links:
                        entity_links[ent] = {"entity": ent, "occurrences": [], "count": 0}
                    if nid not in entity_links[ent]["occurrences"]:
                        entity_links[ent]["occurrences"].append(nid)
                        entity_links[ent]["count"] += 1

            # 只保留出现 ≥ 2 次的实体（跨节点链接才有意义）
            result = [
                v for v in entity_links.values() if v["count"] >= 2
            ]
            result.sort(key=lambda x: x["count"], reverse=True)
            return result
        except Exception:
            logger.exception("Entity linking failed, skipping")
            return []  # 降级：异常时返回空列表

    async def _extract_entities_from_nodes(
        self,
        nodes: list[dict],
        ner_budget: Optional[list[int]] = None,
        ner_fails: Optional[list[int]] = None,
    ) -> dict[str, list[str]]:
        """
        统一实体提取入口。

        优先用 LLM 做命名实体识别（受预算和 fail-fast 约束），
        LLM 未覆盖的节点（cap 外 / 单节点失败 / 预算耗尽）降级到正则提取。
        返回 {node_id: [entity_name, ...], ...}

        LLM 结果优先；正则兜底保证所有节点都有机会提取实体。
        """
        llm_result = await self._ner_with_llm(nodes, ner_budget=ner_budget, ner_fails=ner_fails)

        # Merge: LLM 结果优先，正则填补空缺（cap 外 + LLM 失败 + 预算耗尽节点）
        result: dict[str, list[str]] = dict(llm_result) if llm_result else {}
        for node in nodes:
            nid = node.get("id", "")
            if nid in result:
                continue  # LLM 已覆盖
            content = node.get("content", "")
            if content:
                entities = self._extract_entities_regex(content)
                if entities:
                    result[nid] = entities
        return result

    def _extract_entities_regex(self, content: str) -> list[str]:
        """
        正则提取命名实体。

        提取规则:
        - 中文：连续2个以上的中文字符（排除标点和停用词）
        - 英文：首字母大写的连续单词（专名）
        - 引号内文本

        Args:
            content: 节点文本内容

        Returns:
            去重后的实体名称列表
        """
        entities: set[str] = set()

        # 提取引号内的内容（中英文引号）
        for match in re.finditer(r'["「『""]([^"「』""]{2,50})["」』""]', content):
            name = match.group(1).strip()
            if name and len(name) >= 2:
                entities.add(name)

        # 提取首字母大写的英文专名（2-4个连续大写词）
        for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b', content):
            name = match.group(1).strip()
            if name and len(name) >= 2 and name.lower() not in {
                "this", "that", "the", "what", "when", "where", "which",
                "there", "these", "those", "then", "than", "also", "with", "from",
            }:
                entities.add(name)

        # 提取连续大写缩写（2-8个字符）
        for match in re.finditer(r'\b([A-Z]{2,8})\b', content):
            name = match.group(1)
            if name not in {"AI", "API", "I", "II", "III", "IV", "VI"}:
                entities.add(name)

        return sorted(entities)

    def _build_ner_prompt(self, content: str) -> str:
        return f"""Extract named entities from the following text. Return ONLY a JSON array of entity names (people, organizations, technologies, products, locations, projects).

Rules:
- Include proper nouns, technical terms, project names, organization names
- Exclude common words, generic terms, numbers, dates
- Return as a JSON array of strings only
- If no entities found, return an empty array []

Text:
{content[:500]}"""

    def _parse_ner_response(self, response: str) -> list[str]:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            entities = parsed.get("entities", [])
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if str(e).strip()]
        if isinstance(parsed, list):
            return [str(e).strip() for e in parsed if str(e).strip()]
        return []

    async def _ner_single_node(self, nid: str, content: str, sem: asyncio.Semaphore) -> tuple[str, list[str]]:
        prompt = self._build_ner_prompt(content)
        async with sem:
            response = await asyncio.wait_for(
                self.llm_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                ),
                timeout=15.0,
            )
        if response:
            return (nid, self._parse_ner_response(response))
        return (nid, [])

    async def _ner_with_llm(
        self,
        nodes: list[dict],
        ner_budget: Optional[list[int]] = None,
        ner_fails: Optional[list[int]] = None,
    ) -> dict[str, list[str]]:
        """
        用 LLM 从节点内容中并行提取命名实体（Semaphore(5) 限流）。

        节点数超过 _NER_MAX_NODES_PER_COMMUNITY 时仅处理内容最长的前 N 个，
        其余节点跳过 LLM，由 _extract_entities_from_nodes 的正则降级覆盖。
        单节点失败不中断其他节点。

        ner_budget: 全局 NER 调用预算计数器 (list[int])，每调用减 1；0 时跳过 LLM
        ner_fails: 连续失败计数器 (list[int])，达 _NER_FAIL_FAST_THRESHOLD 时跳过 LLM

        Returns:
            {node_id: [entity_name, ...], ...} 或 {}（LLM 不可用时）
        """
        if not self.llm_client or not self.llm_client.api_key:
            return {}

        # 预算耗尽或 fail-fast 触发 → 跳过 LLM
        if ner_budget is not None and ner_budget[0] <= 0:
            return {}
        if ner_fails is not None and ner_fails[0] >= _NER_FAIL_FAST_THRESHOLD:
            return {}

        eligible = [
            (node.get("id", ""), node.get("content", ""))
            for node in nodes
            if node.get("content", "") and len(node.get("content", "").strip()) >= 5
        ]
        if not eligible:
            return {}

        eligible.sort(key=lambda x: len(x[1]), reverse=True)
        max_nodes = _NER_MAX_NODES_PER_COMMUNITY
        if ner_budget is not None:
            max_nodes = min(max_nodes, ner_budget[0])
        capped = eligible[:max_nodes]

        sem = asyncio.Semaphore(5)
        tasks = [
            self._ner_single_node(nid, content, sem)
            for nid, content in capped
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 扣除预算
        if ner_budget is not None:
            ner_budget[0] -= len(capped)

        # 跟踪失败计数（用于 fail-fast）
        n_failed = sum(1 for item in results if isinstance(item, Exception))
        if ner_fails is not None:
            if n_failed > 0:
                ner_fails[0] += n_failed
            else:
                ner_fails[0] = 0  # 全部成功 → 重置

        entity_map: dict[str, list[str]] = {}
        for item in results:
            if isinstance(item, Exception):
                continue
            nid, entities = item
            if entities:
                entity_map[nid] = entities
        return entity_map

    def _extract_topics(self, nodes: list[dict]) -> list[str]:
        """从社区节点中提取主题关键词。"""
        contents = [n.get("content", "") for n in nodes]
        return self._extract_keywords(contents, max_features=3)

    def _extract_keywords(self, texts: list[str], max_features: int = 10) -> list[str]:
        """提取关键词——优先 TF-IDF，回退到词频统计。"""
        non_empty = [t for t in texts if t.strip()]
        if not non_empty:
            return []

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                max_features=max_features, stop_words="english"
            )
            vectorizer.fit_transform(non_empty)
            return list(vectorizer.get_feature_names_out())
        except (ImportError, ValueError):
            pass

        # 简单词频回退
        word_freq: dict[str, int] = {}
        for text in non_empty:
            for word in text.lower().split():
                word = "".join(c for c in word if c.isalpha())
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:max_features]]

    # ─── Step 4: COMPRESS ─────────────────────────────────

    @staticmethod
    def _count_tokens(text: str) -> int:
        if not text:
            return 0
        if _TIKTOKEN_ENC is not None:
            return len(_TIKTOKEN_ENC.encode(text))
        has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
        if has_cjk:
            return int(len(text) * 2.5 / 1.5)
        return len(text.split())

    @staticmethod
    def _truncate_tokens(text: str, max_tokens: int) -> str:
        if not text:
            return text
        if _TIKTOKEN_ENC is not None:
            tokens = _TIKTOKEN_ENC.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return _TIKTOKEN_ENC.decode(tokens[:max_tokens])
        words = text.split()
        has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf]', text))
        if has_cjk:
            ratio = max_tokens / max(1, len(text) * 2.5 / 1.5)
            cutoff = int(len(text) * min(1.0, ratio))
            return text[:cutoff]
        if len(words) > max_tokens:
            return " ".join(words[:max_tokens])
        return text

    def _compress_step(self, communities: list[dict]) -> tuple[list[dict], int]:
        """
        [Harness Fix] COMPRESS 步骤实现。

        功能：
        - 社区报告限制 500 token 以内
        - 每个情节摘要提取前 20 个 TF-IDF 关键词
        - 每层输出预算：主题 ≤3 个，情节 ≤20 个，事实 ≤50 条
        """
        total_keywords = 0
        for community in communities:
            report = community.get("report", "")
            if self._count_tokens(report) > 500:
                community["report"] = self._truncate_tokens(report, 500)

            episodes = community.get("episodes", [])
            if episodes:
                texts = [ep.get("content", "") for ep in episodes]
                keywords = self._extract_keywords(texts, max_features=20)
                community["keywords"] = keywords
                total_keywords += len(keywords)
            else:
                community["keywords"] = []

            community["topics"] = community.get("topics", [])[:3]
            community["episodes"] = community.get("episodes", [])[:20]
            community["facts"] = community.get("facts", [])[:50]

        return communities, total_keywords

    # ─── Step 5: PRUNE ────────────────────────────────────

    def _prune_step(
        self,
        nodes: list[dict],
        connections: dict[str, dict[str, float]],
    ) -> tuple[list[dict], dict[str, dict[str, float]], int, list]:
        """
        剪枝步骤：删除 τ 值低于阈值的节点及其连接。
        使用 Hebbian 更新器剪枝每个节点的弱连接。
        """

        prune_ops: list = []
        keep_nodes: list[dict] = []
        pruned_ids: set[str] = set()

        for node in nodes:
            tau = node.get("tau_value", 1.0)
            node_id = node["id"]
            degree = len(connections.get(node_id, {}))
            created_at = node.get("created_at", 0)
            age_seconds = time.time() - created_at
            # 保护规则：新节点（< 2h）、高 τ（> 0.3）、或高连接度 不剪枝
            # 【v5.27.0】force_promote 节点打 protected 标记 → 永不剪（方案①）
            is_protected = (
                node.get("protected") in (True, "true", 1)
                or (age_seconds < 7200) or (tau > 0.3) or (degree > 1)
            )
            if not is_protected and self.tau_engine and tau < self.tau_engine.config.decay_threshold:
                pruned_ids.add(node_id)
                prune_ops.append(
                    AuditOperation(
                        op_type="delete",
                        node_id=node_id,
                        old_value=node.get("content", ""),
                        reason="tau_decay",
                    )
                )
            else:
                keep_nodes.append(node)

        # 【v5.27.0】批量剪枝上限保护（方案②）：单次剪枝 > 50% 活跃节点 → 中止，全部保留。
        # 返回原 nodes（而非 keep_nodes，后者已剔除候选且不含保护标记），
        # 下游 RESOLVE/PERSIST 在完整节点集上照常运行；返回 0, [] → _persist_prune 无操作。
        total = len(nodes)
        if total > 0 and len(pruned_ids) > total * _MAX_PRUNE_RATIO:
            logger.warning(
                "PRUNE aborted: %d/%d nodes (%d%%) exceed 50%% batch limit — all kept",
                len(pruned_ids), total, round(len(pruned_ids) / total * 100),
            )
            return nodes, connections, 0, []

        # 清理被剪枝节点的连接
        clean_connections: dict[str, dict[str, float]] = {}
        for src, targets in connections.items():
            if src in pruned_ids:
                continue
            clean_connections[src] = {
                dst: w for dst, w in targets.items() if dst not in pruned_ids
            }

        # Hebbian 剪枝：每个节点只保留 K 个最强连接
        if self.hebbian_updater:
            for nid in list(clean_connections.keys()):
                conns = clean_connections.get(nid, {})
                if len(conns) > self.hebbian_updater.config.k_sparsity:
                    clean_connections[nid] = self.hebbian_updater.prune_connections(conns)

        # 【P4】τ-Hebbian 联动：低 τ 节点连接权重衰减
        if self.hebbian_updater and self.tau_engine:
            tau_map = {n["id"]: n.get("tau_value", n.get("tau_initial", 1.0))
                       for n in keep_nodes}
            clean_connections = self.hebbian_updater.tau_decay_connections(
                clean_connections, tau_map,
                decay_threshold=self.tau_engine.config.decay_threshold,
            )

        return keep_nodes, clean_connections, len(pruned_ids), prune_ops

    # ─── Step 6: RESOLVE ──────────────────────────────────

    def _resolve_step(
        self,
        communities: list[dict],
        nodes: list[dict],
    ) -> tuple[list, int]:
        """
        冲突检测与消歧。

        检测内容高度相似（Jaccard ≥ 0.8）的同社区节点对，
        合并为单条记录。
        """

        merge_ops: list = []
        conflict_count = 0

        for community in communities:
            members = community.get("nodes", [])
            if len(members) < 2:
                continue
            merged, ops = self._find_and_merge_conflicts(members)
            merge_ops.extend(ops)
            conflict_count += len(ops)

        return merge_ops, conflict_count

    def _find_and_merge_conflicts(self, nodes: list[dict]) -> tuple[list[dict], list]:
        """在一个社区内检测并合并冲突节点对。"""

        ops: list = []
        merged: set[str] = set()

        for i in range(len(nodes)):
            if nodes[i]["id"] in merged:
                continue
            # 【v5.27.0】方案①防合并击穿：protected 节点永不参与合并（既不作胜者也
            # 不作被合并方）。否则普通节点 τ 更高时 protected 节点作 loser 被
            # DETACH DELETE → "永久保留"语义被间接击穿。
            if nodes[i].get("protected") in (True, "true", 1):
                continue
            for j in range(i + 1, len(nodes)):
                if nodes[j]["id"] in merged:
                    continue
                if nodes[j].get("protected") in (True, "true", 1):
                    continue
                sim = self._combined_similarity(
                    nodes[i].get("content", ""), nodes[j].get("content", "")
                )
                if sim >= 0.8:
                    # 合并：保留 τ 值更高的节点
                    tau_i = nodes[i].get("tau_value", 0)
                    tau_j = nodes[j].get("tau_value", 0)
                    if tau_i >= tau_j:
                        merged.add(nodes[j]["id"])
                        loser, winner = nodes[j], nodes[i]
                    else:
                        merged.add(nodes[i]["id"])
                        loser, winner = nodes[i], nodes[j]
                    ops.append(
                        AuditOperation(
                            op_type="update",
                            node_id=loser["id"],
                            old_value=loser.get("content", ""),
                            new_value=winner["id"],
                            reason="community_merge",
                        )
                    )

        remaining = [n for n in nodes if n["id"] not in merged]
        return remaining, ops

    # ─── 【FIX】GraphLite持久化方法 ──────────────────────────────

    def _persist_communities(self, graphlite_store, communities: list[dict], dream_id: str) -> int:
        """将CLUSTER结果写回GraphLite CommunityNode。

        使用 MERGE ON id 增量 upsert，避免先 DETACH DELETE 全部再重建
        （若中途崩溃则所有社区数据永久丢失）。
        保留的历史社区由外部周期性清理策略处理。

        先清理旧 COMMUNITY_MEMBER 边再 upsert，避免竞赛条件导致
        一个 EpisodeNode 关联到 2 个 CommunityNode。
        """
        import json
        created = 0
        new_member_sets: dict[str, set[str]] = {}
        # 收集所有成员 ID（用于后续阶段）
        all_member_ids: set[str] = set()
        for comm in communities:
            for member_id in comm.get("members", []):
                all_member_ids.add(member_id)

        # 先清理自己社区的旧 COMMUNITY_MEMBER 边（不碰外部社区）
        # 【FIX 2026-08-09】原实现 MATCH (c:CommunityNode) 无社区过滤 →
        # 对每个 dream 成员删除所有社区（含外部）指向它的边（实证 EXT→epX 被删）。
        # 修复：限定 {id: $cid}，与 dream_candidate_store.py L368-376 一致。
        try:
            for comm in communities:
                for member_id in comm.get("members", []):
                    graphlite_store.query_cypher(
                        "MATCH (c:CommunityNode {id: $cid})"
                        "-[r:COMMUNITY_MEMBER]->"
                        "(e:EpisodeNode {id: $eid}) DELETE r",
                        {"cid": comm["id"], "eid": member_id},
                    )
        except Exception:
            logger.warning("Failed to clean before community upsert", exc_info=True)
        for comm in communities:
            try:
                # GraphLite 不支持 MERGE：MATCH 存在性检查 + INSERT 建节点 / SET 更新属性
                comm_vals = {
                    "id": comm["id"],
                    "name": f"dream_{dream_id[:8]}_comm_{created}",
                    "summary": comm.get("report", "")[:800],
                    "score": 0.0,
                    "created_at": time.time(),
                }
                if graphlite_store.execute_cypher(
                    "MATCH (c:CommunityNode {id: $id}) RETURN c",
                    {"id": comm["id"]},
                ):
                    graphlite_store.execute_cypher(
                        "MATCH (c:CommunityNode {id: $id}) "
                        "SET c.name = $name, c.summary = $summary, "
                        "c.leiden_score = $score, c.created_at = $created_at",
                        comm_vals,
                    )
                else:
                    graphlite_store.execute_cypher(
                        "INSERT (c:CommunityNode {id: $id, name: $name, "
                        "summary: $summary, leiden_score: $score, "
                        "created_at: $created_at})",
                        comm_vals,
                    )
                member_set: set[str] = set()
                for member_id in comm.get("members", []):
                    member_set.add(member_id)
                    try:
                        # GraphLite 不支持 MERGE：MATCH 边存在性检查 + INSERT（幂等）
                        if not graphlite_store.execute_cypher(
                            "MATCH (c:CommunityNode {id: $cid})"
                            "-[:COMMUNITY_MEMBER]->"
                            "(e:EpisodeNode {id: $eid}) RETURN c",
                            {"cid": comm["id"], "eid": member_id},
                        ):
                            graphlite_store.execute_cypher(
                                "MATCH (c:CommunityNode {id: $cid}), "
                                "(e:EpisodeNode {id: $eid}) "
                                "INSERT (c)-[:COMMUNITY_MEMBER]->(e)",
                                {"cid": comm["id"], "eid": member_id},
                            )
                    except Exception:
                        logger.warning("Failed to CREATE COMMUNITY_MEMBER edge", exc_info=True)
                new_member_sets[comm["id"]] = member_set
                created += 1
            except Exception as e:
                logger.warning("Community persist failed: %s", e)
        # 【FIX 2026-08-09】同源湮灭 bug：原实现用 WHERE c.id <> $cid DELETE r
        # 对每个 (cid, member) 删其他所有社区的边（含外部社区）→ 共享成员被互删
        # → 孤儿（属零个社区）。新实现：按 member_count 倒序，每成员只保留最大
        # 社区的边，只删自己社区的边，不动外部社区。
        # 与 dream_candidate_store.py 修复一致。
        #
        # 阶段 3a：按 member_count 倒序建 max_community_by_member 映射
        max_community_by_member: dict[str, str] = {}
        sorted_cids = sorted(
            new_member_sets.keys(),
            key=lambda cid: len(new_member_sets[cid]),
            reverse=True,
        )
        for cid in sorted_cids:
            for mid in new_member_sets[cid]:
                if mid not in max_community_by_member:
                    max_community_by_member[mid] = cid

        # 阶段 3b：只删自己社区到非最大成员的边（不动外部社区）
        try:
            for cid, members in new_member_sets.items():
                for mid in members:
                    if max_community_by_member.get(mid, cid) != cid:
                        graphlite_store.query_cypher(
                            "MATCH (c:CommunityNode {id: $cid})"
                            "-[r:COMMUNITY_MEMBER]->"
                            "(e:EpisodeNode {id: $eid}) DELETE r",
                            {"cid": cid, "eid": mid}
                        )
        except Exception:
            logger.warning("Failed to clean up stale COMMUNITY_MEMBER edges", exc_info=True)
        return created

    def _persist_prune(self, graphlite_store, prune_ops: list) -> tuple[int, list[str]]:
        """将PRUNE剪枝结果执行真实的GraphLite DELETE。

        先删边再删节点，避免GraphLite外键约束错误。
        返回 (删除数量, 被删节点ID列表)
        """
        deleted = 0
        pruned_ids: list[str] = []
        for op in prune_ops:
            if op.op_type == "delete":
                try:
                    # 先用 DETACH 删掉所有指向该节点的边
                    graphlite_store.query_cypher(
                        "MATCH (e:EpisodeNode {id: $id}) DETACH DELETE e",
                        {"id": op.node_id}
                    )
                    deleted += 1
                    pruned_ids.append(op.node_id)
                except Exception:
                    logger.warning("Failed to DETACH DELETE pruned node from GraphLite", exc_info=True)
        return deleted, pruned_ids

    def _persist_merge(self, graphlite_store, merge_ops: list) -> None:
        """将RESOLVE合并结果写回GraphLite（打标记 + DETACH DELETE被合并节点）。"""
        for op in merge_ops:
            if op.op_type == "update" and op.new_value:
                try:
                    # 把被合并节点的内容保存到目标节点
                    graphlite_store.query_cypher(
                        "MATCH (target:EpisodeNode {id: $target}) "
                        "SET target.content = target.content + ' | merged: ' + $content",
                        {"target": op.new_value, "content": op.old_value or ""}
                    )
                    # 删除被合并节点（DETACH先删边）
                    graphlite_store.query_cypher(
                        "MATCH (e:EpisodeNode {id: $id}) DETACH DELETE e",
                        {"id": op.node_id}
                    )
                except Exception:
                    logger.warning("Failed to persist merge resolution in GraphLite", exc_info=True)

    @staticmethod
    def _persist_merge_get_removed(merge_ops: list) -> list[str]:
        """从 merge_ops 中提取被删除的节点ID（给FAISS增量更新用）。"""
        return [op.node_id for op in merge_ops
                if op.op_type == "update" and op.node_id]

    def _persist_hyperedges(self, graphlite_store, communities: list[dict], dream_id: str) -> int:
        """梦境结束后，为每个社区创建HyperedgeNode（Layer4）。"""
        import json
        created = 0
        for comm in communities:
            members = comm.get("members", [])
            if len(members) < 2:
                continue
            try:
                hyperedge_id = str(uuid.uuid4())
                metadata = json.dumps({
                    "dream_id": dream_id,
                    "community_id": comm["id"],
                    "keywords": comm.get("keywords", []),
                }, ensure_ascii=False)
                graphlite_store.query_cypher(
                    "CREATE (h:HyperedgeNode {id: $id, type: 'semantic', "
                    "created_at: $created_at, gate_value: 1.0, metadata: $metadata})",
                    {"id": hyperedge_id, "created_at": time.time(), "metadata": metadata}
                )
                for member_id in members:
                    try:
                        graphlite_store.query_cypher(
                            "MATCH (h:HyperedgeNode {id: $hid}), (e:EpisodeNode {id: $eid}) "
                            "INSERT (h)-[:HYPEREDGE_MEMBER]->(e)",
                            {"hid": hyperedge_id, "eid": member_id}
                        )
                    except Exception:
                        logger.warning("Failed to INSERT HYPEREDGE_MEMBER edge", exc_info=True)
                created += 1
            except Exception as e:
                logger.warning("Hyperedge persist failed: %s", e)
        return created

    def _jaccard_similarity(self, text_a: str, text_b: str) -> float:
        """计算两个文本的 Jaccard 相似度（基于词集）。"""
        if not text_a or not text_b:
            return 0.0
        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union > 0 else 0.0

    def _combined_similarity(self, text_a: str, text_b: str) -> float:
        """加权合并：Jaccard 词集 + 向量余弦相似度。"""
        jac = self._jaccard_similarity(text_a, text_b)
        if self.encoder is not None:
            try:
                emb_a = self.encoder.embed(text_a)
                emb_b = self.encoder.embed(text_b)
                if emb_a is not None and emb_b is not None:
                    norm_a = np.linalg.norm(emb_a)
                    norm_b = np.linalg.norm(emb_b)
                    if norm_a > 0 and norm_b > 0:
                        cos = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                        return 0.4 * jac + 0.6 * max(0.0, cos)
            except Exception:
                pass
        return jac
