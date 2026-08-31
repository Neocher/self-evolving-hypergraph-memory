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


class _EntityView:
    """EntityNode props dict → duck-typing 视图（供 extract_attributes 消费）。

    属性对齐 attribute_extractor 期望的接口：name / entity_type / entity_id / aliases。
    """

    __slots__ = ("name", "entity_type", "entity_id", "aliases")

    def __init__(self, props: dict):
        self.name = str(props.get("name") or props.get("norm_name") or "")
        self.entity_type = str(props.get("entity_type") or "Person")
        self.entity_id = str(props.get("id") or "")
        self.aliases = list(props.get("aliases") or [])

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

# 【P3】RESOLVE Jaccard 预筛阈值：jac < 0.15 时跳过余弦编码。
# 数学保证：sim = 0.4·jac + 0.6·cos ≤ 0.4·0.15 + 0.6 = 0.66 < 0.8（合并阈值）
# → 预筛跳过的文本对合并决策与全量计算完全一致，纯省 encoder.embed 调用。
_JACCARD_PRESCREEN_THRESHOLD = 0.15


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
        write_queue=None,
        ontology_evolution=None,
        retrieval_guard=None,
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
            write_queue: WriteQueue 实例（可选，存在时 PERSIST 经单写线程 submit 串行执行）
            ontology_evolution: OntologyEvolution 实例（可选，v5.38.0 SYNTHESIZE 后 Schema 自演化）
            retrieval_guard: SelfEvolvingRetrieval 实例（可选，检索健康探针信号入口；
                探针直调其内层 _qr，不调 retrieve() 以免自增 _total_calls 干扰周期触发）
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
        self._write_queue = write_queue
        self.ontology_evolution = ontology_evolution
        self.retrieval_guard = retrieval_guard

    async def retrieval_health_probe(self, nodes: list[dict],
                                     sample_size: int = 5) -> float:
        """梦境侧检索健康探针（离线、低频、不污染热路径）。

        抽核心节点（高 τ = 近期写入/重要）→ content 片段作 query → 直调内层
        QueryRouter（不经 SelfEvolvingRetrieval.retrieve，避免自增
        _total_calls 干扰周期触发）→ 计算 top-10 命中率（recall）→
        低召回时喂给 retrieval_guard.report_probe() 作为触发信号。

        【H7 2026-08-28】sample_size 30→5：每次检索为全量 QueryRouter
        (向量+BM25+图+fusion)，30 次≈450s 导致 dream 必超时 + to_thread
        孤儿线程持续吃 CPU。5 次(~75s) 仍可估 recall，回归可接受。
        """
        guard = self.retrieval_guard
        if guard is None or not nodes:
            return 0.0
        inner = getattr(guard, "_qr", None)
        if inner is None or not hasattr(inner, "retrieve"):
            return 0.0

        def _tau(n: dict) -> float:
            try:
                return float(n.get("tau_value") or n.get("tau_initial") or 1.0)
            except (TypeError, ValueError):
                return 1.0

        core = [
            n for n in sorted(nodes, key=_tau, reverse=True)[:sample_size]
            if len((n.get("content") or "").strip()) >= 8
        ]
        if not core:
            return 0.0

        hits = 0
        for node in core:
            content = (node.get("content") or "").strip()
            query = content[:40]
            try:
                results = await asyncio.to_thread(inner.retrieve, query,
                                                  include_archived=False)
            except Exception:
                continue
            results = results if isinstance(results, list) else []
            nid = node.get("id")
            c60 = content[:60]
            if any(
                isinstance(r, dict) and (r.get("node_id") == nid or
                                         c60 in (r.get("content") or ""))
                for r in results[:10]
            ):
                hits += 1

        recall = hits / max(1, len(core))
        guard.report_probe(recall, len(core))
        logger.info("Retrieval health probe: recall@10=%.2f (%d/%d)",
                    recall, hits, len(core))
        return recall

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

        # Step 0: Retrieval 健康探针（离线低频，失败不阻塞梦境）
        # 【H8 2026-08-29】整体 30s 超时：内层 retrieve 无 API 3s 超时保护，
        # 每次硬跑 60-90s，5 次≈450s 导致 dream 必超时+孤儿线程吃 CPU。
        # 超时直接跳过 probe（探针价值 < 阻塞成本），不卡 dream 主流程。
        try:
            await asyncio.wait_for(
                self.retrieval_health_probe(gathered), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Dream %s: retrieval health probe timed out (skipped)",
                           dream_id)
        except Exception:
            logger.warning("Dream %s: retrieval health probe failed (non-fatal)",
                           dream_id, exc_info=True)

        # Step 2: CLUSTER — 社区检测（在独立线程中运行，避免阻塞事件循环）
        communities = await asyncio.to_thread(self._cluster_step, gathered, connections)
        logger.info("Dream %s: CLUSTER — %d communities", dream_id, len(communities))

        # Step 3: SYNTHESIZE — 生成社区摘要
        communities = await self._synthesize_step(communities)
        logger.info("Dream %s: SYNTHESIZE — %d reports generated", dream_id, len(communities))

        # Step 3a: Ontology 自演化 (v5.38.0) — SYNTHESIZE 后聚合社区 topics/report
        # 做 1 次 LLM 判断（LLM 延迟不压写读关键路径；llm_client 空/失败 → 直接返回）
        await self._ontology_evolution_step(communities)

        # Step 3b: SSM 梦境深度巩固 (P2) — 每次 run() 开始时重置 SSM 状态
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

        # Step 3c: CALIBRATE — 信心校准 (Manufactured Confidence, P1)
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
        # 【P3】整块包 asyncio.to_thread：encoder.embed（嵌入）+ 余弦计算均 CPU 密集，
        # 移出事件循环（max_dream_duration 可抢占）。
        merge_ops, conflict_count = await asyncio.to_thread(
            self._resolve_step, communities, gathered
        )
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
            # 【v6.3.1】候选模式下也落库实体（幂等，只增不删）：
            # PERSIST 直接模式的 PRUNE/MERGE 破坏性操作仍经 apply 人工放行，
            # 但实体落库（_persist_entities）与 Schema 演化（_persist_schema_
            # evolution）本质是 sha1 elementKey / blake3 证据键幂等的只增写——
            # 候选模式下不执行则 EntityNode 永不落库（v6.2.0 P0-① 生产缺陷）。
            # 这里经写队列串行（_persist_async），候选保存不受影响。
            if graphlite_store is not None:
                try:
                    await self._persist_async(
                        self._persist_entities, graphlite_store, communities)
                    await self._persist_async(
                        self._persist_schema_evolution, graphlite_store, communities)
                    await self._persist_async(
                        self._persist_atomic_facts, graphlite_store, communities)
                except Exception as persist_exc:
                    persist_degraded = True
                    logger.error(
                        "Dream %s: candidate-mode entity persist partial failure "
                        "(degraded, next dream repairs): %s", dream_id, persist_exc)
        elif graphlite_store is not None:
            # 直接模式（原行为）：直接修改生产数据
            # 【H5】GraphLite 无跨语句事务（TransactionManager 的 rollback 仅做
            # tx_tag 清理，无法撤销裸 GQL 的 CREATE/DELETE），故不做伪事务包装；
            # 改为：任一步骤抛异常 → 打 degraded 标记，下次梦境通过 upsert 修复
            # 【H2】写队列深度守卫（镜像 app.py auto_apply 的错峰逻辑）：PERSIST
            # 会在单写线程上排队多步写，若队列已过半满（积压 > max_pending//2）
            # 说明写压力大，梦境写回会加剧积压 → 本次跳过 PERSIST（degraded），
            # 下次梦境按 H5 upsert 语义自愈。只减少写、不新增写路径。
            persist_degraded = False
            q = self._write_queue
            if q is not None and q.pending_count() > q.max_pending // 2:
                persist_degraded = True
                logger.warning(
                    "Dream %s: PERSIST skipped — write queue busy "
                    "(%d/%d pending, degraded, next dream repairs)",
                    dream_id, q.pending_count(), q.max_pending,
                )
            else:
                (persist_created, persist_deleted, all_removed_ids,
                 persist_degraded) = await self._persist_direct(
                    graphlite_store, communities, prune_ops, merge_ops, dream_id,
                )
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
                tau = self.tau_engine.compute_strength(
                    created_at, node_id=node.get("id"),
                    fact_track=node.get("fact_track", "active"),
                )
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

        # 合并所有分区 — 保留 _detect_communities 的 cid（社区归属），
        # 跨分量用 next_comm 偏移避免 cid 冲突（每个分量内 cid 独立从 0 编号）
        partition: dict[str, int] = {}
        next_comm = 0
        for sp in sub_results:
            for nid, cid in sp.items():
                if nid not in partition:
                    partition[nid] = next_comm + cid
            # 该分量内最大 cid 决定偏移
            if sp:
                next_comm += max(sp.values()) + 1
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
            # 【P9】types 缓存：先遍历 node_items 一次算出 {nid: type_set}（含去空过滤），
            # 内循环只取集合求交——N=2000 时 4M 次正则 → 2000 次，纯等价重构。
            type_sets: dict[str, set] = {}
            for nid, node in node_items:
                content = node.get("content", "")
                types = self.ontology_validator._extract_types(content)
                type_sets[nid] = {t.get("type", "") for t in types if t.get("type")}
            for i, (nid_a, node_a) in enumerate(node_items):
                for nid_b, node_b in node_items[i + 1 :]:
                    if G.has_edge(nid_a, nid_b):
                        continue  # 已有连接，不覆盖
                    shared_types = type_sets[nid_a] & type_sets[nid_b]
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

    async def _ontology_evolution_step(self, communities: list[dict]) -> None:
        """Schema 自演化（v5.38.0）：SYNTHESIZE 后聚合社区 → 1 次 LLM 判断。

        llm_client 空 / LLM 失败 → 直接返回（不阻塞梦境管道）。
        【v5.50.0 P1-5】写盘成功（new_type/merge_existing/attr_op，含正交同轮）后
        刷新活动路由器 alias map —— 新学别名无需重启即生效。
        """
        if self.llm_client is None or self.ontology_evolution is None:
            return
        if not communities:
            return
        try:
            result = await self.ontology_evolution.evolve(communities, self.llm_client)
            action = result.get("action")
            if action in ("new_type", "merge_existing"):
                logger.info("Dream: Ontology evolution %s → %s", action, result.get("type"))
            if action in ("new_type", "merge_existing", "attr_op"):
                self._refresh_attr_aliases()
        except Exception:
            logger.warning("Dream: ontology evolution skipped (non-fatal)", exc_info=True)

    def _refresh_attr_aliases(self) -> None:
        """【v5.50.0 P1-5】从 extended 文件重载 attr_aliases → 刷新活动路由器。

        retrieval_guard（SelfEvolvingRetrieval）委托内层 _qr.set_attr_aliases；
        无 guard / 无 set_attr_aliases / 加载失败 → 静默降级（重启生效）。
        """
        guard = self.retrieval_guard
        if guard is None or not hasattr(guard, "set_attr_aliases"):
            return
        evo = self.ontology_evolution
        if evo is None:
            return
        try:
            aliases = (evo.load() or {}).get("attr_aliases") or {}
            # 【R3 P3-2】文件顶层 attr_aliases 非 dict（list/string 损坏）→ 降级空 dict，
            # 防 _extract_property_terms 的 .items() 抛异常被外层 try 吞掉致属性通道静默降级。
            if not isinstance(aliases, dict):
                aliases = {}
        except Exception:
            logger.warning("Dream: attr alias refresh skipped (load failed)", exc_info=True)
            return
        try:
            guard.set_attr_aliases(aliases)
            logger.info("Dream: attr alias map refreshed (%d canonicals)", len(aliases))
        except Exception:
            logger.warning("Dream: attr alias refresh skipped (guard failed)", exc_info=True)

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
        # 【P3】嵌入记忆化：encoder.embed 结果按文本缓存（dict），重复文本跳过编码。
        # 社区内最坏 O(N²) 对比较 → 每唯一文本只编码 1 次。
        emb_cache: dict[str, np.ndarray] = {}

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
                    nodes[i].get("content", ""), nodes[j].get("content", ""),
                    emb_cache=emb_cache,
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

    async def _persist_direct(
        self,
        graphlite_store,
        communities: list[dict],
        prune_ops: list,
        merge_ops: list,
        dream_id: str,
    ) -> tuple[int, int, list[str], bool]:
        """直接模式 PERSIST 五步（PRUNE → COMMUNITIES → MERGE → HYPEREDGES）。

        【H2】从 run() 内联块提取为方法：写队列深度守卫（队列过半满）时跳过
        整个 PERSIST，不逐步骤判断；本方法保留原 try/except 语义——任一步
        抛异常 → degraded 标记（返回），下次梦境 upsert 自愈。
        返回 (created, deleted, all_removed_ids, degraded)。
        """
        persist_created = 0
        persist_deleted = 0
        all_removed_ids: list[str] = []
        persist_degraded = False
        try:
            persist_deleted, pruned_ids = await self._persist_async(
                self._persist_prune, graphlite_store, prune_ops)
            all_removed_ids.extend(pruned_ids)
            # 【v5.40】社区 PERSIST 切块：每社区单独提交一个 low 任务（~3s/块），
            # 块间写线程排空 high 优先级外部写——不切块则 30-60s 单体任务占死
            # 单写线程，优先级只重排 pending 不重排 in-flight → 外部写 503。
            # MERGE ON id 语义保留；阶段 3 同源湮灭（每成员只保留最大社区边）
            # 依赖全局成员集，作为最后一块 low 任务提交（短任务，不影响切块）。
            member_sets: dict[str, set[str]] = {}
            for idx, comm in enumerate(communities):
                created, member_set = await self._persist_async(
                    self._persist_one_community, graphlite_store, comm,
                    dream_id, idx, priority="low")
                persist_created += created
                if member_set:
                    member_sets[comm["id"]] = member_set
            all_removed_ids.extend(self._persist_merge_get_removed(merge_ops))
            await self._persist_async(self._persist_merge, graphlite_store, merge_ops)
            if member_sets:
                await self._persist_async(
                    self._persist_communities_prune_edges,
                    graphlite_store, member_sets)
            await self._persist_async(
                self._persist_hyperedges, graphlite_store, communities, dream_id)
            await self._persist_async(
                self._persist_entities, graphlite_store, communities)
            await self._persist_async(
                self._persist_schema_evolution, graphlite_store, communities)
            await self._persist_async(
                self._persist_atomic_facts, graphlite_store, communities)
        except Exception as persist_exc:
            persist_degraded = True
            logger.error("Dream %s: PERSIST partial failure (degraded, next dream repairs): %s",
                         dream_id, persist_exc)
        return persist_created, persist_deleted, all_removed_ids, persist_degraded

    def _persist_entities(self, graphlite_store, communities: list[dict]) -> int:
        """Schema 自演化（P0-①）：community.entity_links → EntityNode + MENTIONS 边落库。

        消费 _entity_linking_step 产出的实体链接（entity → occurrences 节点 id），
        逐实体幂等创建 EntityNode 并连 MENTIONS 边（store 侧幂等去重）。
        失败不阻塞（沿用 PERSIST degraded 语义，下次梦境自愈）。
        """
        created = 0
        if graphlite_store is None or not hasattr(graphlite_store, "link_entity_to_episode"):
            return 0
        seen: set[str] = set()
        for comm in communities:
            for link in comm.get("entity_links") or []:
                ent = (link.get("entity") or "").strip()
                if not ent or ent in seen:
                    continue
                seen.add(ent)
                # 【Codex 批1 P2-3】异常打 WARNING（此前裸 except 吞没难排查）；
                # 仅在全部 occurrences 链接成功后计数（防计数虚高）。
                ok = True
                for occ in (link.get("occurrences") or []):
                    try:
                        graphlite_store.link_entity_to_episode(ent, str(occ))
                    except Exception as e:
                        ok = False
                        logger.warning(
                            "Dream PERSIST: link_entity_to_episode failed "
                            "entity=%s occ=%s: %s", ent, occ, e)
                if ok:
                    created += 1
        if created:
            logger.info("Dream PERSIST: %d entities persisted (schema self-evolution)", created)
        return created

    # ─── AtomicFact 事实级中间层（P0-③）─────────────

    _FACT_VERB_PAT = re.compile(
        r"([A-Z][A-Za-z]+)\s+"
        r"((?:is|was|has|had|works\s+as|works\s+at|graduated\s+from|enrolled\s+in|"
        r"started|finished|likes|prefers|attends|attended|moved\s+to|born\s+in))\s+"
        r"([^.,;!?]{1,80})",
        re.IGNORECASE,
    )
    _FACT_TIME_PAT = re.compile(
        r"(?:in|on|since|until)\s+((?:January|February|March|April|May|June|July|"
        r"August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|"
        r"Sep|Oct|Nov|Dec)?\.?\s*\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}|\d{4})",
        re.IGNORECASE,
    )

    def _extract_facts_rules(self, content: str) -> list[dict]:
        """规则抽取 SPO 三元组（英文为主，LoCoMo 评测场景；中文后续扩展）。"""
        if not content:
            return []
        out: list[dict] = []
        for m in self._FACT_VERB_PAT.finditer(content):
            subj = m.group(1).strip()
            pred = m.group(2).strip()
            obj = m.group(3).strip()
            if not subj or not pred or not obj:
                continue
            vt = ""
            tm = self._FACT_TIME_PAT.search(content)
            if tm:
                vt = tm.group(1).strip()
            out.append({
                "subject": subj, "predicate": pred,
                "object": obj, "valid_time": vt,
            })
            if len(out) >= 10:
                break
        return out

    async def _persist_atomic_facts(self, graphlite_store, communities: list[dict]) -> int:
        """AtomicFact 落库（P0-③）：社区 episode 内容 → SPO 事实节点（幂等）。

        独立于实体落库（_persist_entities）——事实级中间层（EverOS 93.05 核心）。
        抽取失败/落库失败不阻塞（PERSIST degraded 语义）。
        """
        created = 0
        if graphlite_store is None or not hasattr(graphlite_store, "create_atomic_fact"):
            return 0
        seen: set[str] = set()
        for comm in communities:
            for node in (comm.get("episodes") or comm.get("nodes") or []):
                content = ""
                if isinstance(node, dict):
                    content = node.get("content") or node.get("summary") or ""
                elif isinstance(node, str):
                    content = node
                if not content:
                    continue
                for fact in self._extract_facts_rules(content):
                    key = "|".join([
                        fact["subject"].lower(), fact["predicate"].lower(),
                        fact["object"].lower(), fact["valid_time"].lower(),
                    ])
                    if key in seen:
                        continue
                    seen.add(key)
                    try:
                        graphlite_store.create_atomic_fact(
                            subject=fact["subject"], predicate=fact["predicate"],
                            object_=fact["object"], valid_time=fact["valid_time"],
                            confidence=0.6,
                        )
                        created += 1
                    except Exception as e:
                        logger.warning(
                            "Dream PERSIST: create_atomic_fact failed %s|%s: %s",
                            fact["subject"], fact["predicate"], e)
        if created:
            logger.info("Dream PERSIST: %d atomic facts persisted (P0-③)", created)
        return created

    # ─── Schema 自进化 P0-②：实体属性/关系演化 ────────────────

    def _persist_schema_evolution(self, graphlite_store, communities: list[dict]) -> int:
        """Schema 自演化（P0-②）：community 内容 → 实体属性/关系演化落库。

        消费已落库 EntityNode（P0-① _persist_entities 之后），对 community
        content 做纯规则属性提取 + 关系抽取 → 分区计票 → sidecar 演化 →
        跨阈值固化（PropertyVerNode / REL_ 边）。失败不阻塞（degraded 自愈）。
        """
        evolved = 0
        failures = 0
        if graphlite_store is None or not hasattr(graphlite_store, "locked_update_entity_props"):
            return 0
        for comm in communities:
            content = comm.get("content") or ""
            ep_id = str(comm.get("episode_id") or comm.get("id") or "")
            links = comm.get("entity_links") or []
            if not content or not links:
                continue
            entities = []
            for link in links:
                ent_name = (link.get("entity") or "").strip()
                if not ent_name:
                    continue
                ent = graphlite_store.get_entity(ent_name)
                if ent:
                    entities.append(_EntityView(ent))
            if not entities:
                continue
            try:
                from core.attribute_extractor import extract_attributes
                attrs = extract_attributes(ep_id, content, entities)
                if attrs:
                    evolved += self._evolve_attrs(graphlite_store, attrs)
                rels = self._extract_entity_relations(graphlite_store, content, entities)
                if rels:
                    evolved += self._evolve_rels(graphlite_store, rels)
            except Exception as exc:  # 单 community 失败 → 聚合，外层标记 degraded 自愈
                failures += 1
                logger.warning("Dream PERSIST schema-evolution failed (%s): %s",
                               str(ep_id)[:12], exc)
        if failures:
            # 抛聚合异常 → _persist_direct 捕获 → persist_degraded=True → 下轮梦境重放
            # （blake3 证据键 + sha1 elementKey 幂等，重放不重复）
            raise RuntimeError(f"schema-evolution failed for {failures}/{len(communities)} communities")
        if evolved:
            logger.info("Dream PERSIST: %d attr/rel evolutions (schema self-evolution P0-②)", evolved)
        return evolved

    def _evolve_attrs(self, store, attrs: list) -> int:
        """属性演化：sidecar 累票 → 固化（PropertyVerNode / sidecar solidified）。"""
        from core.schema_evolver import accumulate_votes, decide, Action, AttrStat, T_SOLIDIFY
        by_ent: dict[str, list] = {}
        for a in attrs:
            by_ent.setdefault(a.entity_id, []).append(a)
        count = 0
        for eid, ex_list in by_ent.items():
            new_props = store.locked_update_entity_props(eid, lambda props, exs=ex_list: self._merge_attrs_sidecar(props, exs))
            sidecar = store._decode_sidecar(new_props, "attrs_json")
            # decide + 固化
            for attr_name, attr_block in sidecar.items():
                solidified = attr_block.get("solidified") or {}
                for vkey, cand in (attr_block.get("candidates") or {}).items():
                    stat = AttrStat(attr_name, cand.get("value", ""), vkey,
                                    cand.get("votes", {}), cand.get("evidence", []),
                                    cand.get("conf", 0.0))
                    action = decide(stat, solidified)
                    if action in (Action.SOLIDIFY, Action.CORRECT):
                        supersedes_id = solidified.get("pvn_key") if action == Action.CORRECT else None
                        try:
                            pvn_key = store.create_property_version(
                                eid, attr_name, stat.value,
                                supersedes_id=supersedes_id)
                        except Exception:
                            pvn_key = ""
                        solidified = {
                            "value": stat.value, "value_blake3": vkey,
                            "version": int(solidified.get("version", 0)) + 1,
                            "conf": stat.confidence,
                            "pvn_key": pvn_key, "active": True,
                        }
                        attr_block["solidified"] = solidified
                        count += 1
                    elif action == Action.STRENGTHEN and solidified:
                        solidified["conf"] = max(float(solidified.get("conf", 0)), stat.confidence)
                        attr_block["solidified"] = solidified
            # 写回（固化后 sidecar 变化）
            store.locked_update_entity_props(eid, lambda props, sc=sidecar: self._write_attrs_sidecar(props, sc))
        return count

    @staticmethod
    def _merge_attrs_sidecar(props: dict, ex_list: list) -> dict:
        from core.schema_evolver import accumulate_votes
        sidecar = dict(props.get("attrs_json") or {}) if isinstance(props.get("attrs_json"), dict) else {}
        new_sc = accumulate_votes(sidecar, ex_list)
        props["attrs_json"] = new_sc
        return props

    @staticmethod
    def _write_attrs_sidecar(props: dict, sidecar: dict) -> dict:
        props["attrs_json"] = sidecar
        return props

    def _extract_entity_relations(self, store, content: str, entities: list) -> list:
        """抽取实体间谓词关系（仅保留两端都是已落库实体的三元组）。"""
        from core.relation_extractor import RelationExtractor
        names = {e.name.lower() for e in entities}
        out = []
        try:
            triples = RelationExtractor().extract(content)
        except Exception:
            return out
        for t in triples:
            subj = (t.subject or "").strip()
            obj = (t.obj or "").strip()
            if not subj or not obj:
                continue
            src = next((e for e in entities if e.name.lower() == subj.lower()), None)
            dst = next((e for e in entities if e.name.lower() == obj.lower()), None)
            if src and dst and src.entity_id != dst.entity_id:
                out.append((src.entity_id, dst.entity_id, t.relation, t.confidence))
        return out

    def _evolve_rels(self, store, rels: list) -> int:
        """关系演化：rels_json 侧车分区计票 → 固化（REL_ 边）。"""
        from core.schema_evolver import confidence as rel_confidence, T_SOLIDIFY
        by_ent: dict[str, list] = {}
        for src_id, dst_id, pred, conf in rels:
            by_ent.setdefault(src_id, []).append((dst_id, pred, conf))
        count = 0
        for src_id, rel_list in by_ent.items():
            new_props = store.locked_update_entity_props(src_id, lambda props, rl=rel_list: self._merge_rels_sidecar(props, rl))
            sidecar = store._decode_sidecar(new_props, "rels_json")
            for dst_id, pred, conf in rel_list:
                slot = sidecar.setdefault(pred, {}).setdefault(dst_id, {})
                slot["target_name"] = slot.get("target_name", "")
                slot["votes"] = slot.get("votes", {})
                slot["evidence"] = slot.get("evidence", [])
                # conf 已由 _merge_rels_sidecar 分区计票计算，不覆盖
                if float(slot.get("conf", 0)) >= T_SOLIDIFY and not slot.get("solidified"):
                    try:
                        store.create_rel_edge(src_id, dst_id, pred, confidence=float(slot["conf"]))
                        slot["solidified"] = True
                        count += 1
                    except Exception:
                        pass
            store.locked_update_entity_props(src_id, lambda props, sc=sidecar: self._write_rels_sidecar(props, sc))
        return count

    @staticmethod
    def _merge_rels_sidecar(props: dict, rel_list: list) -> dict:
        from core.schema_evolver import confidence as rel_confidence
        sidecar = dict(props.get("rels_json") or {}) if isinstance(props.get("rels_json"), dict) else {}
        for dst_id, pred, conf in rel_list:
            slot = sidecar.setdefault(pred, {}).setdefault(dst_id, {})
            slot["votes"] = slot.get("votes", {})
            # 分区计票（与属性侧一致：单分区封顶 CAP=5，≥2 独立分区才高分）
            slot["votes"]["rel_extract"] = int(slot["votes"].get("rel_extract", 0)) + 1
            slot["conf"] = rel_confidence(slot["votes"])
        props["rels_json"] = sidecar
        return props

    @staticmethod
    def _write_rels_sidecar(props: dict, sidecar: dict) -> dict:
        props["rels_json"] = sidecar
        return props

    # ─── 【FIX】GraphLite持久化方法 ──────────────────────────────

    async def _persist_async(self, fn, *args, priority: str = "normal", **kwargs):
        """PERSIST 写入串行化：有 write_queue 时经单写线程 submit（priority 透传），
        否则回退 to_thread（priority 不参与 to_thread 调用）。"""
        if self._write_queue is not None:
            return await self._write_queue.submit(fn, *args, priority=priority, **kwargs)
        return await asyncio.to_thread(fn, *args, **kwargs)

    def _persist_communities(self, graphlite_store, communities: list[dict], dream_id: str) -> int:
        """将CLUSTER结果写回GraphLite CommunityNode（完整语义，供直接调用/测试）。

        【v5.40】内部复用切块原语：逐社区 _persist_one_community（阶段1 清旧边 +
        阶段2 MERGE 幂等 upsert）+ 阶段 3 同源湮灭 _persist_communities_prune_edges。
        生产直接模式 _persist_direct 走逐社区 low 任务提交（块间排空 high），
        本方法保持与 v5.39 一致的原子整体语义。

        使用 MERGE ON id 增量 upsert，避免先 DETACH DELETE 全部再重建
        （若中途崩溃则所有社区数据永久丢失）。
        保留的历史社区由外部周期性清理策略处理。

        先清理旧 COMMUNITY_MEMBER 边再 upsert，避免竞赛条件导致
        一个 EpisodeNode 关联到 2 个 CommunityNode。
        """
        created = 0
        member_sets: dict[str, set[str]] = {}
        for idx, comm in enumerate(communities):
            c, member_set = self._persist_one_community(
                graphlite_store, comm, dream_id, idx)
            created += c
            if member_set:
                member_sets[comm["id"]] = member_set
        if member_sets:
            self._persist_communities_prune_edges(graphlite_store, member_sets)
        return created

    def _persist_one_community(
        self, graphlite_store, comm: dict, dream_id: str, idx: int = 0,
    ) -> tuple[int, set[str]]:
        """【v5.40】单社区持久化块（切块原语）：清理该社区旧 COMMUNITY_MEMBER 边
        （限定 {id: $cid} 不碰外部社区）+ MERGE 幂等 upsert 社区节点与成员边。

        【v6.9 批量化】改用 OverGraph 批量写入（batch_write_txn + WriteTxn.stage
        编排：delete_edge 清旧边 + upsert_node 读-合并 + upsert_edge 建边，单次
        commit 原子落库，异常 rollback）。行为与逐条 execute_cypher 等价：
        created∈{0,1}、member_set 语义不变、不碰外部社区边、EpisodeNode 不存在
        跳过（原 MATCH 无行）、重复 apply 不累积重复边。

        返回 (created, member_set)：created∈{0,1}（upsert 成功数）；
        member_set 供阶段 3 同源湮灭（每成员只保留最大社区边）。
        生产 _persist_direct 每社区单独经写队列提交一个 low 任务（~3s/块），
        块间写线程排空 high 优先级外部写。
        """
        created = 0
        member_set: set[str] = set()
        comm_id = comm.get("id", "")
        if not comm_id:
            return created, member_set
        try:
            with graphlite_store.batch_write_txn() as (txn, db):
                # 阶段 1：清理该社区旧边（限定 cid，不碰外部社区）
                # 【FIX 2026-08-09】原实现 MATCH (c:CommunityNode) 无社区过滤 →
                # 对每个 dream 成员删除所有社区（含外部）指向它的边（实证 EXT→epX 被删）。
                # 修复：限定 {id: $cid}，与 dream_candidate_store.py 一致。
                member_ids = list(comm.get("members", []))
                member_set.update(member_ids)
                try:
                    cview = db.get_node_by_key("CommunityNode", comm_id)
                except Exception:
                    cview = None
                ci = int(cview.id) if cview is not None else None
                edge_ops: list[dict] = []
                for member_id in member_ids:
                    try:
                        mview = db.get_node_by_key("EpisodeNode", member_id)
                    except Exception:
                        mview = None
                    if mview is None:
                        # EpisodeNode 不存在 → 原 MATCH 无行 → 不建边
                        continue
                    if ci is not None:
                        try:
                            old = db.get_edge_by_triple(
                                ci, int(mview.id), "COMMUNITY_MEMBER")
                        except Exception:
                            old = None
                        if old is not None:
                            edge_ops.append({
                                "op": "delete_edge",
                                "target": {"id": int(old.id)},
                            })
                    edge_ops.append({
                        "op": "upsert_edge",
                        "from": {"labels": ["CommunityNode"], "key": comm_id},
                        "to": {"labels": ["EpisodeNode"], "key": member_id},
                        "label": "COMMUNITY_MEMBER",
                        "props": {},
                    })
                # 阶段 2：upsert 社区节点（读-合并 = 原 MATCH-exists INSERT/SET）
                comm_vals = {
                    "id": comm_id,
                    "name": f"dream_{dream_id[:8]}_comm_{idx}",
                    "summary": (comm.get("report", "") or "")[:800],
                    "leiden_score": 0.0,
                    "created_at": time.time(),
                }
                try:
                    existing = db.get_node_by_key("CommunityNode", comm_id)
                    props = dict(existing.props) if existing is not None else {}
                except Exception:
                    props = {}
                props.update(comm_vals)
                ops: list[dict] = [{
                    "op": "upsert_node",
                    "labels": ["CommunityNode"],
                    "key": comm_id,
                    "props": props,
                }]
                ops.extend(edge_ops)
                if ops:
                    txn.stage(ops)
                created = 1
        except Exception as e:
            logger.warning("Community persist failed: %s", e)
        return created, member_set




    def _persist_communities_prune_edges(
        self, graphlite_store, new_member_sets: dict[str, set[str]],
    ) -> None:
        """【v5.40】阶段 3 同源湮灭（切块最后一块）：每成员只保留最大社区的边。

        输入为全部社区切块返回的 member_set 汇总；按 member_count 倒序建
        max_community_by_member，每成员只保留最大社区的边，只删自己社区的边，
        不动外部社区。原 _persist_communities 阶段 3 提取（2026-08-09 同源湮灭
        bug 修复语义保持）。
        """
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

    def _persist_prune(self, graphlite_store, prune_ops: list) -> tuple[int, list[str]]:
        """将 PRUNE 剪枝结果归档（archived=true，替代物理删除）。

        保留 (deleted_count, pruned_ids) 签名不变——deleted_count 语义改为
        「归档数」；pruned_ids 仍返回给下游 FAISS incremental_faiss_update 做
        remove_ids（节点归档后须从向量索引剔除）。
        """
        deleted = 0
        pruned_ids: list[str] = []
        for op in prune_ops:
            if op.op_type == "delete":
                try:
                    if graphlite_store.archive_node(op.node_id):
                        deleted += 1
                        pruned_ids.append(op.node_id)
                except Exception:
                    logger.warning("Failed to archive pruned node in GraphLite", exc_info=True)
        return deleted, pruned_ids

    def _persist_merge(self, graphlite_store, merge_ops: list) -> None:
        """将 RESOLVE 合并结果写回 GraphLite（winner 拼接摘要 + 归档 loser + SUPERSEDES 边）。

        【P8】merge 截断：先读现有 content，Python 侧拼
        `(old + ' | merged: ' + old_value)[:2000]` 再 SET——防止无界追加
        （连续合并下 content 无限增长）。每 merge 多 1 次读（合并本就低频）。
        """
        for op in merge_ops:
            if op.op_type == "update" and op.new_value:
                try:
                    # 先读目标节点现有 content，Python 侧拼接并截断
                    rows = graphlite_store.query_cypher(
                        "MATCH (target:EpisodeNode {id: $target}) "
                        "RETURN target.content AS content",
                        {"target": op.new_value},
                    )
                    old_content = ""
                    if rows:
                        row = rows[0]
                        if isinstance(row, dict):
                            old_content = row.get("content", "") or ""
                        elif isinstance(row, (list, tuple)) and len(row) >= 1:
                            old_content = row[0] or ""
                    merged_content = (
                        str(old_content) + " | merged: " + (op.old_value or "")
                    )[:2000]
                    graphlite_store.query_cypher(
                        "MATCH (target:EpisodeNode {id: $target}) "
                        "SET target.content = $content",
                        {"target": op.new_value, "content": merged_content}
                    )
                    # 归档被合并节点（loser）+ 建 SUPERSEDES 血统边（替代 DETACH DELETE）
                    graphlite_store.archive_node(op.node_id, op.new_value)
                except Exception:
                    logger.warning("Failed to persist merge resolution in GraphLite", exc_info=True)

    @staticmethod
    def _persist_merge_get_removed(merge_ops: list) -> list[str]:
        """从 merge_ops 中提取被删除的节点ID（给FAISS增量更新用）。"""
        return [op.node_id for op in merge_ops
                if op.op_type == "update" and op.node_id]

    def _persist_hyperedges(self, graphlite_store, communities: list[dict], dream_id: str) -> int:
        """梦境结束后，为每个社区创建HyperedgeNode（Layer4）。

        【v6.9 批量化】原逐成员 query_cypher CREATE/INSERT → batch_write_txn +
        WriteTxn.stage 编排（每社区一个事务：HyperedgeNode upsert + 全部
        HYPEREDGE_MEMBER 边），单次 commit 原子落库，异常 rollback。
        EpisodeNode 不存在 → 跳过（原 MATCH 无行）；created 语义不变。
        """
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
                with graphlite_store.batch_write_txn() as (txn, db):
                    ops: list[dict] = [{
                        "op": "upsert_node",
                        "labels": ["HyperedgeNode"],
                        "key": hyperedge_id,
                        "props": {
                            "id": hyperedge_id,
                            "type": "semantic",
                            "created_at": time.time(),
                            "gate_value": 1.0,
                            "metadata": metadata,
                        },
                    }]
                    for member_id in members:
                        try:
                            mview = db.get_node_by_key("EpisodeNode", member_id)
                        except Exception:
                            mview = None
                        if mview is None:
                            continue
                        ops.append({
                            "op": "upsert_edge",
                            "from": {"labels": ["HyperedgeNode"], "key": hyperedge_id},
                            "to": {"labels": ["EpisodeNode"], "key": member_id},
                            "label": "HYPEREDGE_MEMBER",
                            "props": {},
                        })
                    txn.stage(ops)
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

    def _combined_similarity(
        self,
        text_a: str,
        text_b: str,
        emb_cache: Optional[dict] = None,
    ) -> float:
        """加权合并：Jaccard 词集 + 向量余弦相似度。

        【P3】Jaccard 预筛：jac < _JACCARD_PRESCREEN_THRESHOLD 时跳过余弦编码
        （sim ≤ 0.4·jac + 0.6 < 0.8，合并决策不变）；emb_cache 按文本缓存
        encoder.embed 结果，重复文本跳过编码。
        """
        jac = self._jaccard_similarity(text_a, text_b)
        if jac < _JACCARD_PRESCREEN_THRESHOLD:
            return jac
        if self.encoder is not None:
            try:
                emb_a = self._embed_with_cache(text_a, emb_cache)
                emb_b = self._embed_with_cache(text_b, emb_cache)
                if emb_a is not None and emb_b is not None:
                    norm_a = np.linalg.norm(emb_a)
                    norm_b = np.linalg.norm(emb_b)
                    if norm_a > 0 and norm_b > 0:
                        cos = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))
                        return 0.4 * jac + 0.6 * max(0.0, cos)
            except Exception:
                pass
        return jac

    def _embed_with_cache(
        self,
        text: str,
        emb_cache: Optional[dict] = None,
    ):
        """encoder.embed 记忆化：emb_cache 命中直接返回，未命中编码并缓存。"""
        if emb_cache is None:
            return self.encoder.embed(text)
        if text not in emb_cache:
            emb_cache[text] = self.encoder.embed(text)
        return emb_cache[text]
