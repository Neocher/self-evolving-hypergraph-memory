"""
梦境整合管道（Layer5 核心）
=========================
GATHER → CLUSTER → SYNTHESIZE → COMPRESS → PRUNE → RESOLVE → AUDIT
                                      ^^^^^^^^
                                [Harness Fix] 新增 COMPRESS 步骤

在系统空闲时自动执行，将碎片化情节转化为结构化知识。
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from core.audit_chain import AuditOperation

logger = logging.getLogger(__name__)


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


class DreamPipeline:
    """
    梦境八步管道。

    GATHER:     收集所有未处理的节点
    CLUSTER:    运行 Leiden 社区检测
    SYNTHESIZE: 生成社区报告（可选用 LLM 语义摘要）
    COMPRESS:   [Harness Fix] 压缩社区报告，限制 Token 预算
    PRUNE:      删除 τ 低于阈值且低连接度的节点
    RESOLVE:    矛盾检测与消歧
    PERSIST:    写回结果到候选存储或 Kuzu
    AUDIT:      写入 BLAKE3 溯源链

    支持两种模式：
    - 直接模式 (candidate_store=None): 直接修改 Kuzu 生产数据（原行为）
    - 候选模式 (candidate_store 有效): 写入临时候选区，供审查后上线
    """

    def __init__(
        self,
        tau_engine=None,
        hebbian_updater=None,
        audit_chain=None,
        llm_client=None,
        ontology_validator=None,
    ) -> None:
        """
        Args:
            tau_engine: TauDecayEngine 实例
            hebbian_updater: SparseHebbianUpdater 实例
            audit_chain: AuditChain 实例
            llm_client: LLMClient 实例（可选，提供时启用 LLM 语义摘要）
            ontology_validator: OntologyValidator 实例（可选，P1 本体约束社区检测）
        """
        self.tau_engine = tau_engine
        self.hebbian_updater = hebbian_updater
        self.audit_chain = audit_chain
        self.llm_client = llm_client
        self.ontology_validator = ontology_validator

    async def run(
        self,
        nodes: list[dict],
        connections: dict[str, dict[str, float]],
        trigger_mode: str = "explicit",
        kuzu_store=None,  # 接收Kuzu引用，用于持久化结果
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
        7. PERSIST    → 【FIX】将结果写回Kuzu（CommunityNode/DELETE/合并/HyperedgeNode）
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

        # Step 2: CLUSTER — 社区检测
        communities = self._cluster_step(gathered, connections)
        logger.info("Dream %s: CLUSTER — %d communities", dream_id, len(communities))

        # Step 3: SYNTHESIZE — 生成社区摘要
        communities = await self._synthesize_step(communities)
        logger.info("Dream %s: SYNTHESIZE — %d reports generated", dream_id, len(communities))

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

        # Step 7: PERSIST — 将结果写回Kuzu或候选存储
        persist_created = 0
        persist_deleted = 0
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
        elif kuzu_store is not None:
            # 直接模式（原行为）：直接修改生产数据
            persist_deleted, pruned_ids = self._persist_prune(kuzu_store, prune_ops)
            all_removed_ids.extend(pruned_ids)
            persist_created = self._persist_communities(kuzu_store, communities, dream_id)
            all_removed_ids.extend(self._persist_merge_get_removed(merge_ops))
            self._persist_merge(kuzu_store, merge_ops)
            self._persist_hyperedges(kuzu_store, communities, dream_id)
            logger.info("Dream %s: PERSIST — %d created, %d deleted, %d for FAISS cleanup",
                        dream_id, persist_created, persist_deleted, len(all_removed_ids))
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
        )

    # ─── Step 1: GATHER ───────────────────────────────────

    def _gather_step(self, nodes: list[dict]) -> list[dict]:
        """
        收集活跃节点并计算 τ 值。
        过滤掉 τ 值过低且无连接的"死亡"节点。
        """
        gathered: list[dict] = []
        for node in nodes:
            node_copy = dict(node)
            if self.tau_engine:
                created_at = node.get("created_at", time.time())
                tau = self.tau_engine.compute_tau(created_at)
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
            ont_edge_count = 0
            for i, (nid_a, node_a) in enumerate(node_map.items()):
                content_a = node_a.get("content", "")
                for nid_b, node_b in list(node_map.items())[i + 1 :]:
                    if G.has_edge(nid_a, nid_b):
                        continue  # 已有连接，不覆盖
                    content_b = node_b.get("content", "")
                    # 提取实体类型
                    types_a = self.ontology_validator._extract_types(content_a)
                    types_b = self.ontology_validator._extract_types(content_b)
                    if types_a and types_b:
                        # 检查是否有共享实体类型
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
        try:
            import cdlib
            from networkx import convert_node_labels_to_integers

            H = convert_node_labels_to_integers(G, label_attribute="orig_id")
            communities_list = cdlib.algorithms.leiden(H)
            # 还原原始节点 ID
            partition: dict[str, int] = {}
            for comm_idx, comm in enumerate(communities_list.communities):
                for int_id in comm:
                    orig_id = H.nodes[int_id].get("orig_id", str(int_id))
                    partition[orig_id] = comm_idx
            # 未分配的孤立节点各自成社区
            next_comm = len(communities_list.communities)
            for node in G.nodes:
                if node not in partition:
                    partition[node] = next_comm
                    next_comm += 1
            return partition
        except ImportError:
            pass

        try:
            from networkx.algorithms.community import louvain_communities

            partition = {}
            for comm_idx, comm in enumerate(louvain_communities(G)):
                for node_id in comm:
                    partition[node_id] = comm_idx
            next_comm = len(partition) // max(1, len(set(partition.values())))
            for node in G.nodes:
                if node not in partition:
                    partition[node] = next_comm
                    next_comm += 1
            return partition
        except ImportError:
            pass

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
        """
        for i, community in enumerate(communities):
            nodes = community.get("nodes", [])
            # 前 20 个最大社区用 LLM，其余用 TF-IDF 回退（平衡质量与速度）
            if self.llm_client and len(nodes) >= 2 and i < 20:
                # LLM 语义摘要
                contents = [n.get("content", "") for n in nodes]
                llm_result = await self.llm_client.summarize_community(contents)
                community["report"] = llm_result["summary"]
                community["keywords"] = llm_result["keywords"]
                community["llm_patterns"] = llm_result["patterns"]
                community["llm_contradictions"] = llm_result["contradictions"]
            else:
                # 原有模板方法（回退）
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
            # 实体链接：提取命名实体并在社区节点间交叉匹配
            community["entity_links"] = await self._entity_linking_step(nodes)
        return communities

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

    async def _entity_linking_step(self, nodes: list[dict]) -> list[dict]:
        """
        实体链接：提取命名实体并在社区节点间交叉匹配。

        两步策略:
        1. 用 LLM 做命名实体识别（降级：正则提取大写词和引号内专名）
        2. 跨节点匹配相同实体，生成实体链接关系

        Args:
            nodes: 社区内节点列表

        Returns:
            entity_links: [{"entity": str, "occurrences": [node_id, ...], "count": int}, ...]
            按出现次数降序排列，失败时返回空列表
        """
        try:
            # Step 1: 提取实体 — 统一入口，优先 LLM 降级到正则
            entity_map = await self._extract_entities_from_nodes(nodes)
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

    async def _extract_entities_from_nodes(self, nodes: list[dict]) -> dict[str, list[str]]:
        """
        统一实体提取入口。

        优先用 LLM 做命名实体识别，失败时降级到正则提取。
        返回 {node_id: [entity_name, ...], ...}
        """
        # 先尝试 LLM
        llm_result = await self._ner_with_llm(nodes)
        if llm_result:
            return llm_result

        # 降级：正则提取
        result: dict[str, list[str]] = {}
        for node in nodes:
            nid = node.get("id", "")
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

    async def _ner_with_llm(self, nodes: list[dict]) -> dict[str, list[str]]:
        """
        用 LLM 从节点内容中提取命名实体。

        使用 self.llm_client（已配置 DEEPSEEK_API_KEY），
        LLM 不可用或调用失败时返回空 dict（触发降级到正则提取）。

        Args:
            nodes: 社区内节点列表

        Returns:
            {node_id: [entity_name, ...], ...} 或 {}（失败时）
        """
        # 尝试使用内部 llm_client
        llm = None
        if self.llm_client and self.llm_client.api_key:
            llm = self.llm_client
        else:
            # 尝试直接用环境变量创建临时客户端
            key = os.environ.get("DEEPSEEK_API_KEY", "")
            if not key:
                return {}  # 无 API Key，降级
            try:
                from core.llm_client import LLMClient
                llm = LLMClient(api_key=key)
            except Exception:
                return {}  # 创建失败，降级

        result: dict[str, list[str]] = {}
        for node in nodes:
            nid = node.get("id", "")
            content = node.get("content", "")

            if not content or len(content.strip()) < 5:
                continue

            prompt = f"""Extract named entities from the following text. Return ONLY a JSON array of entity names (people, organizations, technologies, products, locations, projects).

Rules:
- Include proper nouns, technical terms, project names, organization names
- Exclude common words, generic terms, numbers, dates
- Return as a JSON array of strings only
- If no entities found, return an empty array []

Text:
{content[:500]}"""

            try:
                response = await llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=256,
                    response_format={"type": "json_object"},
                )
                if response:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict):
                        entities = parsed.get("entities", [])
                        if isinstance(entities, list):
                            result[nid] = [str(e).strip() for e in entities if str(e).strip()]
                    elif isinstance(parsed, list):
                        result[nid] = [str(e).strip() for e in parsed if str(e).strip()]
            except Exception:
                continue  # 单节点失败不中断整体流程

        return result

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
            if len(report.split()) > 500:
                community["report"] = " ".join(report.split()[:500])

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
            is_protected = (age_seconds < 7200) or (tau > 0.3) or (degree > 1)
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
            for j in range(i + 1, len(nodes)):
                if nodes[j]["id"] in merged:
                    continue
                sim = self._jaccard_similarity(
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

    # ─── 【FIX】Kuzu持久化方法 ──────────────────────────────

    def _persist_communities(self, kuzu_store, communities: list[dict], dream_id: str) -> int:
        """将CLUSTER结果写回Kuzu CommunityNode。
        
        先清理旧社区（DETACH DELETE所有CommunityNode及其边），再创建新的。
        """
        import json
        created = 0
        try:
            # 【FIX】先清理旧社区，防止无限累积
            kuzu_store.query_cypher("MATCH (c:CommunityNode) DETACH DELETE c")
        except Exception:
            logger.warning("Community DETACH DELETE failed", exc_info=True)
        for comm in communities:
            try:
                kuzu_store.query_cypher(
                    "CREATE (c:CommunityNode {id: $id, name: $name, summary: $summary, "
                    "leiden_score: $score, created_at: $created_at})",
                    {
                        "id": comm["id"],
                        "name": f"dream_{dream_id[:8]}_comm_{created}",
                        "summary": comm.get("report", "")[:800],
                        "score": 0.0,
                        "created_at": time.time(),
                    }
                )
                for member_id in comm.get("members", []):
                    try:
                        kuzu_store.query_cypher(
                            "MATCH (c:CommunityNode {id: $cid}), (e:EpisodeNode {id: $eid}) "
                            "CREATE (c)-[:COMMUNITY_MEMBER]->(e)",
                            {"cid": comm["id"], "eid": member_id}
                        )
                    except Exception:
                        logger.warning("Failed to CREATE COMMUNITY_MEMBER edge", exc_info=True)
                created += 1
            except Exception as e:
                logger.warning("Community persist failed: %s", e)
        return created

    def _persist_prune(self, kuzu_store, prune_ops: list) -> tuple[int, list[str]]:
        """将PRUNE剪枝结果执行真实的Kuzu DELETE。

        先删边再删节点，避免Kuzu外键约束错误。
        返回 (删除数量, 被删节点ID列表)
        """
        deleted = 0
        pruned_ids: list[str] = []
        for op in prune_ops:
            if op.op_type == "delete":
                try:
                    # 先用 DETACH 删掉所有指向该节点的边
                    kuzu_store.query_cypher(
                        "MATCH (e:EpisodeNode {id: $id}) DETACH DELETE e",
                        {"id": op.node_id}
                    )
                    deleted += 1
                    pruned_ids.append(op.node_id)
                except Exception:
                    logger.warning("Failed to DETACH DELETE pruned node from Kuzu", exc_info=True)
        return deleted, pruned_ids

    def _persist_merge(self, kuzu_store, merge_ops: list) -> None:
        """将RESOLVE合并结果写回Kuzu（打标记 + DETACH DELETE被合并节点）。"""
        for op in merge_ops:
            if op.op_type == "update" and op.new_value:
                try:
                    # 把被合并节点的内容保存到目标节点
                    kuzu_store.query_cypher(
                        "MATCH (target:EpisodeNode {id: $target}) "
                        "SET target.content = target.content + ' | merged: ' + $content",
                        {"target": op.new_value, "content": op.old_value or ""}
                    )
                    # 删除被合并节点（DETACH先删边）
                    kuzu_store.query_cypher(
                        "MATCH (e:EpisodeNode {id: $id}) DETACH DELETE e",
                        {"id": op.node_id}
                    )
                except Exception:
                    logger.warning("Failed to persist merge resolution in Kuzu", exc_info=True)

    @staticmethod
    def _persist_merge_get_removed(merge_ops: list) -> list[str]:
        """从 merge_ops 中提取被删除的节点ID（给FAISS增量更新用）。"""
        return [op.node_id for op in merge_ops
                if op.op_type == "update" and op.node_id]

    def _persist_hyperedges(self, kuzu_store, communities: list[dict], dream_id: str) -> int:
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
                kuzu_store.query_cypher(
                    "CREATE (h:HyperedgeNode {id: $id, type: 'semantic', "
                    "created_at: $created_at, gate_value: 1.0, metadata: $metadata})",
                    {"id": hyperedge_id, "created_at": time.time(), "metadata": metadata}
                )
                for member_id in members:
                    try:
                        kuzu_store.query_cypher(
                            "MATCH (h:HyperedgeNode {id: $hid}), (e:EpisodeNode {id: $eid}) "
                            "CREATE (h)-[:HYPEREDGE_MEMBER]->(e)",
                            {"hid": hyperedge_id, "eid": member_id}
                        )
                    except Exception:
                        logger.warning("Failed to CREATE HYPEREDGE_MEMBER edge", exc_info=True)
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
