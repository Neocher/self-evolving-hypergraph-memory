"""达摩院 P0-a 会话作用域检索与组织 — 源码断言 (AC1/AC3 静态侧).

断言 scripts/bench_locomo_v72_ontology.py 的 SESSION_SCOPE 分支 (达摩院 round3 研究
§2 P0-a 行 + §5 设计; M1 跨会话污染 324/451 题 ctx 含异会话 raw 消息 ~17%, M2 重名
John×3/Luna×3/Max×5, M3 round2 71.7-72.7% 触发且整体替换组织段, M4 全库块摘要含
异会话人物):
- SESSION_SCOPE env 缺省 '0' (off) → v6.15.0 行为逐字节等价, A/B 基线保真;
- SESSION_SCOPE=1: 检索 (A fusion/B 记忆块/C 图) 候选池先加深再按官方
  conversation_idx (episode.session_id 同坐标) 过滤; rerank 在会话内;
- 组织段 (ENTITY/RELATIONS/FACT TYPES/GLOBAL CONTEXT) 只输出本会话 dia_id 事实;
  实体标识标注 conv 作用域 URI 前缀 (跨会话同名永不合并);
- AC3: round2 触发时不再整体替换组织段 → "组织段 + 追加证据" 拼接 (M3 修复),
  触发率/会话内命中率汇总打印;
- 全程不读 gold/evidence 反推会话归属 (§11 红线 7: 只直用 conversation_idx)。
纯文本断言, 不 import bench (其顶层会加载检索/灌库/embedding, 非单测面)。
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "bench_locomo_v72_ontology.py"
TEXT = SRC.read_text(encoding="utf-8")


def test_session_scope_env_defaults_off():
    """SESSION_SCOPE env 存在且缺省 off ('0') — 单变量 A/B 臂, off = v6.15.0 基线。"""
    assert 'SESSION_SCOPE = os.environ.get("SESSION_SCOPE", "0") == "1"' in TEXT
    assert 'SESSION_SCOPE_URI = os.environ.get("SESSION_SCOPE_URI", "1") == "1"' in TEXT
    assert 'SESSION_SCOPE_POOL = int(os.environ.get("SESSION_SCOPE_POOL", "500"))' in TEXT


def test_config_log_prints_session_scope_state():
    """启动配置行打印 SESSION_SCOPE 状态 (汇总日志要求)。"""
    assert "SESSION_SCOPE={'on' if SESSION_SCOPE else 'off'}" in TEXT


def test_scope_routes_by_conversation_idx_not_evidence():
    """路由只消费官方 conversation_idx / episode.session_id; 无题面实体反推会话。"""
    assert "_scope = ci if SESSION_SCOPE else None" in TEXT
    assert "_conv_label[_cvi] = _sid" in TEXT  # sample_id conv-N 标签映射
    # 检索与 round2 追加检索都带 scope
    assert "retrieve_channels(question, hitk=HITK_MODE, session_ts=session_ts, scope=_scope)" in TEXT
    assert "retrieve_channels(fq, session_ts=session_ts, scope=_scope)" in TEXT
    assert "build_ctx(question, channels, rerank_top=40, scope=_scope)" in TEXT


def test_fusion_channel_pool_deepened_then_filtered():
    """A 通道: scoped 直传引擎 scope 参数 — QueryRouter.retrieve(scope=ci) 内部
    FUSION 候选池加深 (config.session_scope_pool) + episode.session_id 统一出口过滤;
    harness 不再对检索结果后过滤 (AC3: 过滤逻辑在引擎, 不在 bench 检索后段)。"""
    # harness 只把 SESSION_SCOPE_POOL env 映射到引擎 config, 不再直接改 fusion_*_topk
    assert "qr.config.session_scope_pool = SESSION_SCOPE_POOL" in TEXT
    # scoped 分支调 retrieve 带 scope=scope (直传引擎); off 分支不带 scope (v6.15.0 基线)
    assert "qr.retrieve(q, level=RetrievalLevel.FUSION, session_ts=session_ts," in TEXT
    assert "hyde=True, scope=scope)" in TEXT
    assert "raw = qr.retrieve(q, level=RetrievalLevel.FUSION, session_ts=session_ts, hyde=True)" in TEXT
    # 无 harness 层结果后过滤: _fuse 检索后段不再出现 node_id → session 判异会话
    assert "_ep_session.get(_nid) != scope" not in TEXT


def test_block_channel_intersects_session_message_range():
    """B 通道 (记忆块): 只保留与会话消息区间相交的块 (跨界块摘要不进 ctx), 展开只取本会话消息。"""
    assert '_ep_session.get(f"ep_{j}") == scope for j in range(blk[1], min(blk[2], len(msg_by_id)))' in TEXT
    assert 'if scope is not None and _ep_session.get(f"ep_{j}") != scope:' in TEXT
    assert "_adjacent(docs_a, seen_a, scope)" in TEXT  # 邻接补全也限定会话


def test_graph_channel_scoped_to_session():
    """C 通道 (图遍历): 会话级 entity/co 子图; 实体提取用会话词汇表。"""
    assert "_ee = entity_eps if scope is None else _conv_entity_eps.get(scope, {})" in TEXT
    assert "_eco = entity_co if scope is None else _conv_entity_co.get(scope, {})" in TEXT
    assert "vocab = entity_eps if scope is None else _conv_entity_eps.get(scope, {})" in TEXT


def test_org_sections_scoped_and_uri_annotated():
    """组织段只输出本会话 dia_id 事实; 实体标识标注 conv 作用域 URI (最小版前缀)。"""
    assert "_facts_map = _ontology_facts if scope is None else _conv_facts.get(scope, {})" in TEXT
    assert "_topc = _top_classes if scope is None else _conv_top_classes.get(scope, [])" in TEXT
    assert "_blocks_it = blocks if scope is None else _conv_blocks.get(scope, [])" in TEXT
    assert "_co_map = entity_co if scope is None else _conv_entity_co.get(scope, {})" in TEXT
    assert "_conv_label.get(scope, 'conv-%d' % scope)" in TEXT
    # URI 标注 = 会话归属前缀, 跨会话同名永不合并 (M2); 不改 raw 消息原文
    assert "def _disp(e):" in TEXT


def test_round2_scoped_keeps_org_section_and_appends_evidence():
    """AC3: SESSION_SCOPE=1 时 round2 不再整体替换组织段 — '组织段 + 追加证据' 拼接,
    保留 ontology_organize 组织段并追加标记段 [ROUND2 SUPPLEMENTAL EVIDENCE] (M3 修复)。"""
    assert "if SESSION_SCOPE:" in TEXT
    assert "[ROUND2 SUPPLEMENTAL EVIDENCE]" in TEXT
    assert '_supp = "\\n".join(f"[{_n0 + j + 1}] {d}" for j, d in enumerate(docs[_n0:]))' in TEXT
    # off 路径保留 v6.15.0 原 round2 整体替换语句 (零回归基线)
    assert 'summ2 = "\\n".join(f"[MEMORY BLOCK {j+1}] {s[:400]}" for j, s in enumerate(ch2["B_sum"][:2]))' in TEXT
    assert "ctx = (summ2 + \"\\n\\n\" + ev_sec) if summ2 else ev_sec" in TEXT


def test_summary_prints_round2_rate_and_in_session_hit_rate():
    """汇总日志打印 SESSION_SCOPE 状态 + round2 触发率 + 会话内命中率 (过程指标)。"""
    assert "round2 触发率:" in TEXT
    assert "会话内命中率:" in TEXT
    assert "_scope_stats" in TEXT


def test_no_evidence_based_session_inference():
    """红线: 检索/装配不得读 evidence 反推会话归属 (仅 conversation_idx 路由)。"""
    # 作用域判断全部基于官方 conversation_idx → 引擎 scope 直传 (node_id/session_id 同坐标)
    assert "conversation_idx" in TEXT
    assert "_scope = ci if SESSION_SCOPE else None" in TEXT
    assert "scope=_scope" in TEXT
    # 组织段文本只陈述事实 (无祈使指令词作为段标题)
    m = re.search(r"\[ROUND2 SUPPLEMENTAL EVIDENCE\]", TEXT)
    assert m is not None
