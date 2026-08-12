#!/usr/bin/env python3
"""
Hermes ↔ SHM Integration Checker v2

Standalone check script for installers. Pure Python stdlib, no third-party deps.

6 checks per design_v2 (CC reviewed):
  1. SHM health endpoint
  2. Plugin directory location
  3. Plugin loadable via real discovery mechanism
  4. Config provider + memory_enabled (indentation-aware YAML parse, no PyYAML)
  5. prefetch end-to-end
  6. System prompt injection

Usage:
  python3 scripts/check_hermes_integration.py [--debug]

Exit codes:
  0 = all PASS
  1 = any FAIL

Design: /home/admin/shm/design_hermes_check.md (v2)
"""

import json
import os
import re
import sys
import traceback
import urllib.error
import urllib.request
from pathlib import Path

# ── CLI ────────────────────────────────────────────────────────────────

DEBUG = "--debug" in sys.argv


# ── Helpers ─────────────────────────────────────────────────────────────


def _resolve_hermes_home() -> Path:
    """Resolve HERMES_HOME: env var or ~/.hermes."""
    raw = os.environ.get("HERMES_HOME", "")
    if raw:
        return Path(raw).expanduser().resolve()
    return Path.home() / ".hermes"


def _log_debug(msg: str) -> None:
    """Print debug info to stderr so it doesn't mix with stdout output."""
    if DEBUG:
        print(f"  [DEBUG] {msg}", file=sys.stderr)


def _traceback_str() -> str:
    """Return current exception traceback as one-line summary."""
    try:
        return traceback.format_exc().strip().split("\n")[-1]
    except Exception:
        return ""


def _format_tb() -> str:
    """Return full exception traceback (for --debug)."""
    return traceback.format_exc().strip()


# ── Check 1: SHM Health ────────────────────────────────────────────────


def check_shm_health() -> tuple[str, str, str | None, dict | None]:
    """GET http://127.0.0.1:8000/health (5s timeout).

    Returns: (verdict, detail, fix_hint | None, health_data | None)
      - PASS + "(vX.Y.Z, N 节点)" when status ∈ {"ok", "degraded"}
      - WARN when status == "degraded"
      - FAIL otherwise
    """
    url = "http://127.0.0.1:8000/health"
    _opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        req = urllib.request.Request(url)
        resp = _opener.open(req, timeout=5)
        raw = resp.read()
        data = json.loads(raw)
    except urllib.error.HTTPError as e:
        _log_debug(f"Health HTTP {e.code}: {_traceback_str()}")
        return (
            "FAIL",
            f"HTTP {e.code}",
            f"SHM 服务返回 HTTP {e.code}，请检查 shm-server 是否正常运行",
            None,
        )
    except urllib.error.URLError as e:
        _log_debug(f"Health URLError: {e.reason}")
        return (
            "FAIL",
            str(e.reason),
            "SHM 服务未启动或无法连接，请运行: systemctl --user start shm-server",
            None,
        )
    except Exception as e:
        _log_debug(f"Health error: {_format_tb()}")
        return (
            "FAIL",
            str(e)[:80],
            "SHM 健康检查异常，请确认 shm-server 在 127.0.0.1:8000 正常运行",
            None,
        )

    status = data.get("status", "error")
    stats = data.get("stats", {})
    version = stats.get("version", "?")
    node_count = stats.get("node_count", 0)
    detail = f"(v{version}, {node_count} 节点)"

    if status == "ok":
        return ("PASS", detail, None, data)
    elif status == "degraded":
        return ("WARN", f"{detail} [status=degraded]", None, data)
    else:
        return (
            "FAIL",
            f"{detail} [status={status}]",
            f"SHM 服务状态异常: status={status}，请检查日志",
            data,
        )


# ── Check 2: Plugin Directory ──────────────────────────────────────────


def check_plugin_dir(hermes_home: Path) -> tuple[str, str, str | None]:
    """Check ~/.hermes/plugins/shm_v5/ exists with __init__.py.

    Also detect wrong location: ~/.hermes/plugins/memory/shm_v5/
    """
    correct = hermes_home / "plugins" / "shm_v5"
    wrong = hermes_home / "plugins" / "memory" / "shm_v5"
    init_file = correct / "__init__.py"

    wrong_exists = wrong.is_dir() and (wrong / "__init__.py").exists()

    if init_file.exists():
        detail = str(correct)
        if wrong_exists:
            detail += " (⚠ 检测到错误位置副本，可删除)"
        return ("PASS", detail, None)

    if wrong_exists:
        return (
            "FAIL",
            f"插件位于错误位置: {wrong}",
            f"请移动: mv {wrong} {correct}",
        )

    return (
        "FAIL",
        f"插件目录不存在: {correct}",
        f"请创建 {correct}/ 并放置 __init__.py 和 plugin.yaml",
    )


# ── Check 3: Plugin Loadable ───────────────────────────────────────────


def check_plugin_loadable(hermes_home: Path) -> tuple[str, str, str | None, object | None]:
    """Load plugin via real discovery mechanism.

    1. os.environ.setdefault("HERMES_HOME", ...)
    2. sys.path.insert(0, $HERMES_HOME/hermes-agent)
    3. from plugins.memory import discover_memory_providers
    4. providers = {n: a for n, d, a in discover_memory_providers()}
    5. PASS: "shm_v5" in providers and available=True

    Also returns the loaded provider instance for use in checks 5/6.
    """
    os.environ.setdefault("HERMES_HOME", str(hermes_home))

    agent_dir = hermes_home / "hermes-agent"
    if not agent_dir.is_dir():
        return (
            "FAIL",
            f"hermes-agent 源码目录不存在: {agent_dir}",
            f"请确认 Hermes 已安装，hermes-agent 目录位于 {agent_dir}",
            None,
        )

    if str(agent_dir) not in sys.path:
        sys.path.insert(0, str(agent_dir))

    try:
        from plugins.memory import discover_memory_providers
    except ImportError as e:
        _log_debug(f"Import discover_memory_providers: {_format_tb()}")
        return (
            "FAIL",
            f"无法导入 discover_memory_providers: {e}",
            f"请确认 {agent_dir}/plugins/memory/__init__.py 存在且无语法错误",
            None,
        )

    try:
        discovered = discover_memory_providers()
        providers = {n: a for n, _d, a in discovered}
    except Exception as e:
        _log_debug(f"discover_memory_providers: {_format_tb()}")
        return (
            "FAIL",
            f"discover_memory_providers() 异常: {e}",
            "请检查插件目录权限和 __init__.py 语法",
            None,
        )

    if "shm_v5" not in providers:
        return (
            "FAIL",
            f"插件列表不含 shm_v5: {list(providers.keys())}",
            f"请确认 {hermes_home}/plugins/shm_v5/__init__.py 存在且包含有效的 MemoryProvider 实现",
            None,
        )

    available = providers["shm_v5"]
    if not isinstance(available, bool):
        return (
            "WARN",
            f"shm_v5 discovered 但 available 类型异常: {type(available).__name__} (期望 bool)",
            "插件 discover_memory_providers() 返回结构异常，请检查第三元是否为 bool",
            None,
        )
    if not available:
        return (
            "FAIL",
            "shm_v5 插件 discovered 但 available=False",
            "请检查插件 __init__.py 中 is_available() 方法返回值",
            None,
        )

    # Try to load the actual provider instance for checks 5/6
    try:
        from plugins.memory import load_memory_provider
        provider_instance = load_memory_provider("shm_v5")
    except Exception as e:
        _log_debug(f"load_memory_provider: {_format_tb()}")
        return (
            "FAIL",
            f"(discovered, load 失败: {e})",
            "插件已发现但加载失败，请检查 MemoryProvider 实现",
            None,
        )

    pname = type(provider_instance).__name__ if provider_instance is not None else "MemoryProvider"
    return ("PASS", pname, None, provider_instance)


# ── Check 4: Config Provider ───────────────────────────────────────────


def check_config(hermes_home: Path) -> tuple[str, str, str | None]:
    """Indentation-aware parse of ~/.hermes/config.yaml.

    No PyYAML dependency — line-by-line scan for top-level "memory:" block.
    Within that block, extract "provider" and "memory_enabled" values.
    Values may be quoted — strip quotes.
    """
    config_path = hermes_home / "config.yaml"
    if not config_path.is_file():
        return (
            "FAIL",
            f"配置文件不存在: {config_path}",
            f"请确认 ~/.hermes/config.yaml 存在，或 HERMES_HOME 环境变量正确",
        )

    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return (
            "FAIL",
            f"无法读取配置文件: {e}",
            "请检查 ~/.hermes/config.yaml 文件权限",
        )

    # Find top-level "memory:" block
    provider: str | None = None
    memory_enabled: str | None = None

    in_memory_block = False
    memory_indent: int | None = None
    for line in lines:
        # Detect top-level "memory:" (no leading whitespace), allow trailing comment
        if re.match(r"^memory:\s*(#.*)?$", line):
            in_memory_block = True
            memory_indent = None
            continue

        if not in_memory_block:
            continue

        # Exit memory block when we hit another top-level key (no indent)
        stripped = line.strip()
        if stripped and not line.startswith((" ", "\t")) and not stripped.startswith("#"):
            break

        # Skip comments and empty lines
        if not stripped or stripped.startswith("#"):
            continue

        # Capture indent of first sub-key in memory block
        line_indent = len(line) - len(line.lstrip())
        if memory_indent is None:
            memory_indent = line_indent

        # Only match keys at the recorded indent level (ignore nested keys)
        if line_indent != memory_indent:
            continue

        # Parse indented keys within the memory block
        m_provider = re.match(r"^\s+provider:\s*(.+?)\s*$", line)
        m_enabled = re.match(r"^\s+memory_enabled:\s*(.+?)\s*$", line)

        if m_provider:
            raw = m_provider.group(1).strip()
            # Strip comments and quotes
            raw = re.sub(r"\s*#.*$", "", raw)
            provider = raw.strip("\"'")
        elif m_enabled:
            raw = m_enabled.group(1).strip()
            raw = re.sub(r"\s*#.*$", "", raw)
            memory_enabled = raw.strip("\"'")

    provider_ok = provider == "shm_v5"
    enabled_ok = memory_enabled is not None and memory_enabled.lower() in ("true", "yes", "on", "1")

    if provider_ok and enabled_ok:
        return ("PASS", f"(shm_v5, memory_enabled=true)", None)

    issues = []
    fix_hints = []
    if not provider_ok:
        detail_prov = provider if provider else "未设置"
        issues.append(f"provider={detail_prov}")
        fix_hints.append('设置 memory.provider = "shm_v5"')
    if not enabled_ok:
        detail_en = memory_enabled if memory_enabled else "未设置"
        issues.append(f"memory_enabled={detail_en}")
        fix_hints.append("设置 memory.memory_enabled = true")

    detail = ", ".join(issues)
    fix = f"在 {config_path} 中: {'; '.join(fix_hints)}"
    return ("FAIL", detail, fix)


# ── Check 5: Prefetch end-to-end ───────────────────────────────────────


def check_prefetch(provider_instance: object) -> tuple[str, str, str | None]:
    """Call p.prefetch("SHM 记忆系统") with 10s timeout.

    Non-empty result = PASS. Compatible with str/list/dict return types.
    Exception/timeout = FAIL.
    """
    if provider_instance is None:
        return ("FAIL", "无 provider 实例", None)

    if not hasattr(provider_instance, "prefetch"):
        return ("FAIL", "provider 实例无 prefetch 方法", None)

    # Initialize if needed (sets _api_available via health check)
    if hasattr(provider_instance, "initialize") and not getattr(
        provider_instance, "_initialized", False
    ):
        try:
            provider_instance.initialize("check_session")
        except Exception as e:
            _log_debug(f"provider initialize: {_format_tb()}")
            return ("FAIL", f"initialize 失败: {e}", None)

    try:
        import signal

        # 10s timeout — SIGALRM works on Unix
        def _timeout_handler(_signum, _frame):
            raise TimeoutError("prefetch timed out")

        signal.signal(signal.SIGALRM, _timeout_handler)
        try:
            signal.alarm(10)
            result = provider_instance.prefetch("SHM 记忆系统")
        finally:
            signal.alarm(0)

    except TimeoutError:
        return (
            "FAIL",
            "请求超时 (10s)",
            "SHM 服务响应过慢，请检查 /memories/retrieve 端点延迟",
        )
    except Exception as e:
        _log_debug(f"prefetch: {_format_tb()}")
        return (
            "FAIL",
            str(e)[:100],
            "prefetch 失败，请检查 SHM 服务日志和 /memories/retrieve 端点",
        )

    # Validate result: handle str, list, dict
    # 兼容两种标题: 官方 "SHM v5 记忆检索" 与本机旧版 "【SHM 记忆检索结果】"
    if isinstance(result, str):
        has_content = bool(result.strip()) and (
            "SHM v5 记忆检索" in result or "SHM 记忆检索结果" in result
        )
        count_hint = None
    elif isinstance(result, (list, tuple)):
        has_content = len(result) > 0
        count_hint = f"{len(result)} 条记忆" if has_content else None
    elif isinstance(result, dict):
        has_content = len(result) > 0
        count_hint = f"{len(result)} 键" if has_content else None
    else:
        has_content = bool(result)
        count_hint = None

    if has_content:
        detail = count_hint if count_hint else ""
        return ("PASS", detail, None)
    else:
        return (
            "FAIL",
            "返回空结果",
            "SHM 连接正常但检索无结果，请确认已写入一些记忆数据",
        )


# ── Check 6: System Prompt ─────────────────────────────────────────────


def check_system_prompt(provider_instance: object) -> tuple[str, str, str | None]:
    """Call p.system_prompt_block() — must contain '记忆系统' or '节点'."""
    if provider_instance is None:
        return ("FAIL", "无 provider 实例", None)

    if not hasattr(provider_instance, "system_prompt_block"):
        return ("FAIL", "provider 实例无 system_prompt_block 方法", None)

    try:
        prompt = provider_instance.system_prompt_block()
    except Exception as e:
        _log_debug(f"system_prompt_block: {_format_tb()}")
        return ("FAIL", str(e)[:100], None)

    if not isinstance(prompt, str) or not prompt.strip():
        return ("FAIL", "system_prompt_block 返回空字符串", None)

    has_memory = "记忆系统" in prompt
    has_node = "节点" in prompt

    if has_memory or has_node:
        found = []
        if has_memory:
            found.append("'记忆系统'")
        if has_node:
            found.append("'节点'")
        return ("PASS", f"含 {', '.join(found)}", None)
    else:
        return ("FAIL", "不含 '记忆系统' 或 '节点'", "system_prompt_block 内容不完整")


# ── Output ──────────────────────────────────────────────────────────────


def main() -> int:
    hermes_home = _resolve_hermes_home()
    _log_debug(f"HERMES_HOME = {hermes_home}")

    checks: list[tuple[int, str, str, str, str | None]] = []
    # Each: (idx, label, verdict, detail, fix_hint)

    provider_instance: object | None = None
    shm_ok = True  # Tracks whether check 1 passed for check 5/6 dependency

    # ── 1 SHM Health ──
    verdict, detail, fix, _ = check_shm_health()
    shm_ok = verdict != "FAIL"
    checks.append((1, "SHM 服务存活", verdict, detail, fix))

    # ── 2 Plugin Dir ──
    verdict, detail, fix = check_plugin_dir(hermes_home)
    checks.append((2, "插件目录位置", verdict, detail, fix))

    # ── 3 Plugin Loadable ──
    verdict, detail, fix, provider_instance = check_plugin_loadable(hermes_home)
    checks.append((3, "插件可加载", verdict, detail, fix))

    # ── 4 Config Provider ──
    verdict, detail, fix = check_config(hermes_home)
    checks.append((4, "config provider", verdict, detail, fix))

    # ── 5 Prefetch ──
    if not shm_ok:
        checks.append(
            (5, "prefetch 端到端", "SKIP", "上游依赖未满足（跳过判级）", None)
        )
    else:
        verdict, detail, fix = check_prefetch(provider_instance)
        checks.append((5, "prefetch 端到端", verdict, detail, fix))

    # ── 6 System Prompt ──
    if not shm_ok:
        checks.append(
            (6, "状态注入", "SKIP", "上游依赖未满足（跳过判级）", None)
        )
    else:
        verdict, detail, fix = check_system_prompt(provider_instance)
        checks.append((6, "状态注入", verdict, detail, fix))

    # ── Print results ──
    print("=== Hermes ↔ SHM 对接检测 ===")
    pass_count = 0
    fail_count = 0
    warn_count = 0
    skip_count = 0

    for idx, label, verdict, detail, fix in checks:
        detail_str = f" {detail}" if detail else ""
        line = f"[{idx}/6] {label:<20} ... {verdict}{detail_str}"
        print(line)
        if fix:
            print(f"  FIX: {fix}")

        if verdict == "PASS":
            pass_count += 1
        elif verdict == "FAIL":
            fail_count += 1
        elif verdict == "WARN":
            warn_count += 1
        elif verdict == "SKIP":
            skip_count += 1

    # ── Summary ──
    parts = []
    if pass_count:
        parts.append(f"{pass_count} PASS")
    if warn_count:
        parts.append(f"{warn_count} WARN")
    if skip_count:
        parts.append(f"{skip_count} SKIP")
    if fail_count:
        parts.append(f"{fail_count} FAIL")

    if fail_count == 0:
        conclusion = f"结论: {'/'.join(parts)} — Hermes ↔ SHM 对接成功"
    else:
        conclusion = f"结论: {'/'.join(parts)} — Hermes ↔ SHM 对接存在问题，请按 FIX 提示修复"

    print(conclusion)

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
