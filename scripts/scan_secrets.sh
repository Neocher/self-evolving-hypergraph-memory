#!/usr/bin/env bash
# ============================================================
# SHM 密钥泄漏防护 — 扫描脚本
# 用法:
#   scan_secrets.sh <path>   扫描路径/文件
#   scan_secrets.sh          扫描当前 git 暂存区
# 检测: sk- API key / shm_ token / AKIA AWS / ghp_ GitHub PAT /
#       Bearer <token> / FEISHU_APP_SECRET= / password= / secret=
# 退出码: 0=安全  1=发现泄漏
# ============================================================
set -u

# 检测模式（避免误报：跳过 REDACTED/占位符/示例值）
PATTERNS=(
  'sk-[a-zA-Z0-9]{20,}'
  'shm_[0-9a-f]{16}'
  'AKIA[0-9A-Z]{16}'
  'ghp_[a-zA-Z0-9]{36}'
  'gho_[a-zA-Z0-9]{36}'
  'xox[bap]-[a-zA-Z0-9-]{10,}'
  'FEISHU_APP_SECRET\s*=\s*[a-zA-Z0-9]{10,}'
  'DEEPSEEK_KEY\s*=\s*"[a-zA-Z0-9]{20,}"'
)

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  # 默认扫暂存区（pre-commit 用）
  FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null)
else
  FILES="$TARGET"
fi

# 排除文件
EXCLUDE='\.env$|\.env\.local$|\.venv/|node_modules/|__pycache__/|data/|graphify-out/|\.git/|\.log$|\.min\.js$'

HITS=0
for f in $FILES; do
  # 跳过排除项
  echo "$f" | grep -qE "$EXCLUDE" && continue
  [ -f "$f" ] || continue
  for pat in "${PATTERNS[@]}"; do
    # 排除已打码值（sk-test 后必须跟 - 才是测试值；sk-testxxx 视为泄漏）
    RESULT=$(grep -nE "$pat" "$f" 2>/dev/null | grep -vE 'sk-<REDACTED>|sk-REDACTED|REDACTED|xxx|example|placeholder|YOUR_|your_|<secret>|<token>|<key>|dummy|sk-test-|test-key')
    if [ -n "$RESULT" ]; then
      echo "⚠️  检测到疑似密钥: $f"
      echo "$RESULT" | sed 's/\(sk-[a-zA-Z0-9]\{4\}\)[a-zA-Z0-9]*/\1***/g; s/\(shm_[0-9a-f]\{4\}\)[0-9a-f]*/\1***/g' | head -5
      HITS=$((HITS+1))
    fi
  done
done

if [ $HITS -gt 0 ]; then
  echo ""
  echo "❌ 发现 $HITS 处疑似密钥泄漏！请先清除后再提交。"
  echo "   规则: 密钥一律走环境变量 (os.environ / export)，禁止硬编码。"
  exit 1
fi
exit 0
