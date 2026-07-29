#!/bin/bash
# ================================================
# 模型推理快捷调用（给 Hermes Agent 自己用的）
# ================================================
# 用法: ./llm.sh <model> <prompt> [max_tokens]
# 例:   ./llm.sh qwen/qwen3-omni-flash "你好" 100

MODEL="${1:-qwen/qwen3-omni-flash}"
PROMPT="$2"
MAX_TOKENS="${3:-500}"

if [ -z "$PROMPT" ]; then
  echo '{"error":"prompt required"}'
  exit 1
fi

curl -s --max-time 60 --noproxy '*' \
  "http://127.0.0.1:8088/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<END
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": $MAX_TOKENS
}
END
)" 2>/dev/null | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    if 'choices' in d:
        print(d['choices'][0]['message']['content'].strip())
    else:
        print(str(d)[:200])
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
