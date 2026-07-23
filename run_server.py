"""SHM v4.0 FastAPI 入口 — 启动服务器"""
import uvicorn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 自动 Kuzu 迁移 — 版本不兼容时自动导出→重建→导入
try:
    from scripts.migrate_kuzu_db import migrate
    migrate(backup=True)
except Exception as e:
    import logging
    logging.warning(f"Kuzu migration skipped (non-fatal): {e}")

# 加载API Key — 与Hermes共享同一个数据源
# 优先级: ①~/.hermes/.env(Hermes主配置) ②./.env(SHM本地)
for _env_file in [
    os.path.expanduser("~/.hermes/.env"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
]:
    if os.path.exists(_env_file):
        for line in open(_env_file):
            line = line.strip()
            if line and '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                if k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY",
                         "ANTHROPIC_API_KEY", "KIMI_API_KEY",
                         "OPENROUTER_API_KEY", "GEMINI_API_KEY"):
                    os.environ[k] = v  # SHM .env 优先，覆盖已有值

from api.app import create_app

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="info",
    )
