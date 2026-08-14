"""SIGTERM/SIGINT 优雅退出处理器注册测试（mock 层，不触发真实信号）。"""
import signal
from unittest.mock import MagicMock, patch

from api.app import _register_signal_handlers


def test_register_signal_handlers_registers_sigterm_sigint():
    loop = MagicMock()
    with patch("api.app.signal.signal") as mock_signal:
        _register_signal_handlers(loop)

    registered = [call.args[0] for call in mock_signal.call_args_list]
    assert signal.SIGTERM in registered
    assert signal.SIGINT in registered


def test_signal_handler_forwards_to_previous_uvicorn_handler():
    """信号到达时转发前驱 handler（uvicorn handle_exit）而非 loop.stop。

    uvicorn 在 lifespan startup 前已安装 handle_exit；直接 loop.stop() 会跳过
    lifespan shutdown 段（写队列不 drain、GraphLite 不 close），转发才是正确路径。
    """
    loop = MagicMock()
    uvicorn_handler = MagicMock()
    with patch("api.app.signal.signal", return_value=uvicorn_handler) as mock_signal:
        _register_signal_handlers(loop)

    sigterm_handler = next(
        call.args[1] for call in mock_signal.call_args_list
        if call.args[0] == signal.SIGTERM
    )
    sigterm_handler(signal.SIGTERM, None)

    uvicorn_handler.assert_called_once_with(signal.SIGTERM, None)
    loop.call_soon_threadsafe.assert_not_called()
