"""EDA Runner 异常定义。"""

class EDARunnerError(Exception):
    """EDA Runner 基础异常。"""


class ConfigError(EDARunnerError):
    """配置错误。"""


class ConnectionError(EDARunnerError):
    """连接错误（SSH、Socket）。"""


class TimeoutError(EDARunnerError):
    """超时错误。"""


class SubmitError(EDARunnerError):
    """任务提交失败。"""


class TaskNotFoundError(EDARunnerError):
    """任务不存在。"""


class BackendError(EDARunnerError):
    """后端执行错误。"""


class ProtocolError(EDARunnerError):
    """协议解析错误。"""
