"""轻量异常定义。"""


class LiteRunnerError(Exception):
    """基础异常。"""


class TaskNotFoundError(LiteRunnerError):
    """任务不存在。"""
