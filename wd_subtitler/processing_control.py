"""处理任务的取消、关闭和阶段状态。"""

from enum import Enum


class ProcessingCancelled(RuntimeError):
    """用户请求取消处理；该异常不属于任务故障。"""


class ShutdownState(str, Enum):
    """应用关闭过程的状态。"""

    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    CLEANING = "CLEANING"
    CLOSED = "CLOSED"


class CheckpointStage(str, Enum):
    """可持久化恢复的处理阶段。"""

    ASR_COMPLETE = "ASR_COMPLETE"
    REVIEW_COMPLETE = "REVIEW_COMPLETE"
    ARBITRATION_COMPLETE = "ARBITRATION_COMPLETE"
    TIMELINE_COMPLETE = "TIMELINE_COMPLETE"
    TRANSLATING = "TRANSLATING"


STAGE_ORDER = {
    CheckpointStage.ASR_COMPLETE: 1,
    CheckpointStage.REVIEW_COMPLETE: 2,
    CheckpointStage.ARBITRATION_COMPLETE: 3,
    CheckpointStage.TIMELINE_COMPLETE: 4,
    CheckpointStage.TRANSLATING: 5,
}
