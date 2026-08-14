"""处理断点的指纹计算、原子存储与恢复判定。"""

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .processing_control import CheckpointStage
from .processing_models import MediaTask, ProcessingOptions, ResumePlan
from .runtime_paths import CHECKPOINT_DIR
from .asr_service import WhisperASRService
from .asr_review import LARGE_V3_MODEL


SCHEMA_VERSION = 2


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stable_hash(value) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _quick_file_hash(path: Path, chunk_size=1024 * 1024) -> str:
    """计算文件首尾区块的快速哈希。"""
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as source:
        digest.update(source.read(chunk_size))
        if size > chunk_size:
            source.seek(max(0, size - chunk_size))
            digest.update(source.read(chunk_size))
    return digest.hexdigest()


def source_fingerprint(task: MediaTask) -> dict:
    source = task.source_path.resolve()
    stat = source.stat()
    return {
        "path": str(source),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "quick_hash": _quick_file_hash(source),
        "track_index": task.selected_track_index,
    }


def stage_fingerprints(options: ProcessingOptions, translation_model: str) -> dict:
    """生成各阶段独立配置指纹，不包含 API Key 和导出格式。"""
    asr = _stable_hash({
        "model": WhisperASRService.DEFAULT_MODEL,
        "hotwords": options.asr_hotwords,
        "quality_mode": options.quality_mode,
        "model_precision": options.model_precision,
    })
    review = _stable_hash({
        "asr": asr,
        "enabled": options.large_v3_review,
        "model": LARGE_V3_MODEL if options.large_v3_review else "",
    })
    arbitration = _stable_hash({
        "review": review,
        "enabled": options.ai_asr_arbitration,
        "base_url": options.base_url.rstrip("/") if options.ai_asr_arbitration else "",
        "model": translation_model if options.ai_asr_arbitration else "",
    })
    timeline = _stable_hash({
        "arbitration": arbitration,
        "enabled": options.timeline_refinement,
    })
    translation = _stable_hash({
        "timeline": timeline,
        "do_translate": options.do_translate,
        "use_context": options.use_context,
        "base_url": options.base_url.rstrip("/"),
        "model": translation_model,
    })
    return {
        "asr": asr,
        "review": review,
        "arbitration": arbitration,
        "timeline": timeline,
        "translation": translation,
    }


def context_input_signature(tasks: list[MediaTask]) -> str:
    """用文件集合和最终日文文本标识术语表或系列总纲输入。"""
    payload = [
        {
            "source": str(task.source_path.resolve()),
            "text": [str(segment.get("text", "")) for segment in task.segments],
        }
        for task in sorted(tasks, key=lambda item: str(item.source_path.resolve()))
    ]
    return _stable_hash(payload)


class CheckpointStore:
    """管理一个媒体文件的可恢复处理断点。"""

    def __init__(self, directory: Path | None = None):
        self.directory = Path(directory or CHECKPOINT_DIR)

    def _checkpoint_path(self, task: MediaTask) -> Path:
        identity = _stable_hash({
            "path": str(task.source_path.resolve()),
            "track_index": task.selected_track_index,
        })[:24]
        return self.directory / f"{identity}.json"

    def _load(self, path: Path) -> dict | None:
        try:
            with path.open("r", encoding="utf-8") as source:
                data = json.load(source)
            if data.get("schema_version") != SCHEMA_VERSION:
                return None
            return data
        except (OSError, ValueError, TypeError):
            return None

    def _write_atomic(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temp_path = Path(raw_path)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                json.dump(data, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temp_path, path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _base_data(
        self,
        task: MediaTask,
        options: ProcessingOptions,
        translation_model: str,
    ) -> dict:
        now = _utc_now().isoformat()
        path = self._checkpoint_path(task)
        current = self._load(path) or {}
        return {
            "schema_version": SCHEMA_VERSION,
            "source": source_fingerprint(task),
            "fingerprints": stage_fingerprints(options, translation_model),
            "snapshots": current.get("snapshots", {}),
            "created_at": current.get("created_at", now),
            "updated_at": now,
            "stage": current.get("stage", CheckpointStage.ASR_COMPLETE.value),
            "translated_count": current.get("translated_count", 0),
            "context_input_signature": current.get("context_input_signature", ""),
            "context_summary": current.get("context_summary", ""),
            "context_ready": current.get("context_ready", False),
        }

    def save_stage(
        self,
        task: MediaTask,
        stage: CheckpointStage,
        options: ProcessingOptions,
        translation_model: str,
    ) -> Path:
        data = self._base_data(task, options, translation_model)
        data["stage"] = stage.value
        stage_rank = {
            CheckpointStage.ASR_COMPLETE: 1,
            CheckpointStage.REVIEW_COMPLETE: 2,
            CheckpointStage.ARBITRATION_COMPLETE: 3,
            CheckpointStage.TIMELINE_COMPLETE: 4,
            CheckpointStage.TRANSLATING: 5,
        }
        data["snapshots"] = {
            key: value
            for key, value in data["snapshots"].items()
            if key in CheckpointStage._value2member_map_
            and stage_rank[CheckpointStage(key)] <= stage_rank[stage]
        }
        data["snapshots"][stage.value] = deepcopy(task.segments)
        data["translated_count"] = 0
        self._write_atomic(self._checkpoint_path(task), data)
        return self._checkpoint_path(task)

    def save_translation_progress(
        self,
        task: MediaTask,
        translated_count: int,
        options: ProcessingOptions,
        translation_model: str,
        context_signature: str = "",
        context_summary: str = "",
        context_ready: bool = False,
    ) -> Path:
        data = self._base_data(task, options, translation_model)
        data["stage"] = CheckpointStage.TRANSLATING.value
        data["snapshots"][CheckpointStage.TRANSLATING.value] = deepcopy(task.segments)
        data["translated_count"] = translated_count
        data["context_input_signature"] = context_signature
        data["context_summary"] = context_summary
        data["context_ready"] = context_ready
        self._write_atomic(self._checkpoint_path(task), data)
        return self._checkpoint_path(task)

    def find_resume(
        self,
        task: MediaTask,
        options: ProcessingOptions,
        translation_model: str,
        expected_context_signature: str | None = None,
    ) -> ResumePlan | None:
        path = self._checkpoint_path(task)
        data = self._load(path)
        if not data:
            return None
        try:
            if data["source"] != source_fingerprint(task):
                return None
            current = stage_fingerprints(options, translation_model)
            stored = data.get("fingerprints", {})
            snapshots = data.get("snapshots", {})
            reusable = None
            for stage, key in (
                (CheckpointStage.ASR_COMPLETE, "asr"),
                (CheckpointStage.REVIEW_COMPLETE, "review"),
                (CheckpointStage.ARBITRATION_COMPLETE, "arbitration"),
                (CheckpointStage.TIMELINE_COMPLETE, "timeline"),
            ):
                if stored.get(key) != current[key] or stage.value not in snapshots:
                    break
                reusable = stage

            translated_count = 0
            if (
                reusable == CheckpointStage.TIMELINE_COMPLETE
                and stored.get("translation") == current["translation"]
                and CheckpointStage.TRANSLATING.value in snapshots
                and expected_context_signature is not None
                and data.get("context_input_signature", "") == expected_context_signature
                and (not options.use_context or bool(data.get("context_ready")))
            ):
                reusable = CheckpointStage.TRANSLATING
                translated_count = int(data.get("translated_count", 0))
            if reusable is None:
                return None
            updated = datetime.fromisoformat(data["updated_at"])
            return ResumePlan(
                checkpoint_path=path,
                stage=reusable,
                segments=deepcopy(snapshots[reusable.value]),
                updated_at=updated,
                translated_count=translated_count,
                context_input_signature=data.get("context_input_signature", ""),
                context_summary=data.get("context_summary", ""),
            )
        except (KeyError, ValueError, TypeError, OSError):
            return None

    def discard(self, task: MediaTask) -> None:
        path = self._checkpoint_path(task)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def delete_completed(self, task: MediaTask) -> None:
        self.discard(task)

    def cleanup_expired(self, days=7) -> int:
        if not self.directory.exists():
            return 0
        cutoff = _utc_now() - timedelta(days=days)
        removed = 0
        for path in self.directory.glob("*.json"):
            data = self._load(path)
            try:
                updated = datetime.fromisoformat(data["updated_at"]) if data else None
            except (KeyError, ValueError, TypeError):
                updated = None
            if updated is None or updated < cutoff:
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed
