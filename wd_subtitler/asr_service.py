import os
import sys
import time
import re
import ctypes
import subprocess
import tempfile

if os.name == "nt":
    _dll_dirs = []
    try:
        for pkg in ["nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime", "nvidia.cuda_nvrtc"]:
            d = os.path.join(sys.prefix, "Lib", "site-packages", pkg.replace(".", os.sep), "bin")
            if os.path.isdir(d):
                _dll_dirs.append(d)
        cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        if os.path.isdir(cuda_base):
            try:
                for v in os.listdir(cuda_base):
                    vp = os.path.join(cuda_base, v)
                    if os.path.isdir(vp) and v.lower().startswith("v"):
                        for sub in ["bin", os.path.join("bin", "x64")]:
                            d = os.path.join(vp, sub)
                            if os.path.isdir(d) and d not in _dll_dirs:
                                _dll_dirs.append(d)
            except OSError:
                pass
        cuda_env = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if cuda_env and os.path.isdir(cuda_env):
            for sub in ["bin", os.path.join("bin", "x64")]:
                d = os.path.join(cuda_env, sub)
                if os.path.isdir(d) and d not in _dll_dirs:
                    _dll_dirs.append(d)
    except Exception:
        pass
    for d in _dll_dirs:
        try:
            os.add_dll_directory(d)
        except OSError:
            pass
        if d not in os.environ.get("PATH", ""):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def _cuda_dlls_ok():
    if os.name != "nt":
        return True
    for dll in ["cublas64_12.dll", "cudart64_12.dll"]:
        try:
            ctypes.WinDLL(dll)
        except OSError:
            return False
    return True


try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class WhisperASRService:
    """基于 Faster-Whisper 的本地语音识别服务"""

    DEFAULT_MODEL = "kotoba-tech/kotoba-whisper-v2.0-faster"
    DEFAULT_DEVICE = "cuda"
    DEFAULT_COMPUTE_TYPE = "float16"

    # Whisper 常见日语幻觉文本（静音/BGM 段凭空生成）
    HALLUCINATION_PATTERNS = [
        "ご視聴ありがとうございました",
        "ありがとうございました",
        "ご清聴ありがとうございました",
        "字幕提供",
        "提供：",
        "ご視聴ありがとうございます",
        "ありがとうございます",
        "チャンネル登録",
        "高評価",
        "フォロー",
    ]

    @staticmethod
    def _evaluate_segment_quality(segment):
        """根据 Whisper 原始指标评估片段质量，供后续复核模型筛选。"""
        score = 100
        reasons = []
        avg_logprob = segment.get("avg_logprob")
        no_speech_prob = segment.get("no_speech_prob")
        compression_ratio = segment.get("compression_ratio")
        decoding_temperature = segment.get("decoding_temperature")

        if avg_logprob is not None:
            if avg_logprob < -1.3:
                score -= 45
                reasons.append("识别概率很低")
            elif avg_logprob < -0.9:
                score -= 30
                reasons.append("识别概率偏低")
            elif avg_logprob < -0.7:
                score -= 15
                reasons.append("识别概率略低")

        if no_speech_prob is not None:
            if no_speech_prob > 0.7:
                score -= 35
                reasons.append("疑似静音或背景声")
            elif no_speech_prob > 0.5:
                score -= 15
                reasons.append("语音存在概率偏低")

        if compression_ratio is not None:
            if compression_ratio > 2.5:
                score -= 35
                reasons.append("疑似重复或幻觉")
            elif compression_ratio > 2.2:
                score -= 15
                reasons.append("文本重复度偏高")

        if decoding_temperature is not None and decoding_temperature >= 0.4:
            score -= 10
            reasons.append("解码触发高温回退")

        quality_score = max(0, min(100, score))
        return {
            "quality_score": quality_score,
            "needs_review": quality_score < 75,
            "review_reasons": reasons,
            "low_confidence": quality_score <= 55,
        }

    @staticmethod
    def _merge_quality_metrics(target, source):
        """合并片段时保留风险更高的一组置信度指标。"""
        for key in ("avg_logprob",):
            values = [value for value in (target.get(key), source.get(key)) if value is not None]
            target[key] = min(values) if values else None
        for key in ("no_speech_prob", "compression_ratio", "decoding_temperature"):
            values = [value for value in (target.get(key), source.get(key)) if value is not None]
            target[key] = max(values) if values else None

    @staticmethod
    def _find_clip_group(start, end, clip_timestamps):
        """返回片段所属的裁剪区间编号，防止离散复核片段被后处理合并。"""
        if not isinstance(clip_timestamps, (list, tuple)):
            return None
        midpoint = (float(start) + float(end)) / 2
        for index in range(0, len(clip_timestamps) - 1, 2):
            if float(clip_timestamps[index]) <= midpoint <= float(clip_timestamps[index + 1]):
                return index // 2
        return None

    def __init__(self, model_name=None, device=None, compute_type=None):
        if WhisperModel is None:
            raise ImportError("请先安装 faster-whisper: pip install faster-whisper")

        self.model_name = model_name or self.DEFAULT_MODEL
        self.device = device or self.DEFAULT_DEVICE
        self.compute_type = compute_type or self.DEFAULT_COMPUTE_TYPE
        self.model = None
        self.model_loaded = False

    def load_model(self, progress_callback=None):
        """加载模型（可重复调用，内部做单例判断）"""
        if self.model_loaded and self.model is not None:
            return

        if progress_callback:
            progress_callback(f"正在加载模型：{self.model_name}（{self.device} / {self.compute_type}）")

        cuda_available = False
        if self.device == "cuda":
            try:
                import ctranslate2
                cuda_count = ctranslate2.get_cuda_device_count()
                dlls_ok = _cuda_dlls_ok()
                if cuda_count > 0 and dlls_ok:
                    cuda_available = True
                    if progress_callback:
                        progress_callback(f"检测到 {cuda_count} 个 CUDA 设备")
                elif cuda_count > 0:
                    if progress_callback:
                        progress_callback("⚠️ CUDA 设备存在，但运行时 DLL 缺失；改用 CPU 模式")
                else:
                    if progress_callback:
                        progress_callback("未检测到 CUDA 设备；改用 CPU 模式")
            except Exception:
                cuda_available = False

        # 构建尝试序列
        attempts = []
        if self.device == "cuda" and cuda_available:
            attempts.append(("cuda", self.compute_type))
            if self.compute_type != "int8_float16":
                attempts.append(("cuda", "int8_float16"))
            attempts.append(("cpu", "int8"))
        else:
            attempts.append(("cpu", "int8"))

        last_error = None
        for device, compute_type in attempts:
            try:
                if progress_callback and attempts.index((device, compute_type)) > 0:
                    if device == "cpu":
                        progress_callback("⚠️ GPU 加载失败；回退到 CPU int8 模式")
                    else:
                        progress_callback(f"⚠️ 当前精度加载失败；正在尝试 {compute_type}")

                self.model = WhisperModel(
                    self.model_name,
                    device=device,
                    compute_type=compute_type,
                )
                self.device = device
                self.compute_type = compute_type
                self.model_loaded = True

                if progress_callback:
                    progress_callback(f"✅ 模型加载完成：设备 {device}；精度 {compute_type}")
                return

            except Exception as e:
                last_error = e
                if progress_callback:
                    err_msg = str(e)[:150]
                    progress_callback(f"⚠️ 模型加载失败：{device} / {compute_type}；{err_msg}")
                continue

        raise Exception(f"模型加载失败（已尝试所有设备/精度组合）: {str(last_error)}")

    def _extract_audio(self, fpath, progress_callback=None):
        """用 ffmpeg 从视频中提取 16kHz 单声道 WAV，提升识别速度"""
        ext = os.path.splitext(fpath)[1].lower()
        # 纯音频格式不需要提取
        if ext in ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma'):
            return None

        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav', prefix='whisper_')
            os.close(tmp_fd)

            cmd = [
                'ffmpeg', '-i', fpath,
                '-vn',             # 不要视频流
                '-ac', '1',        # 单声道
                '-ar', '16000',    # 16kHz
                '-f', 'wav',
                '-y',              # 覆盖
                tmp_path
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                if progress_callback:
                    progress_callback("✅ 音频提取完成：16 kHz / 单声道")
                return tmp_path
            else:
                if progress_callback:
                    progress_callback("⚠️ 音频提取失败；将使用原始文件")
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return None

        except FileNotFoundError:
            if progress_callback:
                progress_callback("⚠️ 未找到 FFmpeg；将使用原始文件")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None
        except Exception as e:
            if progress_callback:
                progress_callback(f"⚠️ 音频提取异常：{e}；将使用原始文件")
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None

    def transcribe(
        self,
        fpath,
        language="ja",
        initial_prompt=None,
        beam_size=2,
        best_of=5,
        patience=1.0,
        temperature=0,
        condition_on_previous_text=False,
        hotwords=None,
        clip_timestamps="0",
        vad_filter=True,
        word_timestamps=True,
        progress_callback=None,
        cancel_check=None,
    ):
        """
        转录音频/视频文件

        Returns:
            list: [{"start", "end", "text", "trans", "low_confidence"}, ...]
        """
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"文件不存在: {fpath}")

        self.load_model(progress_callback=progress_callback)

        fname = os.path.basename(fpath)

        # P2: 视频先提取音频
        audio_path = self._extract_audio(fpath, progress_callback)
        actual_path = audio_path if audio_path else fpath

        start_time = time.time()

        try:
            segments_iter, info = self.model.transcribe(
                actual_path,
                language=language,
                initial_prompt=initial_prompt,
                beam_size=beam_size,
                best_of=best_of,
                patience=patience,
                vad_filter=vad_filter,
                vad_parameters={
                    "threshold": 0.4,                   # P0: 0.5→0.4
                    "min_silence_duration_ms": 350,     # P0: 500→350
                    "min_speech_duration_ms": 250,
                },
                word_timestamps=word_timestamps,
                condition_on_previous_text=condition_on_previous_text,
                hotwords=hotwords,
                clip_timestamps=clip_timestamps,
                repetition_penalty=1.2,                 # P0: 1.1→1.2
                no_speech_threshold=0.5,                # P0: 0.6→0.5
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,        # P0: 新增防幻觉
                temperature=temperature,
            )

            if progress_callback:
                duration = info.duration
                progress_callback(
                    f"音频信息：时长 {duration:.1f} 秒；语言 {info.language}；"
                    f"语言置信度 {info.language_probability:.2f}"
                )

            # P1: 收集词级时间戳 + 低置信度过滤 + 幻觉检测
            raw_segments = []
            hallucination_count = 0
            seen_text_counts = {}
            for i, seg in enumerate(segments_iter):
                # P2: 取消检查
                if cancel_check and cancel_check():
                    if progress_callback:
                        progress_callback("⚠️ 识别已取消")
                    break

                text = seg.text.strip()
                if not text or len(text) < 1:
                    continue

                normalized_text = re.sub(r'[。、？！…\s]', '', text)
                seen_text_counts[normalized_text] = seen_text_counts.get(normalized_text, 0) + 1
                # 常见句子只有同时具备声学风险时才按幻觉过滤，避免误删真实对白。
                is_hallucination = self._is_hallucination(
                    text,
                    avg_logprob=getattr(seg, "avg_logprob", None),
                    no_speech_prob=getattr(seg, "no_speech_prob", None),
                    compression_ratio=getattr(seg, "compression_ratio", None),
                    repetition_count=seen_text_counts[normalized_text],
                )
                if is_hallucination:
                    hallucination_count += 1
                    continue

                # P1: 收集词级时间戳
                words_data = None
                if word_timestamps and hasattr(seg, 'words') and seg.words:
                    words_data = []
                    for w in seg.words:
                        words_data.append({
                            "start": round(w.start, 3),
                            "end": round(w.end, 3),
                            "word": w.word.strip(),
                        })

                segment_data = {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": text,
                    "trans": "",
                    "words": words_data,
                    "avg_logprob": getattr(seg, "avg_logprob", None),
                    "no_speech_prob": getattr(seg, "no_speech_prob", None),
                    "compression_ratio": getattr(seg, "compression_ratio", None),
                    "decoding_temperature": getattr(seg, "temperature", None),
                    "clip_group": self._find_clip_group(
                        seg.start,
                        seg.end,
                        clip_timestamps,
                    ),
                }
                segment_data.update(self._evaluate_segment_quality(segment_data))
                raw_segments.append(segment_data)

                # P2: 实时进度回调（按音频时长计算）
                if progress_callback and info.duration:
                    pct = min(99, int(seg.end / info.duration * 100))
                    progress_callback(f"识别进度 {pct:>3}%｜{text[:30]}…", progress_pct=pct)

            # P1: 后处理（使用词级时间戳精准断句）
            processed_segments = self._post_process_segments(raw_segments)

            duration = int(time.time() - start_time)
            low_conf_count = sum(1 for s in processed_segments if s.get('low_confidence'))
            if progress_callback and (hallucination_count or low_conf_count):
                notes = []
                if hallucination_count:
                    notes.append(f"过滤幻觉 {hallucination_count} 个")
                if low_conf_count:
                    notes.append(f"低置信度 {low_conf_count} 行")
                progress_callback(f"识别后处理：{'；'.join(notes)}")

            return processed_segments

        except RuntimeError as e:
            err = str(e).lower()
            if self.device == "cuda" and (".dll" in err or "library" in err or "not found" in err or "cannot be loaded" in err):
                if progress_callback:
                    progress_callback(f"⚠️ GPU 推理失败，切换到 CPU 模式重试...")
                self.model = None
                self.model_loaded = False
                self.device = "cpu"
                self.compute_type = "int8"
                return self.transcribe(fpath, language=language, initial_prompt=initial_prompt,
                    beam_size=beam_size, best_of=best_of, patience=patience,
                    temperature=temperature,
                    condition_on_previous_text=condition_on_previous_text,
                    hotwords=hotwords, vad_filter=vad_filter,
                    clip_timestamps=clip_timestamps,
                    word_timestamps=word_timestamps,
                    progress_callback=progress_callback, cancel_check=cancel_check)
            raise Exception(f"识别失败: {str(e)}")
        except Exception as e:
            raise Exception(f"识别失败: {str(e)}")
        finally:
            # P2: 清理临时音频文件
            if audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass

    def _is_hallucination(
        self,
        text,
        avg_logprob=None,
        no_speech_prob=None,
        compression_ratio=None,
        repetition_count=1,
    ):
        """
        P0: 检测 Whisper 幻觉文本。
        判断条件：文本完全匹配或仅由幻觉模式组成（去掉标点后）。
        """
        text_clean = text.strip()
        # 去掉标点和空格后检查
        text_no_punct = re.sub(r'[。、？！…\s]', '', text_clean)
        if not text_no_punct:
            return True

        matched = any(
            text_no_punct == re.sub(r'[。、？！…\s]', '', pattern)
            for pattern in self.HALLUCINATION_PATTERNS
        )
        if not matched:
            return False

        risks = sum((
            no_speech_prob is not None and no_speech_prob >= 0.55,
            avg_logprob is not None and avg_logprob <= -0.9,
            compression_ratio is not None and compression_ratio >= 2.2,
        ))
        return risks >= 2 or (repetition_count >= 3 and risks >= 1)

    def _post_process_segments(self, segments, max_len=35):
        """
        后处理：
        1. 清理异常片段（end <= start，空文本）
        2. 合并过短片段（<5字且无标点）
        3. 对超长片段按标点切分（P1: 优先使用词级时间戳）
        4. 确保时间戳不重叠、不倒退
        """
        if not segments:
            return []

        # 第一步：清理异常片段
        cleaned = []
        for seg in segments:
            if seg["end"] <= seg["start"]:
                seg["end"] = seg["start"] + 0.3
            if seg["text"].strip():
                cleaned.append(seg)

        if not cleaned:
            return []

        # 第二步：合并过短片段
        merged = []
        buffer_seg = None
        for seg in cleaned:
            text = seg["text"].strip()
            is_complete = bool(re.search(r'[。？！…?！]$', text))
            is_short = len(text) < 5

            if buffer_seg is not None:
                # 多个离散复核区间之间不能跨越长时间空白合并
                crosses_clip = (
                    buffer_seg.get("clip_group") is not None
                    and seg.get("clip_group") is not None
                    and buffer_seg.get("clip_group") != seg.get("clip_group")
                )
                if crosses_clip or seg["start"] - buffer_seg["end"] > 1.0:
                    merged.append(buffer_seg)
                    buffer_seg = None
                else:
                    buffer_seg["end"] = seg["end"]
                    buffer_seg["text"] += text
                    self._merge_quality_metrics(buffer_seg, seg)
                    buffer_seg.update(self._evaluate_segment_quality(buffer_seg))
                    # P1: 合并词级数据
                    if buffer_seg.get("words") and seg.get("words"):
                        buffer_seg["words"] = buffer_seg["words"] + seg["words"]
                    # 低置信度传递
                    if seg.get("low_confidence"):
                        buffer_seg["low_confidence"] = True
                    if is_complete or len(buffer_seg["text"]) >= 12:
                        merged.append(buffer_seg)
                        buffer_seg = None
                    continue

            if is_short and not is_complete:
                buffer_seg = dict(seg)
            else:
                merged.append(seg)

        if buffer_seg is not None:
            same_clip = (
                buffer_seg.get("clip_group") is None
                or merged[-1].get("clip_group") is None
                or buffer_seg.get("clip_group") == merged[-1].get("clip_group")
            )
            if merged and same_clip and buffer_seg["start"] - merged[-1]["end"] <= 1.0:
                merged[-1]["end"] = buffer_seg["end"]
                merged[-1]["text"] += buffer_seg["text"]
                self._merge_quality_metrics(merged[-1], buffer_seg)
                merged[-1].update(self._evaluate_segment_quality(merged[-1]))
                if merged[-1].get("words") and buffer_seg.get("words"):
                    merged[-1]["words"] = merged[-1]["words"] + buffer_seg["words"]
                if buffer_seg.get("low_confidence"):
                    merged[-1]["low_confidence"] = True
            else:
                merged.append(buffer_seg)

        # 第三步：对超长片段按标点切分
        final = []
        for seg in merged:
            text = seg["text"]
            if len(text) <= max_len:
                final.append(seg)
                continue

            # 按句号、问号、感叹号切分
            parts = re.split(r'([。？！…?！])', text)
            sub_sentences = []
            current = ""
            for p in parts:
                if not p:
                    continue
                if re.match(r'^[。？！…?！]$', p):
                    current += p
                    if current.strip():
                        sub_sentences.append(current.strip())
                    current = ""
                else:
                    current += p
            if current.strip():
                sub_sentences.append(current.strip())

            if not sub_sentences or len(sub_sentences) == 1:
                final.append(seg)
                continue

            # P1: 优先使用词级时间戳切分，失败回退线性插值
            split_result = self._split_with_word_timestamps(seg, sub_sentences)
            final.extend(split_result)

        # 第四步：确保时间戳不重叠
        for i in range(1, len(final)):
            prev = final[i - 1]
            curr = final[i]
            if curr["start"] < prev["end"]:
                curr["start"] = prev["end"]
            if curr["end"] <= curr["start"]:
                curr["end"] = curr["start"] + 0.3

        return final

    def _split_with_word_timestamps(self, seg, sub_sentences):
        """
        P1: 使用词级时间戳精准切分超长片段。
        将子句文本与词序列匹配，使用真实词时间戳代替线性插值。
        匹配失败时回退到线性插值。
        """
        words = seg.get("words")
        if not words or len(sub_sentences) <= 1:
            return self._split_linear(seg, sub_sentences)

        # 构建词文本拼接和字符映射
        word_texts = [w["word"] for w in words]
        full_text = "".join(word_texts)
        full_clean = full_text.replace(" ", "").strip()

        result = []
        search_pos = 0

        for sub in sub_sentences:
            if not sub:
                continue

            sub_clean = sub.replace(" ", "").strip()
            if not sub_clean:
                continue

            # 在全文本中查找子句位置
            found_pos = full_clean.find(sub_clean, search_pos)

            if found_pos < 0:
                # 匹配失败，整体回退到线性插值
                return self._split_linear(seg, sub_sentences)

            sub_end_pos = found_pos + len(sub_clean)

            # 将 clean 位置映射回原始 full_text 位置
            # （因为去除了空格，需要找到对应的原始位置）
            orig_start = self._map_clean_to_orig(full_text, found_pos)
            orig_end = self._map_clean_to_orig(full_text, sub_end_pos - 1) + 1

            # 找到覆盖这个字符范围的词
            char_pos = 0
            seg_start_time = seg["start"]
            seg_end_time = seg["end"]
            found_start = False
            found_end = False

            for wi, wtext in enumerate(word_texts):
                word_start_char = char_pos
                word_end_char = char_pos + len(wtext)

                if not found_start and word_end_char > orig_start:
                    seg_start_time = words[wi]["start"]
                    found_start = True

                if not found_end and word_end_char >= orig_end:
                    seg_end_time = words[wi]["end"]
                    found_end = True
                    break

                char_pos = word_end_char

            if not found_start:
                seg_start_time = seg["start"]
            if not found_end:
                seg_end_time = seg["end"]

            result.append({
                "start": round(seg_start_time, 3),
                "end": round(seg_end_time, 3),
                "text": sub,
                "trans": "",
                "words": None,
                "low_confidence": seg.get("low_confidence", False),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
                "decoding_temperature": seg.get("decoding_temperature"),
                "quality_score": seg.get("quality_score", 100),
                "needs_review": seg.get("needs_review", False),
                "review_reasons": list(seg.get("review_reasons", [])),
                "clip_group": seg.get("clip_group"),
            })

            search_pos = sub_end_pos

        # 如果结果数量不匹配，回退
        if len(result) != len([s for s in sub_sentences if s]):
            return self._split_linear(seg, sub_sentences)

        return result

    def _map_clean_to_orig(self, full_text, clean_pos):
        """将去空格后的字符位置映射回原始文本位置"""
        clean_idx = 0
        for orig_idx, ch in enumerate(full_text):
            if ch == " ":
                continue
            if clean_idx == clean_pos:
                return orig_idx
            clean_idx += 1
        return len(full_text) - 1

    def _split_linear(self, seg, sub_sentences):
        """线性插值切分（回退方案）"""
        total_dur = seg["end"] - seg["start"]
        total_chars = len(seg["text"])
        elapsed = 0.0
        valid_subs = [s for s in sub_sentences if s]

        if not valid_subs:
            return [seg]

        result = []
        for sub in valid_subs:
            char_count = len(sub)
            seg_dur = total_dur * (char_count / total_chars) if total_chars > 0 else total_dur / len(valid_subs)
            seg_start = seg["start"] + elapsed
            seg_end = seg_start + seg_dur

            if seg_end - seg_start < 0.3:
                seg_end = seg_start + 0.3

            result.append({
                "start": round(seg_start, 3),
                "end": round(min(seg_end, seg["end"]), 3),
                "text": sub,
                "trans": "",
                "words": None,
                "low_confidence": seg.get("low_confidence", False),
                "avg_logprob": seg.get("avg_logprob"),
                "no_speech_prob": seg.get("no_speech_prob"),
                "compression_ratio": seg.get("compression_ratio"),
                "decoding_temperature": seg.get("decoding_temperature"),
                "quality_score": seg.get("quality_score", 100),
                "needs_review": seg.get("needs_review", False),
                "review_reasons": list(seg.get("review_reasons", [])),
                "clip_group": seg.get("clip_group"),
            })
            elapsed += seg_dur

        return result
