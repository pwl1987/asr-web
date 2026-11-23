"""处理管线: 音频解码 → ASR 引擎适配层 → LLM 分阶段(A 校对/B 理解/C 成文)。

ASR 引擎适配: transcribe_wav() 是唯一入口, 返回 [{idx,start_ms,end_ms,speaker_id,raw_text}]。
未来换 MOSS-Transcribe 等引擎只需替换此函数内部实现, 上层不感知。

LLM: llm_chat() 统一处理 OpenAI 兼容接口(全局限流 + 网络重试); 显式区分
HTTP 失败/空内容/截断。校对按批并行、逐批回填、支持取消与补缺; 理解按字
数分块并合并, 单次输出截断时自动二分重试; 成文截断自动续写。
所有结构化输出经 validate 后才入库。
"""
import hashlib
import json
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.error import HTTPError, URLError

import config

# ---------------------------------------------------------------- ASR 引擎适配层

_MODEL = None
_MODEL_ERROR: str | None = None


def load_asr_model() -> str | None:
    """后台加载 FunASR; 返回错误信息或 None。幂等。"""
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None or _MODEL_ERROR is not None:
        return _MODEL_ERROR
    try:
        from funasr import AutoModel
        _MODEL = AutoModel(
            model=str(config.MODEL_DIR / "paraformer-zh"),
            vad_model=str(config.MODEL_DIR / "fsmn-vad"),
            punc_model=str(config.MODEL_DIR / "ct-punc-zh"),
            spk_model=str(config.MODEL_DIR / "cam++"),
            disable_update=True, disable_pbar=True,
        )
        _MODEL_ERROR = None
    except Exception as e:  # noqa: BLE001
        _MODEL = None
        _MODEL_ERROR = f"{type(e).__name__}: {e}"
    return _MODEL_ERROR


def asr_ready() -> bool:
    return _MODEL is not None


def asr_error() -> str | None:
    return _MODEL_ERROR


def transcribe_wav(wav_path: str, hotword: str = "") -> list[dict]:
    """运行 ASR, 返回规范化片段列表。引擎无关的统一出口。"""
    if _MODEL is None:
        raise RuntimeError(asr_error() or "ASR 模型未加载")
    res = _MODEL.generate(input=wav_path, batch_size_s=300, hotword=hotword or "")
    r = res[0]
    segments = []
    for i, s in enumerate(r.get("sentence_info") or []):
        segments.append({
            "idx": i,
            "start_ms": int(s.get("start") or 0),
            "end_ms": int(s.get("end") or 0),
            "speaker_id": s.get("spk") if s.get("spk") is not None else None,
            "raw_text": (s.get("text") or "").strip(),
        })
    if not segments:  # 兜底: 无 sentence_info 时整段一条, 时间戳未知
        text = (r.get("text") or "").strip()
        if text:
            segments = [{"idx": 0, "start_ms": None, "end_ms": None,
                         "speaker_id": None, "raw_text": text}]
    return segments


# ---------------------------------------------------------------- ffmpeg 解码

def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _wav_duration(path: str) -> float:
    """用 wave 模块解析(兼容 ffmpeg 输出带 LIST/INFO extra chunk 的情况)。"""
    import wave
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate() or 16000)


def decode_audio(input_path: str, out_wav: str) -> float:
    """解码为 16k mono s16 wav。返回实际时长(按解码后帧数计, 不依赖输入封装头)。
    超过 ASR_MAX_DURATION_MIN 抛 ValueError。"""
    max_sec = config.MAX_DURATION_MIN * 60 + 1
    p = subprocess.run(
        [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-threads", "2",
         "-i", input_path, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
         "-t", str(max_sec), "-f", "wav", out_wav],
        capture_output=True, timeout=config.FFMPEG_TIMEOUT)
    if p.returncode != 0:
        raise ValueError("音频解码失败(文件损坏或格式不支持): "
                         + p.stderr.decode(errors="replace")[-300:])
    duration = _wav_duration(out_wav)
    if duration < 0.05:
        raise ValueError("解码后无有效音频数据")
    if duration > config.MAX_DURATION_MIN * 60:
        raise ValueError(f"音频时长超过 {config.MAX_DURATION_MIN} 分钟上限")
    return round(duration, 1)


# ---------------------------------------------------------------- LLM 客户端

class LLMError(Exception):
    pass


class LLMTruncated(LLMError):
    pass


class LLMEmpty(LLMError):
    pass


class Cancelled(Exception):
    """AI 任务被用户取消(协作式, 只在安全点抛出)。"""


_llm_gate = threading.BoundedSemaphore(max(1, config.LLM_CONCURRENCY))


def llm_chat(messages: list[dict], max_tokens: int) -> tuple[str, str]:
    """返回 (content, finish_reason)。网络错误重试一次; 空内容/结构异常抛 LLMError 系。"""
    body = {
        "model": config.LLM_MODEL, "messages": messages,
        "temperature": 0.1, "max_tokens": max_tokens, "stream": False,
    }
    if config.LLM_DISABLE_THINKING:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{config.LLM_BASE_URL}/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode(), method="POST",
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}",
                 "Content-Type": "application/json"})
    data = None
    for attempt in range(2):
        try:
            with _llm_gate:
                with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
                    data = json.loads(resp.read())
            break
        except HTTPError as e:
            if 500 <= e.code < 600 and attempt == 0:
                time.sleep(1.0)
                continue
            detail = e.read().decode(errors="replace")[:300] if e.fp else ""
            raise LLMError(f"LLM HTTP {e.code}: {detail or e.reason}") from e
        except (URLError, TimeoutError, ConnectionError) as e:
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise LLMError(f"LLM 调用失败: {type(e).__name__}: {e}") from e
        except Exception as e:  # noqa: BLE001
            raise LLMError(f"LLM 调用失败: {type(e).__name__}: {e}") from e
    try:
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        finish = choice.get("finish_reason") or ""
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError("LLM 响应结构异常(缺少 choices/message)") from e
    if not content.strip():
        raise LLMEmpty("LLM 返回空 content(thinking 未关闭或输出为空)")
    return content, finish


def _strip_fences(s: str) -> str:
    return re.sub(r"^```(json)?\s*|\s*```$", "", s.strip(), flags=re.I)


def extract_json(s: str) -> dict:
    s = _strip_fences(s)
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b <= a:
        raise LLMError("LLM 输出中未找到 JSON 对象")
    try:
        return json.loads(s[a:b + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"LLM 输出 JSON 非法: {e.msg} (位置 {e.pos})") from e


def _seg_line(s: dict) -> str:
    spk = f"说话人{(s.get('speaker_id') or 0) + 1}" if s.get("speaker_id") is not None else "未知说话人"
    ts = ""
    if s.get("start_ms") is not None:
        ts = f"[{s['start_ms']//60000:02d}:{s['start_ms']//1000%60:02d}-" \
             f"{s['end_ms']//60000:02d}:{s['end_ms']//1000%60:02d}]"
    return f"[id={s['idx']}][{spk}]{ts}{s.get('raw_text') or s.get('corrected_text') or ''}"


# ---------------------------------------------------------------- 阶段 A: 忠实校对

_SYS_PROOFREAD = (
    "你是严谨的中文会议转写校对员。对给定的逐条转写片段做忠实校对：只修正 ASR 错误"
    "（同音字/错别字/专名写法/标点断句/无中生有的幻觉套话），可以参考上文语境和热词，"
    "但绝不改写原意、不增删信息、不合并或拆分片段、不改变说话人立场，保留口语用词。\n"
    '输出严格 JSON：{"items":[{"id": <片段id>, "text": "<校对后全文>"}]}，items 必须覆盖输入的每一个 id。'
)


def apply_glossary(text: str, glossary: dict[str, str] | None) -> str:
    """术语表确定性纠正(优先于 LLM 校对): 长词优先, 避免互相覆盖。"""
    if not text or not glossary:
        return text
    for wrong in sorted(glossary, key=len, reverse=True):
        right = glossary[wrong]
        if wrong and wrong != right and wrong in text:
            text = text.replace(wrong, right)
    return text


def clean_citations(text: str, valid_ids: set[int]) -> str:
    """清掉文档里引用了不存在片段的 [#id], 保留合法引用。"""
    return re.sub(r"\[#(\d+)\]",
                  lambda m: m.group(0) if int(m.group(1)) in valid_ids else "", text)


def _make_batches(segments: list[dict], max_chars: int) -> list[list[dict]]:
    batches: list[list[dict]] = []
    cur: list[dict] = []
    n_chars = 0
    for s in segments:
        cur.append(s)
        n_chars += len(s.get("raw_text") or "")
        if n_chars >= max_chars:
            batches.append(cur)
            cur, n_chars = [], 0
    if cur:
        batches.append(cur)
    return batches


def _proofread_batch(batch: list[dict], prev_line: str, hotword: str,
                     glossary: dict[str, str] | None = None) -> tuple[dict, list[int]]:
    """单批校对: 缺段重试; 网络/JSON 异常重试; 返回 ({idx:text}, 仍未覆盖 ids)。"""
    ids = [s["idx"] for s in batch]
    ctx = f"（上文仅供理解：{prev_line}）\n" if prev_line else ""
    hw = f"\n热词(专名写法以此为准): {hotword}" if hotword else ""
    base_user = (f"{ctx}以下 {len(batch)} 条转写片段来自同一段录音，请逐条校对：\n\n"
                 + "\n".join(_seg_line(s) for s in batch) + hw)
    prompt = base_user
    got: dict[int, str] = {}
    missing: list[int] = ids
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            content, _ = llm_chat(
                [{"role": "system", "content": _SYS_PROOFREAD},
                 {"role": "user", "content": prompt}], max_tokens=8000)
            data = extract_json(content)
            items = data.get("items")
            if not isinstance(items, list):
                raise LLMError("校对输出缺少 items 数组")
            for it in items:
                if isinstance(it, dict) and it.get("id") in ids and isinstance(it.get("text"), str):
                    got[it["id"]] = apply_glossary(it["text"].strip(), glossary)
        except LLMError as e:
            last_err = e
            if not got:
                continue
        missing = [i for i in ids if i not in got]
        if not missing:
            return got, []
        prompt = (f"上一次输出漏掉了 id={missing}，请只补齐这些 id 的校对结果。"
                  f"仍输出严格 JSON {{\"items\":[...]}}。\n\n" + base_user)
    if last_err is not None and not got:
        raise last_err
    return got, missing


def proofread_segments(segments: list[dict], hotword: str = "", *,
                       progress=None, cancel=None, only_missing: bool = False,
                       force_human: bool = False, on_result=None,
                       glossary: dict[str, str] | None = None) -> tuple[dict, dict]:
    """分批忠实校对(批间并行)。返回 ({idx: corrected_text}, stats)。

    - 默认跳过 corrected_by='human' 的片段, 绝不覆盖人工修改(force_human=True 才覆盖)
    - only_missing=True 只处理尚未校对的片段(增量补缺)
    - on_result({idx:text}) 在每批完成时回调, 供上层立即落库(部分失败/取消不丢)
    """
    pool: list[dict] = []
    skipped_human = 0
    skipped_done = 0
    for s in segments:
        if s.get("corrected_by") == "human" and not force_human:
            skipped_human += 1
            continue
        if only_missing and s.get("corrected_text") is not None:
            skipped_done += 1
            continue
        pool.append(s)
    batches = _make_batches(pool, config.STAGE_A_BATCH_CHARS)
    total = len(batches)
    out: dict[int, str] = {}
    missing: list[int] = []
    failed_batches = 0
    partial_batches = 0
    errors: list[str] = []
    cancelled = False
    done = 0
    if not batches:
        return out, {"batches": 0, "failed_batches": 0, "partial_batches": 0,
                     "missing_ids": [], "skipped_human": skipped_human,
                     "skipped_done": skipped_done, "cancelled": False, "errors": []}

    workers = max(1, min(int(config.PROOFREAD_WORKERS or 1), total))
    ex = ThreadPoolExecutor(max_workers=workers)
    future_map = {}
    for bi, batch in enumerate(batches):
        prev_line = _seg_line(batches[bi - 1][-1]) if bi > 0 else ""
        future_map[ex.submit(_proofread_batch, batch, prev_line, hotword, glossary)] = batch
    try:
        for fut in as_completed(future_map):
            done += 1
            try:
                got, miss = fut.result()
            except Exception as e:  # noqa: BLE001 单批失败不拖垮整轮
                failed_batches += 1
                errors.append(f"{type(e).__name__}: {e}"[:200])
                got, miss = {}, [s["idx"] for s in future_map[fut]]
            if got:
                out.update(got)
                if on_result:
                    on_result(got)
            if miss:
                missing.extend(miss)
                if got:
                    partial_batches += 1
            if progress:
                progress(done, total)
            if cancel and cancel():  # 先把已完成批次落库, 再停(部分结果不丢)
                cancelled = True
                break
    finally:
        for f in future_map:
            f.cancel()
        # 取消时不等待在途批次, 只丢弃其结果
        ex.shutdown(wait=not cancelled, cancel_futures=True)

    stats = {
        "batches": total, "failed_batches": failed_batches,
        "partial_batches": partial_batches, "missing_ids": sorted(set(missing)),
        "skipped_human": skipped_human, "skipped_done": skipped_done,
        "cancelled": cancelled, "errors": errors,
    }
    return out, stats


# ---------------------------------------------------------------- 阶段 B: 会议理解

_SYS_UNDERSTAND = (
    "你是会议分析员。基于带说话人和时间戳的校对转写稿，提取会议理解结构。严格区分："
    "提出建议≠会议通过；个人判断≠已证实事实；讨论方向≠最终决定。"
    "决定/任务只写会议里明确达成的内容；拿不准的一律放进 unconfirmed。"
    "所有条目的 segment_ids 必须引用输入中真实存在的 [id=N]，用于回听溯源。\n"
    '输出严格 JSON：{"topic": "会议主题", "discussion": [{"title","summary","segment_ids":[]}],'
    ' "viewpoints": [{"speaker","point","segment_ids":[]}],'
    ' "disagreements": [{"issue","side_a","side_b","resolved":true/false,"segment_ids":[]}],'
    ' "decisions": [{"decision","owner","deadline","segment_ids":[]}],'
    ' "suggestions": [{"content","proposer","segment_ids":[]}],'
    ' "tasks": [{"content","owner","deadline","segment_ids":[]}],'
    ' "unconfirmed": [{"content","segment_ids":[]}]}, 数组允许为空。'
)

_LIST_KEYS = ["discussion", "viewpoints", "disagreements", "decisions",
              "suggestions", "tasks", "unconfirmed"]

_MERGE_FIELDS = {
    "discussion": ("title", "summary"),
    "viewpoints": ("speaker", "point"),
    "disagreements": ("issue", "side_a", "side_b"),
    "decisions": ("decision", "owner"),
    "suggestions": ("content", "proposer"),
    "tasks": ("content", "owner"),
    "unconfirmed": ("content",),
}


def _clean_understanding(data: dict, valid_ids: set[int]) -> dict:
    # topic 允许为空, 由合并后的最终结果兜底(否则首块空主题会盖掉后面块的真实主题)
    out = {"topic": str(data.get("topic") or "").strip()}
    for k in _LIST_KEYS:
        items = data.get(k)
        if not isinstance(items, list):
            items = []
        cleaned = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item = dict(item)
            orig = item.get("segment_ids") or []
            sids = [i for i in orig if i in valid_ids]
            item["segment_ids"] = sids
            item["invalid_refs"] = len(orig) - len(sids)
            cleaned.append(item)
        out[k] = cleaned
    return out


def _merge_understanding(a: dict | None, b: dict) -> dict:
    if a is None:
        return b
    out = dict(a)
    out["topic"] = out.get("topic") or b.get("topic") or ""
    for k in _LIST_KEYS:
        items = list(out.get(k) or [])
        index = {_item_key(k, it): it for it in items}
        for it in b.get(k) or []:
            key = _item_key(k, it)
            if key in index:
                old = index[key]
                sids = list(dict.fromkeys((old.get("segment_ids") or [])
                                          + (it.get("segment_ids") or [])))
                old["segment_ids"] = sids
                old["invalid_refs"] = (old.get("invalid_refs") or 0) + (it.get("invalid_refs") or 0)
            else:
                items.append(it)
                index[key] = it
        out[k] = items
    return out


def _item_key(kind: str, item: dict) -> tuple:
    fields = _MERGE_FIELDS.get(kind, ("content",))
    return tuple(re.sub(r"\s+", "", str(item.get(f) or "")) for f in fields)


def _understand_lines(lines: list[str], valid_ids: set[int], depth: int = 0) -> dict:
    """一次(或二分后多次)调用 LLM 提取结构; 输出截断时自动二分重试。"""
    content, finish = llm_chat(
        [{"role": "system", "content": _SYS_UNDERSTAND},
         {"role": "user", "content": "会议转写稿如下：\n\n" + "\n".join(lines)}],
        max_tokens=min(config.MAX_OUTPUT_TOKENS_CAP, config.UNDERSTAND_MAX_TOKENS))
    if finish == "length":
        if len(lines) > 1 and depth < 2:
            mid = len(lines) // 2
            return _merge_understanding(
                _understand_lines(lines[:mid], valid_ids, depth + 1),
                _understand_lines(lines[mid:], valid_ids, depth + 1))
        content, finish = llm_chat(
            [{"role": "system", "content": _SYS_UNDERSTAND},
             {"role": "user", "content": "会议转写稿如下：\n\n" + "\n".join(lines)
              + "\n\n输出务必精简：每类不超过 8 条，segment_ids 每条不超过 5 个。"}],
            max_tokens=min(config.MAX_OUTPUT_TOKENS_CAP, config.UNDERSTAND_MAX_TOKENS))
        if finish == "length":
            raise LLMTruncated("会议理解输出被截断(单块内容过长)")
    data = extract_json(content)
    if not isinstance(data, dict):
        raise LLMError("会议理解输出不是 JSON 对象")
    return _clean_understanding(data, valid_ids)


def _chunk_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    if not lines:
        return []
    chunks: list[list[str]] = []
    cur: list[str] = []
    n = 0
    for ln in lines:
        if cur and n + len(ln) > max_chars:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(ln)
        n += len(ln)
    if cur:
        chunks.append(cur)
    return chunks


def understand_meeting(segments: list[dict], speaker_names: dict[int, str], *,
                       progress=None, cancel=None) -> dict:
    """输入已校对片段(有 corrected 用 corrected, 否则 raw)。长稿分块提取后合并。"""
    def name_of(sid) -> str:
        return speaker_names.get(sid, f"说话人{sid + 1}") if sid is not None else "未知说话人"

    lines: list[str] = []
    valid_ids: set[int] = set()
    for s in segments:
        text = (s.get("corrected_text") or s.get("raw_text") or "").strip()
        if not text:
            continue
        spk = name_of(s.get("speaker_id"))
        ts = f"{(s.get('start_ms') or 0)//1000}s" if s.get("start_ms") is not None else "?"
        lines.append(f"[id={s['idx']}][{spk}@{ts}] {text}")
        valid_ids.add(s["idx"])
    if not lines:
        raise LLMError("无有效转写片段")

    chunks = _chunk_lines(lines, config.UNDERSTAND_CHUNK_CHARS)
    merged: dict | None = None
    failed = 0
    errors: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        if cancel and cancel():
            raise Cancelled("已取消")
        try:
            part = _understand_lines(chunk, valid_ids)
            merged = _merge_understanding(merged, part)
        except LLMError as e:
            failed += 1
            errors.append(str(e)[:200])
        if progress:
            progress(i, len(chunks))
        if cancel and cancel():  # 单块也响应取消: 不落半成品
            raise Cancelled("已取消")
    if merged is None:
        raise LLMError("会议理解失败: " + "; ".join(errors[:3]))
    merged["topic"] = (merged.get("topic") or "").strip() or "未识别会议主题"
    merged["_speaker_names"] = speaker_names
    if failed:
        merged["_stats"] = {"chunks": len(chunks), "failed_chunks": failed,
                            "errors": errors}
    return merged


# ---------------------------------------------------------------- 阶段 C: 正式材料

_DOC_KINDS = {
    "transcript": ("完整会议文字稿", "按时间顺序输出全部发言，标注说话人与时间。"),
    "outline": ("会议主题大纲", "以层级大纲呈现会议主题与各议题讨论脉络。"),
    "minutes": ("规范化会议纪要", "按标准会议纪要格式：会议主题/出席/议题/讨论要点/会议决定/待办事项(责任人+时间节点)/待确认问题。"),
    "report": ("正式情况汇报及工作建议", "面向领导的正式汇报：背景/主要情况/会议决定/工作建议/下一步安排。"),
}


def format_transcript(segments: list[dict], speaker_names: dict[int, str]) -> str:
    lines = []
    for s in segments:
        text = (s.get("corrected_text") or s.get("raw_text") or "").strip()
        if not text:
            continue
        sid = s.get("speaker_id")
        if sid is None:
            spk = "未知说话人"
        else:
            spk = speaker_names.get(sid, f"说话人{sid + 1}")
        st = s.get("start_ms") or 0
        ts = f"{st // 60000:02d}:{st // 1000 % 60:02d}"
        lines.append(f"[{ts}] {spk}：{text}")
    return "\n".join(lines)


def transcript_hash(segments: list[dict], speaker_names: dict[int, str]) -> str:
    """材料/理解所依据稿件的指纹(文本+说话人+顺序), 用于发现"基于旧版本"。"""
    h = hashlib.sha256()
    for s in segments:
        text = (s.get("corrected_text") or s.get("raw_text") or "").strip()
        sid = s.get("speaker_id")
        spk = speaker_names.get(sid, f"说话人{sid + 1}") if sid is not None else "未知"
        h.update(f"{s['idx']}|{spk}|{text}\n".encode())
    return h.hexdigest()[:16]


def generate_document(kind: str, understanding: dict, transcript_text: str,
                      options: dict, cancel=None) -> str:
    """生成正式材料(markdown)。截断自动续写(最多 2 次)。"""
    name, requirement = _DOC_KINDS[kind]
    opts = []
    if options.get("title"):
        opts.append(f"标题：《{options['title']}》")
    if options.get("audience"):
        opts.append(f"用途与面向对象：{options['audience']}")
    if options.get("word_count"):
        opts.append(f"字数要求：约 {options['word_count']} 字")
    if options.get("tone"):
        opts.append(f"语气：{options['tone']}")
    if options.get("include_names") is False:
        opts.append("不要出现具体发言人姓名，用角色代称")
    if options.get("include_unconfirmed") is False:
        opts.append("不要附待确认问题清单")
    else:
        opts.append("文末附「待确认问题」清单(来自 unconfirmed)")
    if options.get("cite_segments", True):
        opts.append("在关键结论/决定/待办/数字后用 [#片段id] 标注依据"
                    "(id 必须来自转写稿行首的 [id=N], 不要编造)")
    opt_text = "；".join(opts)

    sys_p = (f"你是公文写作专家。基于会议理解结构与转写稿撰写{requirement}。"
             "严格以材料内容为依据，不得编造会议中不存在的内容；"
             "对 unconfirmed 中的事项明确标注为待确认。输出 markdown 正文。")
    if opt_text:
        sys_p += "\n写作要求：" + opt_text + "。"
    user = ("会议理解结构(JSON)：\n" + json.dumps(understanding, ensure_ascii=False)
            + "\n\n校对后会议转写稿：\n" + transcript_text)
    messages = [{"role": "system", "content": sys_p},
                {"role": "user", "content": user}]
    max_tokens = min(config.MAX_OUTPUT_TOKENS_CAP,
                     max(4096, int(options.get("word_count") or 3000) * 3))
    if cancel and cancel():
        raise Cancelled("已取消")
    content, finish = llm_chat(messages, max_tokens=max_tokens)
    if cancel and cancel():
        raise Cancelled("已取消")
    parts = [content]
    continues = 0
    while finish == "length" and continues < 2:  # 截断续写
        if cancel and cancel():
            raise Cancelled("已取消")
        continues += 1
        messages = messages + [
            {"role": "assistant", "content": parts[-1]},
            {"role": "user", "content": "继续输出，从中断处接续，不要重复已有内容。"}]
        content, finish = llm_chat(messages, max_tokens=max_tokens)
        if cancel and cancel():
            raise Cancelled("已取消")
        parts.append(content)
    doc = "".join(parts)
    if finish == "length":
        doc += "\n\n> ⚠️ 输出因长度截断，可能不完整。"
    return doc


# ---------------------------------------------------------------- legacy /api/verify

_SYS_FAITHFUL = _SYS_PROOFREAD  # 兼容旧语义: 整段忠实校对
_SYS_CLEAN = (
    "你是中文语音转写整理员。在修正 ASR 错误（同音字/错别字/幻觉/标点）的基础上，"
    "去除语气词（嗯、啊、呃、然后、那个等无实义词），清理口语重复，整理为通顺可读的书面语，"
    "但不得改变任何事实内容与说话人立场。\n"
    '输出严格 JSON：{"corrected": "整理后全文", "changes": [{"from":"原","to":"改","reason":"原因"}]}。'
)


def verify_text(text: str, mode: str = "faithful", cleanup_filler: bool = False) -> dict:
    sys_p = _SYS_CLEAN if mode == "clean" else _SYS_FAITHFUL
    if cleanup_filler and mode != "clean":
        sys_p += "\n额外要求：更彻底地清除语气词与填充词。"
    content, _ = llm_chat(
        [{"role": "system", "content": sys_p},
         {"role": "user", "content": "以下是语音转写原文，请按要求处理：\n\n" + text}],
        max_tokens=max(4096, min(config.MAX_OUTPUT_TOKENS_CAP, len(text) * 2)))
    try:
        out = extract_json(content)
        return {"corrected": str(out.get("corrected", "")).strip(),
                "changes": out.get("changes", [])[:80]}
    except LLMError:
        return {"corrected": _strip_fences(content), "changes": []}
