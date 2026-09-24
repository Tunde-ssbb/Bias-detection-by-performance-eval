"""
LLM Batch API Clients

Provides batch processing interfaces for OpenAI and Anthropic APIs.
"""

from __future__ import annotations
import os
import sys
import requests
import json
from json import JSONDecodeError
import threading
import time
import random
import re
import base64
import hashlib
import traceback
from abc import ABC, abstractmethod

from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError, APIConnectionError , APIError
from src.util.util import extract_json_from_str

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal, Union


from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_MODEL = "gpt-4o-mini"
HF_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LOCAL_MODEL = "qwen2.5:7b-instruct"
LOCAL_FAST_MODEL = "llama3.2:3b"

OPENAI_BASE_URL = "https://api.openai.com/v1"
HF_BASE_URL = "https://router.huggingface.co/v1"
OLLAMA_BASE_URL = "http://localhost:11434"

Provider = Literal["openai", "hf", "local", "local_fast"]


import threading
import time
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# RATE LIMITING
# ============================================================================
# Thread-safe global rate limiter for API requests


@dataclass
class RateLimits:
    """Configuration for API rate limits (requests per minute, tokens per minute)."""
    rpm: Optional[float] = 500  # requests per minute
    tpm: Optional[float] = 500000  # tokens per minute


class GlobalRateLimiter:
    """
    Thread-safe global (process-wide) limiter for RPM and TPM.
    Correctly reserves one slot per acquire() call.
    """

    def __init__(self, limits: RateLimits = RateLimits()):
        self._limits = limits
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._next_req_time = 0.0
        self._next_tok_time = 0.0

    def set_limits(self, *, rpm: Optional[float] = None, tpm: Optional[float] = None) -> None:
        with self._cv:
            self._limits = RateLimits(rpm=rpm, tpm=tpm)
            # Wake any waiting threads so they can recompute with new limits
            self._cv.notify_all()

    def acquire(self, *, est_tokens: int = 0) -> None:
        """
        Blocks until a request slot (and token budget slot) is available.
        Reserves exactly one slot per call.
        """
        while True:
            with self._cv:
                now = time.monotonic()

                # Compute the earliest time we may proceed for each dimension
                req_ready = now
                tok_ready = now

                if self._limits.rpm and self._limits.rpm > 0:
                    req_ready = max(now, self._next_req_time)

                if self._limits.tpm and self._limits.tpm > 0 and est_tokens > 0:
                    tok_ready = max(now, self._next_tok_time)

                ready_at = max(req_ready, tok_ready)
                wait_s = ready_at - now

                if wait_s <= 0:
                    # Reserve the next slot(s) exactly once, now that we're proceeding
                    now2 = time.monotonic()

                    if self._limits.rpm and self._limits.rpm > 0:
                        req_interval = 60.0 / self._limits.rpm
                        self._next_req_time = max(now2, self._next_req_time) + req_interval

                    if self._limits.tpm and self._limits.tpm > 0 and est_tokens > 0:
                        tok_interval = 60.0 * (est_tokens / self._limits.tpm)
                        self._next_tok_time = max(now2, self._next_tok_time) + tok_interval

                    return

                # Wait until the time we reserved is reached (or limits change)
                self._cv.wait(timeout=wait_s)

# -------- Base class --------

class LLMBase(ABC):
    """
    Base interface for all LLM backends.
    """

    def __init__(self):
        self.prompt_count = 0

    @abstractmethod
    def _get_response(self, prompt: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def _get_json_response(self, prompt: str) -> str:
        raise NotImplementedError

    def get_response(self, prompt: str) -> str:
        self.prompt_count += 1
        return self._get_response(prompt)

    def get_json_response(self, prompt: str, max_retries = 1) -> str:
        self.prompt_count += 1
        try:
            res = extract_json_from_str(self._get_json_response(prompt))
            print("Using specific json response mode")
            return res
        except NotImplementedError: 
            print("Using normal response mode")
            while retries <= max_retries:
                response = llm.get_response(prompt)
                try:
                    json_concepts = extract_json_from_str(response)
                    return json_concepts
                except JSONDecodeError as e:
                    print(f"JSONDecodeError: {e}. Retrying...")
                    retries += 1
            raise ValueError(f"Failed to extract concepts after {max_retries} retries. Exiting")
                
    def estimate_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)                

    def get_prompt_count(self) -> int:
        return self.prompt_count

    def reset_prompt_count(self) -> None:
        self.prompt_count = 0
        return

# -------- Public factory --------

def initialize_llm(provider: Provider = "local", model = None, temperature: float = 0) -> LLMBase:
    if provider == "local":
        return LocalLLM(model=model if model else LOCAL_MODEL)
        
    if provider == "local_fast":
        return LocalLLM(model=model if model else LOCAL_FAST_MODEL)


    if provider == "openai":
        return OpenAIStyleLLM(
            base_url=OPENAI_BASE_URL,
            api_key=os.environ["OPENAI_API_KEY"],
            model=model if model else OPENAI_MODEL,
            name="OpenAI",
            temperature=temperature
        )

    if provider == "hf":
        return OpenAIStyleLLM(
            base_url=HF_BASE_URL,
            api_key=os.environ["HF_TOKEN"],
            model=model if model else HF_MODEL,
            name="HuggingFace",
        )

    raise ValueError(f"Unknown provider: {provider}")


# -------- Implementations --------
GLOBAL_LIMITER = GlobalRateLimiter()

class OpenAIStyleLLM(LLMBase):
    """
    For OpenAI-compatible chat/completions APIs
    (OpenAI, Hugging Face router, many compatible providers)
    """

    def __init__(self, base_url: str, api_key: str, model: str, name: str, temperature: float = 0):
        self.model = model
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self._tls = threading.local()
        self.prompt_count = 0
        self.temperature = temperature

    def _client(self) -> OpenAI:
        # One client per thread (safe + keeps connection pooling per thread)
        if not hasattr(self._tls, "client"):
            self._tls.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._tls.client

    def _retry_after_seconds(self, err: Exception) -> Optional[float]:
        """
        Best-effort extraction of Retry-After from SDK errors.
        Works when the underlying response headers are exposed.
        """
        resp = getattr(err, "response", None)
        if resp is None:
            return None
        headers = getattr(resp, "headers", None)
        if not headers:
            return None
        ra = headers.get("retry-after") or headers.get("Retry-After")
        if not ra:
            return None
        try:
            return float(ra)
        except ValueError:
            return None

    def _is_retryable(self, err: Exception) -> bool:
        # Retry: rate limits + transient server/network conditions
        if isinstance(err, (RateLimitError, APITimeoutError, APIConnectionError)):
            return True

        if isinstance(err, APIError):
            # APIError often includes status_code
            status = getattr(err, "status_code", None)
            # Retry 5xx; don't retry 4xx (except 429 handled above)
            if status is not None and 500 <= status <= 599:
                return True
            return False

        # For “OpenAI-compatible” providers, you might get generic exceptions.
        # Be conservative: retry common transient text patterns (optional).
        msg = str(err).lower()
        transient = (
            "timeout" in msg
            or "timed out" in msg
            or "temporarily unavailable" in msg
            or "connection reset" in msg
            or "connection aborted" in msg
            or "server error" in msg
        )
        return transient

    def _sleep_backoff(self, attempt: int, retry_after: Optional[float] = None) -> None:
        # Respect server hint first
        if retry_after is not None and retry_after > 0:
            time.sleep(min(20.0, retry_after))
            return

        # Exponential backoff with jitter
        base = 0.5 * (2 ** attempt)          # 0.5, 1, 2, 4, 8...
        cap = 20.0
        sleep_s = min(cap, base) * (0.7 + 0.6 * random.random())  # jitter ~[0.7, 1.3]
        time.sleep(sleep_s)

    def _chat_once(self, *, prompt: str, timeout_s: float, response_format=None) -> str:
        completion = self._client().chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout_s,
            response_format=response_format
            # temperature=self.temperature
        )
        return completion.choices[0].message.content


    def _get_response(self, prompt: str, retries: int = 6) -> str:
        timeout_s = 60.0

        # request rate


        last_err: Exception | None = None
        for attempt in range(retries + 1):
            est = self.estimate_tokens(prompt) + 500 # add output tokens
            print(f"requiesting {est} tokens from rate")
            GLOBAL_LIMITER.acquire(est_tokens=est)
            print("acquired lock")
            try:
                return self._chat_once(prompt=prompt, timeout_s=timeout_s)
            except Exception as e:
                last_err = e
                if attempt == retries or not self._is_retryable(e):
                    raise
                self._sleep_backoff(attempt, retry_after=self._retry_after_seconds(e))

        # unreachable, but keeps type checkers happy
        raise last_err  # type: ignore[misc]

    def _get_json_response(self, prompt: str, retries: int = 6) -> str:
        timeout_s = 60.0
        response_format = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(retries + 1):
            est = self.estimate_tokens(prompt) + 500 # add output tokens
            print(f"requiesting {est} tokens from rate")
            GLOBAL_LIMITER.acquire(est_tokens=est)
            print("acquired lock")
            try:
                return self._chat_once(prompt=prompt, timeout_s=timeout_s, response_format=response_format)
            except Exception as e:
                last_err = e
                if attempt == retries or not self._is_retryable(e):
                    raise RuntimeError(f"{self.name} API error: {e}") from e
                self._sleep_backoff(attempt, retry_after=self._retry_after_seconds(e))

        raise RuntimeError(f"{self.name} API error: {last_err}")  # type: ignore[misc]

class LocalLLM(LLMBase):
    """
    Ollama-backed local LLM
    """

    def __init__(self, model: str, base_url: str = OLLAMA_BASE_URL):
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _get_response(self, prompt: str) -> str:
        r = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]



JsonlLine = str


@dataclass
class BatchResult:
    batch_id: str
    status: str
    output_file_id: Optional[str]
    error_file_id: Optional[str]
    output_lines: List[dict]
    error_lines: List[dict]
    # Raw batch object (as dict) for debugging / logging
    batch_obj: dict


class BatchLLM:
    """
    Clean helper for OpenAI Batch + Responses API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5-mini-2025-08-07",
        endpoint: str = "/v1/responses",
        completion_window: str = "24h",
        poll_interval_s: float = 5.0,
        reasoning_effort: str = "medium",
    ):
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model
        self.endpoint = endpoint  # default: /v1/responses
        self.completion_window = completion_window
        self.poll_interval_s = poll_interval_s
        self.reasoning_effort = reasoning_effort

    # ========================================================================
    # REQUEST BUILDING
    # ========================================================================
    
    def build_request(
        self,
        prompt: str,
        custom_id: str,
        save_path: Optional[Union[str, Path]] = None,
        *,
        # You can pass extra kwargs to the Responses request body (tools, reasoning, etc.)
        extra_body: Optional[dict] = None,
    ) -> JsonlLine:
        """
        Build ONE JSONL request line for Batch.

        For Responses endpoint, the "body" should match what you'd send to POST /v1/responses,
        except it's wrapped in {custom_id, method, url, body}.

        Returns: a JSON string (no trailing newline).
        If save_path is provided, appends the line to that JSONL file.
        """
        body = {
            "model": self.model,
            "input": [
                {"role": "user", "content": prompt},
            ],
            "reasoning": {"effort": self.reasoning_effort},
        }
        if extra_body:
            # Shallow merge is usually enough; override keys explicitly if needed
            body.update(extra_body)

        req = {
            "custom_id": custom_id,
            "method": "POST",
            "url": self.endpoint,
            "body": body,
        }

        line = json.dumps(req, ensure_ascii=False)

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        return line

    def parse_batch_line(self, line: dict):
        """
        Input: one parsed JSONL line from batch output
        Returns: (custom_id, response_text, error)

        response_text -> extracted assistant text if present
        error -> None if success, otherwise readable error string
        """



        custom_id = line.get("custom_id")

        if "text" in line:
            text = line.get("text")
            error = line.get("error")

            if error:
                if isinstance(error, dict):
                    return custom_id, None, error.get("message", str(error))
                return custom_id, None, str(error)

            return custom_id, text, None


        # ---- Transport / request-level failure
        if line.get("error"):
            err = line["error"]
            return custom_id, None, err.get("message", str(err))

        response = line.get("response")
        if not response:
            return custom_id, None, "Missing response object"

        status_code = response.get("status_code")

        # ---- HTTP-level failure
        if status_code != 200:
            body = response.get("body", {})
            err = body.get("error") if isinstance(body, dict) else None
            msg = err.get("message") if err else f"HTTP {status_code}"
            return custom_id, None, msg

        body = response.get("body", {})

        # ---- Model-level failure
        if body.get("error"):
            err = body["error"]
            return custom_id, None, err.get("message", str(err))
        elif body.get("status"):
            status = body.get("status")
            if status == "incomplete":
                # Check for specific incomplete reasons
                incomplete_details = body.get("incomplete_details", {})
                reason = incomplete_details.get("reason", "unknown")
                
                if reason == "max_output_tokens":
                    err = f"INCOMPLETE: Token limit reached (max_output_tokens exceeded)"
                elif reason == "content_filter":
                    err = f"INCOMPLETE: Content filter triggered"
                elif reason == "max_completion_tokens":
                    err = f"INCOMPLETE: Completion token limit exceeded"
                else:
                    err = f"INCOMPLETE: Status incomplete (reason: {reason})"
                return custom_id, None, err

        # ---- Extract assistant text (Responses API structure)
        response_text = None

        try:
            outputs = body.get("output", [])

            for item in outputs:
                if item.get("type") == "message":
                    for content in item.get("content", []):
                        if content.get("type") == "output_text":
                            response_text = content.get("text")
                            break

                if response_text:
                    break

        except Exception as ex:
            return custom_id, None, f"Parsing error: {repr(ex)}"

        if response_text is None:
            return custom_id, None, "No output_text found"

        return custom_id, response_text, None

    # ----------------------------
    # Submit batch
    # ----------------------------
    def submit(
        self,
        input_jsonl: Union[Iterable[JsonlLine], str, Path],
        *,
        metadata: Optional[dict] = None,
        upload_filename: str = "batch_input.jsonl",
    ) -> str:
        """
        Submit a batch job.

        input_jsonl can be:
          - iterable of JSONL lines (strings)
          - a path to an existing .jsonl file (str/Path)

        Returns: batch_id
        """
        input_path: Path

        if isinstance(input_jsonl, (str, Path)):
            input_path = Path(input_jsonl)
            if not input_path.exists():
                raise FileNotFoundError(f"input_jsonl path not found: {input_path}")
        else:
            # Write lines to a temp file
            tmp_dir = Path(".batchllm_tmp")
            tmp_dir.mkdir(exist_ok=True)
            input_path = tmp_dir / upload_filename
            with input_path.open("w", encoding="utf-8") as f:
                for line in input_jsonl:
                    # Safety: ensure one JSON object per line
                    if not isinstance(line, str) or not line.strip():
                        continue
                    f.write(line.rstrip("\n") + "\n")

        # Upload as a file with purpose="batch"
        with input_path.open("rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")

        # Create the batch
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint=self.endpoint,
            completion_window=self.completion_window,
            metadata=metadata or {},
        )

        return batch.id

    # ----------------------------
    # Batch retrieval + waiting
    # ----------------------------
    def get_batch(self, batch_id: str) -> dict:
        """Retrieve batch and return as a plain dict."""
        b = self.client.batches.retrieve(batch_id)
        # SDK objects typically support model_dump()
        return b.model_dump() if hasattr(b, "model_dump") else dict(b)

    def wait_for_completion(
        self,
        batch_id: str,
        *,
        timeout_s: Optional[float] = 3600.0,
    ) -> dict:
        """
        Poll until batch is in a terminal state.
        Terminal states: completed, failed, expired, cancelled
        """
        start = time.time()
        while True:
            batch_obj = self.get_batch(batch_id)
            status = batch_obj.get("status")

            if status in ("completed", "failed", "expired", "cancelled"):
                return batch_obj

            if timeout_s is not None and (time.time() - start) > timeout_s:
                raise TimeoutError(
                    f"Timed out waiting for batch {batch_id}. Last status={status}"
                )

            time.sleep(self.poll_interval_s)

    # ----------------------------
    # Files download helpers
    # ----------------------------
    def _download_file_text(self, file_id: str) -> str:
        content_bytes = self.client.files.content(file_id).read()
        return content_bytes.decode("utf-8", errors="replace")

    def download_file_jsonl(self, file_id: str) -> List[dict]:
        """
        Download a Files API content and parse as JSONL -> List[dict].
        Skips blank lines.
        """
        text = self._download_file_text(file_id)
        lines: List[dict] = []
        for raw in text.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            lines.append(json.loads(raw))
        return lines

    def download_results(
        self,
        batch_id: str,
        *,
        wait: bool = True,
        timeout_s: Optional[float] = 3600.0,
    ) -> BatchResult:
        """
        Returns parsed output_lines and error_lines if present.
        If wait=True, waits until terminal.
        """
        batch_obj = self.wait_for_completion(batch_id, timeout_s=timeout_s) if wait else self.get_batch(batch_id)

        status = batch_obj.get("status")
        output_file_id = batch_obj.get("output_file_id")
        error_file_id = batch_obj.get("error_file_id")

        output_lines: List[dict] = []
        error_lines: List[dict] = []

        if output_file_id:
            output_lines = self.download_file_jsonl(output_file_id)

        if error_file_id:
            error_lines = self.download_file_jsonl(error_file_id)

        return BatchResult(
            batch_id=batch_id,
            status=status,
            output_file_id=output_file_id,
            error_file_id=error_file_id,
            output_lines=output_lines,
            error_lines=error_lines,
            batch_obj=batch_obj,
        )

    # ----------------------------
    # Writing helpers
    # ----------------------------
    def write_jsonl(self, lines: List[dict], path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return p

    def write_results(
        self,
        result: BatchResult,
        *,
        output_path: Optional[Union[str, Path]] = None,
        error_path: Optional[Union[str, Path]] = None,
        log_path: Optional[Union[str, Path]] = None,
    ) -> None:
        if output_path is not None:
            self.write_jsonl(result.output_lines, output_path)
        if error_path is not None:
            self.write_jsonl(result.error_lines, error_path)
        if log_path is not None:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text(self.log_error(result.batch_obj), encoding="utf-8")

    # ----------------------------
    # Error logging helper
    # ----------------------------
    def log_error(self, batch: Union[str, dict]) -> str:
        """
        Return a readable string of batch status + any batch-level errors + file ids.
        Pass either a batch_id or a batch dict (from get_batch / wait_for_completion).
        """
        batch_obj = self.get_batch(batch) if isinstance(batch, str) else batch

        status = batch_obj.get("status")
        batch_id = batch_obj.get("id") or batch_obj.get("batch_id")  # depending on shape
        output_file_id = batch_obj.get("output_file_id")
        error_file_id = batch_obj.get("error_file_id")

        # Batch-level errors are typically under batch_obj["errors"]["data"]
        errors_block = batch_obj.get("errors")
        err_data = None
        if isinstance(errors_block, dict):
            err_data = errors_block.get("data")
        elif errors_block is not None:
            # sometimes SDK may give it already as list
            err_data = errors_block

        lines = []
        lines.append(f"batch_id: {batch_id}")
        lines.append(f"status: {status}")
        lines.append(f"output_file_id: {output_file_id}")
        lines.append(f"error_file_id: {error_file_id}")

        if err_data:
            lines.append("")
            lines.append("batch_errors:")
            for e in err_data:
                # e might be dict-like
                if isinstance(e, dict):
                    code = e.get("code")
                    msg = e.get("message")
                    param = e.get("param")
                    line_no = e.get("line")
                else:
                    code = getattr(e, "code", None)
                    msg = getattr(e, "message", None)
                    param = getattr(e, "param", None)
                    line_no = getattr(e, "line", None)

                lines.append(
                    f"- code={code} line={line_no} param={param} message={msg}"
                )
        else:
            lines.append("")
            lines.append("batch_errors: <none found on batch object>")

        # Useful: include the whole batch json (compact) at the end
        lines.append("")
        lines.append("raw_batch_json:")
        lines.append(json.dumps(batch_obj, ensure_ascii=False, indent=2))

        return "\n".join(lines)
    

    def _asdict(self, obj: Any) -> Dict[str, Any]:
        """Best-effort conversion of SDK objects to plain dict."""
        if obj is None:
            return {}
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):  # pydantic (new SDK)
            return obj.model_dump()
        if hasattr(obj, "dict"):        # older pydantic
            return obj.dict()
        if hasattr(obj, "__dict__"):    # fallback
            return dict(obj.__dict__)
        # last resort: represent it
        return {"_repr": repr(obj)}

    def log_batch(
        self,
        batch_id: str,
        log_path: Union[str, Path],
        *,
        include_request_errors: bool = True,
        max_request_errors: Optional[int] = None,  # None => all
        include_raw_json: bool = True,
    ) -> Path:
        """
        Writes ONE log file with all information available from:
          - the batch object (authoritative via client.batches.retrieve)
          - batch-level errors (batch.errors)
          - request-level errors (downloaded from error_file_id), if present

        No bypassing, no extra API calls beyond retrieve + optional files.content.
        """
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Always retrieve fresh, so you log the latest state
        batch_obj = self.client.batches.retrieve(batch_id)
        batch = self._asdict(batch_obj)

        status = batch.get("status")
        output_file_id = batch.get("output_file_id")
        error_file_id = batch.get("error_file_id")
        request_counts = batch.get("request_counts")
        metadata = batch.get("metadata")

        lines = []
        lines.append(f"batch_id: {batch_id}")
        lines.append(f"status: {status}")
        lines.append(f"output_file_id: {output_file_id}")
        lines.append(f"error_file_id: {error_file_id}")
        lines.append(f"request_counts: {json.dumps(request_counts, ensure_ascii=False)}")
        lines.append(f"metadata: {json.dumps(metadata, ensure_ascii=False)}")
        lines.append("")

        # ---- Batch-level errors (validation failures often show only here)
        errors_block = batch.get("errors")
        batch_errors = None
        if isinstance(errors_block, dict):
            batch_errors = errors_block.get("data")
        elif isinstance(errors_block, list):
            batch_errors = errors_block

        lines.append("BATCH-LEVEL ERRORS:")
        if batch_errors:
            for e in batch_errors:
                e = e if isinstance(e, dict) else self._asdict(e)
                lines.append(
                    f"- code={e.get('code')} line={e.get('line')} param={e.get('param')} message={e.get('message')}"
                )
        else:
            lines.append("<none present>")
        lines.append("")

        # ---- Request-level errors (only if error_file_id exists)
        if include_request_errors:
            lines.append("REQUEST-LEVEL ERRORS (from error_file_id):")
            if error_file_id:
                try:
                    err_lines = self.download_file_jsonl(error_file_id)
                except Exception as ex:
                    err_lines = []
                    lines.append(f"<failed to download/parse error file: {repr(ex)}>")

                if err_lines:
                    shown = 0
                    for item in err_lines:
                        shown += 1
                        if max_request_errors is not None and shown > max_request_errors:
                            break

                        # Common shapes: {error: {...}, request: {...}} etc.
                        err = item.get("error") or {}
                        req = item.get("request") or {}
                        custom_id = req.get("custom_id") or item.get("custom_id")
                        msg = err.get("message") or str(err) or "Unknown error"
                        code = err.get("code")
                        param = err.get("param")

                        # Also include response status if present
                        resp = item.get("response") or {}
                        resp_status = resp.get("status_code") or resp.get("status")

                        lines.append(f"{shown}. custom_id={custom_id}")
                        if resp_status is not None:
                            lines.append(f"   response_status={resp_status}")
                        if code is not None:
                            lines.append(f"   code={code}")
                        if param is not None:
                            lines.append(f"   param={param}")
                        lines.append(f"   message={msg}")
                else:
                    lines.append("<none (error file empty or not present)>")
            else:
                lines.append("<no error_file_id on batch>")
            lines.append("")

        # ---- Full raw batch JSON for “all possible information”
        if include_raw_json:
            lines.append("RAW BATCH JSON (full object):")
            lines.append(json.dumps(batch, indent=2, ensure_ascii=False, default=str))
            lines.append("")

        p.write_text("\n".join(lines), encoding="utf-8")
        return p



@dataclass
class AnthropicBatchResult:
    batch_id: str
    status: str                 # "completed" or "failed"
    output_lines: List[dict]    # list of {"custom_id","text","error":None} on success
    error_lines: List[dict]     # list of {"custom_id","text":None,"error":...} for per-request errors
    batch_obj: dict             # raw batch object for debugging



class AnthropicBatchLLM:
    """
    Anthropic batch class that preserves existing OpenAI BatchLLM workflow:
    """

    def __init__(
        self,
        api_key: str = os.environ["ANTHROPIC_API_KEY"],
        model: str = "claude-3-5-sonnet-latest",
        base_url: str = "https://api.anthropic.com",
        anthropic_version: str = "2023-06-01",
        max_tokens: int = 1024,
        poll_interval_s: float = 5.0,
        temperature: float = 0.0,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.max_tokens = max_tokens
        self.poll_interval_s = poll_interval_s
        self.temperature = temperature

        if not self.api_key or not self.api_key.strip():
            raise ValueError("ANTHROPIC_API_KEY is empty")

        # optional sanity print (don’t log full key)
        print("Anthropic key prefix:", self.api_key[:7])

    # ----------------------------
    # Internal HTTP helpers
    # ----------------------------
    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "content-type": "application/json",
        }



    def encode_custom_id(self, custom_id: str) -> str:
        """
        Convert arbitrary ID → Anthropic-safe ID within 64-char limit.
    
        """
        # Style abbreviation mapping to avoid collisions (concise and casual both start with 'c')
        STYLE_ABBREV_MAP = {
            "concise": "cn",
            "casual": "cs",
            "detailed": "d",
            "formal": "f",
            "realistic": "r"
        }
        
        # Parse the custom_id format: type::batch::occ::style::idx
        parts = custom_id.split("::")  
        if len(parts) == 5:
            type_id, batch_name, occ_code, style, idx = parts
            
            # Abbreviate style using mapping to avoid collisions
            style_abbrev = STYLE_ABBREV_MAP.get(style, style[0] if style else "r")
            
            # Hash batch name if too long (keep first 8 chars of hash)
            if len(batch_name) > 10:
                batch_hash = hashlib.md5(batch_name.encode()).hexdigest()[:8]
            else:
                batch_hash = batch_name
            
            # Create compact format: type:batch:occ:style:idx (single colon separators)
            compact = f"{type_id}:{batch_hash}:{occ_code}:{style_abbrev}:{idx}"
            
            # Store mapping for decoding
            raw = compact.encode("utf-8")
        else:
            # Fallback for non-standard format
            raw = custom_id.encode("utf-8")
        
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


    def decode_custom_id(self, encoded_id: str) -> str:
        """
        Convert Anthropic-safe ID → original ID.
        
        Note: Batch name will be hashed, style abbreviated. 
        This returns the compact format, not the original.
        """
        padding = "=" * (-len(encoded_id) % 4)
        raw = base64.urlsafe_b64decode(encoded_id + padding)
        return raw.decode("utf-8")

    def _get_json(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = Request(url, headers=self._headers(), method="GET")

        try:
            with urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except HTTPError as e:
            print("HTTPError URL:", req.full_url)
            print("HTTPError SENT HEADERS:", dict(req.header_items()))
            print("HTTPError BODY:", e.read().decode("utf-8", errors="replace"))
            print("STACK:\n", "".join(traceback.format_stack(limit=15)))
            raise
        

            


    def _post_json(self, path: str, payload: dict) -> dict:
        if not self.api_key or not str(self.api_key).strip():
            raise ValueError("ANTHROPIC_API_KEY is missing/empty at runtime")

        url = f"{self.base_url}{path}"
        headers = self._headers()

        # TEMP DEBUG
        print("POST", url)
        print("HEADERS", {k: ("<set>" if k == "x-api-key" else v) for k, v in headers.items()})
        # print("PAYLOAD", payload)  # uncomment if needed

        data = json.dumps(payload).encode("utf-8")
        req = Request(url, headers=headers, data=data, method="POST")

        try:
            with urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except HTTPError as e:
            print("HTTPError URL:", req.full_url)
            print("HTTPError SENT HEADERS:", dict(req.header_items()))
            print("HTTPError BODY:", e.read().decode("utf-8", errors="replace"))
            print("STACK:\n", "".join(traceback.format_stack(limit=15)))
            raise
            

    def _get_text(self, path: str) -> str:
        url = f"{self.base_url}{path}"
        headers = self._headers()
        req = Request(url, headers=headers, method="GET")

        
        try:
            with urlopen(req) as resp:
                # If no error, you can still check final URL
                print("FINAL URL:", getattr(resp, "url", None))
                return resp.read().decode("utf-8", errors="replace")
        except HTTPError as e:
            print("HTTPError URL:", req.full_url)
            print("HTTPError SENT HEADERS:", dict(req.header_items()))
            print("HTTPError BODY:", e.read().decode("utf-8", errors="replace"))
            print("STACK:\n", "".join(traceback.format_stack(limit=15)))
            raise


    # ----------------------------
    # Build request JSONL line
    # ----------------------------
    def build_request(self, prompt: str, custom_id: str, save_path=None, *, extra_params=None) -> str:
        safe_id = self.encode_custom_id(custom_id)


        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            "temperature": self.temperature,
        }
        if extra_params:
            params.update(extra_params)

        obj = {"custom_id": safe_id, "params": params}
        line = json.dumps(obj, ensure_ascii=False)

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

        return line

    # ----------------------------
    # Submit batch
    # ----------------------------
    def submit(self, input_jsonl: Union[Iterable[JsonlLine], str, Path]) -> str:
        """
        Creates a message batch by reading JSONL lines and sending:
          POST /v1/messages/batches {"requests":[...]}
        Returns batch_id
        """
        requests_list: List[dict] = []

        if isinstance(input_jsonl, (str, Path)):
            p = Path(input_jsonl)
            if not p.exists():
                raise FileNotFoundError(f"input_jsonl path not found: {p}")
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    requests_list.append(json.loads(line))
        else:
            for line in input_jsonl:
                if not isinstance(line, str) or not line.strip():
                    continue
                requests_list.append(json.loads(line))

        resp = self._post_json("/v1/messages/batches", {"requests": requests_list})
        batch_id = resp.get("id")
        if not batch_id:
            raise RuntimeError(f"Unexpected submit response: {resp}")
        return batch_id

    # ----------------------------
    # Polling
    # ----------------------------
    def get_batch(self, batch_id: str) -> dict:
        return self._get_json(f"/v1/messages/batches/{batch_id}")

    def wait_for_completion(self, batch_id: str, *, timeout_s: Optional[float] = 3600.0) -> dict:
        """
        Wait until processing_status is terminal. Typically "ended".
        """
        start = time.time()
        while True:
            batch = self.get_batch(batch_id)
            status = batch.get("processing_status")

            if status in ("ended", "canceled"):
                return batch

            if timeout_s is not None and (time.time() - start) > timeout_s:
                raise TimeoutError(f"Timed out waiting for batch {batch_id}. Last status={status}")

            time.sleep(self.poll_interval_s)

    # ----------------------------
    # Parse one results line -> (custom_id, text, error)
    # ----------------------------
    def parse_batch_line(self, line: dict) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Anthropic results line shape:
          {"custom_id":"...","result":{"type":"succeeded","message":{...}}}
          {"custom_id":"...","result":{"type":"errored","error":{...}}}

        Returns: (custom_id, response_text, error_string)
        """
        custom_id = line.get("custom_id")
        result = line.get("result") or {}

        rtype = result.get("type")
        if rtype != "succeeded":
            err = result.get("error") or result
            if isinstance(err, dict):
                error_type = err.get("type", "")
                error_msg = err.get("message", "")
                
                # Detect specific error types
                if "max_tokens" in error_msg.lower() or "length" in error_msg.lower():
                    return custom_id, None, f"INCOMPLETE: Token limit - {error_msg}"
                elif error_type == "invalid_request_error" and "max_tokens" in str(err):
                    return custom_id, None, f"INCOMPLETE: Token limit exceeded"
                else:
                    return custom_id, None, error_msg or json.dumps(err, ensure_ascii=False)
            return custom_id, None, str(err)

        msg = result.get("message") or {}
        
        # Check for incomplete messages (Anthropic stop_reason)
        stop_reason = msg.get("stop_reason")
        if stop_reason == "max_tokens":
            # Message succeeded but was cut off
            content = msg.get("content") or []
            texts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                    texts.append(block["text"])
            response_text = "\n".join(texts).strip() if texts else None
            return self.decode_custom_id(custom_id), response_text, "INCOMPLETE: Token limit reached (max_tokens)"
        
        content = msg.get("content") or []

        texts: List[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                texts.append(block["text"])

        response_text = "\n".join(texts).strip() if texts else None
        if not response_text:
            return self.decode_custom_id(custom_id), None, "No text content found"

        return self.decode_custom_id(custom_id), response_text, None

    # ----------------------------
    # Download results (and normalize to your BatchResult shape)
    # ----------------------------
    def download_results(self, batch_id: str, *, wait: bool = True, timeout_s: Optional[float] = 3600.0) -> BatchResult:
        batch_obj = self.wait_for_completion(batch_id, timeout_s=timeout_s) if wait else self.get_batch(batch_id)
        processing_status = batch_obj.get("processing_status")

        # If batch hasn't ended yet and wait=False, we can't fetch complete results reliably
        # but we'll attempt anyway.
        raw = self._get_text(f"/v1/messages/batches/{batch_id}/results")

        output_lines: List[dict] = []
        error_lines: List[dict] = []

        for raw_line in raw.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            line = json.loads(raw_line)

            cid, text, err = self.parse_batch_line(line)
            if err:
                error_lines.append(line)
            else:
                output_lines.append(line)

        # Map Anthropic "ended" + success count to your "completed"/"failed"
        status = "completed" if len(output_lines) > 0 else "failed"

        # If the batch is canceled and there are no successes, keep failed
        if processing_status == "canceled" and len(output_lines) == 0:
            status = "failed"

        return AnthropicBatchResult(
            batch_id=batch_id,
            status=status,
            output_lines=output_lines,
            error_lines=error_lines,
            batch_obj=batch_obj,
        )

    # ----------------------------
    # Write JSONL (same signature as your OpenAI BatchLLM)
    # ----------------------------
    def write_jsonl(self, lines: List[dict], path: Union[str, Path]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for obj in lines:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        return p

    # ----------------------------
    # Log batch to single file (batch-level + summary + raw batch JSON)
    # ----------------------------
    def log_batch(self, batch_id: str, log_path: Union[str, Path], *, include_raw_json: bool = True) -> Path:

        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        try:
            batch = self.get_batch(batch_id)
        except (HTTPError, URLError) as e:
            p.write_text(f"Failed to retrieve batch {batch_id}: {repr(e)}\n", encoding="utf-8")
            return p

        lines = []
        lines.append(f"batch_id: {batch_id}")
        lines.append(f"processing_status: {batch.get('processing_status')}")
        lines.append(f"request_counts: {json.dumps(batch.get('request_counts'), ensure_ascii=False)}")
        lines.append("")

        # Log any extra fields if they exist (future-proof)
        for key in ("error", "errors", "message"):
            if key in batch and batch[key]:
                lines.append(f"{key}: {json.dumps(batch[key], ensure_ascii=False)}")
                lines.append("")

        if include_raw_json:
            lines.append("RAW BATCH JSON:")
            lines.append(json.dumps(batch, indent=2, ensure_ascii=False))
            lines.append("")

        p.write_text("\n".join(lines), encoding="utf-8")
        # Append a sample of per-request errors from results
        try:
            raw = self._get_text(f"/v1/messages/batches/{batch_id}/results")
            lines = [json.loads(l) for l in raw.splitlines() if l.strip()]
            lines = lines[:10]  # sample

            out = []
            for l in lines:
                cid, text, err = self.parse_batch_line(l)
                out.append({"custom_id": cid, "error": err, "text_preview": (text[:120] if text else None)})

            log_extra = "\nRESULTS SAMPLE:\n" + json.dumps(out, indent=2, ensure_ascii=False)
        except Exception as ex:
            log_extra = f"\nRESULTS SAMPLE: <failed to load> {repr(ex)}\n"

        # then write log file including log_extra

        return p

    def debug_first_error(self, batch_id: str) -> None:
        raw = self._get_text(f"/v1/messages/batches/{batch_id}/results")
        for l in raw.splitlines():
            if not l.strip():
                continue
            obj = json.loads(l)
            cid, text, err = self.parse_batch_line(obj)
            if err:
                print("custom_id:", cid)
                print("error:", err)
                print("raw line:", json.dumps(obj, ensure_ascii=False)[:2000])
                return
        print("No errors found in results.")


@dataclass
class BatchJobResult:
    batch_id: str
    status: str
    output_file_id: Optional[str]
    error_file_id: Optional[str]



if __name__ == "__main__":
    llm = BatchLLM(model="gpt-5-mini")  # 1) specify model; 2) uses /v1/responses by default

    # Build a JSONL file
    jsonl_path = Path("inputs/my_batch.jsonl")
    llm.build_request("Say hello.", "req-1", save_path=jsonl_path)
    llm.build_request("Say goodbye.", "req-2", save_path=jsonl_path)

    # 3) submit
    batch_id = llm.submit(jsonl_path, metadata={"purpose": "demo"})
    print("submitted:", batch_id)

    # 6) wait + download + write
    result = llm.download_results(batch_id, wait=True, timeout_s=3600)
    llm.write_results(
        result,
        output_path="outputs/batch_output.jsonl",
        error_path="outputs/batch_errors.jsonl",
        log_path="outputs/batch_log.txt",
    )

    print("final status:", result.status)