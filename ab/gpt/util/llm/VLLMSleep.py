import gc
import os
from pathlib import Path

import torch

from ab.gpt.util.Util import extract_all_to_train

MIN_NEW_TOKENS = 256
SKIPPED_OUTPUT = (None, None, None, "")
GIB = 1024 ** 3

# A park that frees less than this did not really happen. bitsandbytes
# Params4bit can accept .to("cpu") while keeping its quantized storage on the
# device, so the move is verified against the allocator instead of trusted.
MIN_PARK_FREED = 1 * GIB


def _env_flag(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _env_int(name, default):
    try:
        return int(float(os.getenv(name, "").strip() or default))
    except (TypeError, ValueError):
        return default


_ALLOC_VARS = ('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')


def _strip_expandable_segments():
    """Drop expandable_segments from the allocator config; returns the old values.

    Sleep mode allocates through vLLM's CuMemAllocator, which uses the CUDA
    virtual-memory API. PyTorch's expandable_segments uses the same API and the
    two cannot coexist -- vLLM asserts rather than corrupting memory
    (pytorch/pytorch#147851). Unsloth sets expandable_segments:True to limit
    fragmentation during training, which is how it reaches the engine.

    The engine runs in a child process that inherits os.environ, so clearing the
    flag here reaches it. The parent's allocator was configured at its first CUDA
    allocation and is unaffected, so the trainer keeps the behaviour it started
    with; the value is restored once the engine is built.
    """
    saved = {}
    for var in _ALLOC_VARS:
        value = os.environ.get(var)
        if value is None or 'expandable_segments' not in value:
            continue
        saved[var] = value
        kept = [part for part in value.split(',')
                if 'expandable_segments' not in part]
        if kept:
            os.environ[var] = ','.join(kept)
        else:
            os.environ.pop(var, None)
        print(f"[vLLM] {var}: dropped expandable_segments for the engine "
              f"(incompatible with sleep mode)", flush=True)
    return saved


def _restore_alloc_conf(saved):
    for var, value in saved.items():
        os.environ[var] = value


#: Destroy the engine when an epoch's generation is done, instead of parking it
#: with sleep(). A parked engine keeps the CUDA context, the compiled graphs and
#: the pool reserved through gpu_memory_utilization, which is most of the card --
#: memory the evaluator needs to train the networks that were just generated.
#: The cost is one engine build per epoch; set NNGPT_VLLM_TEARDOWN=0 to trade
#: that memory back for the faster wake-up.
TEARDOWN_BETWEEN_EPOCHS = _env_flag("NNGPT_VLLM_TEARDOWN", True)


def sleep_mode_enabled():
    return _env_flag("NNGPT_VLLM_SLEEP", True)


def _free_gpu():
    if not torch.cuda.is_available():
        return 0, 0
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    return free, total


def park_trainer(model):
    """Move the HF training model to host memory. Returns its previous device.

    Returns None when nothing actually left the card, so the caller does not
    later "restore" a model that never moved, and so the engine sizes itself
    against the memory that is really free.
    """
    if model is None:
        return None
    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        return None
    if device.type == "cpu":
        return device

    before, total = _free_gpu()
    try:
        model.to("cpu")
    except Exception as exc:
        print(f"[vLLM] Trainer could not be parked on the host ({exc}); "
              f"the engine will share the card with it.", flush=True)
        return None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    after, _ = _free_gpu()
    freed = after - before

    if freed < MIN_PARK_FREED:
        print(f"[vLLM] Parking the trainer freed only {freed / GIB:.2f} GiB "
              f"({after / GIB:.1f} GiB free of {total / GIB:.1f} GiB). Quantized "
              f"weights normally stay on the device, so the engine will share "
              f"the card with the trainer.", flush=True)
        return None

    print(f"[vLLM] Trainer parked on the host; freed {freed / GIB:.1f} GiB "
          f"({after / GIB:.1f} GiB of {total / GIB:.1f} GiB free).", flush=True)
    return device


def unpark_trainer(model, device):
    if model is None or device is None or device.type == "cpu":
        return
    try:
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[vLLM] Trainer restored to {device}.", flush=True)
    except Exception as exc:
        print(f"[vLLM] Could not restore the trainer to {device}: {exc}", flush=True)
        raise


class VLLMSleepEngine:

    def __init__(self, base_model_name, tokenizer, max_model_len,
                 temperature=0.8, top_k=70, top_p=0.9,
                 gpu_memory_utilization=None, quantization=None,
                 max_lora_rank=None, enforce_eager=None, dtype="bfloat16",
                 served_model=None, max_num_seqs=None, trainer_reserve_gib=None,
                 kv_cache_dtype=None):
        from vllm import LLM

        self.tokenizer = tokenizer
        self.system_prompt = None
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_model_len = int(max_model_len)
        self.suppress_thinking = False
        self.budget_tokens = None

        self._adapter_path = None
        self._adapter_version = 0
        self._asleep = False

        served_model = served_model or os.getenv("NNGPT_VLLM_MODEL", "").strip()
        if served_model and served_model != base_model_name:
            print(f"[vLLM] Inference weights: {served_model} "
                  f"(trainer keeps {base_model_name})", flush=True)
        else:
            served_model = base_model_name

        free, total = _free_gpu()
        reserve = _env_float("NNGPT_VLLM_TRAINER_RESERVE_GIB",
                             trainer_reserve_gib if trainer_reserve_gib is not None else 0.0)
        budget = min(free, total - reserve * GIB) - _env_float("NNGPT_VLLM_MARGIN_GIB", 1.0) * GIB
        if gpu_memory_utilization is None:
            gpu_memory_utilization = _env_float("NNGPT_VLLM_GPU_MEMORY_UTILIZATION", 0.0)
        if not gpu_memory_utilization:
            gpu_memory_utilization = max(0.10, min(0.95, budget / total)) if total else 0.90

        default_quant = "" if served_model != base_model_name else "bitsandbytes"
        quantization = quantization or os.getenv(
            "NNGPT_VLLM_QUANTIZATION", default_quant).strip() or None
        if quantization and quantization.lower() in {"none", "off", "auto"}:
            quantization = None

        max_lora_rank = _env_int("NNGPT_VLLM_MAX_LORA_RANK",
                                 max_lora_rank or 32)
        enforce_eager = (enforce_eager if enforce_eager is not None
                         else _env_flag("NNGPT_VLLM_ENFORCE_EAGER", True))
        max_num_seqs = _env_int("NNGPT_VLLM_MAX_NUM_SEQS", max_num_seqs or 16)
        # "fp8" halves the KV cache, roughly doubling how many sequences fit,
        # at the cost of a small quantization error in attention. Left at "auto"
        # by default because the generated code is the measured output.
        kv_cache_dtype = kv_cache_dtype or os.getenv(
            "NNGPT_VLLM_KV_CACHE_DTYPE", "auto").strip() or "auto"

        kwargs = dict(
            model=served_model,
            enable_sleep_mode=True,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
            max_loras=1,
            max_model_len=self.max_model_len,
            max_num_seqs=max_num_seqs,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype=dtype,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
            swap_space=0,
            kv_cache_dtype=kv_cache_dtype,
        )
        if quantization:
            kwargs["quantization"] = quantization
            if quantization == "bitsandbytes":
                kwargs["load_format"] = "bitsandbytes"

        print(f"[vLLM] Building engine: model={served_model} "
              f"max_model_len={self.max_model_len} max_num_seqs={max_num_seqs} "
              f"quantization={quantization or 'from checkpoint'} "
              f"kv_cache_dtype={kv_cache_dtype} "
              f"gpu_memory_utilization={gpu_memory_utilization:.3f} "
              f"(free={free / GIB:.1f} GiB, total={total / GIB:.1f} GiB)", flush=True)
        _saved_alloc = _strip_expandable_segments()
        try:
            self.llm = LLM(**kwargs)
        finally:
            _restore_alloc_conf(_saved_alloc)
        print("[vLLM] Engine ready: sleep mode on, LoRA hot-swap on.", flush=True)

    def sleep(self, level=1):
        if self._asleep:
            return
        try:
            self.llm.sleep(level=level)
            self._asleep = True
            gc.collect()
            torch.cuda.empty_cache()
            free, total = _free_gpu()
            print(f"[vLLM] Asleep (level={level}); {free / GIB:.1f} GiB of "
                  f"{total / GIB:.1f} GiB free for the trainer.", flush=True)
        except Exception as exc:
            print(f"[vLLM] sleep() failed, engine stays resident: {exc}", flush=True)

    def wake_up(self, adapter_path=None):
        if self._asleep:
            gc.collect()
            torch.cuda.empty_cache()
            self.llm.wake_up()
            self._asleep = False
            free, total = _free_gpu()
            print(f"[vLLM] Awake; {free / GIB:.1f} GiB of {total / GIB:.1f} GiB "
                  f"still free.", flush=True)
        self.set_adapter(adapter_path)

    def set_adapter(self, adapter_path):
        if adapter_path is None:
            return
        path = Path(adapter_path)
        if not (path / "adapter_config.json").is_file():
            print(f"[vLLM] No adapter_config.json under {path}; generating with "
                  f"the base weights.", flush=True)
            return
        self._adapter_path = path
        self._adapter_version += 1
        print(f"[vLLM] Adapter v{self._adapter_version} <- {path}", flush=True)

    def shutdown(self):
        """Destroy the engine outright and give the whole card back.

        sleep() keeps the process able to generate again in milliseconds, but
        the engine still owns its CUDA context, its compiled graphs and the
        pool it reserved through gpu_memory_utilization -- most of the card.
        Evaluating the generated networks needs that memory, so between epochs
        the engine is destroyed rather than parked, and rebuilt when the next
        epoch generates. Never raises: a failure to tear down must not take
        the run with it.
        """
        before, total = _free_gpu()
        llm, self.llm = getattr(self, "llm", None), None
        engine = getattr(llm, "llm_engine", None)
        for owner in (getattr(engine, "engine_core", None), engine, llm):
            if owner is None:
                continue
            for name in ("shutdown", "close"):
                fn = getattr(owner, name, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
        del llm, engine
        # The next build initialises the model-parallel group again; leaving
        # the old one registered makes that build fail outright, so a torn-down
        # engine that was not unregistered here can never be replaced.
        try:
            from vllm.distributed.parallel_state import (
                destroy_distributed_environment, destroy_model_parallel)
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass
        self._asleep = False
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        after, _ = _free_gpu()
        print(f"[vLLM] Engine destroyed; {after / GIB:.1f} GiB of "
              f"{total / GIB:.1f} GiB free "
              f"(+{(after - before) / GIB:.1f} GiB released).", flush=True)

    def _lora_request(self):
        if self._adapter_path is None:
            return None
        from vllm.lora.request import LoRARequest
        return LoRARequest(f"nngpt-v{self._adapter_version}",
                           self._adapter_version, str(self._adapter_path))

    def _build_messages(self, user_content):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def _token_ids(self, prompt_text):
        messages = self._build_messages(prompt_text)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return list(self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True))
        body = "\n".join(f"{m['role'].capitalize()}: {m['content']}"
                         for m in messages)
        return list(self.tokenizer(f"{body}\nAssistant:",
                                   add_special_tokens=False)["input_ids"])

    def token_length(self, prompt_text):
        return len(self._token_ids(prompt_text))

    def output_budget(self, in_len, max_new_tokens):
        want = max_new_tokens or self.max_model_len
        room = self.max_model_len - in_len
        if room < MIN_NEW_TOKENS:
            return None
        return min(want, room)

    def chat_batch(self, prompts, max_len=None, max_new_tokens=None,
                   engineer_prompt=False):
        if not prompts:
            return []
        if self._asleep:
            self.wake_up()

        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        requests, budgets, kept, results = [], [], [], {}
        for pos, prompt_text in enumerate(prompts):
            ids = self._token_ids(prompt_text)
            budget = self.output_budget(len(ids), max_new_tokens)
            if budget is None:
                print(f"[vLLM] SKIPPED prompt {pos}: input_tokens={len(ids)} "
                      f"max_model_len={self.max_model_len} leaves "
                      f"{self.max_model_len - len(ids)} for output.", flush=True)
                results[pos] = SKIPPED_OUTPUT
                continue
            requests.append(TokensPrompt(prompt_token_ids=ids))
            budgets.append(budget)
            kept.append(pos)

        if not kept:
            return [results[pos] for pos in range(len(prompts))]

        sampling = [SamplingParams(temperature=self.temperature,
                                   top_k=self.top_k, top_p=self.top_p,
                                   max_tokens=budget) for budget in budgets]
        print(f"[vLLM] generate: {len(kept)} prompts in one continuous batch, "
              f"max_new_tokens={max(budgets)}", flush=True)
        outputs = self.llm.generate(requests, sampling,
                                    lora_request=self._lora_request())

        for pos, out in zip(kept, outputs):
            text = out.outputs[0].text if out.outputs else ""
            results[pos] = (*extract_all_to_train(text), text)
        return [results[pos] for pos in range(len(prompts))]

    def chat(self, prompt, max_len=None, max_new_tokens=None,
             engineer_prompt=False):
        return self.chat_batch([prompt], max_len=max_len,
                               max_new_tokens=max_new_tokens,
                               engineer_prompt=engineer_prompt)[0]


def build_engine(base_model_name, tokenizer, max_model_len, temperature=0.8,
                 top_k=70, top_p=0.9, trainer_reserve_gib=None):
    try:
        return VLLMSleepEngine(
            base_model_name, tokenizer, max_model_len,
            temperature=temperature, top_k=top_k, top_p=top_p,
            trainer_reserve_gib=trainer_reserve_gib,
        )
    except Exception as exc:
        print(f"[vLLM] ENGINE BUILD FAILED: {exc}", flush=True)
        print("[vLLM] Generation falls back to the HuggingFace path, which is "
              "several times slower. Install a vllm build matching this torch, "
              "or set NNGPT_VLLM_SLEEP=0 to silence this.", flush=True)
        return None


class VLLMGeneration:

    def __init__(self, base_model_name, tokenizer, max_model_len,
                 temperature=0.8, top_k=70, top_p=0.9):
        self.base_model_name = base_model_name
        self.tokenizer = tokenizer
        self.max_model_len = int(max_model_len)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.system_prompt = None
        self.suppress_thinking = False
        self.budget_tokens = None

        self.engine = None
        self.fallback = None
        self._build_failed = False
        self._parked_device = None

    @property
    def llm(self):
        if self.engine is None:
            raise AttributeError("vLLM engine is not built")
        return self.engine.llm

    @property
    def active(self):
        return self.engine is not None

    def _target(self):
        return self.engine if self.engine is not None else self.fallback

    def prepare(self, adapter_path, trainer_model):
        if self._build_failed:
            return
        self._parked_device = park_trainer(trainer_model)
        if self.engine is None:
            self.engine = build_engine(
                self.base_model_name, self.tokenizer, self.max_model_len,
                temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
            )
            if self.engine is None:
                self._build_failed = True
                unpark_trainer(trainer_model, self._parked_device)
                self._parked_device = None
                return
            self.engine.set_adapter(adapter_path)
            return
        self.engine.wake_up(adapter_path)

    def release(self, trainer_model):
        """Called once generation for the epoch is finished.

        The evaluator trains every generated network on the same GPU, so the
        engine is destroyed here and rebuilt by the next prepare(). Set
        NNGPT_VLLM_TEARDOWN=0 to park it with sleep() instead, which wakes
        faster but leaves the reserved pool in place for the whole evaluation.
        """
        if self.engine is not None:
            if TEARDOWN_BETWEEN_EPOCHS:
                self.engine.shutdown()
                self.engine = None
            else:
                self.engine.sleep(level=1)
        if self._parked_device is not None:
            unpark_trainer(trainer_model, self._parked_device)
            self._parked_device = None

    def ensure_asleep(self):
        if self.engine is not None:
            self.engine.sleep(level=1)

    def shutdown(self):
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def token_length(self, prompt_text):
        """Prompt length in tokens, with or without a live engine.

        Between epochs the engine is torn down, but the prompt-fitting pass
        still has to measure prompts. Measuring needs the tokenizer only, so
        it is done here rather than delegated to an engine that may be gone --
        returning None instead would make every prompt look unmeasurable and
        the epoch would generate nothing.
        """
        if self.engine is not None:
            return self.engine.token_length(prompt_text)
        target = self.fallback
        if target is not None:
            rendered = target._templated_from_messages(
                target._build_messages(prompt_text))
            return len(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        messages = self._build_messages(prompt_text)
        if hasattr(self.tokenizer, "apply_chat_template"):
            return len(self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True))
        body = "\n".join(f"{m['role'].capitalize()}: {m['content']}"
                          for m in messages)
        return len(self.tokenizer(f"{body}\nAssistant:",
                                  add_special_tokens=False)["input_ids"])

    def output_budget(self, in_len, max_new_tokens):
        if self.engine is not None:
            return self.engine.output_budget(in_len, max_new_tokens)
        want = max_new_tokens or self.max_model_len
        room = self.max_model_len - in_len
        return None if room < MIN_NEW_TOKENS else min(want, room)

    def _build_messages(self, user_content):
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages

    def chat(self, prompt, max_len=None, max_new_tokens=None, engineer_prompt=False):
        target = self._target()
        if target is None:
            return SKIPPED_OUTPUT
        target.system_prompt = self.system_prompt
        return target.chat(prompt, max_len=max_len, max_new_tokens=max_new_tokens,
                           engineer_prompt=engineer_prompt)

    def chat_batch(self, prompts, max_len=None, max_new_tokens=None,
                   engineer_prompt=False):
        target = self._target()
        if target is None:
            return [SKIPPED_OUTPUT for _ in prompts]
        target.system_prompt = self.system_prompt
        return target.chat_batch(prompts, max_len=max_len,
                                 max_new_tokens=max_new_tokens,
                                 engineer_prompt=engineer_prompt)
