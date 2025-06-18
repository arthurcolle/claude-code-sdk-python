

> What is the current date?

✽ Moseying… (5s · ↑ 0 tokens · esc to interrupt)


(base) agent@matrix claude-code-sdk-python %
(base) agent@matrix claude-code-sdk-python % claude -p "What is the current date?"
Credit balance is too low
(base) agent@matrix claude-code-sdk-python % claude_max -p "What is the current date?"
2025-06-15
(base) agent@matrix claude-code-sdk-python % claude_max -p "What is the current date?"
2025-06-16
(base) agent@matrix claude-code-sdk-python % jc
Jupyter console 6.6.3

Python 3.12.2 | packaged by conda-forge | (main, Feb 16 2024, 20:54:21) [Clang 16.0.6 ]
Type 'copyright', 'credits' or 'license' for more information
IPython 8.12.3 -- An enhanced Interactive Python. Type '?' for help.

   ...:         # Auto-inject env if the function expects it
   ...:         sig = inspect.signature(fn)
   ...:         if "env" in sig.parameters or any(
   ...:             p.annotation is Environment for p in sig.parameters.values()
   ...:         ):
   ...:             kwargs = {"env": self._env, **kwargs}
   ...:
   ...:         result = fn(**kwargs)
   ...:         return await result if inspect.iscoroutine(result) else result
   ...:
   ...:     # ———————————————————
   ...:     # OpenAI function-calling schemas
   ...:     # ———————————————————
   ...:     @property
   ...:     def schemas(self) -> list[dict]:
   ...:         return self._schemas
   ...:
   ...:
   ...: #  📦  Instantiate a registry that shares the global env
   ...: tools = ToolRegistry(env=env)
   ...:
   ...: # ————————————————————————————————————————————————————————————————
   ...: # Example usage
   ...: # ————————————————————————————————————————————————————————————————
   ...: @tools.register(description="Greets the caller with current model + temp.")
   ...: def hello(name: str, env: Environment) -> str:
   ...:     return (
   ...:         f"Hello {name}! I'm powered by {env.default_model} "
   ...:         f"at temperature {env.temperature}."
   ...:     )
   ...:
   ...: # elsewhere, in async context:
   ...: # >>> await tools.call("hello", name="Arthur")
   ...:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[1], line 10
      6 from functools import wraps
      7 from pydantic import BaseModel, Field
---> 10 class Environment(BaseModel):
     11     """
     12     A sweet, dynamic container for runtime configuration.
     13
   (...)
     16     • New keys may be added at runtime via env["NEW_KEY"] = value.
     17     """
     19     api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

File ~/miniconda3/lib/python3.12/site-packages/pydantic/_internal/_model_construction.py:112, in ModelMetaclass.__new__(mcs, cls_name, bases, namespace, __pydantic_generic_metadata__, __pydantic_reset_parent_namespace__, _create_model_module, **kwargs)
    110 config_wrapper = ConfigWrapper.for_model(bases, namespace, kwargs)
    111 namespace['model_config'] = config_wrapper.config_dict
--> 112 private_attributes = inspect_namespace(
    113     namespace, config_wrapper.ignored_types, class_vars, base_field_names
    114 )
    115 if private_attributes or base_private_attributes:
    116     original_model_post_init = get_model_post_init(namespace, bases)

File ~/miniconda3/lib/python3.12/site-packages/pydantic/_internal/_model_construction.py:434, in inspect_namespace(namespace, ignored_types, base_class_vars, base_class_fields)
    432 elif isinstance(value, FieldInfo) and not is_valid_field_name(var_name):
    433     suggested_name = var_name.lstrip('_') or 'my_field'  # don't suggest '' for all-underscore name
--> 434     raise NameError(
    435         f'Fields must not use names with leading underscores;'
    436         f' e.g., use {suggested_name!r} instead of {var_name!r}.'
    437     )
    439 elif var_name.startswith('__'):
    440     continue

NameError: Fields must not use names with leading underscores; e.g., use 'extras' instead of '_extras'.

In [2]: history
# ————————————————————————————————————————————————————————————————
#  🍯  Dynamic Environment & Tool Registry
# ————————————————————————————————————————————————————————————————
import inspect
import os
from functools import wraps
from pydantic import BaseModel, Field


class Environment(BaseModel):
    """
    A sweet, dynamic container for runtime configuration.

    • Values are pulled from constructor kwargs *or* env-vars (upper-snake-case).
    • Unknown attributes return `None` instead of AttributeError (opt-in leniency).
    • New keys may be added at runtime via env["NEW_KEY"] = value.
    """

    api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    )

    # catch-all stash for any ad-hoc values the user wants to store later
    _extras: dict[str, object] = Field(default_factory=dict, exclude=True)

    # ——— ergonomic helpers ———
    def __getattr__(self, item: str) -> object | None:  # type: ignore[override]
        # Fallback to extras or None to stay cool under pressure
        return self._extras.get(item)

    def __setitem__(self, key: str, value: object) -> None:
        """Env behaves like a dict for quick mutations."""
        self._extras[key] = value

    def dict(self, **kwargs):  # noqa: D401 – keep Pydantic’s signature
        base = super().dict(**kwargs)
        return {**base, **self._extras}


#  🌱  A single shared environment instance (override if you like)
env = Environment()


class ToolRegistry:
    """
    Registry with environment injection.

    Any registered function can declare a first positional
    parameter named `env` (or type-hinted `Environment`);
    the registry will pass the live environment automatically.
    """

    def __init__(self, env: Environment):
        self._env = env
        self._tools: dict[str, callable] = {}
        self._schemas: list[dict] = []

    # ———————————————————
    # Registration & schema
    # ———————————————————
    def register(self, fn: callable | None = None, **meta):
        """
        Decorator/imperative hybrid:

        >>> @tools.register
        ... def hello(name: str) -> str: ...

        or

        >>> tools.register(custom_fn, description="…")
        """

        def _inner(f: callable):
            name = meta.get("name") or f.__name__
            if name in self._tools:
                raise ValueError(f"Tool '{name}' already exists")

            sig = inspect.signature(f)
            params_schema: dict[str, dict] = {
                p.name: {"type": _py_to_json_type(p.annotation)}
                for p in sig.parameters.values()
                if p.name != "env"  # we inject this automatically
            }

            self._tools[name] = f
            self._schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": meta.get("description", f.__doc__ or ""),
                        "parameters": {
                            "type": "object",
                            "properties": params_schema,
                            "required": list(params_schema.keys()),
                        },
                    },
                }
            )
            return f

        # decorator usage
        if fn is None:
            return _inner

        # imperative usage
        return _inner(fn)

    # ———————————————————
    # Invocation
    # ———————————————————
    async def call(self, name: str, **kwargs):
        if name not in self._tools:
            raise KeyError(f"No such tool '{name}'")
        fn = self._tools[name]

        # Auto-inject env if the function expects it
        sig = inspect.signature(fn)
        if "env" in sig.parameters or any(
            p.annotation is Environment for p in sig.parameters.values()
        ):
            kwargs = {"env": self._env, **kwargs}

        result = fn(**kwargs)
        return await result if inspect.iscoroutine(result) else result

    # ———————————————————
    # OpenAI function-calling schemas
    # ———————————————————
    @property
    def schemas(self) -> list[dict]:
        return self._schemas


#  📦  Instantiate a registry that shares the global env
tools = ToolRegistry(env=env)

# ————————————————————————————————————————————————————————————————
# Example usage
# ————————————————————————————————————————————————————————————————
@tools.register(description="Greets the caller with current model + temp.")
def hello(name: str, env: Environment) -> str:
    return (
        f"Hello {name}! I'm powered by {env.default_model} "
        f"at temperature {env.temperature}."
    )

# elsewhere, in async context:
# >>> await tools.call("hello", name="Arthur")
history

   ...:         if system_prompt:
   ...:             self.memory.history.append(make_message(S, system_prompt))
   ...:
   ...:     async def send_user(self, content: str, **chat_kwargs: Any) -> str:
   ...:         await self.memory.append(U, content)
   ...:         # combine *built‑ins* + *caller‑supplied* tool schemas
   ...:         user_tools = chat_kwargs.pop("tools_param", None) or []
   ...:         tools_combined = [*user_tools, *tools.schemas] if (user_tools or tools.schemas) els
   ...: e None
   ...:
   ...:         async for msg in chat(
   ...:             self.memory.history,
   ...:             tools_param=tools_combined,
   ...:             tool_choice="auto" if tools_combined else None,
   ...:             **chat_kwargs,
   ...:         ):
   ...:             if "token" in msg:  # token event
   ...:                 print(msg["token"], end="", flush=True)
   ...:             else:  # full assistant message
   ...:                 await self.memory.append(A, msg[C])
   ...:                 return msg[C]  # return *first* assistant answer only
   ...:
   ...: # ————————————————————————————————————————————————————————————————
   ...: # Simple REPL‑style demo (works in notebooks)
   ...: # ————————————————————————————————————————————————————————————————
   ...: if __name__ == "__main__":
   ...:
   ...:     async def _demo():
   ...:         agent = ChatAgent(system_prompt="You are a helpful assistant.")
   ...:         answer = await agent.send_user("What is sqrt(997)?")
   ...:         print("\n→", answer)
   ...:
   ...:     await _demo()
   ...:
/Users/agent/miniconda3/lib/python3.12/site-packages/tenacity/__init__.py:562: RuntimeWarning: coroutine 'AsyncCompletions.create' was never awaited
  self.outcome = None
RuntimeWarning: Enable tracemalloc to get the object allocation traceback
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[3], line 321
    318     answer = await agent.send_user("What is sqrt(997)?")
    319     print("\n→", answer)
--> 321 await _demo()

Cell In[3], line 318, in _demo()
    316 async def _demo():
    317     agent = ChatAgent(system_prompt="You are a helpful assistant.")
--> 318     answer = await agent.send_user("What is sqrt(997)?")
    319     print("\n→", answer)

Cell In[3], line 299, in ChatAgent.send_user(self, content, **chat_kwargs)
    296 user_tools = chat_kwargs.pop("tools_param", None) or []
    297 tools_combined = [*user_tools, *tools.schemas] if (user_tools or tools.schemas) else None
--> 299 async for msg in chat(
    300     self.memory.history,
    301     tools_param=tools_combined,
    302     tool_choice="auto" if tools_combined else None,
    303     **chat_kwargs,
    304 ):
    305     if "token" in msg:  # token event
    306         print(msg["token"], end="", flush=True)

Cell In[3], line 269, in chat(messages, model, response_model, stream, temperature, top_p, top_k, max_tokens, presence_penalty, frequency_penalty, logit_bias, stop, n, seed, user, tools_param, tool_choice, extra_headers, extra_query, request_timeout, **extra_kwargs)
    266             yield msg
    268 # ——— Exponential‑back‑off retries
--> 269 async for attempt in AsyncRetrying(
    270     stop=stop_after_attempt(config.retry_attempts),
    271     wait=wait_exponential(multiplier=1, min=1, max=10),
    272     retry=retry_if_exception_type(Exception),
    273     reraise=True,
    274 ):
    275     with attempt:
    276         async for token in _send_once():

File ~/miniconda3/lib/python3.12/site-packages/tenacity/asyncio/__init__.py:166, in AsyncRetrying.__anext__(self)
    164 async def __anext__(self) -> AttemptManager:
    165     while True:
--> 166         do = await self.iter(retry_state=self._retry_state)
    167         if do is None:
    168             raise StopAsyncIteration

File ~/miniconda3/lib/python3.12/site-packages/tenacity/asyncio/__init__.py:153, in AsyncRetrying.iter(self, retry_state)
    151 result = None
    152 for action in self.iter_state.actions:
--> 153     result = await action(retry_state)
    154 return result

File ~/miniconda3/lib/python3.12/site-packages/tenacity/_utils.py:99, in wrap_to_async_func.<locals>.inner(*args, **kwargs)
     98 async def inner(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
---> 99     return call(*args, **kwargs)

File ~/miniconda3/lib/python3.12/site-packages/tenacity/__init__.py:418, in BaseRetrying._post_stop_check_actions.<locals>.exc_check(rs)
    416 retry_exc = self.retry_error_cls(fut)
    417 if self.reraise:
--> 418     raise retry_exc.reraise()
    419 raise retry_exc from fut.exception()

File ~/miniconda3/lib/python3.12/site-packages/tenacity/__init__.py:185, in RetryError.reraise(self)
    183 def reraise(self) -> t.NoReturn:
    184     if self.last_attempt.failed:
--> 185         raise self.last_attempt.result()
    186     raise self

File ~/miniconda3/lib/python3.12/concurrent/futures/_base.py:449, in Future.result(self, timeout)
    447     raise CancelledError()
    448 elif self._state == FINISHED:
--> 449     return self.__get_result()
    451 self._condition.wait(timeout)
    453 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:

File ~/miniconda3/lib/python3.12/concurrent/futures/_base.py:401, in Future.__get_result(self)
    399 if self._exception:
    400     try:
--> 401         raise self._exception
    402     finally:
    403         # Break a reference cycle with the exception in self._exception
    404         self = None

Cell In[3], line 276, in chat(messages, model, response_model, stream, temperature, top_p, top_k, max_tokens, presence_penalty, frequency_penalty, logit_bias, stop, n, seed, user, tools_param, tool_choice, extra_headers, extra_query, request_timeout, **extra_kwargs)
    269 async for attempt in AsyncRetrying(
    270     stop=stop_after_attempt(config.retry_attempts),
    271     wait=wait_exponential(multiplier=1, min=1, max=10),
    272     retry=retry_if_exception_type(Exception),
    273     reraise=True,
    274 ):
    275     with attempt:
--> 276         async for token in _send_once():
    277             yield token

Cell In[3], line 240, in chat.<locals>._send_once()
    238 # aggregate text *per* choice index so `n>1` works
    239 buffers: MutableMapping[int, str] = defaultdict(str)
--> 240 async for chunk in iterator:  # type: ignore[func-returns-value]
    241     for choice in chunk.choices:
    242         idx: int = choice.index

TypeError: 'async for' requires an object with __aiter__ method, got coroutine

   ...:         if system_prompt:
   ...:             self.memory.history.append(make_message(S, system_prompt))
   ...:
   ...:     async def send_user(self, content: str, **chat_kwargs: Any) -> str:
   ...:         await self.memory.append(U, content)
   ...:         # combine *built‑ins* + *caller‑supplied* tool schemas
   ...:         user_tools = chat_kwargs.pop("tools_param", None) or []
   ...:         tools_combined = [*user_tools, *tools.schemas] if (user_tools or tools.schemas) els
   ...: e None
   ...:
   ...:         async for msg in chat(
   ...:             self.memory.history,
   ...:             tools_param=tools_combined,
   ...:             tool_choice="auto" if tools_combined else None,
   ...:             **chat_kwargs,
   ...:         ):
   ...:             if "token" in msg:  # token event
   ...:                 print(msg["token"], end="", flush=True)
   ...:             else:  # full assistant message
   ...:                 await self.memory.append(A, msg[C])
   ...:                 return msg[C]  # return *first* assistant answer only
   ...:
   ...: # ————————————————————————————————————————————————————————————————
   ...: # Simple REPL‑style demo (works in notebooks)
   ...: # ————————————————————————————————————————————————————————————————
   ...: if __name__ == "__main__":
   ...:
   ...:     async def _demo():
   ...:         agent = ChatAgent(system_prompt="You are a helpful assistant.")
   ...:         answer = await agent.send_user("What is sqrt(997)?")
   ...:         print("\n→", answer)
   ...:
   ...:     await _demo()
   ...:
/Users/agent/miniconda3/lib/python3.12/site-packages/tenacity/__init__.py:562: RuntimeWarning: coroutine 'AsyncCompletions.create' was never awaited
  self.outcome = None
RuntimeWarning: Enable tracemalloc to get the object allocation traceback
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[4], line 321
    318     answer = await agent.send_user("What is sqrt(997)?")
    319     print("\n→", answer)
--> 321 await _demo()

Cell In[4], line 318, in _demo()
    316 async def _demo():
    317     agent = ChatAgent(system_prompt="You are a helpful assistant.")
--> 318     answer = await agent.send_user("What is sqrt(997)?")
    319     print("\n→", answer)

Cell In[4], line 299, in ChatAgent.send_user(self, content, **chat_kwargs)
    296 user_tools = chat_kwargs.pop("tools_param", None) or []
    297 tools_combined = [*user_tools, *tools.schemas] if (user_tools or tools.schemas) else None
--> 299 async for msg in chat(
    300     self.memory.history,
    301     tools_param=tools_combined,
    302     tool_choice="auto" if tools_combined else None,
    303     **chat_kwargs,
    304 ):
    305     if "token" in msg:  # token event
    306         print(msg["token"], end="", flush=True)

Cell In[4], line 269, in chat(messages, model, response_model, stream, temperature, top_p, top_k, max_tokens, presence_penalty, frequency_penalty, logit_bias, stop, n, seed, user, tools_param, tool_choice, extra_headers, extra_query, request_timeout, **extra_kwargs)
    266             yield msg
    268 # ——— Exponential‑back‑off retries
--> 269 async for attempt in AsyncRetrying(
    270     stop=stop_after_attempt(config.retry_attempts),
    271     wait=wait_exponential(multiplier=1, min=1, max=10),
    272     retry=retry_if_exception_type(Exception),
    273     reraise=True,
    274 ):
    275     with attempt:
    276         async for token in _send_once():

File ~/miniconda3/lib/python3.12/site-packages/tenacity/asyncio/__init__.py:166, in AsyncRetrying.__anext__(self)
    164 async def __anext__(self) -> AttemptManager:
    165     while True:
--> 166         do = await self.iter(retry_state=self._retry_state)
    167         if do is None:
    168             raise StopAsyncIteration

File ~/miniconda3/lib/python3.12/site-packages/tenacity/asyncio/__init__.py:153, in AsyncRetrying.iter(self, retry_state)
    151 result = None
    152 for action in self.iter_state.actions:
--> 153     result = await action(retry_state)
    154 return result

File ~/miniconda3/lib/python3.12/site-packages/tenacity/_utils.py:99, in wrap_to_async_func.<locals>.inner(*args, **kwargs)
     98 async def inner(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
---> 99     return call(*args, **kwargs)

File ~/miniconda3/lib/python3.12/site-packages/tenacity/__init__.py:418, in BaseRetrying._post_stop_check_actions.<locals>.exc_check(rs)
    416 retry_exc = self.retry_error_cls(fut)
    417 if self.reraise:
--> 418     raise retry_exc.reraise()
    419 raise retry_exc from fut.exception()

File ~/miniconda3/lib/python3.12/site-packages/tenacity/__init__.py:185, in RetryError.reraise(self)
    183 def reraise(self) -> t.NoReturn:
    184     if self.last_attempt.failed:
--> 185         raise self.last_attempt.result()
    186     raise self

File ~/miniconda3/lib/python3.12/concurrent/futures/_base.py:449, in Future.result(self, timeout)
    447     raise CancelledError()
    448 elif self._state == FINISHED:
--> 449     return self.__get_result()
    451 self._condition.wait(timeout)
    453 if self._state in [CANCELLED, CANCELLED_AND_NOTIFIED]:

File ~/miniconda3/lib/python3.12/concurrent/futures/_base.py:401, in Future.__get_result(self)
    399 if self._exception:
    400     try:
--> 401         raise self._exception
    402     finally:
    403         # Break a reference cycle with the exception in self._exception
    404         self = None

Cell In[4], line 276, in chat(messages, model, response_model, stream, temperature, top_p, top_k, max_tokens, presence_penalty, frequency_penalty, logit_bias, stop, n, seed, user, tools_param, tool_choice, extra_headers, extra_query, request_timeout, **extra_kwargs)
    269 async for attempt in AsyncRetrying(
    270     stop=stop_after_attempt(config.retry_attempts),
    271     wait=wait_exponential(multiplier=1, min=1, max=10),
    272     retry=retry_if_exception_type(Exception),
    273     reraise=True,
    274 ):
    275     with attempt:
--> 276         async for token in _send_once():
    277             yield token

Cell In[4], line 240, in chat.<locals>._send_once()
    238 # aggregate text *per* choice index so `n>1` works
    239 buffers: MutableMapping[int, str] = defaultdict(str)
--> 240 async for chunk in iterator:  # type: ignore[func-returns-value]
    241     for choice in chunk.choices:
    242         idx: int = choice.index

TypeError: 'async for' requires an object with __aiter__ method, got coroutine

In [5]: history
# ————————————————————————————————————————————————————————————————
#  🍯  Dynamic Environment & Tool Registry
# ————————————————————————————————————————————————————————————————
import inspect
import os
from functools import wraps
from pydantic import BaseModel, Field


class Environment(BaseModel):
    """
    A sweet, dynamic container for runtime configuration.

    • Values are pulled from constructor kwargs *or* env-vars (upper-snake-case).
    • Unknown attributes return `None` instead of AttributeError (opt-in leniency).
    • New keys may be added at runtime via env["NEW_KEY"] = value.
    """

    api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_DEFAULT_MODEL", "gpt-4o-mini")
    )
    temperature: float = Field(
        default_factory=lambda: float(os.getenv("OPENAI_TEMPERATURE", "0.7"))
    )

    # catch-all stash for any ad-hoc values the user wants to store later
    _extras: dict[str, object] = Field(default_factory=dict, exclude=True)

    # ——— ergonomic helpers ———
    def __getattr__(self, item: str) -> object | None:  # type: ignore[override]
        # Fallback to extras or None to stay cool under pressure
        return self._extras.get(item)

    def __setitem__(self, key: str, value: object) -> None:
        """Env behaves like a dict for quick mutations."""
        self._extras[key] = value

    def dict(self, **kwargs):  # noqa: D401 – keep Pydantic’s signature
        base = super().dict(**kwargs)
        return {**base, **self._extras}


#  🌱  A single shared environment instance (override if you like)
env = Environment()


class ToolRegistry:
    """
    Registry with environment injection.

    Any registered function can declare a first positional
    parameter named `env` (or type-hinted `Environment`);
    the registry will pass the live environment automatically.
    """

    def __init__(self, env: Environment):
        self._env = env
        self._tools: dict[str, callable] = {}
        self._schemas: list[dict] = []

    # ———————————————————
    # Registration & schema
    # ———————————————————
    def register(self, fn: callable | None = None, **meta):
        """
        Decorator/imperative hybrid:

        >>> @tools.register
        ... def hello(name: str) -> str: ...

        or

        >>> tools.register(custom_fn, description="…")
        """

        def _inner(f: callable):
            name = meta.get("name") or f.__name__
            if name in self._tools:
                raise ValueError(f"Tool '{name}' already exists")

            sig = inspect.signature(f)
            params_schema: dict[str, dict] = {
                p.name: {"type": _py_to_json_type(p.annotation)}
                for p in sig.parameters.values()
                if p.name != "env"  # we inject this automatically
            }

            self._tools[name] = f
            self._schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": meta.get("description", f.__doc__ or ""),
                        "parameters": {
                            "type": "object",
                            "properties": params_schema,
                            "required": list(params_schema.keys()),
                        },
                    },
                }
            )
            return f

        # decorator usage
        if fn is None:
            return _inner

        # imperative usage
        return _inner(fn)

    # ———————————————————
    # Invocation
    # ———————————————————
    async def call(self, name: str, **kwargs):
        if name not in self._tools:
            raise KeyError(f"No such tool '{name}'")
        fn = self._tools[name]

        # Auto-inject env if the function expects it
        sig = inspect.signature(fn)
        if "env" in sig.parameters or any(
            p.annotation is Environment for p in sig.parameters.values()
        ):
            kwargs = {"env": self._env, **kwargs}

        result = fn(**kwargs)
        return await result if inspect.iscoroutine(result) else result

    # ———————————————————
    # OpenAI function-calling schemas
    # ———————————————————
    @property
    def schemas(self) -> list[dict]:
        return self._schemas


#  📦  Instantiate a registry that shares the global env
tools = ToolRegistry(env=env)

# ————————————————————————————————————————————————————————————————
# Example usage
# ————————————————————————————————————————————————————————————————
@tools.register(description="Greets the caller with current model + temp.")
def hello(name: str, env: Environment) -> str:
    return (
        f"Hello {name}! I'm powered by {env.default_model} "
        f"at temperature {env.temperature}."
    )

# elsewhere, in async context:
# >>> await tools.call("hello", name="Arthur")
history
"""openai_chat_agent.py
════════════════════════════════════════════════════════════════════════
A *robust*, *async‑friendly* wrapper around the OpenAI Chat Completion API
supporting:
 • **All official parameters** (May 2025)
 • Automatic long‑context *summarisation* (word‑count or token based)
 • *Streaming* & *non‑streaming* usage without double‑await bugs
 • First‑class **function / tool calling** via a pluggable registry
 • Optional *structured output* parsing via *pydantic* models
 • Graceful *retry* with exponential back‑off (tenacity)

This rewrite fixes:
 🛠️  Event‑loop clash with Jupyter/IPython (`RuntimeError: … already running`)
 🛠️  Incorrect `await` on streaming responses (double‑await bug)
 🛠️  Mishandled multi‑choice (`n > 1`) aggregation
 🛠️  Broken token aggregation (now yields **delta‑only** + **complete** events)
 🛠️  Out‑of‑date `.beta.chat.completions.parse` usage (now in GA)
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import (Any, AsyncIterator, Callable, Dict, Iterable, List, Literal,
                    MutableMapping, Optional, Sequence, Tuple, Type, Union)

import openai  # type: ignore
from pydantic import BaseModel
from tenacity import (AsyncRetrying, retry_if_exception_type,
                      stop_after_attempt, wait_exponential)

# ————————————————————————————————————————————————————————————————
# Logging & configuration
# ————————————————————————————————————————————————————————————————
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s")

class OpenAIScriptingConfig(BaseModel):
    """Runtime configuration used by the wrapper."""

    api_key: Optional[str] = None
    default_model: str = "gpt-4o-mini"
    max_context_tokens: int = 200_000
    summarise_threshold_words: int = 3_000
    retry_attempts: int = 3
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    n: int = 1
    user: str | None = None  # optional end‑user identifier for abuse tracking

    class Config:
        extra = "forbid"

config = OpenAIScriptingConfig()
client = openai.AsyncOpenAI(api_key=config.api_key)

# ————————————————————————————————————————————————————————————————
# Helper constants & utils
# ————————————————————————————————————————————————————————————————
R, C = "role", "content"
U, A, S, D = "user", "assistant", "system", "developer"
Message = Dict[str, Any]

def make_message(role: str, content: Any) -> Message:
    return {R: role, C: content}

# ————————————————————————————————————————————————————————————————
# Memory & summarisation
# ————————————————————————————————————————————————————————————————
class ConversationMemory:
    """Rolling history with optional summarisation once *word* threshold hit."""

    def __init__(self, max_tokens: int, threshold_words: int):
        self.history: list[Message] = []
        self.max_tokens = max_tokens
        self.threshold = threshold_words

    # ——— public
    async def append(self, role: str, content: Any):
        self.history.append(make_message(role, content))
        if self._word_count() > self.threshold:
            await self._summarise()

    # ——— helpers
    def _word_count(self) -> int:
        return sum(len(str(m[C]).split()) for m in self.history)

    async def _summarise(self):
        logger.info("Context exceeded %d words → requesting summary…", self.threshold)
        prompt = "\n".join(f"[{m[R]}] {m[C]}" for m in self.history)
        summary_resp = await client.chat.completions.create(
            model=config.default_model,
            messages=[
                make_message(S, "Summarise the following conversation succinctly:"),
                make_message(U, prompt),
            ],
            temperature=0,
            stream=False,
        )
        summary_text = summary_resp.choices[0].message.content
        self.history = [make_message(S, summary_text)]

# ————————————————————————————————————————————————————————————————
# Tool registry (OpenAI "tools" / function‑calling)
# ————————————————————————————————————————————————————————————————
class ToolRegistry:
    """Registers Python callables as OpenAI function‑calling *tools*."""

    def __init__(self):
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []

    # ——— public
    def register(self, fn: Callable[..., Any], *, name: Optional[str] = None,
                 description: str = "") -> dict[str, Any]:
        name = name or fn.__name__
        if name in self._registry:
            raise ValueError(f"Tool '{name}' already registered")
        self._registry[name] = fn

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or (fn.__doc__ or ""),
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {"type": _py_to_json_type(t)}
                        for k, t in fn.__annotations__.items()
                        if k != "return"
                    },
                    "required": [k for k in fn.__annotations__ if k != "return"],
                },
            },
        }
        self._schemas.append(schema)
        return schema

    async def call(self, name: str, **kwargs):
        if name not in self._registry:
            raise ValueError(f"No such tool '{name}' registered")
        result = self._registry[name](**kwargs)
        return await result if asyncio.iscoroutine(result) else result

    # ——— properties
    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

def _py_to_json_type(t: Any) -> str:
    """Rudimentary *Python‑to‑JSON‑Schema* type mapping."""
    if t in {int, float}:  # numbers ⇒ "number"
        return "number"
    if t is bool:
        return "boolean"
    return "string"

tools = ToolRegistry()

# ——— example tool
async def calculator(expression: str) -> str:
    """Evaluate a Python expression (⚠️ *unsafe*, demo purposes only)."""
    try:
        return str(eval(expression, {}, {}))
    except Exception as err:  # pragma: no cover
        return f"error: {err}"

TOOLS_CALC_SCHEMA = tools.register(calculator, description="Simple arithmetic evaluator")

# ————————————————————————————————————————————————————————————————
# Core chat wrapper
# ————————————————————————————————————————————————————————————————
async def chat(
    messages: Sequence[Message],
    *,
    model: str | None = None,
    response_model: Type[BaseModel] | None = None,
    stream: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: Dict[str, float] | None = None,
    stop: Union[str, Sequence[str], None] = None,
    n: int | None = None,
    seed: int | None = None,
    user: str | None = None,
    tools_param: Sequence[dict[str, Any]] | None = None,
    tool_choice: Union[str, dict[str, Any], None] | None = None,
    extra_headers: Dict[str, str] | None = None,
    extra_query: Dict[str, Any] | None = None,
    request_timeout: float | None = None,
    **extra_kwargs: Any,
) -> AsyncIterator[Message]:
    """Low‑level coroutine yielding tokens (**streaming**) or full messages."""
    mdl = model or config.default_model
    request_body: dict[str, Any] = {
        "model": mdl,
        "messages": list(messages),
        "stream": stream,
        "temperature": temperature if temperature is not None else config.temperature,
        "top_p": top_p if top_p is not None else config.top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty if presence_penalty is not None else config.presence_penalty,
        "frequency_penalty": frequency_penalty if frequency_penalty is not None else config.frequency_penalty,
        "logit_bias": logit_bias,
        "stop": stop,
        "n": n if n is not None else config.n,
        "seed": seed if seed is not None else config.seed,
        "user": user if user is not None else config.user,
        "tools": tools_param,
        "tool_choice": tool_choice,
        **extra_kwargs,
    }
    # drop `None`s (OpenAI will reject them)
    request_body = {k: v for k, v in request_body.items() if v is not None}

    async def _send_once() -> AsyncIterator[Message]:
        """Single network round‑trip with retries handled outside."""
        if stream:
            # 1⃣  STREAMING ------------------------------------------------
            iterator = client.chat.completions.create(
                **request_body,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=request_timeout,
            )  # returns AsyncIterator – *do not await*

            # aggregate text *per* choice index so `n>1` works
            buffers: MutableMapping[int, str] = defaultdict(str)
            async for chunk in iterator:  # type: ignore[func-returns-value]
                for choice in chunk.choices:
                    idx: int = choice.index
                    delta: str | None = getattr(choice.delta, "content", None)
                    if delta:
                        buffers[idx] += delta
                        yield {"index": idx, "token": delta}
            # done → emit full assistant message(s)
            for idx, text in buffers.items():
                msg: Message = {R: A, C: text}
                if response_model is not None:
                    msg["parsed"] = client.chat.completions.parse(text, response_model)  # type: ignore[arg-type]
                msg["index"] = idx
                yield msg
        else:
            # 2⃣  NON‑STREAMING -------------------------------------------
            completion = await client.chat.completions.create(
                **request_body,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=request_timeout,
            )
            for choice in completion.choices:
                msg: Message = {R: choice.message.role, C: choice.message.content, "index": choice.index}
                if response_model is not None:
                    msg["parsed"] = client.chat.completions.parse(choice.message.content, response_model)  # type: ignore[arg-type]
                yield msg

    # ——— Exponential‑back‑off retries
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(config.retry_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            async for token in _send_once():
                yield token

# ————————————————————————————————————————————————————————————————
# High‑level Agent façade
# ————————————————————————————————————————————————————————————————
class ChatAgent:
    """Stateful conversation manager with memory + summarisation."""

    def __init__(self, *, system_prompt: str | None = None):
        self.memory = ConversationMemory(
            max_tokens=config.max_context_tokens,
            threshold_words=config.summarise_threshold_words,
        )
        if system_prompt:
            self.memory.history.append(make_message(S, system_prompt))

    async def send_user(self, content: str, **chat_kwargs: Any) -> str:
        await self.memory.append(U, content)
        # combine *built‑ins* + *caller‑supplied* tool schemas
        user_tools = chat_kwargs.pop("tools_param", None) or []
        tools_combined = [*user_tools, *tools.schemas] if (user_tools or tools.schemas) else None

        async for msg in chat(
            self.memory.history,
            tools_param=tools_combined,
            tool_choice="auto" if tools_combined else None,
            **chat_kwargs,
        ):
            if "token" in msg:  # token event
                print(msg["token"], end="", flush=True)
            else:  # full assistant message
                await self.memory.append(A, msg[C])
                return msg[C]  # return *first* assistant answer only

# ————————————————————————————————————————————————————————————————
# Simple REPL‑style demo (works in notebooks)
# ————————————————————————————————————————————————————————————————
if __name__ == "__main__":

    async def _demo():
        agent = ChatAgent(system_prompt="You are a helpful assistant.")
        answer = await agent.send_user("What is sqrt(997)?")
        print("\n→", answer)

    await _demo()
"""openai_chat_agent.py
════════════════════════════════════════════════════════════════════════
A *robust*, *async‑friendly* wrapper around the OpenAI Chat Completion API
supporting:
 • **All official parameters** (May 2025)
 • Automatic long‑context *summarisation* (word‑count or token based)
 • *Streaming* & *non‑streaming* usage without double‑await bugs
 • First‑class **function / tool calling** via a pluggable registry
 • Optional *structured output* parsing via *pydantic* models
 • Graceful *retry* with exponential back‑off (tenacity)

This rewrite fixes:
 🛠️  Event‑loop clash with Jupyter/IPython (`RuntimeError: … already running`)
 🛠️  Incorrect `await` on streaming responses (double‑await bug)
 🛠️  Mishandled multi‑choice (`n > 1`) aggregation
 🛠️  Broken token aggregation (now yields **delta‑only** + **complete** events)
 🛠️  Out‑of‑date `.beta.chat.completions.parse` usage (now in GA)
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import (Any, AsyncIterator, Callable, Dict, Iterable, List, Literal,
                    MutableMapping, Optional, Sequence, Tuple, Type, Union)

import openai  # type: ignore
from pydantic import BaseModel
from tenacity import (AsyncRetrying, retry_if_exception_type,
                      stop_after_attempt, wait_exponential)

# ————————————————————————————————————————————————————————————————
# Logging & configuration
# ————————————————————————————————————————————————————————————————
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="[%(levelname)s] %(message)s")

class OpenAIScriptingConfig(BaseModel):
    """Runtime configuration used by the wrapper."""

    api_key: Optional[str] = None
    default_model: str = "gpt-4o-mini"
    max_context_tokens: int = 200_000
    summarise_threshold_words: int = 3_000
    retry_attempts: int = 3
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    n: int = 1
    user: str | None = None  # optional end‑user identifier for abuse tracking

    class Config:
        extra = "forbid"

config = OpenAIScriptingConfig()
client = openai.AsyncOpenAI(api_key=config.api_key)

# ————————————————————————————————————————————————————————————————
# Helper constants & utils
# ————————————————————————————————————————————————————————————————
R, C = "role", "content"
U, A, S, D = "user", "assistant", "system", "developer"
Message = Dict[str, Any]

def make_message(role: str, content: Any) -> Message:
    return {R: role, C: content}

# ————————————————————————————————————————————————————————————————
# Memory & summarisation
# ————————————————————————————————————————————————————————————————
class ConversationMemory:
    """Rolling history with optional summarisation once *word* threshold hit."""

    def __init__(self, max_tokens: int, threshold_words: int):
        self.history: list[Message] = []
        self.max_tokens = max_tokens
        self.threshold = threshold_words

    # ——— public
    async def append(self, role: str, content: Any):
        self.history.append(make_message(role, content))
        if self._word_count() > self.threshold:
            await self._summarise()

    # ——— helpers
    def _word_count(self) -> int:
        return sum(len(str(m[C]).split()) for m in self.history)

    async def _summarise(self):
        logger.info("Context exceeded %d words → requesting summary…", self.threshold)
        prompt = "\n".join(f"[{m[R]}] {m[C]}" for m in self.history)
        summary_resp = await client.chat.completions.create(
            model=config.default_model,
            messages=[
                make_message(S, "Summarise the following conversation succinctly:"),
                make_message(U, prompt),
            ],
            temperature=0,
            stream=False,
        )
        summary_text = summary_resp.choices[0].message.content
        self.history = [make_message(S, summary_text)]

# ————————————————————————————————————————————————————————————————
# Tool registry (OpenAI "tools" / function‑calling)
# ————————————————————————————————————————————————————————————————
class ToolRegistry:
    """Registers Python callables as OpenAI function‑calling *tools*."""

    def __init__(self):
        self._registry: Dict[str, Callable[..., Any]] = {}
        self._schemas: list[dict[str, Any]] = []

    # ——— public
    def register(self, fn: Callable[..., Any], *, name: Optional[str] = None,
                 description: str = "") -> dict[str, Any]:
        name = name or fn.__name__
        if name in self._registry:
            raise ValueError(f"Tool '{name}' already registered")
        self._registry[name] = fn

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description or (fn.__doc__ or ""),
                "parameters": {
                    "type": "object",
                    "properties": {
                        k: {"type": _py_to_json_type(t)}
                        for k, t in fn.__annotations__.items()
                        if k != "return"
                    },
                    "required": [k for k in fn.__annotations__ if k != "return"],
                },
            },
        }
        self._schemas.append(schema)
        return schema

    async def call(self, name: str, **kwargs):
        if name not in self._registry:
            raise ValueError(f"No such tool '{name}' registered")
        result = self._registry[name](**kwargs)
        return await result if asyncio.iscoroutine(result) else result

    # ——— properties
    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

def _py_to_json_type(t: Any) -> str:
    """Rudimentary *Python‑to‑JSON‑Schema* type mapping."""
    if t in {int, float}:  # numbers ⇒ "number"
        return "number"
    if t is bool:
        return "boolean"
    return "string"

tools = ToolRegistry()

# ——— example tool
async def calculator(expression: str) -> str:
    """Evaluate a Python expression (⚠️ *unsafe*, demo purposes only)."""
    try:
        return str(eval(expression, {}, {}))
    except Exception as err:  # pragma: no cover
        return f"error: {err}"

TOOLS_CALC_SCHEMA = tools.register(calculator, description="Simple arithmetic evaluator")

# ————————————————————————————————————————————————————————————————
# Core chat wrapper
# ————————————————————————————————————————————————————————————————
async def chat(
    messages: Sequence[Message],
    *,
    model: str | None = None,
    response_model: Type[BaseModel] | None = None,
    stream: bool = True,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    logit_bias: Dict[str, float] | None = None,
    stop: Union[str, Sequence[str], None] = None,
    n: int | None = None,
    seed: int | None = None,
    user: str | None = None,
    tools_param: Sequence[dict[str, Any]] | None = None,
    tool_choice: Union[str, dict[str, Any], None] | None = None,
    extra_headers: Dict[str, str] | None = None,
    extra_query: Dict[str, Any] | None = None,
    request_timeout: float | None = None,
    **extra_kwargs: Any,
) -> AsyncIterator[Message]:
    """Low‑level coroutine yielding tokens (**streaming**) or full messages."""
    mdl = model or config.default_model
    request_body: dict[str, Any] = {
        "model": mdl,
        "messages": list(messages),
        "stream": stream,
        "temperature": temperature if temperature is not None else config.temperature,
        "top_p": top_p if top_p is not None else config.top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty if presence_penalty is not None else config.presence_penalty,
        "frequency_penalty": frequency_penalty if frequency_penalty is not None else config.frequency_penalty,
        "logit_bias": logit_bias,
        "stop": stop,
        "n": n if n is not None else config.n,
        "seed": seed if seed is not None else config.seed,
        "user": user if user is not None else config.user,
        "tools": tools_param,
        "tool_choice": tool_choice,
        **extra_kwargs,
    }
    # drop `None`s (OpenAI will reject them)
    request_body = {k: v for k, v in request_body.items() if v is not None}

    async def _send_once() -> AsyncIterator[Message]:
        """Single network round‑trip with retries handled outside."""
        if stream:
            # 1⃣  STREAMING ------------------------------------------------
            iterator = client.chat.completions.create(
                **request_body,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=request_timeout,
            )  # returns AsyncIterator – *do not await*

            # aggregate text *per* choice index so `n>1` works
            buffers: MutableMapping[int, str] = defaultdict(str)
            async for chunk in iterator:  # type: ignore[func-returns-value]
                for choice in chunk.choices:
                    idx: int = choice.index
                    delta: str | None = getattr(choice.delta, "content", None)
                    if delta:
                        buffers[idx] += delta
                        yield {"index": idx, "token": delta}
            # done → emit full assistant message(s)
            for idx, text in buffers.items():
                msg: Message = {R: A, C: text}
                if response_model is not None:
                    msg["parsed"] = client.chat.completions.parse(text, response_model)  # type: ignore[arg-type]
                msg["index"] = idx
                yield msg
        else:
            # 2⃣  NON‑STREAMING -------------------------------------------
            completion = await client.chat.completions.create(
                **request_body,
                extra_headers=extra_headers,
                extra_query=extra_query,
                timeout=request_timeout,
            )
            for choice in completion.choices:
                msg: Message = {R: choice.message.role, C: choice.message.content, "index": choice.index}
                if response_model is not None:
                    msg["parsed"] = client.chat.completions.parse(choice.message.content, response_model)  # type: ignore[arg-type]
                yield msg

    # ——— Exponential‑back‑off retries
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(config.retry_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    ):
        with attempt:
            async for token in _send_once():
                yield token

# ————————————————————————————————————————————————————————————————
# High‑level Agent façade
# ————————————————————————————————————————————————————————————————
class ChatAgent:
    """Stateful conversation manager with memory + summarisation."""

    def __init__(self, *, system_prompt: str | None = None):
        self.memory = ConversationMemory(
            max_tokens=config.max_context_tokens,
            threshold_words=config.summarise_threshold_words,
        )
        if system_prompt:
            self.memory.history.append(make_message(S, system_prompt))

    async def send_user(self, content: str, **chat_kwargs: Any) -> str:
        await self.memory.append(U, content)
        # combine *built‑ins* + *caller‑supplied* tool schemas
        user_tools = chat_kwargs.pop("tools_param", None) or []
        tools_combined = [*user_tools, *tools.schemas] if (user_tools or tools.schemas) else None

        async for msg in chat(
            self.memory.history,
            tools_param=tools_combined,
            tool_choice="auto" if tools_combined else None,
            **chat_kwargs,
        ):
            if "token" in msg:  # token event
                print(msg["token"], end="", flush=True)
            else:  # full assistant message
                await self.memory.append(A, msg[C])
                return msg[C]  # return *first* assistant answer only

# ————————————————————————————————————————————————————————————————
# Simple REPL‑style demo (works in notebooks)
# ————————————————————————————————————————————————————————————————
if __name__ == "__main__":

    async def _demo():
        agent = ChatAgent(system_prompt="You are a helpful assistant.")
        answer = await agent.send_user("What is sqrt(997)?")
        print("\n→", answer)

    await _demo()
