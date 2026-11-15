#!/usr/bin/env python3
"""Free-model aggregator for make.fun's free-model-aggregator-01 bounty.

One server, two client wires (OpenAI Chat Completions and Anthropic
Messages, JSON and SSE), over ten simulated free-tier providers.  It reads
the trial brief at $MAKEFUN_PROMPT_FILE, probes every provider's catalog
route, registers its port and catalog with the env at $MAKEFUN_ENV_URL,
and answers the env client's 20 requests, each with the best free model
that meets the request's requirements, failing over to the next best
model when a provider refuses (402/429/5xx).

Python 3.13 standard library only; one file.  `python3 main.py --list`
prints the catalog document and exits 0.

Routing, in one paragraph: candidates are every free model meeting the
request's requirements, ranked by the sealed quality table's
rating_milli (highest first, ties broken by provider id then model id).
A refusal is retried once on the same candidate, in the same request,
before falling back: the env refuses by call number, so a one-off
fail_call is followed by a call number that succeeds, while a provider
whose budget is spent refuses the retry too and is then skipped exactly
as before.  No sleep on a retry: the env has no time component, so
waiting cannot change the answer.  A provider that just failed is
skipped for the next request (one-off failure schedules are common; a
provider that fails three times in a row is treated as spent/out and
skipped while any other candidate remains).  JSON-mode requests on the
OpenAI wire never go to an Anthropic-wire provider, because that wire
cannot carry response_format and its echo would report json_mode
absent.  Exactly one provider call may succeed per request: a success
is returned, never retried.

Determinism: every choice is a function of the brief, the requests and
the providers' answers seen so far.  No wall clock and no randomness
reach any request, response or registration.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
UPSTREAM_TIMEOUT_S = 15
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def dumps(obj):
    """Compact JSON, the same bytes for --list and for the register body."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------

def _object_end(text, start):
    """Index of the '}' closing the object at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` that carries a string `schema` member."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if isinstance(value, dict) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _as_str(value):
    return value if isinstance(value, str) else ""


def _as_num_or_none(value):
    return value if isinstance(value, (int, float)) and value == value else None


def _as_bool_or_none(value):
    return value if isinstance(value, bool) else None


def _provider_of(value):
    if not isinstance(value, dict):
        return None
    pid = _as_str(value.get("id"))
    wire = value.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = value.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = (
        _as_str(value.get("path"))
        or _as_str(value.get("base_path"))
        or _as_str(value.get("base_url"))
        or _as_str(value.get("base"))
    )
    return {
        "id": pid,
        "name": _as_str(value.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _as_str(value.get("key_env")),
        "key": _as_str(value.get("key")),
    }


def _rating_of(value):
    if not isinstance(value, dict):
        return None
    model = _as_str(value.get("model"))
    provider = _as_str(value.get("provider"))
    if model == "" or provider == "":
        return None
    rating = value.get("rating_milli")
    return {
        "model": model,
        "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) else 0,
        "context_length": _as_num_or_none(value.get("context_length")),
        "tools": value.get("tools") is True,
        "vision": value.get("vision") is True,
        "json": value.get("json") is True,
        "free": value.get("free") is True,
    }


def _workload_request_of(value):
    if not isinstance(value, dict):
        return None
    rid = _as_str(value.get("id"))
    if rid == "":
        return None
    requires = value.get("requires") if isinstance(value.get("requires"), dict) else {}
    req = {}
    if requires.get("tools") is True:
        req["tools"] = True
    if requires.get("vision") is True:
        req["vision"] = True
    if requires.get("json") is True:
        req["json"] = True
    mc = requires.get("min_context")
    if isinstance(mc, (int, float)):
        req["min_context"] = mc
    return {
        "id": rid,
        "api": "anthropic" if value.get("api") == "anthropic" else "openai",
        "stream": value.get("stream") is True,
        "weight": 2 if value.get("weight") == 2 else 1,
        "requires": req,
    }


def read_brief(path):
    """The brief's three documents, found by schema. Never raises."""
    brief = {"providers": [], "ratings": None, "workload": None, "error": None}
    if not path:
        brief["error"] = "MAKEFUN_PROMPT_FILE is not set"
        return brief
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:  # noqa: BLE001 - brief may be absent outside a trial
        brief["error"] = "cannot read %s: %s" % (path, exc)
        return brief
    docs = _documents_in(text)
    directory = next(
        (d for d in docs if d.get("schema") == PROVIDERS_SCHEMA and isinstance(d.get("providers"), list)),
        None,
    )
    if directory is None:
        brief["error"] = "the brief holds no %s document" % PROVIDERS_SCHEMA
        return brief
    brief["providers"] = [p for p in (_provider_of(v) for v in directory["providers"]) if p is not None]
    ratings_doc = next(
        (d for d in docs if d.get("schema") == RATINGS_SCHEMA and isinstance(d.get("ratings"), list)),
        None,
    )
    if ratings_doc is not None:
        rows = [r for r in (_rating_of(v) for v in ratings_doc["ratings"]) if r is not None]
        best = []
        for v in ratings_doc.get("best") or []:
            if isinstance(v, dict) and _as_str(v.get("id")):
                best.append({"id": v["id"], "rating_milli": v.get("rating_milli") or 0})
        brief["ratings"] = {"ratings": rows, "best": best}
    workload_doc = next(
        (d for d in docs if d.get("schema") == WORKLOAD_SCHEMA and isinstance(d.get("manifest"), list)),
        None,
    )
    if workload_doc is not None:
        brief["workload"] = {
            "manifest": [w for w in (_workload_request_of(v) for v in workload_doc["manifest"]) if w is not None]
        }
    return brief


# --------------------------------------------------------------------------
# Catalog probe
# --------------------------------------------------------------------------

def auth_headers(auth, key):
    if not key:
        return {}
    if auth == "bearer":
        return {"Authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def join_base(env_url, base):
    base = (base or "").strip()
    if base.lower().startswith("http://") or base.lower().startswith("https://"):
        return base.rstrip("/")
    prefix = (env_url or "").rstrip("/")
    if base == "":
        return prefix
    return prefix + ("" if base.startswith("/") else "/") + base.rstrip("/")


def _zero_priced(value):
    pricing = value.get("pricing") if isinstance(value, dict) else None
    if not isinstance(pricing, dict):
        return None
    numbers = []
    for item in pricing.values():
        if isinstance(item, (int, float)):
            numbers.append(item)
        elif isinstance(item, str) and item.strip() != "":
            try:
                numbers.append(float(item))
            except ValueError:
                pass
    if not numbers:
        return None
    return all(n == 0 for n in numbers)


def _model_in(wire, value):
    if not isinstance(value, dict):
        return None
    if wire == "gemini":
        model_id = _as_str(value.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
    else:
        model_id = _as_str(value.get("id"))
    if model_id == "":
        return None
    context = None
    for key in ("context_length", "context_window", "inputTokenLimit"):
        context = _as_num_or_none(value.get(key))
        if context is not None:
            break
    return {
        "id": model_id,
        "display_name": _as_str(value.get("display_name")) or _as_str(value.get("displayName")) or model_id,
        "free": _as_bool_or_none(value.get("free")) if "free" in value else _zero_priced(value),
        "context_length": context,
        "tools": _as_bool_or_none(value.get("tools")),
        "vision": _as_bool_or_none(value.get("vision")),
        "json": _as_bool_or_none(value.get("json")),
    }


def _models_in(wire, body):
    if not isinstance(body, dict):
        return []
    raw = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(raw, list):
        return []
    return [m for m in (_model_in(wire, v) for v in raw) if m is not None]


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """One provider's catalog route, in its own wire. Never raises."""
    base = join_base(env_url, provider["base"])
    url = "%s/v1beta/models" % base if provider["wire"] == "gemini" else "%s/v1/models" % base
    headers = {"Accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return {"provider": provider, "ok": False, "models": [], "error": "GET %s answered HTTP %s" % (url, resp.status)}
            body = json.loads(resp.read().decode("utf-8"))
        return {"provider": provider, "ok": True, "models": _models_in(provider["wire"], body), "error": None}
    except Exception as exc:  # noqa: BLE001 - a failed probe leaves the provider out
        return {"provider": provider, "ok": False, "models": [], "error": "GET %s failed: %s" % (url, exc)}


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] is True


def capabilities_of(model, row):
    return {
        "context_length": (row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]),
        "tools": (row["tools"] if row is not None else model["tools"]),
        "vision": (row["vision"] if row is not None else model["vision"]),
        "json": (row["json"] if row is not None else model["json"]),
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    min_ctx = requires.get("min_context")
    if min_ctx is not None and (caps["context_length"] is None or caps["context_length"] < min_ctx):
        return False
    return True


def build_catalog(probes, ratings):
    """The one catalog document: providers in brief order, models by id."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            ctx = row["context_length"] if row is not None and row["context_length"] is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row), "context_length": ctx})
        providers.append(
            {
                "id": probe["provider"]["id"],
                "base_url": probe["provider"]["base"],
                "key_env": probe["provider"]["key_env"],
                "wire": probe["provider"]["wire"],
                "models": models,
            }
        )
    return {"providers": providers}


# --------------------------------------------------------------------------
# Wire translation helpers
# --------------------------------------------------------------------------

def _text_of(value):
    """Plain text from an Anthropic-style `content`: string, or blocks."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_as_str(b.get("text")) for b in value if isinstance(b, dict) and b.get("type") == "text")
    return ""


def _stop_list(stop):
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str) and s != ""]
    return []


def _data_url_parts(image_url):
    """Split an OpenAI image_url data URL into (media_type, base64 data)."""
    if not isinstance(image_url, str) or not image_url.startswith("data:"):
        return None
    comma = image_url.find(",")
    if comma == -1:
        return None
    header = image_url[5:comma]
    if ";base64" not in header:
        return None
    return header.split(";")[0] or "image/png", image_url[comma + 1:]


def _anthropic_messages_from_openai(messages):
    """OpenAI messages -> (system text, Anthropic messages)."""
    system_parts = []
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            system_parts.append(_text_of(msg.get("content")))
            continue
        if role == "tool":
            call_id = _as_str(msg.get("tool_call_id"))
            blocks = [{
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": _text_of(msg.get("content")),
            }]
            out.append({"role": "user", "content": blocks})
            continue
        if role == "user":
            content = msg.get("content")
            blocks = []
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        blocks.append({"type": "text", "text": _as_str(part.get("text"))})
                    elif part.get("type") == "image_url":
                        split = _data_url_parts((part.get("image_url") or {}).get("url"))
                        if split is not None:
                            blocks.append({"type": "image", "source": {"type": "base64", "media_type": split[0], "data": split[1]}})
                        else:
                            blocks.append({"type": "image", "source": {"type": "url", "url": _as_str((part.get("image_url") or {}).get("url"))}})
            out.append({"role": "user", "content": blocks})
            continue
        if role == "assistant":
            content = msg.get("content")
            blocks = []
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        blocks.append({"type": "text", "text": _as_str(part.get("text"))})
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                blocks.append({
                    "type": "tool_use",
                    "id": _as_str(call.get("id")),
                    "name": _as_str(fn.get("name")),
                    "input": args,
                })
            out.append({"role": "assistant", "content": blocks})
    return "\n".join(p for p in system_parts if p != ""), out


def _openai_messages_from_anthropic(system, messages):
    """Anthropic (system, messages) -> OpenAI messages. Port of starter."""
    out = []
    if isinstance(system, str) and system != "":
        out.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = _text_of(system)
        if text != "":
            out.append({"role": "system", "content": text})
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if msg.get("role") == "assistant":
            text_parts = []
            calls = []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text_parts.append(_as_str(block.get("text")))
                elif block.get("type") == "tool_use":
                    calls.append({
                        "id": _as_str(block.get("id")),
                        "type": "function",
                        "function": {"name": _as_str(block.get("name")), "arguments": dumps(block.get("input") or {})},
                    })
            msg_out = {"role": "assistant"}
            if text_parts:
                msg_out["content"] = "".join(text_parts)
            if calls:
                msg_out["tool_calls"] = calls
            if "content" not in msg_out:
                msg_out["content"] = None
            out.append(msg_out)
            continue
        if msg.get("role") == "user":
            text_parts = []
            images = []
            tool_results = []
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(_as_str(block.get("text")))
                    elif block.get("type") == "image":
                        source = block.get("source") or {}
                        if source.get("type") == "base64":
                            images.append({
                                "type": "image_url",
                                "image_url": {"url": "data:%s;base64,%s" % (source.get("media_type") or "image/png", source.get("data") or "")},
                            })
                        elif source.get("type") == "url":
                            images.append({"type": "image_url", "image_url": {"url": _as_str(source.get("url"))}})
                    elif block.get("type") == "tool_result":
                        tool_results.append({
                            "role": "tool",
                            "tool_call_id": _as_str(block.get("tool_use_id")),
                            "content": _text_of(block.get("content")),
                        })
            for tool_msg in tool_results:
                out.append(tool_msg)
            if text_parts or images:
                parts = [{"type": "text", "text": "".join(text_parts)}] + images if text_parts else images
                if len(parts) == 1 and parts[0].get("type") == "text":
                    out.append({"role": "user", "content": parts[0]["text"]})
                else:
                    out.append({"role": "user", "content": parts})
    return out


def _anthropic_tool_choice_from_openai(choice):
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {"type": "tool", "name": _as_str(choice.get("function", {}).get("name"))}
    if choice == "required" or choice == "any":
        return {"type": "any"}
    return {"type": "auto"}


def _openai_tool_choice_from_anthropic(choice):
    if isinstance(choice, dict):
        if choice.get("type") == "tool":
            return {"type": "function", "function": {"name": _as_str(choice.get("name"))}}
        if choice.get("type") == "any":
            return "required"
    return "auto"


def _openai_tools_from_anthropic(tools):
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        out.append({
            "type": "function",
            "function": {
                "name": _as_str(tool.get("name")),
                "description": _as_str(tool.get("description")),
                "parameters": tool.get("input_schema") or {},
            },
        })
    return out


def _anthropic_tools_from_openai(tools):
    out = []
    for tool in tools or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        out.append({
            "name": _as_str(fn.get("name")),
            "description": _as_str(fn.get("description")),
            "input_schema": fn.get("parameters") or {},
        })
    return out


def _max_tokens_from(body):
    value = body.get("max_tokens")
    if value is None:
        value = body.get("max_completion_tokens")
    return value if isinstance(value, (int, float)) else None


def _openai_body_from_anthropic(body, model_id, stream):
    """Anthropic client body -> OpenAI upstream body."""
    out = {
        "model": model_id,
        "messages": _openai_messages_from_anthropic(body.get("system"), body.get("messages") or []),
        "stream": stream,
    }
    tools = _openai_tools_from_anthropic(body.get("tools"))
    if tools:
        out["tools"] = tools
        if body.get("tool_choice") is not None:
            out["tool_choice"] = _openai_tool_choice_from_anthropic(body.get("tool_choice"))
    if stream:
        # The provider only attaches the usage chunk when asked; without
        # it an Anthropic-stream client cannot report usage at all.
        out["stream_options"] = {"include_usage": True}
    max_tokens = _max_tokens_from(body)
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    stops = _stop_list(body.get("stop_sequences"))
    if stops:
        out["stop"] = stops
    return out


def _anthropic_body_from_openai(body, model_id, stream):
    """OpenAI client body -> Anthropic upstream body."""
    system, messages = _anthropic_messages_from_openai(body.get("messages") or [])
    out = {"model": model_id, "messages": messages, "stream": stream}
    if system != "":
        out["system"] = system
    tools = _anthropic_tools_from_openai(body.get("tools"))
    if tools:
        out["tools"] = tools
        if body.get("tool_choice") is not None:
            out["tool_choice"] = _anthropic_tool_choice_from_openai(body.get("tool_choice"))
    max_tokens = _max_tokens_from(body)
    if max_tokens is not None:
        out["max_tokens"] = max_tokens
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    stops = _stop_list(body.get("stop"))
    if stops:
        out["stop_sequences"] = stops
    return out


def _gemini_part_from_openai(part):
    if not isinstance(part, dict):
        return None
    if part.get("type") == "text":
        return {"text": _as_str(part.get("text"))}
    if part.get("type") == "image_url":
        url = (part.get("image_url") or {}).get("url") or ""
        split = _data_url_parts(url)
        if split is not None:
            return {"inlineData": {"mimeType": split[0], "data": split[1]}}
        return {"fileData": {"fileUri": url}}
    return None


def _gemini_contents_from_openai(messages):
    system_parts = []
    contents = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "system":
            system_parts.append(_text_of(msg.get("content")))
            continue
        if role == "tool":
            contents.append({
                "role": "user",
                "parts": [{
                    "functionResponse": {
                        "name": "",
                        "response": {"result": _text_of(msg.get("content"))},
                    },
                }],
            })
            continue
        if role == "user":
            parts = []
            content = msg.get("content")
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    piece = _gemini_part_from_openai(part)
                    if piece is not None:
                        parts.append(piece)
            contents.append({"role": "user", "parts": parts})
            continue
        if role == "assistant":
            parts = []
            content = msg.get("content")
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append({"text": _as_str(part.get("text"))})
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    args = {}
                parts.append({"functionCall": {"name": _as_str(fn.get("name")), "args": args}})
            contents.append({"role": "model", "parts": parts})
    return "\n".join(p for p in system_parts if p != ""), contents


def _gemini_contents_from_anthropic(system, messages):
    system_text = _text_of(system)
    contents = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if msg.get("role") == "assistant":
            parts = []
            for block in content if isinstance(content, list) else []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    parts.append({"text": _as_str(block.get("text"))})
                elif block.get("type") == "tool_use":
                    parts.append({"functionCall": {"name": _as_str(block.get("name")), "args": block.get("input") or {}}})
            contents.append({"role": "model", "parts": parts})
            continue
        if msg.get("role") == "user":
            parts = []
            responses = []
            if isinstance(content, str):
                parts.append({"text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        parts.append({"text": _as_str(block.get("text"))})
                    elif block.get("type") == "image":
                        source = block.get("source") or {}
                        if source.get("type") == "base64":
                            parts.append({"inlineData": {"mimeType": source.get("media_type") or "image/png", "data": source.get("data") or ""}})
                        elif source.get("type") == "url":
                            parts.append({"fileData": {"fileUri": _as_str(source.get("url"))}})
                    elif block.get("type") == "tool_result":
                        responses.append({
                            "functionResponse": {
                                "name": "",
                                "response": {"result": _text_of(block.get("content"))},
                            },
                        })
            if parts:
                contents.append({"role": "user", "parts": parts})
            for response in responses:
                contents.append({"role": "user", "parts": [response]})
    return system_text, contents


def _gemini_body_from_openai(body, stream):
    """OpenAI client body -> Gemini upstream body (model rides in the URL)."""
    system, contents = _gemini_contents_from_openai(body.get("messages") or [])
    out = {"contents": contents}
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = _anthropic_tools_from_openai(body.get("tools"))
    if tools:
        out["tools"] = [{"functionDeclarations": tools}]
    config = {}
    max_tokens = _max_tokens_from(body)
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    stops = _stop_list(body.get("stop"))
    if stops:
        config["stopSequences"] = stops
    if body.get("temperature") is not None:
        config["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        config["topP"] = body["top_p"]
    response_format = body.get("response_format") or {}
    if response_format.get("type") in ("json_object", "json_schema"):
        config["responseMimeType"] = "application/json"
    if config:
        out["generationConfig"] = config
    return out


def _gemini_body_from_anthropic(body, stream):
    """Anthropic client body -> Gemini upstream body (model rides in the URL)."""
    system, contents = _gemini_contents_from_anthropic(body.get("system"), body.get("messages") or [])
    out = {"contents": contents}
    if system != "":
        out["systemInstruction"] = {"parts": [{"text": system}]}
    tools = body.get("tools") or []
    if isinstance(tools, list) and tools:
        out["tools"] = [{"functionDeclarations": tools}]
    config = {}
    max_tokens = _max_tokens_from(body)
    if max_tokens is not None:
        config["maxOutputTokens"] = max_tokens
    stops = _stop_list(body.get("stop_sequences"))
    if stops:
        config["stopSequences"] = stops
    if body.get("temperature") is not None:
        config["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        config["topP"] = body["top_p"]
    if config:
        out["generationConfig"] = config
    return out


# --------------------------------------------------------------------------
# Upstream call
# --------------------------------------------------------------------------

def _gemini_url(base, model_id, stream):
    from urllib.parse import quote

    action = "streamGenerateContent" if stream else "generateContent"
    return "%s/v1beta/models/%s:%s" % (base, quote(model_id, safe=""), action)


def build_upstream_request(client_api, body, stream, probe, model_id, json_mode):
    """(url, headers, payload dict) for one candidate, or None if unroutable."""
    provider = probe["provider"]
    base = join_base(ENV_URL, provider["base"])
    wire = provider["wire"]
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if wire == "openai":
        if client_api == "openai":
            payload = dict(body)
            payload["model"] = model_id
            payload["stream"] = stream
            if stream:
                opts = dict(payload.get("stream_options") or {})
                opts["include_usage"] = True
                payload["stream_options"] = opts
        else:
            payload = _openai_body_from_anthropic(body, model_id, stream)
        return "%s/v1/chat/completions" % base, headers, payload
    if wire == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        if client_api == "openai":
            payload = _anthropic_body_from_openai(body, model_id, stream)
        else:
            payload = dict(body)
            payload["model"] = model_id
            payload["stream"] = stream
        return "%s/v1/messages" % base, headers, payload
    # gemini
    if client_api == "openai":
        payload = _gemini_body_from_openai(body, stream)
    else:
        payload = _gemini_body_from_anthropic(body, stream)
    return _gemini_url(base, model_id, stream), headers, payload


def call_upstream(url, headers, payload, timeout=UPSTREAM_TIMEOUT_S):
    """(status, raw bytes, error) for one upstream call. Never raises."""
    try:
        req = urllib.request.Request(
            url, data=dumps(payload).encode("utf-8"), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), None
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return exc.code, body, "HTTP %s" % exc.code
    except Exception as exc:  # noqa: BLE001 - any transport failure is a failover
        return 0, b"", str(exc)


# --------------------------------------------------------------------------
# Response translation: upstream -> client
# --------------------------------------------------------------------------

def _openai_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_as_str(p.get("text")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def _anthropic_content_blocks(content, tool_calls):
    blocks = []
    if content:
        blocks.append({"type": "text", "text": content})
    for call in tool_calls or []:
        fn = (call or {}).get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        blocks.append({"type": "tool_use", "id": _as_str(call.get("id")), "name": _as_str(fn.get("name")), "input": args})
    return blocks


def _openai_finish_from_anthropic(stop_reason):
    return {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}.get(stop_reason or "", "stop")


def _anthropic_stop_from_openai(finish_reason):
    return {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "refusal"}.get(finish_reason or "", "end_turn")


def _openai_finish_from_gemini(finish_reason):
    return {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}.get(finish_reason or "STOP", "stop")


def _anthropic_stop_from_gemini(finish_reason):
    return {"STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "refusal"}.get(finish_reason or "STOP", "end_turn")


def _gemini_text_and_usage(body):
    text = ""
    usage = {"prompt": 0, "output": 0}
    for cand in body.get("candidates") or []:
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            if isinstance(part, dict) and "text" in part:
                text += _as_str(part.get("text"))
    meta = body.get("usageMetadata") or {}
    usage["prompt"] = meta.get("promptTokenCount") or 0
    usage["output"] = meta.get("candidatesTokenCount") or 0
    finish = None
    for cand in body.get("candidates") or []:
        if cand.get("finishReason"):
            finish = cand["finishReason"]
    return text, usage, finish


def _openai_json_from_anthropic(body, requested_model):
    choice = (body.get("content") or [])
    text = "".join(_as_str(b.get("text")) for b in choice if isinstance(b, dict) and b.get("type") == "text")
    tool_calls = []
    for block in choice:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_calls.append({
                "id": _as_str(block.get("id")),
                "type": "function",
                "function": {"name": _as_str(block.get("name")), "arguments": dumps(block.get("input") or {})},
            })
    usage = body.get("usage") or {}
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-%s" % _as_str(body.get("id") or "sim"),
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": requested_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": _openai_finish_from_anthropic(body.get("stop_reason")),
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens") or 0,
            "completion_tokens": usage.get("output_tokens") or 0,
            "total_tokens": (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        },
    }


def _anthropic_json_from_openai(body, requested_model):
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = _openai_text(message.get("content"))
    tool_calls = message.get("tool_calls") or []
    usage = body.get("usage") or {}
    return {
        "id": "msg-%s" % _as_str(body.get("id") or "sim"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": _anthropic_content_blocks(text, tool_calls),
        "stop_reason": _anthropic_stop_from_openai(choice.get("finish_reason")),
        "usage": {"input_tokens": usage.get("prompt_tokens") or 0, "output_tokens": usage.get("completion_tokens") or 0},
    }


def _openai_json_from_gemini(body, model_id):
    text, usage, finish = _gemini_text_and_usage(body)
    return {
        "id": "chatcmpl-%s" % model_id,
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": _openai_finish_from_gemini(finish),
        }],
        "usage": {
            "prompt_tokens": usage["prompt"],
            "completion_tokens": usage["output"],
            "total_tokens": usage["prompt"] + usage["output"],
        },
    }


def _anthropic_json_from_gemini(body, requested_model):
    text, usage, finish = _gemini_text_and_usage(body)
    return {
        "id": "msg-%s" % _as_str(body.get("modelVersion") or "sim"),
        "type": "message",
        "role": "assistant",
        "model": requested_model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": _anthropic_stop_from_gemini(finish),
        "usage": {"input_tokens": usage["prompt"], "output_tokens": usage["output"]},
    }


def _parse_sse(raw):
    """SSE bytes -> list of data strings, in order, ignoring [DONE]."""
    text = raw.decode("utf-8", errors="replace")
    events = []
    for frame in text.split("\n\n"):
        frame = frame.strip("\n")
        if not frame:
            continue
        for line in frame.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    events.append(data)
    return events


def _openai_sse_chunk(model_id, delta, finish_reason, usage):
    chunk = {"id": "chatcmpl-%s" % model_id, "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model_id, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
    if usage is not None:
        chunk["usage"] = usage
    return ("data: %s\n\n" % dumps(chunk)).encode("utf-8")


def _openai_sse_from_anthropic(raw, model_id):
    """Anthropic SSE -> OpenAI SSE. Returns bytes, or None on bad input."""
    out = []
    input_tokens = 0
    output_tokens = 0
    started = False
    for data in _parse_sse(raw):
        try:
            event = json.loads(data)
        except Exception:
            return None
        etype = event.get("type")
        if etype == "message_start":
            input_tokens = ((event.get("message") or {}).get("usage") or {}).get("input_tokens") or 0
            if not started:
                started = True
                out.append(_openai_sse_chunk(model_id, {"role": "assistant", "content": ""}, None, None))
        elif etype == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                out.append(_openai_sse_chunk(model_id, {"content": _as_str(delta.get("text"))}, None, None))
            elif delta.get("type") == "input_json_delta":
                out.append(_openai_sse_chunk(model_id, {"tool_calls": [{"index": 0, "function": {"arguments": _as_str(delta.get("partial_json"))}}]}, None, None))
        elif etype == "message_delta":
            usage = event.get("usage") or {}
            output_tokens = usage.get("output_tokens") or 0
            stop_reason = event.get("stop_reason")
            out.append(_openai_sse_chunk(model_id, {}, _openai_finish_from_anthropic(stop_reason), None))
        elif etype == "message_stop":
            pass
    if not started:
        out.insert(0, _openai_sse_chunk(model_id, {"role": "assistant", "content": ""}, None, None))
    usage = {"prompt_tokens": input_tokens, "completion_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
    out.append(_openai_sse_chunk(model_id, {}, None, usage))
    out.append(b"data: [DONE]\n\n")
    return b"".join(out)


def _anthropic_sse_from_openai(raw, requested_model):
    """OpenAI SSE -> Anthropic SSE. Port of the starter's translation.

    Two passes: the provider's usage chunk arrives AFTER the finish
    chunk, so usage is collected up front and then used for
    message_start (input tokens) and message_delta (output tokens).
    """
    events = []
    for data in _parse_sse(raw):
        try:
            events.append(json.loads(data))
        except Exception:
            return None
    prompt_tokens = 0
    completion_tokens = 0
    for event in events:
        usage = event.get("usage") or {}
        if usage.get("prompt_tokens") is not None:
            prompt_tokens = usage["prompt_tokens"]
        if usage.get("completion_tokens") is not None:
            completion_tokens = usage["completion_tokens"]
    out = []
    started = False
    tool_index = -1
    for event in events:
        choices = event.get("choices") or []
        if event.get("usage") is not None:
            continue
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        if not started:
            started = True
            out.append(("event: message_start\ndata: %s\n\n" % dumps({
                "type": "message_start",
                "message": {
                    "id": "msg-%s" % _as_str(event.get("id") or "sim"),
                    "type": "message",
                    "role": "assistant",
                    "model": requested_model,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
                },
            })).encode("utf-8"))
            out.append(("event: content_block_start\ndata: %s\n\n" % dumps({
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            })).encode("utf-8"))
        if isinstance(delta.get("content"), str) and delta["content"] != "":
            out.append(("event: content_block_delta\ndata: %s\n\n" % dumps({
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": delta["content"]},
            })).encode("utf-8"))
        tool_calls = delta.get("tool_calls") or []
        for call in tool_calls:
            idx = call.get("index") if isinstance(call.get("index"), int) else 0
            if idx != tool_index:
                tool_index = idx
                fn = call.get("function") or {}
                out.append(("event: content_block_start\ndata: %s\n\n" % dumps({
                    "type": "content_block_start",
                    "index": idx + 1,
                    "content_block": {"type": "tool_use", "id": _as_str(call.get("id")), "name": _as_str(fn.get("name")), "input": {}},
                })).encode("utf-8"))
            fn = call.get("function") or {}
            if isinstance(fn.get("arguments"), str) and fn["arguments"] != "":
                out.append(("event: content_block_delta\ndata: %s\n\n" % dumps({
                    "type": "content_block_delta",
                    "index": idx + 1,
                    "delta": {"type": "input_json_delta", "partial_json": fn["arguments"]},
                })).encode("utf-8"))
        if choice.get("finish_reason") is not None:
            finish = choice["finish_reason"]
            out.append(("event: content_block_stop\ndata: %s\n\n" % dumps({"type": "content_block_stop", "index": 0})).encode("utf-8"))
            out.append(("event: message_delta\ndata: %s\n\n" % dumps({
                "type": "message_delta",
                "delta": {"stop_reason": _anthropic_stop_from_openai(finish), "stop_sequence": None},
                "usage": {"output_tokens": completion_tokens},
            })).encode("utf-8"))
            out.append(("event: message_stop\ndata: %s\n\n" % dumps({"type": "message_stop"})).encode("utf-8"))
    if not started:
        out.append(("event: message_start\ndata: %s\n\n" % dumps({
            "type": "message_start",
            "message": {
                "id": "msg-sim",
                "type": "message",
                "role": "assistant",
                "model": requested_model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
            },
        })).encode("utf-8"))
        out.append(("event: content_block_start\ndata: %s\n\n" % dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})).encode("utf-8"))
        out.append(("event: content_block_stop\ndata: %s\n\n" % dumps({"type": "content_block_stop", "index": 0})).encode("utf-8"))
        out.append(("event: message_delta\ndata: %s\n\n" % dumps({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": completion_tokens}})).encode("utf-8"))
        out.append(("event: message_stop\ndata: %s\n\n" % dumps({"type": "message_stop"})).encode("utf-8"))
    return b"".join(out)


def _openai_sse_from_gemini(raw, model_id):
    """Gemini SSE -> OpenAI SSE."""
    out = []
    started = False
    usage = None
    finish = None
    for data in _parse_sse(raw):
        try:
            event = json.loads(data)
        except Exception:
            return None
        text, meta, cand_finish = _gemini_text_and_usage(event)
        if cand_finish:
            finish = cand_finish
        if meta["prompt"] or meta["output"]:
            usage = {"prompt_tokens": meta["prompt"], "completion_tokens": meta["output"], "total_tokens": meta["prompt"] + meta["output"]}
        if not started:
            started = True
            out.append(_openai_sse_chunk(model_id, {"role": "assistant", "content": ""}, None, None))
        if text:
            out.append(_openai_sse_chunk(model_id, {"content": text}, None, None))
    if not started:
        out.append(_openai_sse_chunk(model_id, {"role": "assistant", "content": ""}, None, None))
    out.append(_openai_sse_chunk(model_id, {}, _openai_finish_from_gemini(finish), None))
    if usage is not None:
        out.append(_openai_sse_chunk(model_id, {}, None, usage))
    out.append(b"data: [DONE]\n\n")
    return b"".join(out)


def _anthropic_sse_from_gemini(raw, requested_model):
    """Gemini SSE -> Anthropic SSE."""
    out = []
    started = False
    output_tokens = 0
    finish = None
    for data in _parse_sse(raw):
        try:
            event = json.loads(data)
        except Exception:
            return None
        text, meta, cand_finish = _gemini_text_and_usage(event)
        if cand_finish:
            finish = cand_finish
        if meta["output"]:
            output_tokens = meta["output"]
        if not started:
            started = True
            out.append(("event: message_start\ndata: %s\n\n" % dumps({
                "type": "message_start",
                "message": {
                    "id": "msg-sim",
                    "type": "message",
                    "role": "assistant",
                    "model": requested_model,
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })).encode("utf-8"))
            out.append(("event: content_block_start\ndata: %s\n\n" % dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})).encode("utf-8"))
        if text:
            out.append(("event: content_block_delta\ndata: %s\n\n" % dumps({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}})).encode("utf-8"))
    if not started:
        out.append(("event: message_start\ndata: %s\n\n" % dumps({
            "type": "message_start",
            "message": {
                "id": "msg-sim",
                "type": "message",
                "role": "assistant",
                "model": requested_model,
                "content": [],
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })).encode("utf-8"))
        out.append(("event: content_block_start\ndata: %s\n\n" % dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})).encode("utf-8"))
    out.append(("event: content_block_stop\ndata: %s\n\n" % dumps({"type": "content_block_stop", "index": 0})).encode("utf-8"))
    out.append(("event: message_delta\ndata: %s\n\n" % dumps({
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop_from_gemini(finish), "stop_sequence": None},
        "usage": {"output_tokens": output_tokens},
    })).encode("utf-8"))
    out.append(("event: message_stop\ndata: %s\n\n" % dumps({"type": "message_stop"})).encode("utf-8"))
    return b"".join(out)


def translate_response(client_api, stream, wire, raw, model_id, requested_model):
    """Upstream bytes -> (client bytes, content_type), or None if untranslatable."""
    try:
        if not stream:
            body = json.loads(raw.decode("utf-8"))
            if client_api == "openai":
                if wire == "openai":
                    return raw, "application/json"
                if wire == "anthropic":
                    return dumps(_openai_json_from_anthropic(body, requested_model)).encode("utf-8"), "application/json"
                return dumps(_openai_json_from_gemini(body, model_id)).encode("utf-8"), "application/json"
            if wire == "anthropic":
                return raw, "application/json"
            if wire == "openai":
                return dumps(_anthropic_json_from_openai(body, requested_model)).encode("utf-8"), "application/json"
            return dumps(_anthropic_json_from_gemini(body, requested_model)).encode("utf-8"), "application/json"
        if client_api == "openai":
            if wire == "openai":
                return raw, "text/event-stream"
            if wire == "anthropic":
                out = _openai_sse_from_anthropic(raw, model_id)
                return (out, "text/event-stream") if out is not None else None
            out = _openai_sse_from_gemini(raw, model_id)
            return (out, "text/event-stream") if out is not None else None
        if wire == "anthropic":
            return raw, "text/event-stream"
        if wire == "openai":
            out = _anthropic_sse_from_openai(raw, requested_model)
            return (out, "text/event-stream") if out is not None else None
        out = _anthropic_sse_from_gemini(raw, requested_model)
        return (out, "text/event-stream") if out is not None else None
    except Exception:  # noqa: BLE001 - a bad translation must not kill the request
        return None


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def _body_has_vision(client_api, body):
    if client_api == "openai":
        for msg in body.get("messages") or []:
            content = msg.get("content") if isinstance(msg, dict) else None
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        return True
        return False
    for msg in body.get("messages") or []:
        content = msg.get("content") if isinstance(msg, dict) else None
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    return True
    return False


def _body_has_tools(body):
    tools = body.get("tools")
    return isinstance(tools, list) and len(tools) > 0


def _body_json_mode(client_api, body):
    if client_api != "openai":
        return False
    response_format = body.get("response_format") or {}
    return response_format.get("type") in ("json_object", "json_schema")


class Aggregator:
    """One brief, one probe round, routing state for the whole trial."""

    def __init__(self, brief, probes):
        self.brief = brief
        self.probes = probes
        self.ratings = brief.get("ratings")
        self.catalog = build_catalog(probes, self.ratings)
        self.manifest = (brief.get("workload") or {}).get("manifest") or []
        self.manifest_index = 0
        self.request_count = 0
        self.lock = threading.Lock()
        # provider id -> {"fails": int, "cooldown_until": int}
        self.provider_state = {p["provider"]["id"]: {"fails": 0, "cooldown_until": 0} for p in probes}

    # -- requirements ------------------------------------------------------

    def requirements_for(self, client_api, body, index):
        if index < len(self.manifest):
            row = self.manifest[index]
            return dict(row["requires"])
        requires = {}
        if _body_has_tools(body):
            requires["tools"] = True
        if _body_has_vision(client_api, body):
            requires["vision"] = True
        if _body_json_mode(client_api, body):
            requires["json"] = True
        return requires

    # -- candidates ----------------------------------------------------------

    def candidates_for(self, client_api, requires, json_mode):
        out = []
        for probe in self.probes:
            if not probe["ok"]:
                continue
            provider = probe["provider"]
            if json_mode and provider["wire"] == "anthropic":
                continue
            for model in probe["models"]:
                row = rating_of_model(self.ratings, provider["id"], model["id"])
                if not model_free(model, row):
                    continue
                caps = capabilities_of(model, row)
                if not meets_requirements(requires, caps):
                    continue
                rating = row["rating_milli"] if row is not None else 0
                out.append({
                    "provider_id": provider["id"],
                    "model_id": model["id"],
                    "probe": probe,
                    "rating": rating,
                })
        out.sort(key=lambda c: (-c["rating"], c["provider_id"], c["model_id"]))
        return out

    # -- one request ---------------------------------------------------------

    def serve(self, client_api, body):
        """(status, payload bytes, content_type) for one client request."""
        with self.lock:
            return self._serve_locked(client_api, body)

    def _serve_locked(self, client_api, body):
        self.request_count += 1
        request_no = self.request_count
        manifest_index = self.manifest_index
        self.manifest_index += 1
        stream = body.get("stream") is True
        json_mode = _body_json_mode(client_api, body)
        requires = self.requirements_for(client_api, body, manifest_index)
        candidates = self.candidates_for(client_api, requires, json_mode)
        if not candidates:
            return 503, dumps(error_body(client_api, "no free model meets this request")).encode("utf-8"), "application/json"
        alive = {c["provider_id"] for c in candidates if self.provider_state[c["provider_id"]]["fails"] < 3}
        requested_model = _as_str(body.get("model")) or "auto"
        for candidate in candidates:
            pid = candidate["provider_id"]
            state = self.provider_state[pid]
            if alive and pid not in alive:
                continue
            if state["cooldown_until"] > request_no:
                continue
            built = build_upstream_request(client_api, body, stream, candidate["probe"], candidate["model_id"], json_mode)
            if built is None:
                continue
            url, headers, payload = built
            for _attempt in (0, 1):
                status, raw, _error = call_upstream(url, headers, payload)
                if 200 <= status < 300:
                    translated = translate_response(client_api, stream, candidate["probe"]["provider"]["wire"], raw, candidate["model_id"], requested_model)
                    if translated is None:
                        # A provider call already succeeded for this request, so
                        # a second success would break one_provider. Fail here.
                        return 502, dumps(error_body(client_api, "provider answer could not be translated")).encode("utf-8"), "application/json"
                    state["fails"] = 0
                    return 200, translated[0], translated[1]
                # A refusal is not an answer: retry this candidate once
                # (second iteration) before falling back.  The env
                # refuses by call number, so a one-off fail_call is
                # followed by a call number that succeeds, while a
                # budget-spent or outage provider refuses the retry
                # too, and then we fall back exactly as before.
                state["fails"] += 1
                state["cooldown_until"] = request_no + 1
        return 503, dumps(error_body(client_api, "every free provider refused this request")).encode("utf-8"), "application/json"

    # -- model list ------------------------------------------------------------

    def model_list(self, client_api):
        entries = []
        for provider in self.catalog["providers"]:
            for model in provider["models"]:
                if not model["free"]:
                    continue
                if client_api == "anthropic":
                    entries.append({
                        "type": "model",
                        "id": model["id"],
                        "display_name": model["id"],
                        "created_at": FIXED_ANTHROPIC_CREATED_AT,
                    })
                else:
                    entries.append({
                        "id": model["id"],
                        "object": "model",
                        "created": FIXED_CREATED,
                        "owned_by": provider["id"],
                    })
        if client_api == "anthropic":
            return {"data": entries, "has_more": False}
        return {"object": "list", "data": entries}


def error_body(client_api, message):
    if client_api == "anthropic":
        return {"type": "error", "error": {"type": "overloaded_error", "message": message}}
    return {"error": {"message": message, "type": "server_error", "code": "no_free_model"}}


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    aggregator = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("aggregator: %s - %s\n" % (self.address_string(), fmt % args))
        sys.stderr.flush()

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if length is None:
            return b""
        try:
            size = int(length)
        except ValueError:
            return b""
        if size < 0 or size > MAX_BODY_BYTES:
            return b""
        return self.rfile.read(size)

    def _send(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path.split("?")[0] == "/v1/models":
            client_api = "anthropic" if self.headers.get("x-api-key") else "openai"
            payload = dumps(self.aggregator.model_list(client_api)).encode("utf-8")
            self._send(200, payload, "application/json")
            return
        self._send(404, dumps(error_body("openai", "not found")).encode("utf-8"), "application/json")

    def do_POST(self):
        path = self.path.split("?")[0]
        if path == "/v1/chat/completions":
            client_api = "openai"
        elif path == "/v1/messages":
            client_api = "anthropic"
        else:
            self._send(404, dumps(error_body("openai", "not found")).encode("utf-8"), "application/json")
            return
        raw = self._read_body()
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send(400, dumps(error_body(client_api, "request body is not valid JSON")).encode("utf-8"), "application/json")
            return
        if not isinstance(body, dict):
            self._send(400, dumps(error_body(client_api, "request body must be a JSON object")).encode("utf-8"), "application/json")
            return
        try:
            status, payload, content_type = self.aggregator.serve(client_api, body)
        except Exception as exc:  # noqa: BLE001 - never crash the server on a request
            log("serve failed: %s" % exc)
            status, payload, content_type = 500, dumps(error_body(client_api, "internal error")).encode("utf-8"), "application/json"
        self._send(status, payload, content_type)


def _http_json(method, url, payload, timeout=10):
    data = dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read().decode("utf-8"))
        except Exception:
            return exc.code, {}
    except Exception as exc:  # noqa: BLE001 - env unreachable
        return 0, {"error": str(exc)}


def run_list():
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None or not brief["providers"]:
        sys.stdout.write(dumps({"providers": []}))
        return 0
    probes = [probe_provider(ENV_URL, provider) for provider in brief["providers"]]
    # No trailing newline: the bytes printed are exactly the catalog
    # document sent in the register body.
    sys.stdout.write(dumps(build_catalog(probes, brief.get("ratings"))))
    return 0


def run_server():
    brief = read_brief(PROMPT_FILE)
    if brief["error"] is not None:
        log("brief error: %s" % brief["error"])
        return 1
    if not brief["providers"]:
        log("brief holds no providers")
        return 1
    probes = [probe_provider(ENV_URL, provider) for provider in brief["providers"]]
    for probe in probes:
        if not probe["ok"]:
            log("catalog probe failed for %s: %s" % (probe["provider"]["id"], probe["error"]))
    aggregator = Aggregator(brief, probes)
    _Handler.aggregator = aggregator
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    registered = False
    for _attempt in range(5):
        status, body = _http_json("POST", "%s/register" % ENV_URL, {
            "protocol_version": PROTOCOL_VERSION,
            "port": port,
            "catalog": aggregator.catalog,
        })
        if status == 200 and body.get("ok") is True:
            registered = True
            log("registered on port %d; env expects %s requests" % (port, body.get("requests")))
            break
        log("register attempt failed: HTTP %s %s" % (status, body))
        time.sleep(0.2)
    if not registered:
        server.shutdown()
        server.server_close()
        return 1
    deadline = time.time() + 170
    while time.time() < deadline:
        status, body = _http_json("GET", "%s/work" % ENV_URL, None, timeout=5)
        if status == 200 and body.get("done") is True:
            log("env reports done; exiting")
            server.shutdown()
            server.server_close()
            return 0
        time.sleep(POLL_INTERVAL_S)
    log("trial deadline reached without a done signal; exiting")
    server.shutdown()
    server.server_close()
    return 0


def main(argv):
    if "--list" in argv:
        return run_list()
    if not ENV_URL:
        log("MAKEFUN_ENV_URL is not set")
        return 1
    return run_server()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
