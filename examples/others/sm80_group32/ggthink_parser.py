# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Use the exported model's ggthink delimiters with the v0.29 parser engine."""

from vllm.parser.engine.adapters import ParserEngineReasoningAdapter
from vllm.parser.qwen3 import Qwen3Parser
from vllm.reasoning import ReasoningParserManager


class Group32GGThinkParser(Qwen3Parser):
    CONFIG_NAME = "group32_ggthink"
    THINK_START = "<ggthink>"
    THINK_END = "</ggthink>"


@ReasoningParserManager.register_module("group32_ggthink")
class Group32GGThinkAdapter(ParserEngineReasoningAdapter):
    _parser_engine_cls = Group32GGThinkParser
