from typing import List, Dict, Any
from collections import defaultdict
import json

import aiofiles
import tiktoken  # OpenAI's token counter, you can use different tokenizers
import re

class KeywordContextManager:
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",  # or any other model you're using
                 max_total_tokens: int = 3000,  # adjust based on your needs
                 max_contexts: int = 5):
        self.keyword_map = defaultdict(list)
        self.contexts = []
        self.max_contexts = max_contexts
        self.max_total_tokens = max_total_tokens
        self.tokenizer = tiktoken.encoding_for_model(model_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        return len(self.tokenizer.encode(text))

    def load_contexts(self, json_file: str):
        utf = "utf-8"
        self.contexts = []
        self.keyword_map.clear()
        with open(json_file, 'r',encoding=utf) as f:
            contexts = json.load(f)

        # Pre-process and index all contexts
        for idx, ctx in enumerate(contexts):
            if not ctx.get('enabled', True):
                continue

            # Pre-calculate token count for each context
            token_count = self.count_tokens(ctx['content'])
            ctx['token_count'] = token_count

            # Pre-sort contexts by insertion_order for faster runtime sorting
            # self.contexts.sort(key=lambda x: (-x.get('insertion_order', 100)))

            if "bookVersion" in ctx and ctx['bookVersion'] == 2:
                ctx['keys'] = ctx['key'].split(", ")
                ctx['case_sensitive'] = ctx["extentions"]["risu_case_sensitive"]
                ctx['constant'] = ctx["alwaysActive"]
                ctx['use_regex'] = ctx["useRegex"]
                ctx['name'] = ctx["comment"]

            keywordlist = ctx['keys']
            # Store the full context data
            current_index = len(self.contexts)  # Get current index before appending
            self.contexts.append(ctx)

            # Create inverted index for fast keyword lookup
            for keyword in keywordlist:
                keyword = keyword.lower().strip()
                self.keyword_map[keyword].append(current_index)  # Use stored index

    def get_relevant_contexts(self,
                              user_input: str,
                              current_prompt_tokens: int = 0) -> List[str]:
        """
        Get relevant contexts while respecting token limits.

        Args:
            user_input: The user's input text
            current_prompt_tokens: Number of tokens already in the prompt
        """
        print(f"text used for context finding: {user_input}")

        matched_indices = set()

        # Find all matching contexts based on keywords
        for idx, ctx in enumerate(self.contexts):
            if ctx.get('constant', False):
                continue

            # Get context properties
            case_sensitive = ctx.get('case_sensitive', False)
            use_regex = ctx.get('use_regex', False)
            keys = ctx['keys']

            # Prepare input text based on case sensitivity
            input_text = user_input if case_sensitive else user_input.lower()

            for key in keys:
                if not key:  # Skip empty keys
                    continue

                key = key.strip()

                if use_regex:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    if re.search(key, input_text, flags=flags):
                        matched_indices.add(idx)
                        print(f"found matching regex {key} for idx {idx}")
                        break
                else:
                    # Convert key to lowercase if not case sensitive
                    key = key if case_sensitive else key.lower()
                    if key in input_text:
                        matched_indices.add(idx)
                        print(f"found matching keyword {key} for idx {idx}")
                        break

        if not matched_indices and not any(ctx.get('constant', False) for ctx in self.contexts):
            return []

        # Track which contexts we've already added to avoid duplicates
        added_contexts = set()
        constant_contexts = []
        matched_contexts = []  # Store tuples of (insertion_order, content) for sorting
        total_tokens = current_prompt_tokens

        # First, handle constant contexts as they take priority
        for idx, ctx in enumerate(self.contexts):
            if ctx.get('constant', False):
                if idx not in added_contexts and total_tokens + ctx['token_count'] <= self.max_total_tokens:
                    constant_contexts.append(ctx['content'])
                    added_contexts.add(idx)
                    total_tokens += ctx['token_count']
                    print(f"added constant {ctx['name']}")
            else:
                # For non-constant contexts, only add if they have matching non-empty keywords
                if idx in matched_indices and idx not in added_contexts:
                    matched_contexts.append((
                        -ctx.get('insertion_order', 100),  # Negative for descending order
                        ctx['token_count'],
                        ctx['content'],
                        ctx['name']
                    ))

        # Sort matched contexts by insertion_order (higher order first)
        matched_contexts.sort()
        regular_contexts = []

        # Add matched contexts respecting token limit
        for _, token_count, content, name in matched_contexts:
            if total_tokens + token_count <= self.max_total_tokens:
                regular_contexts.append(content)
                total_tokens += token_count
                print(f"added match {name}")

        # Combine contexts respecting max_contexts limit
        final_contexts = constant_contexts
        remaining_slots = min(
            self.max_contexts - len(final_contexts),
            len(regular_contexts)
        )

        if remaining_slots > 0:
            final_contexts.extend(regular_contexts[:remaining_slots])

        return final_contexts

    def format_contexts(self, contexts: List[str]) -> str:
        """Format contexts for insertion into prompt"""
        if not contexts:
            return ""

        return "\n\n".join(contexts)

    def get_formatted_relevent_context(self, user_input: str, current_prompt_tokens: int = 0):
        """Format contexts for insertion into prompt"""
        contexts = self.get_relevant_contexts(user_input, current_prompt_tokens)

        return self.format_contexts(contexts)
