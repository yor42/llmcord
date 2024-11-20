import asyncio
import json
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional

from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI, ChatOllama, ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openrouter.openrouter import OpenRouterLLM
from langchain_mistralai import ChatMistralAI
from langchain.chat_models.base import BaseChatModel
from typing import Dict, Type
import functools

import discord
import httpx
from langchain_xai import ChatXAI
from openai import AsyncOpenAI
import yaml

from KeywordContextManager import KeywordContextManager

context_manager = KeywordContextManager(
    model_name="gpt-4o",
    max_total_tokens=5000,
    max_contexts= 10
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4o", "claude-3", "gemini", "pixtral", "llava", "vision")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

ALLOWED_FILE_TYPES = ("image", "text")
ALLOWED_CHANNEL_TYPES = (discord.ChannelType.text, discord.ChannelType.public_thread, discord.ChannelType.private_thread, discord.ChannelType.private)

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 100

encoding="utf-8"


PROVIDER_MODEL_MAP: Dict[str, Type[BaseChatModel]] = {
    "openai": ChatOpenAI,
    "ollama": ChatOllama,
    "anthropic": ChatAnthropic,
    "mistral": ChatMistralAI,
    "x-ai": ChatXAI,
    "groq": ChatGroq,
    "openrouter": OpenRouterLLM
    # etc.
}


def get_langchain_model(config: dict) -> BaseChatModel:
    """
    Factory function to create appropriate LangChain chat model based on config
    """
    provider, model_name = config["model"].split("/", 1)

    if provider not in PROVIDER_MODEL_MAP:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(PROVIDER_MODEL_MAP.keys())}")

    model_class = PROVIDER_MODEL_MAP[provider]

    # Get provider-specific config
    provider_config = config["providers"][provider]

    # Common parameters for all models
    common_params = {
        "streaming": True,
        "model": model_name,
    }

    # Provider-specific parameter mapping
    if provider == "openai":
        common_params.update({
            "openai_api_key": provider_config.get("api_key", "sk-no-key-required"),
            "base_url": provider_config["base_url"],
            "model_kwargs": config.get("extra_api_parameters", {})
        })
    elif provider == "ollama":
        common_params.update({
            "base_url": provider_config["base_url"],
            "model_kwargs": config.get("extra_api_parameters", {})
        })
    # Add more provider-specific mappings as needed

    return model_class(**common_params)


@functools.lru_cache()
def get_config(filename="config.yaml") -> dict:
    """
    Load and parse config file with caching
    """
    with open(filename, "r", encoding=encoding) as file:
        config = yaml.safe_load(file)

        # Validate required fields
        required_fields = ["model", "providers"]
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")

        # Validate model format
        if "/" not in config["model"]:
            raise ValueError("Model must be in format: provider/model_name")

        provider = config["model"].split("/")[0]
        if provider not in config["providers"]:
            raise ValueError(f"Provider '{provider}' from model string not found in providers config")

        return config

def load_character(name):
    chardef = ""
    charname = ""
    with open("./characters/"+name+".json", "r", encoding=encoding) as file:
        readjson = json.load(file)
        chardef = readjson["description"]
        charname = readjson["name"]
    print(f"loaded character {charname}")
    context_manager.load_contexts("./lorebooks/"+name+".json")

    return chardef, charname

Character_definition, character_name = load_character("kal\'tsit")

cfg = get_config()

if client_id := cfg["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = None


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    next_msg: Optional[discord.Message] = None

    has_bad_attachments: bool = False
    fetch_next_failed: bool = False

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    if (
        new_msg.channel.type not in ALLOWED_CHANNEL_TYPES
        or (new_msg.channel.type != discord.ChannelType.private and discord_client.user not in new_msg.mentions)
        or new_msg.author.bot
    ):
        return

    config_file = get_config()

    allowed_channel_ids = config_file["allowed_channel_ids"]
    allowed_role_ids = config_file["allowed_role_ids"]

    if (allowed_channel_ids and not any(id in allowed_channel_ids for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None)))) or (
        allowed_role_ids and not any(role.id in allowed_role_ids for role in getattr(new_msg.author, "roles", []))
    ):
        return

    provider, model = config_file["model"].split("/", 1)
    base_url = config_file["providers"][provider]["base_url"]
    api_key = config_file["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images: bool = any(x in model for x in VISION_MODEL_TAGS)
    accept_usernames: bool = any(x in provider for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config_file["max_text"]
    max_images = config_file["max_images"] if accept_images else 0
    max_messages = config_file["max_messages"]

    use_plain_responses: bool = config_file["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))
    messages = []
    # Build message chain and set user warnings
    if pre_system_prompt := config_file["pre_history_system_prompt"]:
        full_pre_system_prompt = dict(role="system", content=str(pre_system_prompt).replace("{{char}}", Character_definition).replace("{{name}}", character_name))
        messages.append(full_pre_system_prompt)

    contexttext = ""
    appendsize = 0
    user_warnings = set()
    curr_msg = new_msg
    while curr_msg and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                good_attachments = {type: [att for att in curr_msg.attachments if att.content_type and type in att.content_type] for type in ALLOWED_FILE_TYPES}

                curr_node.text = "\n".join(
                    ([curr_msg.content] if curr_msg.content else [])
                    + [embed.description for embed in curr_msg.embeds if embed.description]
                    + [(await httpx_client.get(att.url)).text for att in good_attachments["text"]]
                )
                if curr_node.text.startswith(discord_client.user.mention):
                    curr_node.text = curr_node.text.replace(discord_client.user.mention, "", 1).lstrip()

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode((await httpx_client.get(att.url)).content).decode('utf-8')}"))
                    for att in good_attachments["image"]
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > sum(len(att_list) for att_list in good_attachments.values())

                try:
                    if (
                        curr_msg.reference is None
                            and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and any(prev_msg_in_channel.type == type for type in (discord.MessageType.default, discord.MessageType.reply))
                        and prev_msg_in_channel.author == (discord_client.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.next_msg = prev_msg_in_channel
                    else:
                        next_is_thread_parent: bool = curr_msg.reference is None and curr_msg.channel.type == discord.ChannelType.public_thread
                        if next_msg_id := curr_msg.channel.id if next_is_thread_parent else getattr(curr_msg.reference, "message_id", None):
                            if next_is_thread_parent:
                                curr_node.next_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(next_msg_id)
                            else:
                                curr_node.next_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(next_msg_id)

                except (discord.NotFound, discord.HTTPException, AttributeError):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_next_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if appendsize < config_file["max_context_depth"]:
                contexttext = contexttext+" "+curr_node.text[:max_text]
                appendsize=+1

            if content:
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_next_failed or (curr_node.next_msg and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.next_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    contexts = context_manager.get_relevant_contexts(contexttext)
    formatted_contexts = context_manager.format_contexts(contexts)

    if system_prompt := config_file["system_prompt"]:


        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names follows Discord ID format and should be typed as '<@{USER_ID}>'.")

        full_system_prompt = dict(role="system", content="\n".join([system_prompt] + system_prompt_extras).replace("{{lorebook}}", formatted_contexts))
        messages.append(full_system_prompt)

    print(messages)

    # Generate and send response message(s) (can be multiple if response is long)
    response_msgs = []
    response_contents = []
    prev_chunk = None
    edit_task = None

    kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_body=config_file["extra_api_parameters"])
    try:
        async with new_msg.channel.typing():
            async for curr_chunk in await openai_client.chat.completions.create(**kwargs):
                prev_content = prev_chunk.choices[0].delta.content if prev_chunk is not None and prev_chunk.choices[0].delta.content else ""
                curr_content = curr_chunk.choices[0].delta.content or ""

                if response_contents or prev_content:
                    if response_contents == [] or len(response_contents[-1] + prev_content) > max_message_length:
                        response_contents.append("")

                        if not use_plain_responses:
                            embed = discord.Embed(description=(prev_content + STREAMING_INDICATOR), color=EMBED_COLOR_INCOMPLETE)
                            for warning in sorted(user_warnings):
                                embed.add_field(name=warning, value="", inline=False)

                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                            response_msgs.append(response_msg)
                            last_task_time = dt.now().timestamp()

                    response_contents[-1] += prev_content

                    if not use_plain_responses:
                        finish_reason = curr_chunk.choices[0].finish_reason

                        ready_to_edit: bool = (
                                                          edit_task is None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                        msg_split_incoming: bool = len(response_contents[-1] + curr_content) > max_message_length
                        is_final_edit: bool = finish_reason is not None or msg_split_incoming
                        is_good_finish: bool = finish_reason is not None and any(finish_reason.lower() == x for x in ("stop", "end_turn"))

                        if ready_to_edit or is_final_edit:
                            while edit_task is not None and not edit_task.done():
                                await asyncio.sleep(0)

                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                            last_task_time = dt.now().timestamp()

                prev_chunk = curr_chunk

        if use_plain_responses:
            for content in response_contents:
                reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                await msg_nodes[response_msg.id].lock.acquire()
                response_msgs.append(response_msg)
    except:
        logging.exception("Error while generating response")

    for msg in response_msgs:
        msg_nodes[msg.id].text = "".join(response_contents)
        msg_nodes[msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main():
    await discord_client.start(cfg["bot_token"])


asyncio.run(main())
