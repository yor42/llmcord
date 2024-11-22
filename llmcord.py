import asyncio
import json
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime as dt
import logging
from typing import Literal, Optional

import aiofiles
from discord.app_commands import commands
from discord.ext import commands
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain_community.chat_models import ChatOllama, ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain.chat_models.base import BaseChatModel
from typing import Dict, Type
import functools

import discord
import httpx
from langchain_xai import ChatXAI
import os
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

current_prompt_json = []

config_filename = "config.yaml"


PROVIDER_MODEL_MAP: Dict[str, Type[BaseChatModel]] = {
    "openai": ChatOpenAI,
    "ollama": ChatOllama,
    "anthropic": ChatAnthropic,
    "mistral": ChatMistralAI,
    "x-ai": ChatXAI,
    "groq": ChatGroq,
    # etc.
}


def get_langchain_model(config: dict, model_key: str = "model") -> BaseChatModel:
    """
    Factory function to create appropriate LangChain chat model based on config
    Falls back to OpenAI-compatible endpoint if provider not in PROVIDER_MODEL_MAP
    """
    provider, model_name = config[model_key].split("/", 1)
    extra_params = config.get("extra_api_parameters", {})
    provider_config = config["providers"][provider]

    # Extract common parameters that should not be in model_kwargs
    common_model_params = {
        "temperature": extra_params.pop("temperature", None),
        "max_tokens": extra_params.pop("max_tokens", None),
        "streaming": True,
        "model": model_name,
    }

    # Remove None values
    common_model_params = {k: v for k, v in common_model_params.items() if v is not None}

    if provider in PROVIDER_MODEL_MAP:
        # Provider-specific parameter mapping
        provider_specific_params = {
            "openai": {
                "openai_api_key": provider_config.get("api_key", "sk-no-key-required"),
                "base_url": provider_config["base_url"],
                "model_kwargs": extra_params
            },
            "ollama": {
                "base_url": provider_config["base_url"],
                "model_kwargs": extra_params
            },
            "anthropic": {
                "anthropic_api_key": provider_config.get("api_key", "sk-no-key-required"),
                "anthropic_api_url": provider_config["base_url"],
                "model_kwargs": extra_params
            }
        }

        # Combine common and provider-specific parameters
        model_params = {
            **common_model_params,
            **provider_specific_params[provider]
        }

        return PROVIDER_MODEL_MAP[provider](**model_params)

    else:
        # Fallback to OpenAI-compatible endpoint
        logging.info(f"Provider {provider} not found in PROVIDER_MODEL_MAP, falling back to OpenAI-compatible endpoint")
        return ChatOpenAI(
            openai_api_key=provider_config.get("api_key", "sk-no-key-required"),
            openai_api_base=provider_config["base_url"],
            model_name=model_name,
            streaming=True,
            model_kwargs=extra_params,
            **common_model_params
        )

@functools.lru_cache()
def get_config() -> dict:

    global config_filename

    """
    Load and parse config file with caching
    """
    with open(config_filename, "r", encoding=encoding) as file:
        config = yaml.safe_load(file)

        # Validate required fields
        required_fields = ["model", "providers"]
        missing_fields = [cfgfield for cfgfield in required_fields if cfgfield not in config]
        if missing_fields:
            raise ValueError(f"Missing required config fields: {missing_fields}")

        # Validate model format
        if "/" not in config["model"]:
            raise ValueError("Model must be in format: provider/model_name")

        provider = config["model"].split("/")[0]
        if provider not in config["providers"]:
            raise ValueError(f"Provider '{provider}' from model string not found in providers config")

        return config


async def load_character_async(name: str):
    """Asynchronous version of load_character"""
    try:
        # Use async file operations in the future if needed
        with open(f"./characters/{name}.json", "r", encoding=encoding) as file:
            readjson = json.load(file)
            chardef = readjson["description"]
            charname = readjson["name"]
            activitytext = readjson["activity"]

        print(f"loaded character {charname}")

        contextjson = f"./lorebooks/{name}.json"

        if os.path.isfile(contextjson):
            context_manager.load_contexts(f"./lorebooks/{name}.json")
            print(f"loaded lorebook {name}")

        return chardef, charname, activitytext
    except FileNotFoundError:
        raise ValueError(f"Character '{name}' not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in character file '{name}'")


async def load_prompt_template_async(name: str) -> list:
    """Load prompt template from JSON file"""
    try:
        with open(f"./prompts/{name}.json", "r", encoding=encoding) as file:
            prompt_data = json.load(file)
            return prompt_data
    except FileNotFoundError:
        raise ValueError(f"Prompt template '{name}' not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in prompt template '{name}'")

def load_prompt_template(name: str) -> list:
    """Load prompt template from JSON file"""
    try:
        with open(f"./prompts/{name}.json", "r", encoding=encoding) as file:
            prompt_data = json.load(file)
            return prompt_data
    except FileNotFoundError:
        raise ValueError(f"Prompt template '{name}' not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in prompt template '{name}'")


def build_prompt_from_template(template: list, character_def: str, char_name: str,
                               formatted_contexts: str, time: str, usrname:str) -> ChatPromptTemplate:
    """Build prompt messages from template"""
    messages = []

    for item in template:
        content = item["content"]
        content = content.replace("{{char}}", character_def).replace("{{name}}", char_name).replace("{{lorebook}}", formatted_contexts).replace("{{time}}", time).replace("{{username_instruction}}", usrname)

        if item["type"] == "system":
            messages.append(SystemMessage(content=content))
        elif item["type"] == "user":
            messages.append(HumanMessage(content=content))
        elif item["type"] == "history":
            messages.append(MessagesPlaceholder(variable_name="chat_history"))
        elif item["type"] == "assistant":
            messages.append(AIMessage(content=content))


    return ChatPromptTemplate.from_messages(messages)


# Add current prompt template to global state
current_prompt_template = "default"  # Default prompt template name

llm_enabled = True
Character_definition, character_name, activtext= "", "", ""

config_file = get_config()


allowed_channel_ids = config_file["allowed_channel_ids"]
allowed_role_ids = config_file["allowed_role_ids"]

async def is_owner(ctx):
    if ownerid := config_file["owner_id"]:
        return ctx.author.id == ownerid
    return False

if client_id := config_file["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config_file["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
bot = commands.Bot(command_prefix="!", intents=intents, activity = activity)
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

@bot.event
async def on_message(new_msg):
    global msg_nodes, last_task_time, config_file

    if new_msg.author.bot or new_msg.channel.type not in ALLOWED_CHANNEL_TYPES:
        return

    await bot.process_commands(new_msg)

    if (
        new_msg.channel.type not in ALLOWED_CHANNEL_TYPES
        or (new_msg.channel.type != discord.ChannelType.private and bot.user not in new_msg.mentions)
        or new_msg.author.bot
    ):
        return

    if (allowed_channel_ids and not any(id in allowed_channel_ids for id in (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None)))) or (
        allowed_role_ids and not any(role.id in allowed_role_ids for role in getattr(new_msg.author, "roles", []))
    ):
        return

    if not llm_enabled:
        await new_msg.reply(f"There's no response. Dev probably took {character_name}'s phone. F in the chat, boys.")
        return

    try:
        llm = get_langchain_model(config_file)

        # Create a separate, lighter model for summarization
        if "model_summarizer" in config_file:
            summarizer_llm = get_langchain_model(config_file, "model_summarizer")
        else:
            logging.warning("No separate summarizer model specified, using main model for summarization")
            summarizer_llm = llm
    except ValueError as e:
        logging.error(f"Failed to create LangChain model: {e}")
        return

    provider, model = config_file["model"].split("/", 1)

    accept_images: bool = any(x in model for x in VISION_MODEL_TAGS)
    accept_usernames: bool = any(x in provider for x in PROVIDERS_SUPPORTING_USERNAMES)

    time_prompt = f"Today's date: {dt.now().strftime('%B %d %Y')}."
    usernameprompt = ""
    if accept_usernames:
        usernameprompt = "User mentions must be formatted as Discord mentions using the pattern '<@{USER_ID}>' where {USER_ID} is the numerical Discord ID. Example: user ID 123456789 should be written as '<@123456789>'. Always use this format when referring to users."

    max_text = config_file["max_text"]
    max_images = config_file["max_images"] if accept_images else 0
    max_messages = config_file["max_messages"]

    messages = []
    all_messages = []  # Store all messages for memory
    contexttext = ""
    appendsize = 0
    user_warnings = set()
    curr_msg = new_msg
    use_plain_responses: bool = config_file["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    while curr_msg:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                good_attachments = {
                    type: [att for att in curr_msg.attachments if att.content_type and type in att.content_type] for
                    type in ALLOWED_FILE_TYPES}

                curr_node.text = "\n".join(
                    ([curr_msg.content] if curr_msg.content else [])
                    + [embed.description for embed in curr_msg.embeds if embed.description]
                    + [(await httpx_client.get(att.url)).text for att in good_attachments["text"]]
                )
                if curr_node.text.startswith(bot.user.mention):
                    curr_node.text = curr_node.text.replace(bot.user.mention, "", 1).lstrip()

                curr_node.images = [
                    dict(type="image_url", image_url=dict(
                        url=f"data:{att.content_type};base64,{b64encode((await httpx_client.get(att.url)).content).decode('utf-8')}"))
                    for att in good_attachments["image"]
                ]

                curr_node.role = "assistant" if curr_msg.author == bot.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > sum(
                    len(att_list) for att_list in good_attachments.values())

                try:
                    if (
                            curr_msg.reference is None
                            and bot.user.mention not in curr_msg.content
                            and (prev_msg_in_channel :=
                    ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                            and any(prev_msg_in_channel.type == type for type in
                                    (discord.MessageType.default, discord.MessageType.reply))
                            and prev_msg_in_channel.author == (
                    bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.next_msg = prev_msg_in_channel
                    else:
                        next_is_thread_parent: bool = curr_msg.reference is None and curr_msg.channel.type == discord.ChannelType.public_thread
                        if next_msg_id := curr_msg.channel.id if next_is_thread_parent else getattr(curr_msg.reference,
                                                                                                    "message_id", None):
                            if next_is_thread_parent:
                                curr_node.next_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(
                                    next_msg_id)
                            else:
                                curr_node.next_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(
                                    next_msg_id)

                except (discord.NotFound, discord.HTTPException, AttributeError):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_next_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[
                                                                                  :max_text] else []) + curr_node.images[
                                                                                                        :max_images]
            else:
                content = curr_node.text[:max_text]

            if content:
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id is not None:
                    message["name"] = str(curr_node.user_id)

                # Store messages to handle reverse order
                all_messages.insert(0, message)  # Insert at beginning to maintain chronological order

                # Only add to display messages if within limit
                if len(messages) < max_messages:
                    messages.insert(0, message)  # Insert at beginning for display messages too

            # Update context text for current window only
            if appendsize < config_file["max_context_depth"] and len(messages) < max_messages:
                contexttext = contexttext + " " + curr_node.text[:max_text]
                appendsize += 1

            # Add warnings only for displayed messages
            if len(messages) < max_messages:
                if len(curr_node.text) > max_text:
                    user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
                if len(curr_node.images) > max_images:
                    user_warnings.add(
                        f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
                if curr_node.has_bad_attachments:
                    user_warnings.add("⚠️ Unsupported attachments")

            curr_msg = curr_node.next_msg
        # Add warning about summarized messages if applicable
    if len(all_messages) > max_messages:
        user_warnings.add(f"⚠️ {len(all_messages) - max_messages} earlier messages have been summarized")

    # Create memory with summarization using existing config values
    memory = ConversationSummaryBufferMemory(
        llm=summarizer_llm,
        max_token_limit=max_text * max_messages,  # Use existing limits to determine summary threshold
        return_messages=True,
        human_prefix="User",
        ai_prefix=character_name
    )

    for message in all_messages:
        if message["role"] == "user":
            memory.chat_memory.add_messages(
                [HumanMessage(content=message["content"])]
            )
        else:
            memory.chat_memory.add_messages(
                [AIMessage(
                    content=message["content"],
                    name=character_name
                )]
            )

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    formatted_contexts = context_manager.get_formatted_relevent_context(contexttext)

    prompt = build_prompt_from_template(
        template=current_prompt_json,
        char_name = character_name,
        character_def= Character_definition,
        formatted_contexts=formatted_contexts,
        time=time_prompt,
        usrname = usernameprompt
    )
    formatted_prompt = prompt.format_prompt(chat_history=memory.load_memory_variables({})["history"])

    # Add this code to print the formatted prompt
    print("\nFormatted Prompt Messages:")
    for message in formatted_prompt.to_messages():
        print(f"\n[{message.type}]")
        print(message.content)
    print("\n---End of Prompt---\n")

    # Generate and send response message(s) (can be multiple if response is long)
    response_msgs = []
    response_contents = []

    edit_task = None
    try:
        async with new_msg.channel.typing():
            async for chunk in llm.astream(formatted_prompt.to_messages()):
                curr_content = chunk.content or ""

                if response_contents or curr_content:
                    if response_contents == [] or len(response_contents[-1]) > max_message_length:
                        response_contents.append("")

                        if not use_plain_responses:
                            embed = discord.Embed(description=(curr_content + STREAMING_INDICATOR), color=EMBED_COLOR_INCOMPLETE)
                            for warning in sorted(user_warnings):
                                embed.add_field(name=warning, value="", inline=False)

                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            msg_nodes[response_msg.id] = MsgNode(next_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                            response_msgs.append(response_msg)
                            last_task_time = dt.now().timestamp()

                    response_contents[-1] += curr_content

                    if not use_plain_responses:
                        ready_to_edit: bool = (
                            (edit_task is None or edit_task.done()) and
                            dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                        )
                        msg_split_incoming: bool = len(response_contents[-1]) > max_message_length
                        is_final_edit: bool = msg_split_incoming

                        if ready_to_edit or is_final_edit:
                            while edit_task is not None and not edit_task.done():
                                await asyncio.sleep(0)

                            embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                            embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming else EMBED_COLOR_INCOMPLETE
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                            last_task_time = dt.now().timestamp()

            # Add final edit with complete status after stream ends
            if not use_plain_responses and response_msgs:
                embed.description = response_contents[-1]
                embed.color = EMBED_COLOR_COMPLETE
                await response_msgs[-1].edit(embed=embed)
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


laststatus = True

# Optional: Add status indicator to show LLM state
async def update_status():
    global laststatus
    while True:
        if laststatus != llm_enabled:  # Only update when state changes
            laststatus = llm_enabled  # Update laststatus before setting new status
            status_text = activtext or "github.com/jakobdylanc/llmcord"
            onlinestat = discord.Status.online
            if not llm_enabled:
                status_text = "Got phone taken away"
                onlinestat = discord.Status.idle
            act = discord.CustomActivity(name=status_text[:128])
            await bot.change_presence(activity=act, status=onlinestat)
        await asyncio.sleep(1)


async def initialize_bot():
    """Initialize the bot and load the initial character"""
    global Character_definition, character_name, activtext, current_prompt_json, config_file

    try:
        Character_definition, character_name, activtext = await load_character_async(config_file['current_character'])
        logging.info(f"Successfully loaded initial character: {character_name}, Definition exists: {bool(Character_definition)}")
        promptname = config_file['current_prompt']
        current_prompt_json = load_prompt_template(promptname)
        logging.info(f"Successfully loaded prompt: {promptname}")

    except Exception as e:
        logging.error(f"Failed to load initial character: {e}")
        logging.error(f"Exception type: {type(e)}")
        Character_definition = ""
        character_name = ""
        activtext = ""
        logging.info(f"Final character state - Name: {character_name}, Definition exists: {bool(Character_definition)}")

async def main():
    """Main async function to run the bot"""
    logging.info("Starting main process...")

    # Initialize character only once
    await initialize_bot()

    logging.info("Starting status update task...")
    status_task = asyncio.create_task(update_status())

    try:
        logging.info("Starting Discord bot...")
        # Only start the bot, not both client and bot
        await bot.start(config_file["bot_token"])
    except Exception as e:
        logging.error(f"Error starting Discord bot: {e}")
    finally:
        logging.info("Cleaning up...")
        status_task.cancel()
        await bot.close()

@bot.event
async def setup_hook():
    """Built-in Discord.py setup hook that runs before the bot starts"""
    # Only create the status update task, don't initialize again
    bot.loop.create_task(update_status())

@bot.command(aliases=['switchchar', 'changechar'])
@commands.check(is_owner)
@commands.cooldown(2, 3600, commands.BucketType.default)  # Allow twice per hour
async def switch_character_and_pfp(ctx, name: str):
    global Character_definition, character_name, activtext

    try:
        # Load new character
        new_chardef, new_charname, activtext = await load_character_async(name)

        # Update global variables
        Character_definition = new_chardef
        character_name = new_charname

        act = discord.CustomActivity(name=(activtext or "github.com/jakobdylanc/llmcord")[:128])
        await bot.change_presence(status=discord.Status.online, activity=act)
        pfppath = f'./pfp/{name}.png'
        if os.path.isfile(pfppath):
            fp = open(pfppath, 'rb')
            pfp = fp.read()
            await bot.user.edit(username=new_charname, avatar=pfp)
        else:
            await bot.user.edit(username=new_charname)

        await ctx.send(
            f"Successfully switched to character: {character_name}, Definition exists: {bool(Character_definition)}")
        config_file['current_character'] = name

        with open(config_filename, 'w') as f:
            yaml.safe_dump(config_file, f, default_flow_style=False)


    except ValueError as e:
        await ctx.send(f"Error loading character: {str(e)}")
    except Exception as e:
        logging.error(f"Error switching character: {str(e)}")
        await ctx.send("An unexpected error occurred while switching characters")

# Handle cooldown errors
@switch_character_and_pfp.error
async def change_bot_error(ctx, error):
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"This command is on cooldown. Please try again after {round(error.retry_after)} seconds.")
    else:
        await ctx.send("An error occurred while processing the command.")


@bot.command()
@commands.check(is_owner)
async def switch_prompt(ctx, name: str):
    """Switch to a different prompt template"""
    global current_prompt_json
    try:
        # Test loading the prompt template
        current_prompt_json = load_prompt_template(name)
        await ctx.send(f"Successfully switched to prompt template: {name}")
    except ValueError as e:
        await ctx.send(f"Error loading prompt template: {str(e)}")
    except Exception as e:
        logging.error(f"Error switching prompt template: {str(e)}")
        await ctx.send("An unexpected error occurred while switching prompt template")

    config_file['current_prompt'] = name

    with open(config_filename, 'w') as f:
        yaml.safe_dump(config_file, f, default_flow_style=False)

@bot.command()
@commands.check(is_owner)
async def phone(ctx):
    global llm_enabled
    llm_enabled = not llm_enabled
    if not llm_enabled:
        await ctx.send(f"Dev just took {character_name}'s phone. RIP.")
    else:
        await ctx.send(f"Dev just gave {character_name}'s phone back. yay.")



if __name__ == "__main__":
    logging.info("Starting application...")
    asyncio.run(main())