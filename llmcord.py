import asyncio
from base64 import b64encode, b64decode
from dataclasses import dataclass, field
from datetime import datetime as dt
import io
import json
import logging
from typing import Literal, Optional

import discord
from discord import app_commands
import httpx
from openai import AsyncOpenAI
from PIL import Image
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gpt-4", "o3", "o4", "claude-3", "gemini", "gemma", "llama", "pixtral", "mistral-small", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def get_config(filename="config.yaml"):
    with open(filename, "r") as file:
        return yaml.safe_load(file)


cfg = get_config()

if client_id := cfg["client_id"]:
    logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot%20applications.commands\n")

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(cfg["status_message"] or "github.com/jakobdylanc/llmcord")[:128])
discord_client = discord.Client(intents=intents, activity=activity)
tree = app_commands.CommandTree(discord_client)

httpx_client = httpx.AsyncClient()

msg_nodes = {}
last_task_time = 0

async def generate_image(prompt, negative_prompt=None, steps=None, cfg_scale=None, width=None, height=None, 
                    sampler_name=None, seed=None, batch_size=None, restore_faces=None, tiling=None, 
                    denoising_strength=None, enable_hr=None, hr_scale=None):
    """Generate an image using Stable Diffusion via A1111 API."""
    cfg = get_config()
    sd_config = cfg.get("stable_diffusion", {})
    base_url = sd_config.get("base_url", "http://127.0.0.1:7860")
    default_params = sd_config.get("default_params", {})
    # Apply prompt template if available
    prompt_template = default_params.get("positive_prompt_template", "{prompt}")
    final_prompt = prompt_template.replace("{prompt}", prompt) if "{prompt}" in prompt_template else prompt
    
    # Apply negative prompt template if available
    neg_prompt = negative_prompt or ""
    neg_prompt_template = default_params.get("negative_prompt_template", "{prompt}")
    final_negative_prompt = neg_prompt_template.replace("{prompt}", neg_prompt) if "{prompt}" in neg_prompt_template and neg_prompt else default_params.get("negative_prompt", "")
    
    # Use provided parameters or fall back to defaults    
    payload = {
        "prompt": final_prompt,
        "negative_prompt": final_negative_prompt,
        "steps": steps or default_params.get("steps", 20),
        "cfg_scale": cfg_scale or default_params.get("cfg_scale", 7.0),
        "width": width or default_params.get("width", 512),
        "height": height or default_params.get("height", 512),
        "sampler_name": sampler_name or default_params.get("sampler_name", "Euler a"),
        "sampler_index": sampler_name or default_params.get("sampler_name", "Euler a"),
        "seed": seed if seed is not None else default_params.get("seed", -1),
        "batch_size": batch_size or default_params.get("batch_size", 1),
        "n_iter": 1,
        "restore_faces": restore_faces if restore_faces is not None else default_params.get("restore_faces", False),
        "tiling": tiling if tiling is not None else default_params.get("tiling", False),
        "denoising_strength": denoising_strength or default_params.get("denoising_strength", 0.7),
        "scheduler": default_params.get("scheduler", "karras"),  # Added scheduler parameter
        "send_images": True,
        "save_images": True,
    }
    
    # Add high-res fix settings if enabled
    if enable_hr or default_params.get("enable_hr", False):
        payload["enable_hr"] = True
        payload["hr_scale"] = hr_scale or default_params.get("hr_scale", 2)
        payload["hr_upscaler"] = default_params.get("hr_upscaler", "Latent")
        payload["hr_second_pass_steps"] = default_params.get("hr_second_pass_steps", 0)
    
    logging.info(f"Sending image generation request with prompt: {prompt}")
    logging.debug(f"Full payload: {payload}")
    
    try:
        response = await httpx_client.post(
            f"{base_url}/sdapi/v1/txt2img",
            json=payload,
            timeout=180.0  # Images can take time to generate, especially with high-res fix
        )
        response.raise_for_status()
        result = response.json()
        
        # Check if images array exists and is not empty
        if "images" not in result or not result["images"]:
            logging.error("No images returned from API")
            return None
            
        # Extract and decode the image
        try:
            image_data = b64decode(result["images"][0])
            
            # Extract generation info if available
            generation_info = {}
            if "info" in result:
                try:
                    # The info might be a JSON string that needs parsing
                    if isinstance(result["info"], str):
                        generation_info = json.loads(result["info"])
                    else:
                        generation_info = result["info"]
                    logging.info(f"Generation info: {generation_info}")
                except:
                    logging.warning("Could not parse generation info")
            
            return {
                "image_data": image_data,
                "info": generation_info,
                "seed": generation_info.get("seed", payload["seed"]) if isinstance(generation_info, dict) else payload["seed"]
            }
        except Exception as decode_error:
            logging.error(f"Failed to decode image: {decode_error}")
            return None
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error: {http_err.response.status_code} - {http_err.response.text}")
        return None
    except Exception as e:
        logging.exception(f"Failed to generate image: {e}")
        return None

async def generate_img2img(init_image_data, prompt, negative_prompt=None, steps=None, cfg_scale=None, width=None, height=None, 
                      sampler_name=None, seed=None, batch_size=None, restore_faces=None, tiling=None, 
                      denoising_strength=None, resize_mode=0):
    """Generate an image using Stable Diffusion img2img API with an initial image."""
    cfg = get_config()
    sd_config = cfg.get("stable_diffusion", {})
    base_url = sd_config.get("base_url", "http://127.0.0.1:7860")
    default_params = sd_config.get("default_params", {})
    
    # Apply prompt template if available
    prompt_template = default_params.get("positive_prompt_template", "{prompt}")
    final_prompt = prompt_template.replace("{prompt}", prompt) if "{prompt}" in prompt_template else prompt
    
    # Encode the initial image to base64
    if isinstance(init_image_data, bytes):
        encoded_image = b64encode(init_image_data).decode('utf-8')
    else:
        # If it's already a string, assume it's already base64 encoded
        encoded_image = init_image_data
    
    # Use provided parameters or fall back to defaults
    payload = {
        "init_images": [encoded_image],
        "resize_mode": resize_mode,
        "prompt": final_prompt,
        "negative_prompt": negative_prompt or default_params.get("negative_prompt", ""),
        "steps": steps or default_params.get("steps", 20),
        "cfg_scale": cfg_scale or default_params.get("cfg_scale", 7.0),
        "width": width or default_params.get("width", 512),
        "height": height or default_params.get("height", 512),
        "sampler_name": sampler_name or default_params.get("sampler_name", "Euler a"),
        "sampler_index": sampler_name or default_params.get("sampler_name", "Euler a"),
        "seed": seed if seed is not None else default_params.get("seed", -1),
        "batch_size": batch_size or default_params.get("batch_size", 1),
        "n_iter": 1,
        "restore_faces": restore_faces if restore_faces is not None else default_params.get("restore_faces", False),
        "tiling": tiling if tiling is not None else default_params.get("tiling", False),
        "denoising_strength": denoising_strength or default_params.get("denoising_strength", 0.7),
        "scheduler": default_params.get("scheduler", "karras"),
        "send_images": True,
        "save_images": True,
    }
    
    logging.info(f"Sending img2img generation request with prompt: {prompt}")
    logging.debug(f"Full payload: {payload}")
    
    try:
        response = await httpx_client.post(
            f"{base_url}/sdapi/v1/img2img",
            json=payload,
            timeout=180.0  # Images can take time to generate
        )
        response.raise_for_status()
        result = response.json()
        
        # Check if images array exists and is not empty
        if "images" not in result or not result["images"]:
            logging.error("No images returned from API")
            return None
            
        # Extract and decode the image
        try:
            image_data = b64decode(result["images"][0])
            
            # Extract generation info if available
            generation_info = {}
            if "info" in result:
                try:
                    # The info might be a JSON string that needs parsing
                    if isinstance(result["info"], str):
                        generation_info = json.loads(result["info"])
                    else:
                        generation_info = result["info"]
                    logging.info(f"Generation info: {generation_info}")
                except:
                    logging.warning("Could not parse generation info")
            
            return {
                "image_data": image_data,
                "info": generation_info,
                "seed": generation_info.get("seed", payload["seed"]) if isinstance(generation_info, dict) else payload["seed"]
            }
        except Exception as decode_error:
            logging.error(f"Failed to decode image: {decode_error}")
            return None
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error: {http_err.response.status_code} - {http_err.response.text}")
        return None
    except Exception as e:
        logging.exception(f"Failed to generate image: {e}")
        return None


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@discord_client.event
async def on_message(new_msg):
    global msg_nodes, last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_client.user not in new_msg.mentions) or new_msg.author.bot:
        return

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    cfg = get_config()

    allow_dms = cfg["allow_dms"]
    permissions = cfg["permissions"]

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_all_channels = not allowed_channel_ids
    is_good_channel = allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    # Check for clear command after permission checks
    cleaned_content_for_command = new_msg.content.removeprefix(discord_client.user.mention).lstrip()
    if cleaned_content_for_command.lower().startswith("clear"):
        try:
            # Reply to the user's "clear" message to signify the context break
            clear_confirmation_msg = await new_msg.reply("🧹 Context cleared. I will start a new conversation from your next message.", silent=True)

            # Add a node for the confirmation message, pointing back to the user's clear command.
            # This ensures the next user reply starts fresh, referencing this confirmation.
            if clear_confirmation_msg:
                 msg_nodes[clear_confirmation_msg.id] = MsgNode(
                     text=clear_confirmation_msg.content,
                     role="assistant",
                     parent_msg=new_msg # Link back to the user's !clear command
                 )
                 # Lock is not strictly needed here as we aren't processing it further, but good practice
                 await msg_nodes[clear_confirmation_msg.id].lock.acquire()
                 msg_nodes[clear_confirmation_msg.id].lock.release() # Release immediately

            # Remove the user's "clear" message from cache if present
            msg_nodes.pop(new_msg.id, None)
            logging.info(f"Context cleared by user {new_msg.author.id} in channel {new_msg.channel.id}")
        except discord.HTTPException as e:
            logging.error(f"Failed to send context clear confirmation: {e}")
        return # Stop processing this message further

    provider_slash_model = cfg["model"]
    provider, model = provider_slash_model.split("/", 1)
    base_url = cfg["providers"][provider]["base_url"]
    api_key = cfg["providers"][provider].get("api_key", "sk-no-key-required")
    openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(x in provider_slash_model.lower() for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = cfg["max_text"]
    max_images = cfg["max_images"] if accept_images else 0
    max_messages = cfg["max_messages"]

    use_plain_responses = cfg["use_plain_responses"]
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR))

    # Build message chain and set user warnings
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg != None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text == None:
                cleaned_content = curr_msg.content.removeprefix(discord_client.user.mention).lstrip()

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]

                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + ["\n".join(filter(None, (embed.title, embed.description, embed.footer.text))) for embed in curr_msg.embeds]
                    + [resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text")]
                )

                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]

                curr_node.role = "assistant" if curr_msg.author == discord_client.user else "user"

                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference == None
                        and discord_client.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_client.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference == None and curr_msg.channel.parent.type == discord.ChannelType.text

                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                if accept_usernames and curr_node.user_id != None:
                    message["name"] = str(curr_node.user_id)

                messages.append(message)

            if len(curr_node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg != None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    if system_prompt := cfg["system_prompt"]:
        system_prompt_extras = [f"Today's date: {dt.now().strftime('%B %d %Y')}."]
        if accept_usernames:
            system_prompt_extras.append("User's names are their Discord IDs and should be typed as '<@ID>'.")

        full_system_prompt = "\n".join([system_prompt] + system_prompt_extras)
        messages.append(dict(role="system", content=full_system_prompt))

    # Generate and send response message(s) (can be multiple if response is long)
    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []

    embed = discord.Embed()
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_body=cfg["extra_api_parameters"])
    try:
        async with new_msg.channel.typing():
            async for curr_chunk in await openai_client.chat.completions.create(**kwargs):
                if finish_reason != None:
                    break

                finish_reason = curr_chunk.choices[0].finish_reason

                prev_content = curr_content or ""
                curr_content = curr_chunk.choices[0].delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    ready_to_edit = (edit_task == None or edit_task.done()) and dt.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        if edit_task != None:
                            await edit_task

                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)

                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                        else:
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))

                        last_task_time = dt.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    reply_to_msg = new_msg if response_msgs == [] else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=content, suppress_embeds=True)
                    response_msgs.append(response_msg)

                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Delete oldest MsgNodes (lowest message IDs) from the cache
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


@tree.command(name="imagine", description="Generate an image using Stable Diffusion")
async def imagine(interaction: discord.Interaction, 
                 prompt: str, 
                 negative_prompt: str = None, 
                 cfg_scale: app_commands.Range[float, 1.0, 30.0] = None,
                 seed: int = -1,
                 sampler: str = None,
                 restore_faces: bool = False,
                 highres_fix: bool = False):
    """Generate an image using Stable Diffusion with the specified parameters."""
    # Defer the response since image generation might take time
    await interaction.response.defer(thinking=True)
    
    try:
        # Log the image generation request
        logging.info(f"Image generation requested by {interaction.user.id} with prompt: {prompt}")
        
        # First show queue status
        status_message = None
        progress_message = None
        last_progress = -1
        last_preview = None
          # Check queue status first
        queue_status = await get_queue_status()
        if queue_status:
            queue_size = queue_status.get("queue_size", 0) or 0  # Ensure we have integers, not None
            rank = queue_status.get("rank", 0) or 0
            rank_eta = queue_status.get("rank_eta", 0) or 0
            
            if queue_size > 0 or rank > 0:
                status_embed = discord.Embed(
                    title="Image Generation Queued",
                    description="Your image is queued for generation.",
                    color=EMBED_COLOR_INCOMPLETE
                )
                
                status_embed.add_field(name="Queue Position", value=str(rank), inline=True)
                status_embed.add_field(name="Queue Size", value=str(queue_size), inline=True)
                status_embed.add_field(name="Estimated Wait", value=f"{rank_eta:.1f}s" if rank_eta else "Calculating...", inline=True)
                status_embed.add_field(name="Prompt", value=prompt, inline=False)
                
                status_message = await interaction.followup.send(embed=status_embed)
                
                # Wait for queue position to reach 0 or low rank
                wait_count = 0
                while rank > 0 and wait_count < 30:  # Limit to 30 checks (5 minutes)
                    await asyncio.sleep(10)  # Check every 10 seconds
                    wait_count += 1
                    
                    new_status = await get_queue_status()
                    if not new_status:
                        break
                        
                    new_rank = new_status.get("rank", 0)
                    new_eta = new_status.get("rank_eta", 0)
                    
                    if new_rank != rank:
                        rank = new_rank
                        status_embed.set_field_at(0, name="Queue Position", value=str(rank), inline=True)
                        status_embed.set_field_at(2, name="Estimated Wait", value=f"{new_eta:.1f}s" if new_eta else "Calculating...", inline=True)
                        
                        if status_message:
                            await status_message.edit(embed=status_embed)
                    
                    if rank == 0:
                        break
        
        # Start monitoring progress while generating
        generation_started = True
        
        # Create progress embed
        progress_embed = discord.Embed(
            title="Generating Image", 
            description="Your image is being generated...",
            color=EMBED_COLOR_INCOMPLETE
        )
        progress_embed.add_field(name="Prompt", value=prompt, inline=False)
        if negative_prompt:
            progress_embed.add_field(name="Negative Prompt", value=negative_prompt, inline=False)
        progress_embed.add_field(name="Progress", value="Starting...", inline=True)
        
        # Send initial progress message or update the status message
        if status_message:
            progress_message = await status_message.edit(embed=progress_embed)
        else:
            progress_message = await interaction.followup.send(embed=progress_embed)
          # Start the image generation
        generation_task = asyncio.create_task(generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=None,  # Use default from config
            cfg_scale=cfg_scale,
            width=None,  # Use default from config
            height=None,  # Use default from config
            sampler_name=sampler,
            seed=seed,
            restore_faces=restore_faces,
            enable_hr=highres_fix
        ))
        
        # Monitor progress while the image is being generated
        while not generation_task.done():
            progress_info = await get_generation_progress(skip_current_image=False)
            
            if progress_info:
                progress_percent = progress_info.get("progress", 0) * 100
                eta = progress_info.get("eta_relative", 0)
                text_info = progress_info.get("textinfo", "")
                current_image = progress_info.get("current_image")
                
                # Only update if progress has changed significantly or we have a new image
                if abs(progress_percent - last_progress) >= 5 or current_image != last_preview:
                    last_progress = progress_percent
                    last_preview = current_image
                    
                    # Update progress embed
                    progress_embed.set_field_at(
                        2 if negative_prompt else 1, 
                        name="Progress", 
                        value=f"{progress_percent:.1f}% - {text_info}\nETA: {eta:.1f}s" if eta else f"{progress_percent:.1f}%", 
                        inline=True
                    )
                    
                    # If we have a preview image, add it to the embed
                    if current_image:
                        try:
                            image_data = b64decode(current_image)
                            file = discord.File(io.BytesIO(image_data), filename="progress_preview.png")
                            progress_embed.set_image(url="attachment://progress_preview.png")
                            
                            # Edit the message with the new file
                            if progress_message:
                                await progress_message.delete()
                                progress_message = await interaction.followup.send(embed=progress_embed, file=file)
                        except Exception as img_err:
                            logging.error(f"Error processing preview image: {img_err}")
                            if progress_message:
                                await progress_message.edit(embed=progress_embed)
                    else:
                        if progress_message:
                            await progress_message.edit(embed=progress_embed)
            
            # Check every second but don't spam the API
            await asyncio.sleep(1)
        
        # Get the generation result
        result = await generation_task
        
        if result and "image_data" in result:
            # Create a file-like object from the image data
            file = discord.File(io.BytesIO(result["image_data"]), filename="generated_image.png")
            
            # Create an embed with the prompt and generation details
            embed = discord.Embed(title="Generated Image", color=EMBED_COLOR_COMPLETE)
            embed.add_field(name="Prompt", value=prompt, inline=False)
            if negative_prompt:
                embed.add_field(name="Negative Prompt", value=negative_prompt, inline=False)
                
            # Add generation parameters
            params = []
            params.append(f"Seed: {result.get('seed', seed)}")

            # Get default values from config for the removed parameters
            cfg = get_config()
            sd_config = cfg.get("stable_diffusion", {})
            default_params = sd_config.get("default_params", {})

            # Use defaults from config
            default_steps = default_params.get("steps", 20)
            default_width = default_params.get("width", 512)
            default_height = default_params.get("height", 512)

            params.append(f"Steps: {default_steps}")
            params.append(f"CFG Scale: {cfg_scale}" if cfg_scale else f"CFG Scale: {default_params.get('cfg_scale', 7.0)}")
            params.append(f"Size: {default_width}x{default_height}")
            params.append(f"Sampler: {sampler}" if sampler else f"Sampler: {default_params.get('sampler_name', 'Euler a')}")

            # Filter out empty strings and join
            params_text = "\n".join([p for p in params if p])
            if params_text:
                embed.add_field(name="Parameters", value=params_text, inline=False)
              # Handle progress message differently based on channel type
            is_dm = isinstance(interaction.channel, discord.DMChannel)
            
            if progress_message:
                if is_dm:
                    # In DMs, edit the existing message instead of deleting it
                    try:
                        await progress_message.edit(embed=embed, attachments=[file])
                        return  # Exit early as we've already sent the response
                    except Exception as e:
                        logging.warning(f"Could not edit progress message in DM: {e}")
                        # Continue to fallback method
                else:
                    # In servers, delete the progress message as before
                    try:
                        await progress_message.delete()
                    except Exception as e:
                        logging.warning(f"Could not delete progress message: {e}")
            
            # Send the response with the generated image (fallback method or if no progress message)
            await interaction.followup.send(embed=embed, file=file)
        else:
            if progress_message:
                await progress_message.delete()
            await interaction.followup.send("Failed to generate image. Check logs for details.")
    
    except Exception as e:
        logging.exception(f"Error in image generation command: {e}")
        await interaction.followup.send(f"Error generating image: {str(e)}")

@tree.command(name="queue", description="Check the Stable Diffusion queue status")
async def queue_status(interaction: discord.Interaction):
    """Check the current queue status of Stable Diffusion."""
    await interaction.response.defer(thinking=True)
    
    try:
        status = await get_queue_status()
        
        if status:
            embed = discord.Embed(
                title="Stable Diffusion Queue Status", 
                color=EMBED_COLOR_COMPLETE
            )
            
            embed.add_field(name="Queue Size", value=str(status.get("queue_size", 0)), inline=True)
            embed.add_field(name="Your Position", value=str(status.get("rank", 0)), inline=True)
            
            # Add estimated times
            queue_eta = status.get("queue_eta", 0)
            rank_eta = status.get("rank_eta", 0)
            embed.add_field(name="Queue ETA", value=f"{queue_eta:.1f}s" if queue_eta else "0s", inline=True)
            embed.add_field(name="Your ETA", value=f"{rank_eta:.1f}s" if rank_eta else "0s", inline=True)
            
            # Add processing time metrics
            avg_time = status.get("avg_event_process_time", 0)
            avg_concurrent_time = status.get("avg_event_concurrent_process_time", 0)
            embed.add_field(name="Avg. Process Time", value=f"{avg_time:.1f}s" if avg_time else "0s", inline=True)
            embed.add_field(name="Avg. Concurrent Time", value=f"{avg_concurrent_time:.1f}s" if avg_concurrent_time else "0s", inline=True)
            
            await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send("Failed to fetch queue status. The server may be offline.")
    
    except Exception as e:
        logging.exception(f"Error in queue status command: {e}")
        await interaction.followup.send(f"Error checking queue status: {str(e)}")

@tree.command(name="progress", description="Check the progress of the current image generation")
async def check_progress(interaction: discord.Interaction, skip_preview: bool = False):
    """Check the progress of the current image generation task."""
    await interaction.response.defer(thinking=True)
    
    try:
        progress_info = await get_generation_progress(skip_current_image=skip_preview)
        
        if progress_info:
            embed = discord.Embed(
                title="Image Generation Progress", 
                color=EMBED_COLOR_COMPLETE
            )
            
            # Add progress percentage and ETA
            progress = progress_info.get("progress", 0) * 100  # Convert to percentage
            eta = progress_info.get("eta_relative", 0)
            embed.add_field(name="Progress", value=f"{progress:.1f}%", inline=True)
            embed.add_field(name="ETA", value=f"{eta:.1f}s" if eta else "0s", inline=True)
            
            # Add text info if available
            if text_info := progress_info.get("textinfo"):
                embed.add_field(name="Status", value=text_info, inline=False)
            
            # Add preview image if available and not skipped
            if not skip_preview and (current_image := progress_info.get("current_image")):
                try:
                    # Decode the image and attach it
                    image_data = b64decode(current_image)
                    file = discord.File(io.BytesIO(image_data), filename="progress_preview.png")
                    embed.set_image(url="attachment://progress_preview.png")
                    await interaction.followup.send(embed=embed, file=file)
                except Exception as img_err:
                    logging.error(f"Error processing preview image: {img_err}")
                    await interaction.followup.send(embed=embed)
            else:
                await interaction.followup.send(embed=embed)
        else:
            await interaction.followup.send("Failed to fetch generation progress. No active generation or server offline.")
    
    except Exception as e:
        logging.exception(f"Error in progress check command: {e}")
        await interaction.followup.send(f"Error checking generation progress: {str(e)}")

async def get_queue_status():
    """Get the current queue status from Stable Diffusion API."""
    cfg = get_config()
    sd_config = cfg.get("stable_diffusion", {})
    base_url = sd_config.get("base_url", "http://127.0.0.1:7860")
    
    try:
        response = await httpx_client.get(
            f"{base_url}/queue/status",
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error getting queue status: {http_err.response.status_code} - {http_err.response.text}")
        return None
    except Exception as e:
        logging.exception(f"Failed to get queue status: {e}")
        return None

async def get_generation_progress(skip_current_image=False):
    """Get the progress of the current image generation task."""
    cfg = get_config()
    sd_config = cfg.get("stable_diffusion", {})
    base_url = sd_config.get("base_url", "http://127.0.0.1:7860")
    
    try:
        response = await httpx_client.get(
            f"{base_url}/sdapi/v1/progress",
            params={"skip_current_image": str(skip_current_image).lower()},
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error getting generation progress: {http_err.response.status_code} - {http_err.response.text}")
        return None
    except Exception as e:
        logging.exception(f"Failed to get generation progress: {e}")
        return None

async def check_internal_progress(task_id, live_preview=True, id_live_preview=-1):
    """Check the internal progress of a specific task."""
    cfg = get_config()
    sd_config = cfg.get("stable_diffusion", {})
    base_url = sd_config.get("base_url", "http://127.0.0.1:7860")
    
    payload = {
        "id_task": task_id,
        "id_live_preview": id_live_preview,
        "live_preview": live_preview
    }
    
    try:
        response = await httpx_client.post(
            f"{base_url}/internal/progress",
            json=payload,
            timeout=10.0
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as http_err:
        logging.error(f"HTTP error checking internal progress: {http_err.response.status_code} - {http_err.response.text}")
        return None
    except Exception as e:
        logging.exception(f"Failed to check internal progress: {e}")
        return None


@discord_client.event
async def on_ready():
    await tree.sync()
    logging.info(f"Connected as {discord_client.user} (ID: {discord_client.user.id})")

async def main():
    await discord_client.start(cfg["bot_token"])


asyncio.run(main())
