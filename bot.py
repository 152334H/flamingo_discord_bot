import discord

intents = discord.Intents.default()
intents.message_content = True

from typing import Optional

import discord
from discord import app_commands

from dotenv import dotenv_values
ENV = dotenv_values(".env")

MY_GUILD = discord.Object(id=ENV['MY_GUILD'])  # replace with your guild id
class MyClient(discord.Client):
    def __init__(self, *, intents: discord.Intents):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    # In this basic example, we just synchronize the app commands to one guild.
    # Instead of specifying a guild to every command, we copy over our global commands instead.
    # By doing so, we don't have to wait up to an hour until they are shown to the end-user.
    async def setup_hook(self):
        # This copies the global commands over to your guild.
        self.tree.copy_global_to(guild=MY_GUILD)
        await self.tree.sync(guild=MY_GUILD)


client = MyClient(intents=intents)

from api import ICLRequest, CAPTION_examples, CAPTION_req_example, INSTRUCT_examples, generate_prompt

@client.event
async def on_ready():
    # test a req for the sake of it
    req = ICLRequest(**CAPTION_req_example)
    print(req.handle())
    print(f'Logged in as {client.user} (ID: {client.user.id})')
    print('------')


@client.tree.command()
async def hello(interaction: discord.Interaction):
    """Says hello!"""
    await interaction.response.send_message(f'Hi, {interaction.user.mention}')


TASKS = {'caption': CAPTION_examples, 'unconditional': [], 'instruct': INSTRUCT_examples}
@client.tree.command()
async def list_tasks(interaction: discord.Interaction):
    """lists all currently supported tasks"""
    await interaction.response.send_message("__**Currently supported tasks**__:\n" + "\n".join(' - ' + s for s in TASKS))

@client.tree.command()
async def test111(interaction: discord.Interaction):
    """Runs a Flamingo task"""
    with io.BytesIO() as bio:
        embedVar, file = handle_req(CAPTION_req_example, 'caption', bio, interaction.user)
        await interaction.response.send_message(embed=embedVar, file=file)

import io
def handle_req(d: dict, task: str, bio: io.BytesIO, user, hf_kwargs={}):
    req = ICLRequest(**d)
    result = req.handle(**hf_kwargs)
    embedVar = discord.Embed(title='Flamingo-9B', description=task, color=0x00ff00)
    embedVar.set_author(name=user.name, url="https://aowfjafiwoajwoi.com", icon_url=user.display_avatar.url)
    embedVar.set_thumbnail(url="attachment://thumb.webp")
    #
    req.query.thumb.save(bio, format='webp')
    bio.seek(0)
    file = discord.File(bio, filename='thumb.webp')
    #
    embedVar.add_field(name="Query", value=req.query.text, inline=False)
    embedVar.add_field(name="Response", value=result, inline=False)
    return embedVar, file

@client.tree.command()
@app_commands.rename(task='task')
@app_commands.describe(
    task='The task to perform',
    image='The image to use',
    prompt='The prompt to use',
)
async def comprehend(
    interaction: discord.Interaction, image: discord.Attachment, prompt: str,
    convert_to_instruction: bool=False, task: str='caption',
    max_new_tokens: int=20,
    num_beams: int=3,
    temperature: float=1.0,
    top_k: int=0,
    top_p: float=1.0,
    length_penalty: float=1.0,
    do_sample: bool=False,
    #early_stopping: bool=False,
):
    """Runs a Flamingo task"""
    if task not in TASKS:
        await interaction.response.send_message(f'Task {task} not supported')
        return
    if '<image>' not in prompt:
        await interaction.response.send_message(f'Prompt must contain <image> placeholder.')
        return
    if convert_to_instruction:
        prompt = generate_prompt(prompt)
    hf_kwargs = {
        'max_new_tokens': max_new_tokens, 'num_beams': num_beams, 'temperature': temperature,
        'top_k': top_k, 'top_p': top_p, 'length_penalty': length_penalty, 'do_sample': do_sample,
    }
    req = {
        "examples": TASKS[task],
        "query": {
            "image_src": await image.read(),
            "text": prompt
        }
    }
    await interaction.response.defer()
    with io.BytesIO() as bio:
        embedVar, file = handle_req(req, task, bio, interaction.user, hf_kwargs)
        await interaction.followup.send(embed=embedVar, file=file)
    #await interaction.response.send_message(req.handle())

@client.tree.command()
@app_commands.describe(
    task='New task name to add',
    image='The image to use',
    prompt='The response to use',
)
async def register_task(interaction: discord.Interaction, image: discord.Attachment, prompt: str, task: str):
    """add ICL example (tbd)"""
    if interaction.user.id != ENV["OWNER_ID"]:
        await interaction.response.send_message(f'User {interaction.user.id} is not authorized to register tasks.')
        return

    if '<image>' not in prompt:
        await interaction.response.send_message(f'Prompt must contain <image> placeholder.')
        return
    #
    ICL = {
        "image_src": await image.read(),
        "text": prompt,
    }
    if task in TASKS: TASKS[task].append(ICL)
    else: TASKS[task] = [ICL]
    await interaction.response.send_message(f"Registered new In-Context Learning example for `{task=}` with image `{image.filename}` and response `{prompt}`")

# A Context Menu command is an app command that can be run on a member or on a message by
# accessing a menu within the client, usually via right clicking.
# It always takes an interaction as its first parameter and a Member or Message as its second parameter.

# This context menu command only works on messages
@client.tree.context_menu(name='Report to Moderators')
async def report_message(interaction: discord.Interaction, message: discord.Message):
    # We're sending this response message with ephemeral=True, so only the command executor can see it
    await interaction.response.send_message(
        f'Thanks for reporting this message by {message.author.mention} to our moderators.', ephemeral=True
    )

    # Handle report by sending it into a log channel
    log_channel = interaction.guild.get_channel(0)  # replace with your channel id

    embed = discord.Embed(title='Reported Message')
    if message.content:
        embed.description = message.content

    embed.set_author(name=message.author.display_name, icon_url=message.author.display_avatar.url)
    embed.timestamp = message.created_at

    url_view = discord.ui.View()
    url_view.add_item(discord.ui.Button(label='Go to Message', style=discord.ButtonStyle.url, url=message.jump_url))

    await log_channel.send(embed=embed, view=url_view)

# Replace TOKEN_HERE with your bot's token
client.run(ENV["BOT_TOKEN"])
