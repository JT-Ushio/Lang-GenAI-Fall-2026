"""
Functions such as (e.g., note, image, link) populate the list of renderings,
which will be shown in place of the line of code in the interface.
"""

import os
import inspect
import re
import subprocess
from urllib.parse import urlparse, parse_qs, urlencode
from file_util import cached, relativize
from dataclasses import dataclass
from arxiv_util import is_arxiv_link, arxiv_reference
from reference import Reference

@dataclass(frozen=True)
class CodeLocation:
    """Refers to a specific line of code."""
    path: str
    line_number: int


@dataclass(frozen=True)
class Rendering:
    """
    Specifies what to display instead of a line of code.  Types:
    - text: plain text (verbatim)
    - markdown: to be rendered as markdown
    - image: an image (data = url)
    - video: a local video or HTTP(S) media URL (data = url)
    - bilibili: an official embedded player (data = player URL)
    - link: an link to internal code or external URL
    """
    type: str
    data: str | None = None
    style: dict | None = None
    external_link: Reference | None = None
    internal_link: CodeLocation | None = None

############################################################

def text(message: str, style: dict | None = None, verbatim: bool = False):
    """Make a note (bullet point) with `message`."""
    style = style or {}
    if verbatim:
        messages = message.split("\n")
        style = {
            "fontFamily": "monospace",
            "whiteSpace": "pre",
            **style
        }
    else:
        messages = [message]

    for message in messages:
        _current_renderings.append(Rendering(type="markdown", data=message, style=style))


def image(url: str, style: dict | None = None, width: int | str | None = None):
    """Show the image at `url`."""
    style = style or {}
    if width is not None:
        style["width"] = width

    if is_url(url):
        path = cached(url, "image")
    else:
        path = url
        if not os.path.exists(path):
            raise ValueError(f"Image not found: {path}")

    _current_renderings.append(Rendering(type="image", data=path, style=style))


def video(url: str, style: dict | None = None, width: int | str | None = None):
    """Embed local video, HTTP(S) media, or a Bilibili BV video page."""
    if not url.startswith(("http://", "https://")) and not os.path.isfile(url):
        raise ValueError(f"Video not found: {url}")
    style = dict(style or {})
    if width is not None:
        style["width"] = width
    parsed = urlparse(url)
    if parsed.hostname in ("bilibili.com", "www.bilibili.com", "m.bilibili.com"):
        match = re.fullmatch(r"/video/(BV[0-9A-Za-z]{10})/?", parsed.path)
        if not match:
            raise ValueError("Use a full Bilibili video URL: https://www.bilibili.com/video/BV...")
        page = parse_qs(parsed.query).get("p", ["1"])[0]
        if not page.isascii() or not page.isdecimal() or int(page) < 1:
            raise ValueError("Bilibili page p must be a positive integer")
        bvid = match.group(1)
        query = urlencode({"bvid": bvid, "page": int(page), "autoplay": 0})
        player_url = f"https://player.bilibili.com/player.html?{query}"
        original_url = f"https://www.bilibili.com/video/{bvid}/?p={int(page)}"
        _current_renderings.append(Rendering(
            type="bilibili", data=player_url, style=style,
            external_link=Reference(title="在 B站打开", url=original_url),
        ))
        return
    _current_renderings.append(Rendering(type="video", data=url, style=style))


def is_url(url: str) -> bool:
    """Check if `url` looks like a URL."""
    return url.startswith("http")


def link(arg: type | Reference | str | None = None, style: dict | None = None, **kwargs):
    """
    Shows a link.  There are four possible usages:
    1. link(title="...", url="...") [Creates a new reference]
    2. link(arg: Reference) [Shows an existing reference]
    3. link(arg: type) [Shows a link to the code]
    4. link(arg: str) [Creates a new reference with the given URL]
    """
    style = style or {}

    if arg is None:
        reference = Reference(**kwargs)
        _current_renderings.append(Rendering(type="link", style=style, external_link=reference))
    elif isinstance(arg, Reference):
        _current_renderings.append(Rendering(type="link", style=style, external_link=arg))
    elif isinstance(arg, type) or callable(arg):
        path = inspect.getfile(arg)
        _, line_number = inspect.getsourcelines(arg)
        anchor = CodeLocation(relativize(path), line_number)
        _current_renderings.append(Rendering(type="link", data=arg.__name__, style=style, internal_link=anchor))
    elif isinstance(arg, str):
        if is_arxiv_link(arg):
            reference = arxiv_reference(arg)
            _current_renderings.append(Rendering(type="link", style=style, external_link=reference))
        else:
            reference = Reference(url=arg)
            _current_renderings.append(Rendering(type="link", style=style, external_link=reference))
    else:
        raise ValueError(f"Invalid argument: {arg}")

    style = {"color": "gray"}


############################################################

# Accumulate the renderings during execution (gets flushed).
_current_renderings: list[Rendering] = []

def pop_renderings() -> list[Rendering]:
    """Return the renderings and clear the list."""
    renderings = _current_renderings.copy()
    _current_renderings.clear()
    return renderings


def system_text(command: list[str]):
    output = subprocess.check_output(command).decode('utf-8')
    output = remove_ansi_escape_sequences(output)
    text(output, verbatim=True)


def remove_ansi_escape_sequences(input_text: str) -> str:
    ansi_escape_pattern = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape_pattern.sub('', input_text)
