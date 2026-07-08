"""plugins/core/__init__.py — Core plugin: registers all built-in Aria capabilities."""
from __future__ import annotations

import logging
from typing import Optional  # used in browser_task closures
from urllib.parse import quote_plus

import aria.core.plugin as _plugin_base
from aria.core.tool import ToolDescriptor, ToolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _str_prop(desc: str) -> dict:
    return {"type": "string", "description": desc}


def _schema(*required_props: tuple[str, str], **optional_props: str) -> dict:
    props = {k: _str_prop(v) for k, v in required_props}
    props.update({k: _str_prop(v) for k, v in optional_props.items()})
    return {
        "type": "object",
        "properties": props,
        "required": [k for k, _ in required_props],
    }


# ---------------------------------------------------------------------------
# CorePlugin
# ---------------------------------------------------------------------------

class CorePlugin(_plugin_base.PluginBase):
    """
    Wraps all existing Aria capabilities as ToolDescriptors.
    speaker, voice_capture, transcriber, menubar are optional — tools
    that need them will degrade gracefully if they are None.
    """

    def __init__(
        self,
        browser=None,
        speaker=None,
        voice_capture=None,
        transcriber=None,
        menubar=None,
        keyterms_prompt: str = "",
    ) -> None:
        self._browser = browser
        self._speaker = speaker
        self._voice_capture = voice_capture
        self._transcriber = transcriber
        self._menubar = menubar
        self._keyterms_prompt = keyterms_prompt

    @classmethod
    def from_context(cls, ctx) -> "CorePlugin":
        return cls(
            browser=ctx.browser,
            speaker=ctx.speaker,
            voice_capture=ctx.voice_capture,
            transcriber=ctx.transcriber,
            menubar=ctx.menubar,
            keyterms_prompt=ctx.keyterms_prompt,
        )

    def register(self, registry: ToolRegistry) -> None:
        registry.register(self._knowledge_tool())
        registry.register(self._web_search_tool())
        registry.register(self._web_direct_tool())
        registry.register(self._navigate_tool())
        registry.register(self._app_tool())
        # media_tool retired — superseded by plugins.media.MediaPlugin
        registry.register(self._app_control_tool())
        registry.register(self._briefing_tool())
        registry.register(self._jobs_tool())
        registry.register(self._browser_task_tool())
        registry.register(self._coder_tool())

    # ------------------------------------------------------------------
    # Knowledge
    # ------------------------------------------------------------------

    def _knowledge_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            from aria.features.summarizer import answer_knowledge
            return answer_knowledge(params["query"])

        return ToolDescriptor(
            name="knowledge",
            description=(
                "Answer factual questions, definitions, explanations, coding help, math, "
                "history, science, or any question answerable from training data. "
                "Use this for general knowledge — do NOT use for live/current data."
            ),
            input_schema=_schema(("query", "The question to answer")),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Web search
    # ------------------------------------------------------------------

    def _web_search_tool(self) -> ToolDescriptor:
        browser = self._browser

        def execute(params: dict) -> str:
            import aria.core.config as config
            import aria.web.websearch as websearch
            from aria.features.summarizer import summarize
            query = params["query"]
            location = params.get("location", "")
            if location:
                query = f"{query} {location}"
            elif params.get("location_sensitive"):
                loc = config.CURRENT_LOCATION
                if loc and loc != "Unknown Location":
                    query = f"{query} {loc}"

            # Primary: DuckDuckGo HTML — one HTTP call, no browser, hard to block
            snippets = websearch.snippets_text(query)
            if snippets:
                return summarize(page_text=snippets, query=query,
                                 instructions=f"Answer from these search results: {query}")

            # Fallback: Google SERP scrape through the background browser
            url = f"https://www.google.com/search?q={quote_plus(query)}"
            if browser is None:
                return "Web search unavailable — browser not initialized."
            links = browser.extract_links(url)
            for link in links[:3]:
                text = browser.fetch(link)
                if text:
                    return summarize(page_text=text, query=query,
                                     instructions=f"Extract the most relevant information to answer: {query}")
            return "I couldn't find a good result for that search."

        return ToolDescriptor(
            name="web_search",
            description=(
                "Search the web for current or live information: news, weather, stock prices, "
                "sports scores, product prices, showtimes, anything needing up-to-date data. "
                "Use for 'latest', 'current', 'today', 'now' queries."
            ),
            input_schema=_schema(
                ("query", "Search terms"),
                location_sensitive="true if query needs local context (weather, nearby restaurants, etc.)",
            ),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Web direct (named site)
    # ------------------------------------------------------------------

    def _web_direct_tool(self) -> ToolDescriptor:
        browser = self._browser
        _SITE_TEMPLATES = {
            "youtube":    "https://www.youtube.com/results?search_query={}",
            "reddit":     "https://www.reddit.com/search/?q={}",
            "linkedin":   "https://www.linkedin.com/search/results/all/?keywords={}",
            "github":     "https://github.com/search?q={}",
            "hackernews": "https://hn.algolia.com/?q={}",
            "hn":         "https://hn.algolia.com/?q={}",
            "spotify":    "https://open.spotify.com/search/{}",
        }

        def execute(params: dict) -> str:
            from aria.features.summarizer import summarize
            query = params["query"]
            site = params.get("site", "").lower()
            template = _SITE_TEMPLATES.get(site)
            if template:
                url = template.format(quote_plus(query))
            else:
                url = f"https://www.google.com/search?q={quote_plus(query + ' ' + site)}"
            if browser is None:
                return "Web fetch unavailable — browser not initialized."
            text = browser.fetch(url)
            if text:
                return summarize(page_text=text, query=query,
                                 instructions=f"Extract the most relevant information to answer: {query}")
            return f"I couldn't load {site or 'that page'}."

        return ToolDescriptor(
            name="web_direct",
            description=(
                "Search a specific named website: YouTube, Reddit, LinkedIn, GitHub, HackerNews, Spotify. "
                "Use when the user explicitly names a site AND wants to search it."
            ),
            input_schema=_schema(
                ("query", "Search terms"),
                site="Site name: youtube | reddit | linkedin | github | hackernews | spotify",
            ),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Navigate
    # ------------------------------------------------------------------

    def _navigate_tool(self) -> ToolDescriptor:
        _KNOWN_SITES = {
            "claude": "https://claude.ai", "claud": "https://claude.ai",
            "youtube": "https://youtube.com", "reddit": "https://reddit.com",
            "github": "https://github.com", "google": "https://google.com",
            "linkedin": "https://linkedin.com", "twitter": "https://x.com",
            "x": "https://x.com", "gmail": "https://mail.google.com",
            "notion": "https://notion.so", "spotify": "https://open.spotify.com",
        }

        def execute(params: dict) -> str:
            from aria.web.browser import goto as browser_goto
            site = params.get("site", "").lower()
            url = _KNOWN_SITES.get(site, params.get("url", f"https://{site}.com"))
            display = params.get("display_name") or site.capitalize()
            try:
                browser_goto(url)
                return f"Opening {display}."
            except Exception as exc:
                logger.error("navigate failed: %s", exc)
                return f"I couldn't open {display}."

        return ToolDescriptor(
            name="navigate",
            description=(
                "Open a website in the browser. Use when user says 'open', 'go to', "
                "'take me to', 'navigate to' a specific site."
            ),
            input_schema=_schema(
                ("site", "Short site name: youtube, github, reddit, claude, gmail, etc."),
                url="Direct URL if known",
                display_name="Human-readable display name for the site",
            ),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # App launcher
    # ------------------------------------------------------------------

    def _app_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            import aria.features.mac_controller as mac_controller
            from aria.features.app_launcher import open_app
            app_name = params.get("app_name", "")
            contact = params.get("contact") or None
            result = open_app(app_name, contact)
            if "couldn't find" in result and not contact:
                folder_result = mac_controller._finder_open(app_name)
                if "couldn't find" not in folder_result:
                    return folder_result
            return result

        return ToolDescriptor(
            name="app",
            description=(
                "Launch a macOS application. Use for 'open Safari', 'open Spotify', "
                "'call/text/message [contact]', 'open [app] and message [contact]'."
            ),
            input_schema=_schema(
                ("app_name", "macOS application name as it appears in the Applications folder"),
                contact="Contact name for messaging/calling apps",
            ),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Media
    # ------------------------------------------------------------------

    def _media_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            import aria.features.media as media
            return media.handle_media_command(params["command"])

        return ToolDescriptor(
            name="media",
            description=(
                "Music playback and YouTube. Use for: play/pause/skip/stop music, "
                "'what's playing', 'play X on YouTube', 'play X on Apple Music', "
                "any media playback command."
            ),
            input_schema=_schema(("command", "The full media command as spoken by the user")),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # App control (AppleScript)
    # ------------------------------------------------------------------

    def _app_control_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            import aria.features.mac_controller as mac_controller
            return mac_controller.handle_app_command(params["command"])

        return ToolDescriptor(
            name="app_control",
            description=(
                "Native Mac control via AppleScript: Spotify playback, system volume, "
                "brightness, dark mode, focus mode, wifi, bluetooth, calendar, email, "
                "reminders, Finder, quit/hide apps, read screen content."
            ),
            input_schema=_schema(("command", "The full Mac control command as spoken")),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Briefing
    # ------------------------------------------------------------------

    def _briefing_tool(self) -> ToolDescriptor:
        speaker = self._speaker

        def execute(params: dict) -> str:
            import aria.features.briefing as briefing
            if speaker is not None:
                try:
                    speaker.say("Getting your briefing, one moment.")
                except Exception:
                    pass
            return briefing.build_briefing()

        return ToolDescriptor(
            name="briefing",
            description=(
                "Build a morning briefing with weather, calendar, email, and news. "
                "Use for: 'give me my briefing', 'morning briefing', 'what's on my day', "
                "'daily summary', 'what do I have today'."
            ),
            input_schema={"type": "object", "properties": {}},
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    def _jobs_tool(self) -> ToolDescriptor:
        speaker = self._speaker

        def execute(params: dict) -> str:
            import memory, jobs
            query = params["query"]
            cached = memory.get_cached_jobs(query)
            if cached is not None:
                memory.store_jobs(cached)
                return jobs.format_spoken_results(cached)
            if speaker is not None:
                try:
                    speaker.say("Searching for jobs, one moment.")
                except Exception:
                    pass
            results = jobs.search_jobs(query)
            memory.store_jobs(results)
            memory.store_last_search(query)
            memory.store_cached_jobs(query, results)
            return jobs.format_spoken_results(results)

        return ToolDescriptor(
            name="jobs",
            description=(
                "Search for job listings on LinkedIn and Indeed. Use for any job search: "
                "'find me jobs', 'any [role] openings', 'search for [role] positions'. "
                "Include filters in query: remote, hybrid, posted this week, salary range."
            ),
            input_schema=_schema(("query", "Job search query including role, location, filters")),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Browser task (multi-step research)
    # ------------------------------------------------------------------

    def _browser_task_tool(self) -> ToolDescriptor:
        speaker = self._speaker
        voice_capture = self._voice_capture
        transcriber = self._transcriber
        keyterms_prompt = self._keyterms_prompt

        def execute(params: dict) -> str:
            import time, computer_use, notifier
            goal = params["goal"]

            def _confirm(summary: str) -> bool:
                if speaker is None:
                    return True
                try:
                    speaker.say(summary)
                    if voice_capture is None or transcriber is None:
                        return True
                    wav = voice_capture.record_once(max_seconds=5)
                    if wav is None:
                        return False
                    reply = transcriber.transcribe(wav).lower().strip()
                    _YES = {"yes", "yeah", "yep", "sure", "do it", "go ahead",
                            "confirm", "proceed", "ok", "okay"}
                    return any(w in reply for w in _YES)
                except Exception:
                    return True

            def _get_input(field: str) -> Optional[str]:
                if speaker is None or voice_capture is None or transcriber is None:
                    return None
                try:
                    speaker.say(f"I need {field}. Please say it now.")
                    wav = voice_capture.record_once(max_seconds=10)
                    if wav is None:
                        return None
                    return transcriber.transcribe(wav, initial_prompt=keyterms_prompt).strip() or None
                except Exception:
                    return None

            _last: list[str] = [""]
            def _on_progress(msg: str) -> None:
                if speaker and msg and msg != _last[0]:
                    _last[0] = msg
                    try:
                        speaker.say(msg)
                    except Exception:
                        pass

            if speaker is not None:
                try:
                    speaker.say("On it.")
                except Exception:
                    pass

            t0 = time.monotonic()
            try:
                result = computer_use.research_loop(
                    goal=goal, max_steps=80,
                    confirm_fn=_confirm, input_fn=_get_input, progress_fn=_on_progress,
                )
            except Exception as exc:
                logger.error("research_loop failed: %s", exc)
                return "I ran into an error. Try again with more detail."
            else:
                notifier.notify_if_slow(time.monotonic() - t0, goal, result)
            return result

        return ToolDescriptor(
            name="browser_task",
            description=(
                "Multi-step browser research requiring navigation and data extraction: "
                "price comparisons, rental searches, flight/hotel searches, apartment listings, "
                "anything needing sequential browser actions to gather and compare data."
            ),
            input_schema=_schema(("goal", "Full research goal with all specific details (dates, location, price range)")),
            execute=execute,
        )

    # ------------------------------------------------------------------
    # Coder
    # ------------------------------------------------------------------

    def _coder_tool(self) -> ToolDescriptor:
        menubar = self._menubar

        def execute(params: dict) -> str:
            import aria.features.coder as coder
            if menubar is not None:
                try:
                    menubar.set_state("CODING")
                except Exception:
                    pass
            return coder.run(params["command"], menubar=menubar)

        return ToolDescriptor(
            name="coder",
            description=(
                "Agentic code execution: write scripts, create files, run tests, fix bugs, "
                "install packages, start servers, build apps. Use for any coding task."
            ),
            input_schema=_schema(("command", "The coding task as described by the user")),
            execute=execute,
        )
