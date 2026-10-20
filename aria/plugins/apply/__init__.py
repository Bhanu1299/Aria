"""plugins/apply/__init__.py — ApplyPlugin: job applications driven by the job sheet.

Tool:
  job_from_sheet — read the next not-yet-handled row (blank Status) from the job
                   sheet (MUSE_SHEET_ID / MUSE_SHEET_TAB) and remember it as the
                   current job for the apply pipeline.
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

import aria.core.plugin as _plugin_base
import aria.state.memory as memory
from aria.core.tool import ToolDescriptor, ToolRegistry
from aria.integrations import sheets

logger = logging.getLogger(__name__)

CURRENT_JOB_KEY = "sheet_job"


def describe_job(job: dict) -> str:
    """One spoken sentence for a sheet row."""
    where = f" in {job['location']}" if job.get("location") else ""
    if job.get("url"):
        link = f"Link is on {urlparse(job['url']).netloc.replace('www.', '')}."
    elif job.get("apply_link"):
        link = f"No apply URL — the sheet says: {job['apply_link']}."
    else:
        link = "There's no apply link on that row."
    return f"Next job from your sheet: {job['role']} at {job['company']}{where}, row {job['row']}. {link}"


class ApplyPlugin(_plugin_base.PluginBase):

    def register(self, registry: ToolRegistry) -> None:
        registry.register(self._job_from_sheet_tool())

    def _job_from_sheet_tool(self) -> ToolDescriptor:
        def execute(params: dict) -> str:
            try:
                job = sheets.get_next_job()
            except sheets.SheetError as exc:
                return str(exc)
            except Exception as exc:
                logger.error("job_from_sheet failed: %s", exc)
                return "I couldn't read your job sheet right now."
            if job is None:
                return "There are no new jobs in your sheet — every row already has a status."
            memory.set_persistent(CURRENT_JOB_KEY, job)
            return describe_job(job)

        return ToolDescriptor(
            name="job_from_sheet",
            description=(
                "Get the next job to apply to from the user's job sheet (the Muse sheet): "
                "the first row whose Status is blank. Use for 'next job from muse', "
                "'what's the next job on my list', 'apply to the next job'."
            ),
            input_schema={"type": "object", "properties": {}, "required": []},
            execute=execute,
        )
