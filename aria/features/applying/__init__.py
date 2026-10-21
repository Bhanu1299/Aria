"""features/applying — applications in front: one visible tab per job, filled one at a time.

  tabs.py      registry of application tabs (~/.aria/apply_tabs.json): the source of truth
  walls.py     CAPTCHA / login wall / email-code detection (code only)
  control.py   pause / resume / stop for the run in progress
  session.py   the run: pick new sheet rows in fit order, fill each in its own tab
  commands.py  the user's words → dispatcher pre-handler (never agent tools)
  startup.py   "start applications": GateLLM, Coldstart, dashboard, Chrome — health-checked

Standing rules: ARCHITECTURE.md "Applications in front".
"""
