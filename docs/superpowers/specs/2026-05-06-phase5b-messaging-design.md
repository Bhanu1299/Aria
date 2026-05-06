# Phase 5B — Messaging Layer Design

**Date:** 2026-05-06  
**Status:** Approved  
**Depends on:** Phase 5A (core engine)  
**Unlocks:** Voice-driven iMessage and WhatsApp send/read

---

## Goal

Aria can read recent messages and send replies across iMessage and WhatsApp by voice. Other platforms (Telegram, Discord, Slack) are registered as stubs — they return a helpful setup message rather than crashing.

---

## Voice examples

- "Read my last iMessage from mom"
- "Reply to Priya on WhatsApp saying I'll be 10 minutes late"
- "What did Rahul say on WhatsApp?"
- "Send the meeting link to the team group on iMessage"

---

## Architecture

### Plugin structure

```
plugins/
  messaging/
    __init__.py          ← MessagingPlugin(PluginBase)
    imessage.py          ← iMessageClient
    whatsapp.py          ← WhatsAppClient
    stubs/
      __init__.py
      telegram.py        ← TelegramStub
      discord.py         ← DiscordStub
      slack.py           ← SlackStub
```

`MessagingPlugin.register(registry)` registers all tools. Unavailable tools (WhatsApp not set up) still register but return a setup prompt on first use.

---

## iMessage

### Mechanism

Two-layer approach:
1. **Read** — query `~/Library/Messages/chat.db` directly via SQLite. The DB is owned by the user so no special permissions beyond Full Disk Access (already needed by Aria for other features).
2. **Send** — AppleScript via `osascript`. No external binary required.

### Tools registered

**`imessage_read`**
```
params:
  contact: str        # name or phone number
  limit: int          # default 5, max 20
returns:
  list of {sender, text, timestamp, is_from_me}
```

**`imessage_send`**
```
params:
  contact: str        # name or phone number
  message: str
  service: str        # "iMessage" | "SMS" | "auto" (default "auto")
returns:
  "Sent." or error string
```

**`imessage_get_contacts`**
```
params: none
returns:
  list of recent contacts with display names
```

### Contact resolution

`iMessageClient` builds a contacts index from the chat.db `handle` table. Fuzzy name matching via `difflib.get_close_matches(name, contact_names, n=1, cutoff=0.6)` — "mom" resolves to "+1-555-xxx-xxxx" from contact display names. Falls back to direct phone/email if no match.

### Permissions

macOS Full Disk Access required for chat.db. If permission denied, tool returns: `"I need Full Disk Access to read iMessages. Go to System Settings → Privacy → Full Disk Access and add Terminal (or Aria)."` 

---

## WhatsApp

### Mechanism

`whatsapp-web.py` library (Python port of the Baileys WebSocket client). Maintains a persistent authenticated WebSocket connection to WhatsApp Web servers.

### Setup (one-time)

```bash
python -m aria.setup whatsapp
```

Opens a QR code in terminal. User scans with WhatsApp on phone. Session saved to `~/.aria/credentials/whatsapp/`. On Aria restart, session restores automatically. If session expires (WhatsApp invalidates after ~14 days of inactivity), setup prompt shown again.

### Tools registered

**`whatsapp_read`**
```
params:
  contact: str        # name or phone number
  limit: int          # default 5, max 20
returns:
  list of {sender, text, timestamp, is_from_me}
```

**`whatsapp_send`**
```
params:
  contact: str
  message: str
returns:
  "Sent." or error string
```

**`whatsapp_get_contacts`**
```
params: none
returns:
  list of recent chats with display names
```

### Not-yet-setup behavior

If `~/.aria/credentials/whatsapp/` doesn't exist, all WhatsApp tools return:
`"WhatsApp isn't set up yet. Run: python -m aria.setup whatsapp"`

### Deduplication

Incoming message listener uses a 500ms debounce window + message ID set to prevent echo loops (sending triggers a receive event on the same client).

---

## Platform stubs

Telegram, Discord, and Slack register tools that return:

```
"Telegram isn't configured yet. 
To enable it, add TELEGRAM_BOT_TOKEN to your .env and restart Aria."
```

Each stub is a complete `ToolDescriptor` with the same interface as the real implementation — swapping in the real implementation later requires no changes to calling code.

---

## Privacy

- Messages never leave the device (iMessage reads local DB, WhatsApp session is local)
- No message content stored in Aria's memory unless the user explicitly asks Aria to remember something from a conversation
- `~/.aria/credentials/whatsapp/` is chmod 700

---

## Error handling

| Error | Response |
|---|---|
| chat.db permission denied | Explain Full Disk Access setup |
| Contact not found | "I couldn't find a contact named X. Try using their phone number." |
| WhatsApp disconnected | "WhatsApp lost its connection. I'll try to reconnect." + auto-reconnect attempt |
| Send failed | "I couldn't send that message: {reason}" |
| osascript error | "iMessage isn't responding. Is the Messages app running?" |

---

## Testing

- `tests/test_imessage.py` — contact resolution, DB query mocking, AppleScript send mocking
- `tests/test_whatsapp.py` — session restore, send/receive with mock client, dedup logic
- `tests/test_messaging_stubs.py` — stubs return correct setup strings

---

## Dependencies

```
whatsapp-web.py>=0.4.0
```

iMessage requires no new dependencies (sqlite3 + subprocess are stdlib).

---

## Definition of done

- [ ] `iMessageClient` with read + send + contact resolution
- [ ] `WhatsAppClient` with session management + read + send
- [ ] One-time setup script `python -m aria.setup whatsapp`
- [ ] All 3 stubs (Telegram, Discord, Slack) returning correct messages
- [ ] `MessagingPlugin` registered and loading in core engine
- [ ] Tests passing for iMessage, WhatsApp, stubs
