#
# Copyright (C) 2021-2022 by Yuki77394@Github, < https://github.com/yuki77394 >.
# This file is part of < https://github.com/Yuki77394/SWAGGYHELPER > project,
# and is released under the "MIT License".
# Please see < https://github.com/Yuki77394/SWAGGYHELPER/blob/main/LICENSE >
# All rights reserved.

from pyrogram import Client
from pyrogram.enums import ChatMemberStatus, ParseMode

import config

from ..logging import LOGGER


# Background tasks spawned by start(); kept referenced so asyncio does
# not garbage-collect them mid-flight.
_BACKGROUND_TASKS = set()


class Alone(Client):
    def __init__(self):
        LOGGER(__name__).info("Starting Bot...")
        super().__init__(
            name="SWAGGYMUSIC",
            api_id=config.API_ID,
            api_hash=config.API_HASH,
            bot_token=config.BOT_TOKEN,
            in_memory=True,
            parse_mode=ParseMode.HTML,
            max_concurrent_transmissions=7,
        )

    async def start(self):
        await super().start()
        self.id = self.me.id
        self.name = self.me.first_name + " " + (self.me.last_name or "")
        self.username = self.me.username
        self.mention = self.me.mention

        # Centralized logger — BOT STARTED event (the only logger-group
        # send outside the logger helpers is none: everything routes
        # through SWAGGYMUSIC.utils.logger).  The delivery gate keeps the
        # original semantics: an unreachable logger group aborts boot.
        started = False
        try:
            # Imported lazily: importing SWAGGYMUSIC.utils at module load
            # would create a circular import while the package is still
            # initialising (this file is imported by SWAGGYMUSIC.__init__).
            from ..utils.logger import log_bot_started

            started = await log_bot_started(self)
        except Exception:
            started = False
        if not started:
            try:
                import config as _cfg
                if not _cfg.LOGGER_ID:
                    LOGGER(__name__).error(
                        "LOGGER_ID is not configured — set it to the chat ID "
                        "of your log group/channel and add the bot there."
                    )
                else:
                    LOGGER(__name__).error(
                        "Bot has failed to access the log group/channel. Make "
                        "sure that you have added your bot to your log "
                        "group/channel and promoted it as an admin."
                    )
            except Exception:
                LOGGER(__name__).error(
                    "Bot has failed to access the log group/channel."
                )
            exit()

        a = await self.get_chat_member(config.LOGGER_ID, self.id)
        if a.status != ChatMemberStatus.ADMINISTRATOR:
            LOGGER(__name__).error(
                "Please promote your bot as an admin in your log group/channel."
            )
            exit()

        # The deletion archive channel is deliberately separate from LOGGER_ID.
        # Validate it only when configured, so existing deployments remain
        # backward-compatible until AUDIT_CHANNEL_ID is added to the env.
        if config.AUDIT_CHANNEL_ID:
            if config.AUDIT_CHANNEL_ID == config.LOGGER_ID:
                LOGGER(__name__).error(
                    "AUDIT_CHANNEL_ID must be different from LOGGER_ID: "
                    "the audit channel stores forwards; LOGGER_ID stores details."
                )
                exit()
            try:
                audit_member = await self.get_chat_member(config.AUDIT_CHANNEL_ID, self.id)
                audit_privileges = getattr(audit_member, "privileges", None)
                can_post = bool(
                    audit_member.status == ChatMemberStatus.ADMINISTRATOR
                    and audit_privileges
                    and getattr(audit_privileges, "can_post_messages", False)
                )
                if not can_post:
                    LOGGER(__name__).error(
                        "Please add the bot as an administrator with Post Messages "
                        "permission in AUDIT_CHANNEL_ID."
                    )
                    exit()
            except Exception as ex:
                LOGGER(__name__).error(
                    "Bot cannot access AUDIT_CHANNEL_ID. Check the chat ID and "
                    "administrator permissions (%s).", type(ex).__name__
                )
                exit()

        LOGGER(__name__).info(f"Channel-management Bot Started as {self.name}")

        # Recover deletion logs that a previous process left interrupted
        # (crashed between the Telegram send and its delivery bookkeeping).
        # Fire-and-forget: boot never waits on it and it never raises — a
        # recovery that cannot run must not keep the bot offline.
        try:
            import asyncio

            from ..utils.deletion_audit import recover_interrupted_deletion_logs

            task = asyncio.get_running_loop().create_task(
                recover_interrupted_deletion_logs()
            )
            _BACKGROUND_TASKS.add(task)
            task.add_done_callback(_BACKGROUND_TASKS.discard)
        except Exception:
            LOGGER(__name__).warning(
                "Interrupted-deletion-log recovery could not be scheduled."
            )

        # Periodic delivery-retry worker: failed / uncertain / stale-claimed
        # delivery records are re-swept from MongoDB on a fixed cadence with
        # per-record exponential backoff — recovery no longer depends on a
        # restart.  Fire-and-forget, idempotent per process, never raises.
        try:
            from ..utils.deletion_audit import start_delivery_retry_worker

            retry_task = start_delivery_retry_worker()
            if retry_task is not None:
                _BACKGROUND_TASKS.add(retry_task)
                retry_task.add_done_callback(_BACKGROUND_TASKS.discard)
        except Exception:
            LOGGER(__name__).warning(
                "Periodic delivery-retry worker could not be scheduled."
            )

    async def stop(self):
        await super().stop()
