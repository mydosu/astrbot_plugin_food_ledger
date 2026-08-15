"""
餐饮记账插件 (astrbot_plugin_food_ledger)

发给机器人文字或图片账单，让大模型识别金额和类别，确认后入库，随时可查账。
需要 AstrBot >= v4.16。
"""
from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.message_components import Image
from astrbot.api.star import Context, Star
from astrbot.core.utils.astrbot_path import get_astrbot_data_path

PLUGIN_NAME = "astrbot_plugin_food_ledger"
DEFAULT_CATEGORIES = "早餐,午餐,晚餐,夜宵,外卖,零食,饮品,水果,聚餐,食材采购,其他"
DRAFT_KV_KEY = "food_ledger_drafts"

# 识别账单时使用的系统提示词
SYSTEM_PROMPT_TEMPLATE = """你是一个专业的餐饮记账助手。用户会发送餐饮消费的账单（可能是文字描述、小票或手机支付截图），你需要从中提取每一笔消费并分类。

【分类体系】用户自定义的合法类别如下，分类用词必须严格一致：
{categories}

【输出要求】
1. 只输出一个 JSON 对象，不要包含任何其他文字、解释或 markdown 代码块标记。
2. JSON 格式如下：
{{
  "record_date": "YYYY-MM-DD，账单里没有明确日期时填 null，有日期则填账单上的日期",
  "items": [
    {{
      "amount": 数字金额（只保留数字，如 25 或 12.5）,
      "category": "类别（必须从分类体系中选择，无法归入任何类别时用「其他」）",
      "description": "简短描述（10 字以内，如 牛肉面+可乐）"
    }}
  ]
}}
3. 一张账单可能包含多笔（如小票有多行），请逐笔提取，不要合并。
4. 金额以元为单位；如果账单上是外币或单位异常，按字面数字记录并在 description 中注明。
5. 识别不出的模糊项不要瞎编，直接跳过。
6. 如果用户明确说了「AA」「每人」「人均」，按人均金额记录并在 description 中注明。
7. 如果内容中识别不出任何有效消费，items 输出空数组 []；无论如何只输出 JSON，禁止输出任何解释、抱歉或额外文字。"""

USER_PROMPT_TEMPLATE = """请识别以下餐饮账单并输出 JSON（只输出 JSON 对象本身，不要代码块标记，不要任何其他文字）：
{content}"""

# 图片转述模型用的提示词：把账单图片转成文字（省 token、方便后续解析）
VISION_TRANSCRIBE_PROMPT = """请转述这张餐饮账单图片（小票/支付截图）里的文字。
要求：
- 把看到的文字尽量原样分行抄写出来，重点是每一笔的：项目名称 + 金额
- 金额必须写清楚，如：牛肉面 25、合计 45.8
- 商家名、日期如有也写出来
- 只输出文字内容，不要任何分析或评论
- 看不清的字用 ? 代替
- 如果图片里看不到任何文字，只输出：无文字"""


class FoodLedgerPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig) -> None:
        super().__init__(context)
        self.config = config

        # 数据目录：data/plugin_data/<插件名>/（持久化数据要放 data 下，防止更新插件时被覆盖）
        self.plugin_name = getattr(self, "name", None) or PLUGIN_NAME
        self.data_dir = Path(get_astrbot_data_path()) / "plugin_data" / self.plugin_name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "ledger.db"
        self._init_db()

        # 待确认草稿：key = unified_msg_origin（不同群/私聊互不干扰）
        self._drafts: dict[str, dict[str, Any]] = {}
        self._drafts_loaded = False

    # ============================================================
    # 数据库
    # ============================================================
    def _get_conn(self) -> sqlite3.Connection:
        """获取一个 SQLite 连接（WAL 模式，读多写少场景并发更稳）。"""
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:
            pass
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS records (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id           TEXT    NOT NULL,
                    session_id        TEXT    NOT NULL,
                    category          TEXT    NOT NULL,
                    amount            REAL    NOT NULL,
                    description       TEXT    DEFAULT '',
                    record_date       TEXT    NOT NULL,
                    created_at        INTEGER NOT NULL,
                    raw_message       TEXT    DEFAULT '',
                    provider_id       TEXT    DEFAULT '',
                    vision_provider_id TEXT   DEFAULT ''
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_user_date ON records(user_id, record_date)"
            )
            conn.commit()
        finally:
            conn.close()

    def _insert_records(self, rows: list[tuple]) -> int:
        """批量写入记录，返回写入条数。"""
        conn = self._get_conn()
        try:
            cur = conn.executemany(
                "INSERT INTO records (user_id, session_id, category, amount, description, "
                "record_date, created_at, raw_message, provider_id, vision_provider_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            conn.commit()
            return len(rows)
        finally:
            conn.close()

    def _query_records(
        self, user_id: str, start: str, end: str, category: Optional[str] = None
    ) -> list[sqlite3.Row]:
        """按用户、日期范围（含端点）、可选类别查询记录。"""
        conn = self._get_conn()
        try:
            sql = (
                "SELECT * FROM records WHERE user_id=? AND record_date BETWEEN ? AND ?"
            )
            params: list[Any] = [user_id, start, end]
            if category:
                sql += " AND category=?"
                params.append(category)
            # TODO: 目前只取最近 200 条，账目多了之后考虑加分页
            sql += " ORDER BY record_date DESC, created_at DESC LIMIT 200"
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    # ============================================================
    # 草稿（待确认账单）：内存 + KV 双持久化，重启不丢
    # ============================================================
    async def _ensure_drafts(self) -> None:
        if not self._drafts_loaded:
            try:
                data = await self.get_kv_data(DRAFT_KV_KEY, None)
                if isinstance(data, dict):
                    self._drafts = data
            except Exception as e:
                logger.warning(f"草稿加载失败: {e}")
            self._drafts_loaded = True

    async def _save_drafts(self) -> None:
        try:
            await self.put_kv_data(DRAFT_KV_KEY, self._drafts)
        except Exception as e:
            logger.warning(f"草稿保存失败: {e}")

    # ============================================================
    # 工具函数
    # ============================================================
    @staticmethod
    def _extract_args(event: AstrMessageEvent, subnames: tuple[str, ...]) -> str:
        """从消息文本中剥离「/记账 <子指令>」前缀，返回剩余参数。

        兼容：指令前可有可无的斜杠、全角/半角空格、大小写别名。
        """
        s = event.message_str.strip()
        m = re.match(
            r"^[/／]?\s*(记账|ledger|账本)[\s　]+(" + "|".join(subnames) + r")[\s　]*(.*)$",
            s,
            re.I | re.S,
        )
        return m.group(3).strip() if m else ""

    @staticmethod
    def _get_images(event: AstrMessageEvent) -> list[Image]:
        """从消息链中提取所有图片组件（保留对象，便于取本地文件做压缩）。"""
        return [comp for comp in event.message_obj.message if isinstance(comp, Image)]

    def _get_categories(self) -> list[str]:
        """读取配置中的类别列表，保证至少包含「其他」。"""
        raw = self.config.get("categories") or DEFAULT_CATEGORIES
        cats = [c.strip() for c in str(raw).replace("，", ",").split(",") if c.strip()]
        if "其他" not in cats:
            cats.append("其他")
        return cats

    async def _resolve_text_provider(self, umo: str) -> Optional[str]:
        """解析本次「账单解析模型」（文本 → 结构化账目）。

        优先级：WebUI/指令配置的 llm_provider > 当前会话正在使用的模型。
        """
        configured = str(self.config.get("llm_provider") or "").strip()
        if configured:
            return configured
        try:
            return await self.context.get_current_chat_provider_id(umo)
        except Exception as e:
            logger.warning(f"获取当前会话模型提供商失败: {e}")
            return None

    def _get_vision_provider(self) -> Optional[str]:
        """获取配置的「图片转述模型」（图片 → 文字描述），未配置返回 None。"""
        return str(self.config.get("image_provider") or "").strip() or None

    # ============================================================
    # LLM 账单识别（两阶段：图片转述 → 结构化解析）
    # ============================================================
    async def _parse_bill(
        self,
        text_provider_id: str,
        vision_provider_id: Optional[str],
        text: str,
        images: list[Image],
    ) -> dict:
        """两阶段识别账单，返回解析后的 JSON dict 或 {"error": ...}。

        - 有图片：先由「图片转述模型」把图片转成文字描述（图片会先压缩，
          微信截图这类原图很大，小模型的上下文往往装不下）；
        - 再由「解析模型」把全部文字（用户描述 + 图片转述）整理成结构化账目。
        - 未配置图片转述模型时，回退为让解析模型直接看图（若其支持多模态）。
        """
        transcribed: Optional[str] = None
        if images:
            if vision_provider_id:
                transcribed, err = await self._transcribe_images(vision_provider_id, images)
                if err:
                    return {"error": err}
                logger.info(f"[记账] 图片转述结果: {transcribed[:300]!r}")
                if not transcribed or transcribed.strip() in ("无文字", "[无文字]"):
                    return {"error": "图片转述模型没有识别到任何文字。请检查 llama.cpp 是否加载了视觉投影文件（--mmproj，否则模型看不到图片），或换一个视觉模型"}
            else:
                logger.info("未配置图片转述模型，尝试用解析模型直接识别图片")
                image_urls = await self._prepare_image_urls(images)
                if image_urls is None:
                    return {"error": "图片处理失败（下载或压缩出错）"}
                return await self._llm_parse(
                    text_provider_id, text or "（图片账单，请仔细识别图片中的消费明细）", image_urls
                )
        parts: list[str] = []
        if text:
            parts.append(f"用户文字描述：\n{text}")
        if transcribed:
            parts.append(f"账单图片转述内容：\n{transcribed}")
        content = "\n\n".join(parts) if parts else "（无有效内容）"
        return await self._llm_parse(text_provider_id, content)

    async def _transcribe_images(
        self, vision_provider_id: str, images: list[Image]
    ) -> tuple[Optional[str], Optional[str]]:
        """调用图片转述模型把图片转成文字，返回 (转述文本, 错误信息)。

        图片会先压缩，避免大图把模型的上下文塞爆（本地小模型 context 通常只有 2k）。
        """
        image_urls = await self._prepare_image_urls(images)
        if image_urls is None:
            return None, "图片处理失败（下载或压缩出错）"
        logger.info(f"[记账] 发送给转述模型的图片格式: {image_urls[0][:80]!r}（共 {len(image_urls)} 张）")
        try:
            resp = await self.context.llm_generate(
                chat_provider_id=vision_provider_id,
                prompt=VISION_TRANSCRIBE_PROMPT,
                image_urls=image_urls,
            )
            return resp.completion_text, None
        except Exception as e:
            logger.error(f"图片转述失败: {e}")
            return None, f"图片转述失败（{vision_provider_id}）：{e}"

    async def _prepare_image_urls(self, images: list[Image]) -> Optional[list[str]]:
        """把图片组件统一转成压缩后的 data URL 列表；任何一张处理失败返回 None。"""
        urls: list[str] = []
        for img in images:
            try:
                path = await img.convert_to_file_path()
                urls.append(self._compress_image_to_base64(path))
            except Exception as e:
                logger.warning(f"图片处理失败: {e}")
                return None
        return urls

    def _compress_image_to_base64(self, path: str) -> str:
        """把本地图片压缩（最长边 ≤ 配置值，JPEG）并返回 data URL。

        微信账单截图分辨率很高，直接 base64 传给小模型会把 context 撑爆。
        """
        from PIL import Image as PILImage

        max_side = int(self.config.get("vision_max_side") or 768)
        quality = int(self.config.get("vision_jpeg_quality") or 80)
        with PILImage.open(path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = max_side / max(w, h)
            if scale < 1:
                im = im.resize(
                    (max(1, int(w * scale)), max(1, int(h * scale))), PILImage.LANCZOS
                )
            import base64
            import io

            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=quality)
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"

    async def _llm_parse(
        self, provider_id: str, content: str, image_urls: Optional[list[str]] = None
    ) -> dict:
        """调用解析模型把文字账单整理成结构化 JSON。"""
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            categories="、".join(self._get_categories())
        )
        kwargs: dict[str, Any] = {
            "chat_provider_id": provider_id,
            "prompt": USER_PROMPT_TEMPLATE.format(content=content),
            "system_prompt": system_prompt,
        }
        if image_urls:
            kwargs["image_urls"] = image_urls
        try:
            resp = await self.context.llm_generate(**kwargs)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return {"error": str(e)}
        parsed = self._parse_llm_json(resp.completion_text)
        if parsed.get("error"):
            snippet = (resp.completion_text or "").strip()[:120]
            logger.warning(f"[记账] 解析模型输出无法解析为 JSON，原文开头: {snippet!r}")
            parsed["error"] = f"{parsed['error']}（模型回复开头：{snippet!r}）"
        return parsed

    @staticmethod
    def _parse_llm_json(text: str) -> dict:
        """容错解析模型输出中的 JSON（兼容 ```json 代码块、前后缀杂文）。"""
        text = (text or "").strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
        if m:
            text = m.group(1).strip()
        try:
            obj = json.loads(text)
            return obj if isinstance(obj, dict) else {"error": "模型输出不是 JSON 对象"}
        except (json.JSONDecodeError, TypeError):
            pass
        m2 = re.search(r"\{.*\}", text, re.S)
        if m2:
            try:
                obj = json.loads(m2.group(0))
                return obj if isinstance(obj, dict) else {"error": "模型输出不是 JSON 对象"}
            except (json.JSONDecodeError, TypeError):
                pass
        return {"error": "模型输出无法解析为 JSON"}

    def _validate_items(self, data: dict) -> tuple[list[dict], list[str]]:
        """校验并规范化模型识别出的条目，返回 (有效条目, 警告信息)。"""
        max_items = int(self.config.get("max_items") or 20)
        items = data.get("items") if isinstance(data, dict) else None
        warnings: list[str] = []
        if not isinstance(items, list):
            return [], ["模型输出缺少 items 字段"]
        if not items:
            return [], ["未识别到任何账单条目"]
        cats = self._get_categories()
        valid: list[dict] = []
        for i, it in enumerate(items, 1):
            if not isinstance(it, dict):
                warnings.append(f"第 {i} 条格式无效，已跳过")
                continue
            try:
                raw_amount = str(it.get("amount", "")).replace("￥", "").replace("¥", "").replace("元", "").strip()
                amount = float(raw_amount)
            except (TypeError, ValueError):
                warnings.append(f"第 {i} 条金额「{it.get('amount')}」无效，已跳过")
                continue
            if amount <= 0:
                warnings.append(f"第 {i} 条金额 {amount} 需大于 0，已跳过")
                continue
            cat = str(it.get("category", "")).strip()
            if cat not in cats:
                warnings.append(f"第 {i} 条类别「{cat}」不在预设类别中，已归入「其他」")
                cat = "其他"
            desc = str(it.get("description", "")).strip()[:50]
            valid.append({"amount": round(amount, 2), "category": cat, "description": desc})
            if len(valid) >= max_items:
                warnings.append(f"条目数超过上限 {max_items}，其余已忽略")
                break
        return valid, warnings

    # ============================================================
    # 输出格式化
    # ============================================================
    @staticmethod
    def _fmt_dt(ts: int) -> str:
        """unix 时间戳 → MM-DD HH:MM（本地时区），查账/草稿展示用。"""
        return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")

    def _format_draft(self, draft: dict, warnings: list[str] | None = None) -> str:
        items = draft["items"]
        total = sum(it["amount"] for it in items)
        currency = str(self.config.get("currency_symbol") or "元")
        lines = [f"📋 待确认账单（共 {len(items)} 笔）｜{self._fmt_dt(draft.get('created_at') or int(time.time()))}"]
        if draft.get("record_date"):
            lines[0] += f"｜账单日期：{draft['record_date']}"
        for i, it in enumerate(items, 1):
            lines.append(f"{i}. {it['category']}｜{it['amount']:.2f} {currency}｜{it['description'] or '—'}")
        lines.append("——————————————")
        lines.append(f"💰 合计：{total:.2f} {currency}")
        if warnings:
            lines.append("")
            lines.extend("⚠️ " + w for w in warnings)
        lines.append("")
        lines.append("✅ 回复「/记账 确认」入账")
        lines.append("✏️ 修改：「/记账 修改 1 金额 15」「/记账 修改 2 类别 晚餐」「/记账 修改 3 描述 牛肉面」")
        lines.append("🚫 放弃：「/记账 取消」")
        return "\n".join(lines)

    def _format_query_result(
        self, start: str, end: str, rows: list[sqlite3.Row], category: Optional[str]
    ) -> str:
        currency = str(self.config.get("currency_symbol") or "元")
        total = sum(r["amount"] for r in rows)
        lines = ["📊 餐饮账目统计"]
        lines.append(f"📅 {start} ~ {end}" + (f"｜类别：{category}" if category else ""))
        lines.append(f"💰 总支出：{total:.2f} {currency}（{len(rows)} 笔）")
        if not rows:
            lines.append("\n暂无记录。")
            return "\n".join(lines)

        # 按类别汇总（金额降序）
        by_cat: dict[str, list[sqlite3.Row]] = {}
        for r in rows:
            by_cat.setdefault(r["category"], []).append(r)
        lines.append("\n🗂️ 按类别：")
        for cat, rs in sorted(by_cat.items(), key=lambda kv: -sum(x["amount"] for x in kv[1])):
            lines.append(f"· {cat}：{sum(x['amount'] for x in rs):.2f} {currency}（{len(rs)} 笔）")

        lines.append("\n📋 明细：")
        for idx, r in enumerate(reversed(rows), 1):  # 倒序查询，反转回时间正序
            lines.append(
                f"#{idx} {self._fmt_dt(r['created_at'])} {r['category']} {r['amount']:.2f} {r['description'] or '—'}"
            )
        lines.append("\n提示：可查询昨天/本月/上月，如「/记账 查账 本月」")
        return "\n".join(lines)

    # ============================================================
    # 指令组：/记账（alias: ledger / 账本）
    # ============================================================
    @filter.command_group("记账", alias={"ledger", "账本"})
    def ledger(self) -> None:
        """餐饮记账插件：AI 识别账单并分类记账。"""
        pass

    # ---------- 自动识别：收到带图片的消息就直接尝试记账，不用输指令 ----------
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_any_message(self, event: AstrMessageEvent):
        """带图片的消息自动尝试记账。

        - 识别出账单 → 输出草稿并 stop_event，避免 AstrBot 再当普通聊天回复
        - 识别失败：
          - 配置/模型问题（未配置模型、调用失败）→ 明确提示用户，方便排查
          - 图里没有账单内容 → 静默放行，让 AstrBot 正常处理
        - 可通过插件配置「自动识别图片账单」关闭
        """
        if not self.config.get("auto_image_ledger", True):
            return
        images = self._get_images(event)
        if not images:
            logger.debug("[记账] 消息无图片，跳过自动识别")
            return
        text = event.message_str.strip()
        # 微信等平台纯图片消息的文本占位符，不是用户输入
        if text in ("[未知消息]", "[图片]", "[Image]"):
            text = ""
        # 指令消息交给指令 handler 处理，这里跳过
        if re.match(r"^[/／]?\s*(记账|ledger|账本)", text, re.I):
            return
        await self._ensure_drafts()
        yield event.plain_result("🧾 收到图片，正在识别账单，可能需要一点时间…")
        logger.info(f"[记账] 收到图片消息，共 {len(images)} 张，尝试自动识别")
        draft, warnings, err, err_kind = await self._recognize(event, text, images)
        if err:
            if err_kind == "config":
                # 配置或模型问题：提示用户，但仍放行给 AstrBot 正常处理
                yield event.plain_result(f"⚠️ 图片自动记账没成功：{err}")
            logger.warning(f"[记账] 自动识别未成功（{err_kind}）: {err}")
            return
        for w in warnings:
            yield event.plain_result("⚠️ " + w)
        yield event.plain_result(self._format_draft(draft))
        event.stop_event()

    async def _recognize(
        self, event: AstrMessageEvent, text: str, images: list[Image]
    ) -> tuple[Optional[dict], list[str], Optional[str], Optional[str]]:
        """识别账单并生成待确认草稿（指令和自动识别共用）。

        返回 (draft, warnings, error_msg, error_kind)：
        - error_kind = "config"：模型配置或调用问题，需要用户处理
        - error_kind = "content"：没识别出账单内容（图里没有账单，属正常情况）
        - error_msg 为 None 时 error_kind 也为 None，draft 已存入 self._drafts 并持久化。
        """
        umo = event.unified_msg_origin
        provider_id = await self._resolve_text_provider(umo)
        if not provider_id:
            return None, [], "未找到可用的解析模型。请在 AstrBot WebUI 插件配置中设置「账单解析模型」。", "config"
        vision_provider_id = self._get_vision_provider()
        warnings: list[str] = []
        if images and not vision_provider_id:
            warnings.append("尚未配置「图片转述模型」，正在用解析模型直接识别图片，效果可能一般")
        logger.info(f"[记账] 图片转述模型: {vision_provider_id or '未配置（回退给解析模型）'}，解析模型: {provider_id}")
        data = await self._parse_bill(provider_id, vision_provider_id, text, images)
        if data.get("error"):
            return None, [], f"识别失败：{data['error']}", "config"
        items, validate_warnings = self._validate_items(data)
        warnings.extend(validate_warnings)
        if not items:
            msg = "未能从账单中识别出有效条目，请确认图片清晰或文字描述完整"
            if validate_warnings:
                msg += "（" + "；".join(validate_warnings) + "）"
            return None, [], msg, "content"
        draft = {
            "provider_id": provider_id,
            "vision_provider_id": vision_provider_id or "",
            "created_at": int(time.time()),
            "items": items,
            "record_date": data.get("record_date") or None,
            "raw": text[:100] if text else "图片账单",
        }
        self._drafts[umo] = draft
        await self._save_drafts()
        return draft, warnings, None, None

    # ---------- 记：识别账单 ----------
    @ledger.command("记", alias={"record", "add", "记一笔"})
    async def record(self, event: AstrMessageEvent):
        """识别并记录账单：/记账 记 <文字或账单图片>"""
        await self._ensure_drafts()
        args = self._extract_args(event, ("记", "record", "add", "记一笔"))
        images = self._get_images(event)
        text = args
        if not text and not images:
            yield event.plain_result(
                "📝 请发送账单内容，例如：\n"
                "· /记账 记 早餐包子豆浆 8.5 元\n"
                "· /记账 记 帮我记一下昨晚聚餐 AA 每人 66\n"
                "· 直接发送账单截图（图片消息，会自动识别）"
            )
            return
        yield event.plain_result("🧾 正在识别账单，请稍候…")
        draft, warnings, err, _ = await self._recognize(event, text, images)
        if err:
            yield event.plain_result("❌ " + err)
            return
        yield event.plain_result(self._format_draft(draft, warnings))

    # ---------- 确认：入账 ----------
    @ledger.command("确认", alias={"确认入账", "ok", "commit"})
    async def confirm(self, event: AstrMessageEvent):
        """确认草稿并写入数据库：/记账 确认"""
        await self._ensure_drafts()
        umo = event.unified_msg_origin
        draft = self._drafts.get(umo)
        if not draft:
            yield event.plain_result("❌ 当前没有待确认的账单。请先发送「/记账 记 …」或账单图片。")
            return
        items = draft["items"]
        rd = draft.get("record_date") or date.today().isoformat()
        now = int(time.time())
        rows = [
            (
                event.get_sender_id(),
                umo,
                it["category"],
                it["amount"],
                it["description"],
                rd,
                now,
                draft.get("raw", ""),
                draft.get("provider_id", ""),
                draft.get("vision_provider_id", ""),
            )
            for it in items
        ]
        try:
            n = await asyncio.to_thread(self._insert_records, rows)
        except Exception as e:
            logger.error(f"账单入库失败: {e}")
            yield event.plain_result(f"❌ 入库失败：{e}")
            return
        del self._drafts[umo]
        await self._save_drafts()
        total = sum(it["amount"] for it in items)
        currency = str(self.config.get("currency_symbol") or "元")
        yield event.plain_result(
            f"✅ 已入账 {n} 笔，共 {total:.2f} {currency}（{rd}，{self._fmt_dt(now)} 记）\n"
            f"「/记账 查账」查看统计，或直接发图片继续记账。"
        )

    # ---------- 修改：调整草稿 ----------
    @ledger.command("修改", alias={"edit"})
    async def edit(self, event: AstrMessageEvent):
        """修改待确认草稿：/记账 修改 <序号> <金额|类别|描述> <新值>"""
        await self._ensure_drafts()
        umo = event.unified_msg_origin
        draft = self._drafts.get(umo)
        if not draft:
            yield event.plain_result("❌ 当前没有待确认的账单。")
            return
        args = self._extract_args(event, ("修改", "edit"))
        parts = re.split(r"[\s　]+", args.strip(), maxsplit=2)
        if len(parts) < 3:
            yield event.plain_result(
                "用法：/记账 修改 <序号> <金额|类别|描述> <新值>\n"
                "示例：/记账 修改 1 金额 15.5｜/记账 修改 2 类别 晚餐｜/记账 修改 3 描述 牛肉面"
            )
            return
        try:
            idx = int(parts[0])
        except ValueError:
            yield event.plain_result("❌ 序号必须是数字。")
            return
        if not 1 <= idx <= len(draft["items"]):
            yield event.plain_result(f"❌ 序号超出范围（1-{len(draft['items'])}）。")
            return
        field, value = parts[1], parts[2].strip()
        item = draft["items"][idx - 1]
        if field in ("金额", "amount", "价格", "钱"):
            try:
                v = round(float(value), 2)
            except ValueError:
                yield event.plain_result("❌ 金额必须是数字。")
                return
            if v <= 0:
                yield event.plain_result("❌ 金额必须大于 0。")
                return
            item["amount"] = v
        elif field in ("类别", "category", "分类"):
            cats = self._get_categories()
            if value not in cats:
                yield event.plain_result(
                    f"❌ 类别「{value}」不存在。可用：{'、'.join(cats)}\n"
                    f"或先添加：/记账 类别 添加 {value}"
                )
                return
            item["category"] = value
        elif field in ("描述", "description", "备注", "desc"):
            item["description"] = value[:50]
        else:
            yield event.plain_result("❌ 未知字段，仅支持：金额 / 类别 / 描述")
            return
        await self._save_drafts()
        yield event.plain_result("✏️ 已修改：\n" + self._format_draft(draft))

    # ---------- 取消：放弃草稿 ----------
    @ledger.command("取消", alias={"放弃", "cancel"})
    async def cancel(self, event: AstrMessageEvent):
        """放弃当前待确认账单：/记账 取消"""
        await self._ensure_drafts()
        umo = event.unified_msg_origin
        if umo in self._drafts:
            del self._drafts[umo]
            await self._save_drafts()
            yield event.plain_result("🗑️ 已取消本次记录。")
        else:
            yield event.plain_result("❌ 当前没有待确认的账单。")

    # ---------- 查账 ----------
    @ledger.command("查账", alias={"查看", "查询", "账单", "stats"})
    async def query(self, event: AstrMessageEvent):
        """查账：/记账 查账 [今天|昨天|本月|上月|YYYY-MM-DD|起始日 结束日] [类别]"""
        args = self._extract_args(event, ("查账", "查看", "查询", "账单", "stats"))
        rng, cat = self._parse_query(args)
        if rng is None:
            yield event.plain_result(
                "❌ 无法解析查询条件。用法：/记账 查账 [今天|昨天|本月|上月|YYYY-MM-DD|YYYY-MM-DD YYYY-MM-DD] [类别名]"
            )
            return
        start, end = rng
        user_id = event.get_sender_id()
        try:
            rows = await asyncio.to_thread(self._query_records, user_id, start, end, cat)
        except Exception as e:
            logger.error(f"查账失败: {e}")
            yield event.plain_result(f"❌ 查询失败：{e}")
            return
        yield event.plain_result(self._format_query_result(start, end, rows, cat))

    def _parse_query(self, args: str) -> tuple[Optional[tuple[str, str]], Optional[str]]:
        """解析查账参数，返回 ((start, end) | None, category | None)。"""
        args = args.strip()
        today = date.today()
        if not args:
            return (today.isoformat(), today.isoformat()), None
        tokens = re.split(r"[\s　]+", args)
        cat: Optional[str] = None
        # 仅当最后一个 token 命中预设类别时才视为类别过滤
        cats = self._get_categories()
        if len(tokens) >= 2 and tokens[-1] in cats:
            cat = tokens[-1]
            tokens = tokens[:-1]
        if not tokens:
            return (today.isoformat(), today.isoformat()), cat
        tok = tokens[0]
        if tok in ("今天", "今日", "today"):
            return (today.isoformat(), today.isoformat()), cat
        if tok in ("昨天", "yesterday"):
            d = today - timedelta(days=1)
            return (d.isoformat(), d.isoformat()), cat
        if tok in ("本月", "this_month"):
            return (today.replace(day=1).isoformat(), today.isoformat()), cat
        if tok in ("上月", "last_month"):
            last_day = today.replace(day=1) - timedelta(days=1)
            return (last_day.replace(day=1).isoformat(), last_day.isoformat()), cat
        if len(tokens) >= 2:
            try:
                d1 = datetime.strptime(tokens[0], "%Y-%m-%d").date()
                d2 = datetime.strptime(tokens[1], "%Y-%m-%d").date()
                if d1 > d2:
                    d1, d2 = d2, d1
                return (d1.isoformat(), d2.isoformat()), cat
            except ValueError:
                pass
        try:
            d = datetime.strptime(tok, "%Y-%m-%d").date()
            return (d.isoformat(), d.isoformat()), cat
        except ValueError:
            return None, None

    # ---------- 类别管理 ----------
    @ledger.command("类别", alias={"category", "分类"})
    async def category(self, event: AstrMessageEvent):
        """查看/添加类别：/记账 类别 ｜ /记账 类别 添加 咖啡"""
        args = self._extract_args(event, ("类别", "category", "分类"))
        cats = self._get_categories()
        m = re.match(r"^(添加|add)\s*(.+)$", args, re.I | re.S)
        if m:
            name = m.group(2).strip()
            if name in cats:
                yield event.plain_result(f"⚠️ 类别「{name}」已存在。")
                return
            cats.append(name)
            self.config["categories"] = ",".join(cats)
            self.config.save_config()
            yield event.plain_result(
                f"✅ 已添加类别「{name}」。\n当前类别：{'、'.join(cats)}"
            )
            return
        yield event.plain_result(
            f"当前类别（{len(cats)} 个）：\n{'、'.join(cats)}\n\n"
            "添加类别：/记账 类别 添加 咖啡\n"
            "类别也可在 WebUI 插件配置中修改。"
        )

    # ---------- 帮助 ----------
    @ledger.command("帮助", alias={"help", "用法", "?"})
    async def help_cmd(self, event: AstrMessageEvent):
        """餐饮记账插件使用说明"""
        yield event.plain_result(
            "📒 餐饮记账插件使用说明\n"
            "——————————————\n"
            "🧾 记账（AI 识别）：\n"
            "· 直接发账单截图/支付截图，自动识别记账\n"
            "· /记账 记 早餐包子豆浆 8.5 元\n"
            "· /记账 记 昨晚聚餐 AA 每人 66\n"
            "图片账单：先由「图片转述模型」转成文字，\n"
            "再由「解析模型」整理成账目，结果先审查后入库。\n"
            "——————————————\n"
            "✅ /记账 确认 —— 确认草稿并入账\n"
            "✏️ /记账 修改 1 金额 15 —— 修改草稿条目\n"
            "🚫 /记账 取消 —— 放弃草稿\n"
            "📊 /记账 查账 [今天|昨天|本月|上月|日期|日期 日期] [类别]\n"
            "🏷️ /记账 类别 [添加 xxx] —— 管理类别\n"
            "🤖 模型设置请在 AstrBot WebUI 插件配置中进行\n"
            "   （图片转述模型 / 账单解析模型）\n"
            "——————————————\n"
            "数据保存在 AstrBot data/plugin_data 目录下，卸载插件不丢失。"
        )

    # ============================================================
    # 生命周期
    # ============================================================
    @filter.on_astrbot_loaded()
    async def on_loaded(self) -> None:
        """AstrBot 初始化完成后恢复草稿。"""
        await self._ensure_drafts()

    async def terminate(self) -> None:
        """插件卸载/停用时保存草稿。"""
        await self._save_drafts()
