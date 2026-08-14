"""本地验证 astrbot_plugin_food_ledger 核心逻辑（mock astrbot 环境）。"""
import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

# ---------------- mock astrbot 包 ----------------
def make_module(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m

astrbot = make_module("astrbot")
astrbot_api = make_module("astrbot.api")
astrbot_event = make_module("astrbot.api.event")
astrbot_star = make_module("astrbot.api.star")
astrbot_mc = make_module("astrbot.api.message_components")
astrbot_core = make_module("astrbot.core")
astrbot_utils = make_module("astrbot.core.utils")
astrbot_path = make_module("astrbot.core.utils.astrbot_path")

class AstrBotConfig(dict):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._saved = False
    def save_config(self):
        self._saved = True

class Image:
    def __init__(self, file=None, url=None, **kw):
        self.file = file
        self.url = url

class Star:
    _shared_kv: dict = {}

    def __init__(self, context):
        self.context = context
        self.name = "astrbot_plugin_food_ledger"
    async def put_kv_data(self, k, v):
        type(self)._shared_kv[k] = v
    async def get_kv_data(self, k, default=None):
        return type(self)._shared_kv.get(k, default)
    async def delete_kv_data(self, k):
        type(self)._shared_kv.pop(k, None)

# filter 装饰器：直接返回原函数（不真正注册）
def _passthrough(*a, **kw):
    def deco(fn):
        return fn
    return deco

class _CmdGroup:
    def __init__(self, name, alias):
        self.name = name
        self.alias = alias
    def command(self, name, alias=None, **kw):
        return _passthrough()
    def group(self, name, **kw):
        return _passthrough()

class _EventMessageType:
    ALL = "all"
    PRIVATE_MESSAGE = "private"
    GROUP_MESSAGE = "group"

filter_mod = types.ModuleType("astrbot.api.event.filter")
filter_mod.EventMessageType = _EventMessageType
filter_mod.command = _passthrough
def _cmd_group(name, alias=None, **kw):
    def deco(fn):
        return _CmdGroup(name, alias)
    return deco
filter_mod.command_group = _cmd_group
filter_mod.event_message_type = _passthrough
filter_mod.platform_adapter_type = _passthrough
filter_mod.permission_type = _passthrough
filter_mod.on_astrbot_loaded = _passthrough
astrbot_event.filter = filter_mod
sys.modules["astrbot.api.event.filter"] = filter_mod

class AstrMessageEvent:
    def __init__(self, message_str="", message=None, umo="test:group:123", sender_id="u1"):
        self.message_str = message_str
        self.message_obj = types.SimpleNamespace(message=message or [])
        self.unified_msg_origin = umo
        self._sender_id = sender_id
        self._stopped = False
    def get_sender_id(self):
        return self._sender_id
    def get_sender_name(self):
        return "tester"
    def plain_result(self, text):
        return text
    def stop_event(self):
        self._stopped = True

astrbot_event.AstrMessageEvent = AstrMessageEvent

class Context:
    def __init__(self):
        self.providers = []
    def get_all_providers(self):
        return self.providers
    async def get_current_chat_provider_id(self, umo):
        return "default"
    async def llm_generate(self, **kwargs):
        raise NotImplementedError

class FakeProvider:
    def __init__(self, pid, model, name=None):
        self.provider_config = {"id": pid, "name": name or pid}
        self._model = model
    def meta(self):
        return types.SimpleNamespace(id=self.provider_config["id"], model=self._model)

astrbot_path.get_astrbot_data_path = lambda: str(TMP_DATA)
astrbot_api.AstrBotConfig = AstrBotConfig
astrbot_api.logger = types.SimpleNamespace(info=print, warning=print, error=print, debug=print)
astrbot_mc.Image = Image
astrbot_star.Context = Context
astrbot_star.Star = Star
sys.modules["astrbot.api"] = astrbot_api
sys.modules["astrbot.api.event"] = astrbot_event
sys.modules["astrbot.api.star"] = astrbot_star
sys.modules["astrbot.api.message_components"] = astrbot_mc
sys.modules["astrbot.core"] = astrbot_core
sys.modules["astrbot.core.utils"] = astrbot_utils
sys.modules["astrbot.core.utils.astrbot_path"] = astrbot_path

TMP_DATA = tempfile.mkdtemp(prefix="ledger_test_")

# ---------------- 导入插件 ----------------
sys.path.insert(0, str(Path(__file__).parent))
import main as plugin_mod  # noqa: E402


class TestLLMParsing(unittest.TestCase):
    def test_plain_json(self):
        out = plugin_mod.FoodLedgerPlugin._parse_llm_json('{"items": [{"amount": 12.5, "category": "早餐", "description": "包子"}]}')
        self.assertIn("items", out)

    def test_fenced_json(self):
        out = plugin_mod.FoodLedgerPlugin._parse_llm_json('```json\n{"items": []}\n```')
        self.assertEqual(out["items"], [])

    def test_json_with_surrounding_text(self):
        out = plugin_mod.FoodLedgerPlugin._parse_llm_json(
            '好的，已识别：\n{"items": [{"amount": 25, "category": "午餐"}]}\n以上是结果。'
        )
        self.assertEqual(out["items"][0]["amount"], 25)

    def test_invalid_output(self):
        out = plugin_mod.FoodLedgerPlugin._parse_llm_json("抱歉我无法识别")
        self.assertIn("error", out)


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.plugin = plugin_mod.FoodLedgerPlugin(
            Context(), AstrBotConfig({})
        )

    def test_valid_items(self):
        items, warns = self.plugin._validate_items({
            "items": [
                {"amount": 12.5, "category": "早餐", "description": "包子"},
                {"amount": "25元", "category": "午餐", "description": "牛肉面"},
                {"amount": "¥8", "category": "饮品", "description": "奶茶"},
            ]
        })
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0]["amount"], 12.5)
        self.assertEqual(items[1]["amount"], 25.0)
        self.assertEqual(items[2]["amount"], 8.0)
        self.assertEqual(warns, [])

    def test_unknown_category_falls_back(self):
        items, warns = self.plugin._validate_items({
            "items": [{"amount": 10, "category": "游戏充值", "description": "x"}]
        })
        self.assertEqual(items[0]["category"], "其他")
        self.assertTrue(any("游戏充值" in w for w in warns))

    def test_invalid_amount_skipped(self):
        items, warns = self.plugin._validate_items({
            "items": [
                {"amount": "abc", "category": "早餐", "description": "x"},
                {"amount": -5, "category": "早餐", "description": "x"},
                {"amount": 0, "category": "早餐", "description": "x"},
            ]
        })
        self.assertEqual(items, [])
        self.assertEqual(len(warns), 3)

    def test_missing_items(self):
        items, warns = self.plugin._validate_items({})
        self.assertEqual(items, [])
        self.assertEqual(warns, ["模型输出缺少 items 字段"])

    def test_max_items(self):
        self.plugin.config["max_items"] = 2
        items, warns = self.plugin._validate_items({
            "items": [{"amount": 1, "category": "早餐"}, {"amount": 2, "category": "午餐"},
                      {"amount": 3, "category": "晚餐"}]
        })
        self.assertEqual(len(items), 2)
        self.assertTrue(any("上限" in w for w in warns))


class TestQueryParsing(unittest.TestCase):
    def setUp(self):
        self.plugin = plugin_mod.FoodLedgerPlugin(
            Context(), AstrBotConfig({})
        )

    def test_default_today(self):
        rng, cat = self.plugin._parse_query("")
        self.assertEqual(rng[0], rng[1])

    def test_this_month(self):
        rng, cat = self.plugin._parse_query("本月")
        self.assertTrue(rng[0].endswith("-01"))

    def test_date_range(self):
        rng, cat = self.plugin._parse_query("2026-08-01 2026-08-31")
        self.assertEqual(rng, ("2026-08-01", "2026-08-31"))

    def test_reversed_range(self):
        rng, cat = self.plugin._parse_query("2026-08-31 2026-08-01")
        self.assertEqual(rng, ("2026-08-01", "2026-08-31"))

    def test_date_with_category(self):
        rng, cat = self.plugin._parse_query("2026-08-01 午餐")
        self.assertEqual(rng, ("2026-08-01", "2026-08-01"))
        self.assertEqual(cat, "午餐")

    def test_garbage(self):
        rng, cat = self.plugin._parse_query("不是日期")
        self.assertIsNone(rng)


class TestExtractArgs(unittest.TestCase):
    def test_basic(self):
        ev = AstrMessageEvent(message_str="/记账 记 午餐牛肉面25元")
        out = plugin_mod.FoodLedgerPlugin._extract_args(ev, ("记", "record", "add", "记一笔"))
        self.assertEqual(out, "午餐牛肉面25元")

    def test_alias_and_no_slash(self):
        ev = AstrMessageEvent(message_str="ledger record 咖啡 15")
        out = plugin_mod.FoodLedgerPlugin._extract_args(ev, ("记", "record", "add", "记一笔"))
        self.assertEqual(out, "咖啡 15")

    def test_fullwidth_space(self):
        ev = AstrMessageEvent(message_str="／记账　记　夜宵烧烤 50")
        out = plugin_mod.FoodLedgerPlugin._extract_args(ev, ("记", "record", "add", "记一笔"))
        self.assertEqual(out, "夜宵烧烤 50")


class TestTwoStage(unittest.TestCase):
    """两阶段识别：图片转述模型 + 解析模型"""

    def test_two_stage_flow(self):
        """配置了图片转述模型：先转述（带图）再解析（纯文字）"""

        async def run():
            calls = []
            ctx = Context()
            ctx.providers = [FakeProvider("vision-x", "qwen-vl"), FakeProvider("text-y", "gpt-4o")]

            async def llm(**kw):
                calls.append(kw)
                if kw.get("image_urls"):
                    # 阶段1：图片转述模型
                    self.assertEqual(kw["chat_provider_id"], "vision-x")
                    self.assertIn("转述", kw["prompt"])
                    self.assertEqual(kw["image_urls"], ["https://img/1.jpg"])
                    return types.SimpleNamespace(completion_text="早点铺：豆浆2元、包子6元，合计8元")
                # 阶段2：解析模型（纯文字，不带图）
                self.assertEqual(kw["chat_provider_id"], "text-y")
                self.assertIn("账单图片转述内容", kw["prompt"])
                self.assertIsNone(kw.get("image_urls"))
                return types.SimpleNamespace(completion_text=json.dumps({
                    "items": [{"amount": 8, "category": "早餐", "description": "豆浆包子"}]
                }))

            ctx.llm_generate = llm
            p = plugin_mod.FoodLedgerPlugin(ctx, AstrBotConfig({}))
            data = await p._parse_bill("text-y", "vision-x", "", ["https://img/1.jpg"])
            self.assertNotIn("error", data)
            self.assertEqual(len(calls), 2)
            items, _ = p._validate_items(data)
            self.assertEqual(items[0]["amount"], 8.0)

        asyncio.run(run())

    def test_fallback_no_vision_provider(self):
        """未配置图片转述模型：图片直接交给解析模型（一次调用，带图）"""

        async def run():
            calls = []
            ctx = Context()

            async def llm(**kw):
                calls.append(kw)
                self.assertIsNotNone(kw.get("image_urls"))
                return types.SimpleNamespace(completion_text='{"items": []}')

            ctx.llm_generate = llm
            p = plugin_mod.FoodLedgerPlugin(ctx, AstrBotConfig({}))
            data = await p._parse_bill("text-y", None, "", ["https://img/1.jpg"])
            self.assertEqual(len(calls), 1)
            self.assertIn("image_urls", calls[0])

        asyncio.run(run())

    def test_text_only_single_call(self):
        """纯文字账单：只调解析模型一次，不带图，包含用户描述"""

        async def run():
            calls = []
            ctx = Context()

            async def llm(**kw):
                calls.append(kw)
                return types.SimpleNamespace(completion_text='{"items": [{"amount": 15, "category": "午餐"}]}')

            ctx.llm_generate = llm
            p = plugin_mod.FoodLedgerPlugin(ctx, AstrBotConfig({}))
            data = await p._parse_bill("text-y", "vision-x", "午餐 15", [])
            self.assertEqual(len(calls), 1)
            self.assertNotIn("image_urls", calls[0])
            self.assertIn("用户文字描述", calls[0]["prompt"])

        asyncio.run(run())

    def test_vision_failure_stops_pipeline(self):
        """图片转述失败：直接返回错误，不再调用解析模型"""

        async def run():
            calls = []
            ctx = Context()

            async def llm(**kw):
                calls.append(kw)
                if kw.get("image_urls"):
                    raise RuntimeError("vision provider down")
                return types.SimpleNamespace(completion_text="{}")

            ctx.llm_generate = llm
            p = plugin_mod.FoodLedgerPlugin(ctx, AstrBotConfig({}))
            data = await p._parse_bill("text-y", "vision-x", "", ["https://img/1.jpg"])
            self.assertIn("error", data)
            self.assertIn("图片转述失败", data["error"])
            self.assertEqual(len(calls), 1)

        asyncio.run(run())


class TestAutoLedger(unittest.TestCase):
    """图片自动记账（无需指令）+ 记录时间展示"""

    def setUp(self):
        plugin_mod.FoodLedgerPlugin._shared_kv.clear()

    def _plugin(self, cfg=None, llm_ok=True):
        ctx = Context()
        ctx.providers = [FakeProvider("text-y", "gpt-4o")]

        async def llm(**kw):
            if not llm_ok:
                raise RuntimeError("provider down")
            return types.SimpleNamespace(completion_text=json.dumps({
                "items": [{"amount": 15, "category": "午餐", "description": "牛肉面"}]
            }))

        ctx.llm_generate = llm
        return plugin_mod.FoodLedgerPlugin(ctx, AstrBotConfig(cfg or {}))

    def test_image_auto_ledger(self):
        """纯图片消息 → 自动识别出草稿 + stop_event"""

        async def run():
            p = self._plugin()
            ev = AstrMessageEvent(message_str="", message=[Image(url="https://img/bill.jpg")], umo="t:g:auto1")
            out = [r async for r in p.on_any_message(ev)]
            self.assertTrue(any("待确认账单" in str(r) for r in out))
            self.assertTrue(ev._stopped)          # 阻止 AstrBot 再当普通聊天回复
            self.assertIn("t:g:auto1", p._drafts)  # 草稿已建
        asyncio.run(run())

    def test_plain_text_ignored(self):
        """纯文字消息不触发"""

        async def run():
            p = self._plugin()
            ev = AstrMessageEvent(message_str="今天天气不错", umo="t:g:auto2")
            out = [r async for r in p.on_any_message(ev)]
            self.assertEqual(out, [])
            self.assertFalse(ev._stopped)
        asyncio.run(run())

    def test_command_message_skipped(self):
        """带记账指令前缀的消息跳过（交给指令 handler）"""

        async def run():
            p = self._plugin()
            ev = AstrMessageEvent(
                message_str="/记账 记 午餐 15",
                message=[Image(url="https://img/b.jpg")],
                umo="t:g:auto3",
            )
            out = [r async for r in p.on_any_message(ev)]
            self.assertEqual(out, [])
        asyncio.run(run())

    def test_disabled_by_config(self):
        """配置关闭后不触发"""

        async def run():
            p = self._plugin(cfg={"auto_image_ledger": False})
            ev = AstrMessageEvent(message_str="", message=[Image(url="https://img/b.jpg")], umo="t:g:auto4")
            out = [r async for r in p.on_any_message(ev)]
            self.assertEqual(out, [])
        asyncio.run(run())

    def test_config_failure_notifies(self):
        """模型调用失败（配置类问题）→ 提示用户，但不 stop，让 AstrBot 正常处理"""

        async def run():
            p = self._plugin(llm_ok=False)
            ev = AstrMessageEvent(message_str="", message=[Image(url="https://img/photo.jpg")], umo="t:g:auto5")
            out = [r async for r in p.on_any_message(ev)]
            self.assertTrue(any("图片自动记账没成功" in str(r) for r in out))
            self.assertFalse(ev._stopped)
        asyncio.run(run())

    def test_content_failure_silent(self):
        """图里没有账单（内容类失败）→ 静默放行，不打扰用户，不 stop"""

        async def run():
            ctx = Context()
            ctx.providers = [FakeProvider("text-y", "gpt-4o")]

            async def llm(**kw):
                return types.SimpleNamespace(completion_text=json.dumps({"items": []}))

            ctx.llm_generate = llm
            p = plugin_mod.FoodLedgerPlugin(ctx, AstrBotConfig({}))
            ev = AstrMessageEvent(message_str="", message=[Image(url="https://img/cat.jpg")], umo="t:g:auto7")
            out = [r async for r in p.on_any_message(ev)]
            self.assertEqual(out, [])
            self.assertFalse(ev._stopped)
        asyncio.run(run())

    def test_query_result_has_time(self):
        """查账明细包含 MM-DD HH:MM 记账时间"""

        async def run():
            p = self._plugin()
            ts = 1750000000
            p._insert_records([("u1", "s1", "早餐", 8.5, "包子", "2026-08-12", ts, "raw", "t", "")])
            rows = p._query_records("u1", "2026-08-01", "2026-08-31")
            fmt = p._format_query_result("2026-08-12", "2026-08-12", rows, None)
            self.assertRegex(fmt, r"#1 \d{2}-\d{2} \d{2}:\d{2} 早餐 8.50")
            self.assertIn("总支出：8.50", fmt)
        asyncio.run(run())

    def test_draft_shows_time(self):
        """草稿标题显示记账时刻"""

        async def run():
            p = self._plugin()
            ev = AstrMessageEvent(message_str="", message=[Image(url="https://img/bill2.jpg")], umo="t:g:auto6")
            out = [r async for r in p.on_any_message(ev)]
            joined = "\n".join(str(r) for r in out)
            self.assertRegex(joined, r"待确认账单（共 1 笔）｜\d{2}-\d{2} \d{2}:\d{2}")
        asyncio.run(run())


class TestDB(unittest.TestCase):
    def setUp(self):
        self.plugin = plugin_mod.FoodLedgerPlugin(Context(), AstrBotConfig({}))
        # 清空数据库与共享 KV，保证用例隔离
        conn = self.plugin._get_conn()
        conn.execute("DELETE FROM records")
        conn.commit()
        conn.close()
        plugin_mod.FoodLedgerPlugin._shared_kv.clear()

    def test_insert_and_query(self):
        rows = [("u1", "s1", "早餐", 8.5, "包子", "2026-08-12", 1750000000, "raw", "default", "vision-x")]
        n = self.plugin._insert_records(rows)
        self.assertEqual(n, 1)
        res = self.plugin._query_records("u1", "2026-08-01", "2026-08-31")
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["amount"], 8.5)
        res_cat = self.plugin._query_records("u1", "2026-08-01", "2026-08-31", "午餐")
        self.assertEqual(len(res_cat), 0)

    def test_draft_flow(self):
        """完整流程：识别→草稿→确认入库→查账"""
        async def run():
            ctx = Context()
            ctx.providers = [FakeProvider("default", "gpt-4o", "OpenAI")]
            async def llm(**kw):
                return types.SimpleNamespace(completion_text=json.dumps({
                    "record_date": "2026-08-12",
                    "items": [
                        {"amount": 12.5, "category": "早餐", "description": "包子豆浆"},
                        {"amount": 33.8, "category": "外卖", "description": "炸鸡"},
                    ],
                }))
            ctx.llm_generate = llm
            p = plugin_mod.FoodLedgerPlugin(ctx, AstrBotConfig({}))
            await p._ensure_drafts()
            # 解析（纯文字，未配置图片转述模型）
            data = await p._parse_bill("default", None, "早餐包子豆浆12.5 外卖炸鸡33.8", [])
            self.assertNotIn("error", data)
            items, warns = p._validate_items(data)
            self.assertEqual(len(items), 2)
            # 草稿
            p._drafts["test:group:123"] = {
                "provider_id": "default", "created_at": 1750000000,
                "items": items, "record_date": "2026-08-12", "raw": "x",
            }
            await p._save_drafts()
            # 模拟重启恢复
            p2 = plugin_mod.FoodLedgerPlugin(ctx, AstrBotConfig({}))
            await p2._ensure_drafts()
            self.assertIn("test:group:123", p2._drafts)
            # 确认入库
            rows = [
                (ev_sender_id := "u1", "test:group:123", it["category"], it["amount"],
                 it["description"], "2026-08-12", 1750000000, "x", "default", "")
                for it in p2._drafts["test:group:123"]["items"]
            ]
            n = await asyncio.to_thread(p2._insert_records, rows)
            self.assertEqual(n, 2)
            res = p2._query_records("u1", "2026-08-12", "2026-08-12")
            self.assertEqual(len(res), 2)
            total = sum(r["amount"] for r in res)
            self.assertAlmostEqual(total, 46.3)
            # 查账格式输出
            fmt = p2._format_query_result("2026-08-12", "2026-08-12", res, None)
            self.assertIn("46.30", fmt)
            self.assertIn("按类别", fmt)
            print("\n" + fmt)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main(verbosity=2)
