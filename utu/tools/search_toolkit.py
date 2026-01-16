import asyncio

from ..config import ToolkitConfig
from ..utils import SimplifiedAsyncOpenAI, get_logger, oneline_object
from .base import TOOL_PROMPTS, AsyncBaseToolkit, register_tool

logger = get_logger(__name__)


# 搜索工具包类，提供网页搜索和网页内容问答功能
class SearchToolkit(AsyncBaseToolkit):
    # 搜索工具包
    """Search Toolkit

    注意：
    NOTE:
        - 请配置必要的环境变量！参见 `configs/agents/tools/search.yaml`
        - Please configure the required env variables! See `configs/agents/tools/search.yaml`

    方法：
    Methods:
        - search(query: str, num_results: int = 5)
        - web_qa(url: str, query: str)
    """

    # 初始化搜索工具包
    def __init__(self, config: ToolkitConfig = None):
        # 调用父类初始化
        super().__init__(config)
        # 从配置中获取搜索引擎类型，默认为 "google"
        search_engine = self.config.config.get("search_engine", "google")
        # 根据搜索引擎类型动态加载并初始化对应的搜索实现
        match search_engine:
            case "google":
                from .search.google_search import GoogleSearch

                self.search_engine = GoogleSearch(self.config.config)
            case "jina":
                from .search.jina_search import JinaSearch

                self.search_engine = JinaSearch(self.config.config)
            case "baidu":
                from .search.baidu_search import BaiduSearch

                self.search_engine = BaiduSearch(self.config.config)
            case "duckduckgo":
                from .search.duckduckgo_search import DuckDuckGoSearch

                self.search_engine = DuckDuckGoSearch(self.config.config)
            case _:
                # 不支持的搜索引擎抛出异常
                raise ValueError(f"Unsupported search engine: {search_engine}")
        
        # 获取网页抓取引擎类型，默认为 "jina"
        crawl_engine = self.config.config.get("crawl_engine", "jina")
        # 根据抓取引擎类型动态加载对应的实现
        match crawl_engine:
            case "jina":
                from .search.jina_crawl import JinaCrawl

                self.crawl_engine = JinaCrawl(self.config.config)
            case "crawl4ai":
                from .search.crawl4ai_crawl import Crawl4aiCrawl

                self.crawl_engine = Crawl4aiCrawl(self.config.config)
            case _:
                # 不支持的抓取引擎抛出异常
                raise ValueError(f"Unsupported crawl engine: {crawl_engine}")
        
        # 用于网页问答（web_qa）的 LLM 客户端初始化
        # llm for web_qa
        self.llm = SimplifiedAsyncOpenAI(
            **self.config.config_llm.model_provider.model_dump() if self.config.config_llm else {}
        )
        # 摘要生成的 token 限制
        self.summary_token_limit = self.config.config.get("summary_token_limit", 1_000)

    # 注册为代理可用工具：执行网页搜索
    @register_tool
    async def search(self, query: str, num_results: int = 5) -> dict:
        # 执行网页搜索以获取网络信息。
        """web search to gather information from the web.

        提示：
        Tips:
        1. 搜索查询应当具体，避免模糊或过长
        1. search query should be concrete and not vague or super long
        2. 必要时可以在查询中使用 Google 搜索运算符，例如：
        2. try to add Google search operators in query if necessary,
        - " " 用于精确匹配；
        - " " for exact match;
        - -xxx 用于排除内容；
        - -xxx for exclude;
        - * 通配符匹配；
        - * wildcard matching;
        - filetype:xxx 指定文件类型；
        - filetype:xxx for file types;
        - site:xxx 指定站点内搜索；
        - site:xxx for site search;
        - before:YYYY-MM-DD, after:YYYY-MM-DD 指定时间范围。
        - before:YYYY-MM-DD, after:YYYY-MM-DD for time range.

        参数：
        Args:
            query (str): 待搜索的查询语句。
            query (str): The query to search for.
            num_results (int, optional): 返回结果的数量。默认为 5。
            num_results (int, optional): The number of results to return. Defaults to 5.
        """
        # 参考：https://serper.dev/playground
        # https://serper.dev/playground
        # 记录搜索工具调用的日志
        logger.info(f"[tool] search: {oneline_object(query)}")
        # 调用选定的搜索引擎执行搜索
        res = await self.search_engine.search(query, num_results)
        # 记录搜索结果概要日志
        logger.info(oneline_object(res))
        return res

    # 注册为代理可用工具：对指定网页进行问答
    @register_tool
    async def web_qa(self, url: str, query: str) -> str:
        # 对网页进行提问，你将从指定的 URL 中获取答案和相关链接。
        """Ask question to a webpage, you will get the answer and related links from the specified url.

        提示：
        Tips:
        - 使用场景：从网页中收集信息，提出详细问题。
        - Use cases: gather information from a webpage, ask detailed questions.

        参数：
        Args:
            url (str): 待提问的网页 URL。
            url (str): The url to ask question to.
            query (str): 待提问的问题。应当清晰、简练且具体。
            query (str): The question to ask. Should be clear, concise, and specific.
        """
        # 记录网页问答工具调用的日志
        logger.info(f"[tool] web_qa: {oneline_object({url, query})}")
        # 使用抓取引擎获取网页内容
        content = await self.crawl_engine.crawl(url)
        # 如果问题为空，则默认为总结网页内容，并使用与网页相同的语言
        query = (
            query or "Summarize the content of this webpage, in the same language as the webpage."
        )  # use the same language
        # 并发执行：内容问答和相关链接提取
        res_summary, res_links = await asyncio.gather(
            self._qa(content, query), self._extract_links(url, content, query)
        )
        # 整合汇总结果和相关链接并返回
        result = f"Summary: {res_summary}\n\nRelated Links: {res_links}"
        return result

    # 内部方法：执行基于网页内容的 LLM 问答
    async def _qa(self, content: str, query: str) -> str:
        # 格式化问答模板
        template = TOOL_PROMPTS["search_qa"].format(content=content, query=query)
        # 调用 LLM 获取回答
        return await self.llm.query_one(
            messages=[{"role": "user", "content": template}], **self.config.config_llm.model_params.model_dump()
        )

    # 内部方法：从网页内容中提取与查询相关的链接
    async def _extract_links(self, url: str, content: str, query: str) -> str:
        # 格式化相关链接提取模板
        template = TOOL_PROMPTS["search_related"].format(url=url, content=content, query=query)
        # 调用 LLM 提取链接
        return await self.llm.query_one(
            messages=[{"role": "user", "content": template}], **self.config.config_llm.model_params.model_dump()
        )
