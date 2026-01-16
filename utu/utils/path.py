import hashlib
import json
import os
import pathlib
import re
import tempfile
from typing import Any
from urllib.parse import urlparse

import requests
import yaml
from jinja2 import Environment, FileSystemLoader, Template


# 获取当前包所在的根目录路径
def get_package_path() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent.parent


# 项目根目录路径
DIR_ROOT = get_package_path()
# 缓存目录路径
CACHE_DIR = DIR_ROOT / ".cache"


# 文件操作工具类
class FileUtils:
    # 检查字符串是否为网络 URL
    @staticmethod
    def is_web_url(url: str) -> bool:
        parsed_url = urlparse(url)
        # 如果包含协议（scheme）和网络位置（netloc），则认为是网络 URL
        return all([parsed_url.scheme, parsed_url.netloc])

    # 获取文件扩展名，支持本地路径和网络 URL
    @staticmethod
    def get_file_ext(file_path: str) -> str:
        # 如果是网络 URL，先解析路径再获取后缀
        if FileUtils.is_web_url(file_path):
            return pathlib.Path(urlparse(file_path).path).suffix
        # 否则直接从本地路径获取后缀
        return pathlib.Path(file_path).suffix

    # 从网络下载文件并返回保存路径
    @staticmethod
    def download_file(url: str, save_path: str = None) -> str:
        # 从网络下载文件。返回保存的路径
        """Download file from web. Return the saved path"""
        # 如果未指定保存路径，则使用临时文件
        # if not save_path, use tempfile
        if not save_path:
            save_path = tempfile.NamedTemporaryFile(
                suffix=FileUtils.get_file_ext(url),
                delete=False,
            ).name
        # 执行 GET 请求下载文件
        response = requests.get(url)
        # 检查响应状态码，如果出错则抛出异常
        response.raise_for_status()
        # 将下载的内容以二进制形式写入本地文件
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path

    # 计算文件的 MD5 哈希值，支持本地文件和网络文件
    @staticmethod
    def get_file_md5(file_path: str) -> str:
        # 计算本地或网络文件的 md5
        """Clac md5 for local or web file"""
        hash_md5 = hashlib.md5()
        # 如果是网络 URL，流式下载并逐步计算 MD5
        if FileUtils.is_web_url(file_path):
            with requests.get(file_path, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=4096):
                    hash_md5.update(chunk)
        else:
            # 如果是本地文件，按块读取并逐步计算 MD5
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        # 返回十六进制格式的 MD5 结果
        return hash_md5.hexdigest()

    # 加载 YAML 格式的配置文件
    @staticmethod
    def load_yaml(file_path: pathlib.Path | str) -> dict[str, Any]:
        # 确保路径是 Path 对象
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        # 检查文件是否存在
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        # 打开文件并使用安全模式加载 YAML 内容
        with file_path.open() as f:
            return yaml.safe_load(f)

    # 从 YAML 文件加载提示词（Prompts）
    @staticmethod
    def load_prompts(fn: str | pathlib.Path) -> dict[str, str]:
        # 从 yaml 文件加载提示词。
        """Load prompts from yaml file.

        - 默认路径：`DIR_ROOT / "utu/prompts" / fn`
        - Default path: `DIR_ROOT / "utu/prompts" / fn`
        """
        if isinstance(fn, str):
            # 自动补全 .yaml 后缀
            if not fn.endswith(".yaml"):
                fn += ".yaml"
            # 构造完整路径
            fn = DIR_ROOT / "utu" / "prompts" / fn
        # 确保提示词文件存在
        assert fn.exists(), f"File {fn} does not exist!"
        # 以 UTF-8 编码读取并加载 YAML 内容
        with fn.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    # 获取 Jinja2 环境对象，用于加载指定目录下的模板
    @staticmethod
    def get_jinja_env(directory: str | pathlib.Path) -> Environment:
        if isinstance(directory, str):
            # 默认为 prompts 目录下的子目录
            directory = DIR_ROOT / "utu" / "prompts" / directory
        if not directory.exists():
            raise FileNotFoundError(f"Directory {directory} does not exist")
        # 创建文件系统加载器并初始化 Jinja2 环境
        return Environment(loader=FileSystemLoader(directory))

    # 获取指定的 Jinja2 模板对象
    @staticmethod
    def get_jinja_template(template_path: str | pathlib.Path) -> Template:
        if isinstance(template_path, str):
            # 自动补全 .j2 后缀
            if not template_path.endswith(".j2"):
                template_path += ".j2"
            # 默认为 prompts 目录下的路径
            template_path = DIR_ROOT / "utu" / "prompts" / template_path
        if not template_path.exists():
            raise FileNotFoundError(f"File {template_path} does not exist")
        # 读取文件内容并创建 Template 对象
        with template_path.open(encoding="utf-8") as f:
            return Template(f.read())

    # 从模板字符串直接创建 Jinja2 模板对象
    @staticmethod
    def get_jinja_template_str(template_str: str) -> Template:
        return Template(template_str)

    # 加载 JSON 格式的文件
    @staticmethod
    def load_json(file_path: str | pathlib.Path) -> dict[str, Any]:
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        # 打开文件并加载 JSON 内容
        with file_path.open() as f:
            return json.load(f)

    # 加载项目内部的 JSON 数据文件（默认在 utu/data 目录下）
    @staticmethod
    def load_json_data(file_path: str | pathlib.Path) -> list[dict[str, Any]]:
        # 如果文件在指定路径不存在，则尝试在默认的 data 目录下查找
        if isinstance(file_path, str) and not os.path.exists(file_path):
            file_path = DIR_ROOT / "utu" / "data" / file_path
        return FileUtils.load_json(file_path)

    # 将数据保存为 JSON 格式的文件
    @staticmethod
    def save_json(file_path: str | pathlib.Path, data: dict[str, Any]) -> None:
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        # 以 UTF-8 编码写入文件，美化输出并保留非 ASCII 字符
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    # 应用差异（Diff）块到文本内容中，支持 SEARCH/REPLACE 格式
    @staticmethod
    def apply_diff(content: str, diff: str) -> str:
        modified_content = content
        # 定义正则表达式匹配 SEARCH/REPLACE 差异块
        pattern = r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE"
        matches = re.findall(pattern, diff, re.DOTALL)
        if not matches:
            raise ValueError("No valid diff blocks found in the provided diff")

        # 应用每一组查找/替换对
        # Apply each search/replace pair
        for search_text, replace_text in matches:
            # 如果原文中包含待替换内容，则执行替换
            if search_text in modified_content:
                modified_content = modified_content.replace(search_text, replace_text)
            else:
                # 否则抛出内容未找到异常
                raise ValueError(f"Search text not found in content: {search_text[:50]}...")
        return modified_content

    # 检查文件是否存在
    @staticmethod
    def file_exists(file_path: str | pathlib.Path) -> bool:
        if isinstance(file_path, str):
            file_path = pathlib.Path(file_path)
        return file_path.exists()
