import json
import logging
import pathlib
from logging.handlers import TimedRotatingFileHandler
from typing import Literal

from colorlog import ColoredFormatter

# 日志目录路径，位于项目根目录下的 logs 文件夹
DIR_LOGS = pathlib.Path(__file__).parent.parent.parent / "logs"
# 确保日志目录存在，如果不存在则创建
DIR_LOGS.mkdir(exist_ok=True)

# 标记日志系统是否已经初始化，防止重复初始化
# Flag to track if logging has been set up
_LOGGING_INITIALIZED = False


# 设置全局日志配置
def setup_logging(level: Literal["WARNING", "INFO", "DEBUG"] = "WARNING") -> None:
    # 检查日志是否已经初始化
    # Check if logging has already been initialized
    global _LOGGING_INITIALIZED
    # 如果已经初始化，记录警告信息并直接返回
    if _LOGGING_INITIALIZED:
        logging.getLogger().warning("Logging has already been initialized! Skipping...")
        return

    # 获取名为 "utu" 的根日志记录器
    utu_logger = logging.getLogger("utu")
    # 设置日志记录器的最低级别为 DEBUG
    utu_logger.setLevel(logging.DEBUG)
    # 如果已有处理器，先清空，避免重复输出
    if utu_logger.handlers:
        utu_logger.handlers.clear()

    # 创建控制台输出处理器
    console_handler = logging.StreamHandler()
    # 设置控制台输出的日志级别
    console_handler.setLevel(level)
    # 使用彩色格式化器美化控制台输出
    color_formatter = ColoredFormatter(
        "%(green)s%(asctime)s%(reset)s[%(blue)s%(name)s%(reset)s] - "
        "%(log_color)s%(levelname)s%(reset)s - %(filename)s:%(lineno)d - %(green)s%(message)s%(reset)s",
        # " - %(cyan)s%(threadName)s%(reset)s",
        # 配置不同日志级别对应的颜色
        log_colors={
            "DEBUG": "cyan",
            "INFO": "white",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        # 配置辅助信息的颜色
        secondary_log_colors={"asctime": {"green": "green"}, "name": {"blue": "blue"}},
    )
    # 为控制台处理器设置格式化器
    console_handler.setFormatter(color_formatter)

    # 创建按时间滚动的文件处理器，每天凌晨滚动一次日志文件
    file_handler = TimedRotatingFileHandler(
        DIR_LOGS / "utu.log", when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    # 设置文件处理器的日志级别
    file_handler.setLevel(logging.DEBUG)
    # 设置文件输出的格式
    formatter = logging.Formatter(
        "%(asctime)s[%(name)s] - %(levelname)s - %(filename)s:%(lineno)d - %(message)s - %(threadName)s"
    )
    # 为文件处理器设置格式化器
    file_handler.setFormatter(formatter)

    # 将控制台和文件处理器添加到根记录器
    utu_logger.addHandler(console_handler)
    utu_logger.addHandler(file_handler)
    # 记录日志系统初始化成功的信息
    utu_logger.info(f"Logging initialized with level {level}.")

    # 标记日志系统为已初始化状态
    # Mark logging as initialized
    _LOGGING_INITIALIZED = True


# 获取指定名称的日志记录器
def get_logger(name: str, level: int | str = logging.DEBUG) -> logging.Logger:
    # 根据名称获取或创建日志记录器
    logger = logging.getLogger(name)
    # 设置记录器的日志级别
    logger.setLevel(level)

    # 辅助方法：记录包含异常回溯信息的错误日志
    def log_error_with_exc(msg, *args, **kwargs):
        # 强制开启异常堆栈信息的记录
        kwargs["exc_info"] = True
        logger.error(msg, *args, **kwargs)

    # 为 logger 对象绑定自定义的 error_exc 方法
    logger.error_exc = log_error_with_exc
    return logger


# 将对象转换为单行字符串形式，用于精简日志输出
def oneline_object(obj: object, limit: int = 100) -> str:
    try:
        # 尝试将对象转换为 JSON 字符串，保留非 ASCII 字符
        s = json.dumps(obj, ensure_ascii=False)
    except TypeError:
        # 如果对象不可 JSON 序列化，则先转为字符串再序列化
        s = json.dumps(str(obj), ensure_ascii=False)
    # 如果字符串长度超过限制，则截断并添加省略号
    return f"{s[:limit]}..." if len(s) > limit else s
