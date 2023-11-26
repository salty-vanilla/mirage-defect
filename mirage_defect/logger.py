import logging
import os


def get_logger():
    # 環境変数からログレベルを取得し、対応するloggingレベルを設定する
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Loggerの設定
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    # ログのフォーマット設定
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 標準出力へのハンドラ設定
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = get_logger()
