import os.path
import time
import logging

def get_logger(log_dir):

    # 创建一个logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Log等级总开关

    # 创建一个handler，用于写入日志文件
    file_name = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = os.path.join(log_dir,file_name+'.log')
    
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) 

    # 定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

    # # 使用logger.XX来记录错误,这里的"error"可以根据所需要的级别进行修改
    # try:
    #     open('/path/to/does/not/exist', 'rb')
    # except (SystemExit, KeyboardInterrupt):
    #     raise
    # except Exception, e:
    #     logger.error('Failed to open file', exc_info=True)