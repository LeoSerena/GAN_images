import logging
import os

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(os.path.join(__path__[0], 'logs', 'logs.txt'))
fileHandler.setLevel(logging.INFO)
fileHandler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)