import logging

logging.basicConfig(level=logging.INFO, format='%(module)-15s - %(lineno)d - %(levelname)s - %(message)s')
logger = logging.getLogger()