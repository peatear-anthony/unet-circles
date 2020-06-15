import logging.config
from os.path import join
from keras_unet.models import unet, unet_mini

if __name__ == '__main__':
    # Logs
    logging.config.fileConfig(join('log', 'logging.conf'))
    logger = logging.getLogger(__name__)
    logger.info("********** main.py **********")
