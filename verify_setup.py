from utils.cuda_check import check_cuda_setup
from utils.logger import logger

def main():
    if check_cuda_setup():
        logger.info("CUDA setup verified successfully!")
    else:
        logger.error("CUDA setup verification failed!")

if __name__ == "__main__":
    main()
