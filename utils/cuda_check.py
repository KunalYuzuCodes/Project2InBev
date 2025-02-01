import torch
import sys
from utils.logger import logger

def check_cuda_setup():
    """Verify CUDA setup and print diagnostic information"""
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        
        # Print CUDA information
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA memory allocation
        test_tensor = torch.zeros(1).cuda()
        del test_tensor
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"CUDA setup error: {e}")
        return False

def get_cuda_memory_usage():
    """Get current CUDA memory usage"""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated()/1024**2,
            'cached': torch.cuda.memory_reserved()/1024**2
        }
    return None
