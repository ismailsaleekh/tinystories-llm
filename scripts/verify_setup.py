"""Verify environment setup - run after installing requirements."""
import sys

def main():
    print("=" * 50)
    print("Environment Verification")
    print("=" * 50)

    # Check Python version
    print(f"\nPython version: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("WARNING: No GPU detected. Training will be slow on CPU.")
    except ImportError:
        print("ERROR: PyTorch not installed")
        return False

    # Check tiktoken
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print(f"\ntiktoken: OK (vocab size: {enc.n_vocab})")
    except ImportError:
        print("ERROR: tiktoken not installed")
        return False

    # Check datasets
    try:
        import datasets
        print(f"datasets: OK (version: {datasets.__version__})")
    except ImportError:
        print("ERROR: datasets not installed")
        return False

    # Check config module
    try:
        sys.path.insert(0, ".")
        from config import GPTConfig, set_seed, SMALL_CONFIG
        set_seed(42)
        print(f"\nConfig module: OK")
        print(f"SMALL_CONFIG params: ~{SMALL_CONFIG.estimate_params() / 1e6:.1f}M")
    except ImportError as e:
        print(f"ERROR: Config module not working: {e}")
        return False

    print("\n" + "=" * 50)
    print("All checks passed! Ready for Phase 1.")
    print("=" * 50)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
