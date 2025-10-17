"""
Utilities for uploading models to Hugging Face Hub.
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List
from huggingface_hub import HfApi, create_repo, upload_file, login
from tqdm import tqdm


def upload_model_to_hf(
    model_path: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    ignore_patterns: Optional[List[str]] = None,
    create_pr: bool = False,
    revision: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    repo_type: str = "model",
    path_in_repo: Optional[str] = None,
) -> str:
    """
    Upload a model folder to Hugging Face Hub.
    
    Args:
        model_path: Path to the local model folder to upload
        repo_id: Repository ID on HuggingFace (format: "username/repo-name")
        token: HuggingFace API token. If None, will use token from HF_TOKEN env var or cached login
        private: Whether to create a private repository
        commit_message: Custom commit message for the upload
        ignore_patterns: List of patterns to ignore (e.g., ["*.pyc", "__pycache__", ".git"])
        create_pr: Whether to create a PR instead of committing to main branch
        revision: The git revision to commit to (e.g., "main", "dev")
    allow_patterns: List of patterns to include (if specified, only these will be uploaded)
    repo_type: Type of repo on the Hub: "model" or "dataset" (default: model)
    path_in_repo: Optional subdirectory path inside the repo to upload into
    
    Returns:
        URL of the uploaded repository
        
    Example:
        >>> upload_model_to_hf(
        ...     model_path="./checkpoints/model-epoch-10",
        ...     repo_id="myusername/my-awesome-model",
        ...     token="hf_xxxxx",
        ...     private=False,
        ...     ignore_patterns=["*.pyc", "__pycache__", ".git", "*.log"]
        ... )
    """
    # Validate model path
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    if not model_path.is_dir():
        raise ValueError(f"Model path is not a directory: {model_path}")
    
    # Get token from environment if not provided
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    # Validate repo_type
    if repo_type not in {"model", "dataset"}:
        raise ValueError(f"Invalid repo_type: {repo_type}. Must be 'model' or 'dataset'.")

    # Login to Hugging Face
    if token:
        login(token=token, add_to_git_credential=True)
        tqdm.write("✓ Logged in to Hugging Face")
    else:
        tqdm.write("⚠ No token provided, using cached credentials")
    
    # Initialize HF API
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type=repo_type,
        )
        tqdm.write(f"✓ Repository created/verified: {repo_id} (type={repo_type})")
    except Exception as e:
        tqdm.write(f"⚠ Repository creation warning: {e}")
    
    # Default ignore patterns
    if ignore_patterns is None:
        ignore_patterns = [
            "*.pyc",
            "__pycache__",
            ".git",
            ".gitignore",
            "*.log",
            ".DS_Store",
            "*.swp",
            "*.swo",
            ".vscode",
            ".idea",
        ]
    
    # Set default commit message
    if commit_message is None:
        commit_message = f"Upload model from {model_path.name}"
    
    tqdm.write(f"\n📤 Uploading from: {model_path}")
    tqdm.write(f"   To repository: {repo_id}")
    tqdm.write(f"   Repo type:     {repo_type}")
    tqdm.write(f"   Ignore patterns: {ignore_patterns}")
    
    # List all files to upload (respect ignore/allow patterns)
    files_to_upload = []
    for root, dirs, files in os.walk(model_path):
        for fname in files:
            fpath = os.path.join(root, fname)
            relpath = os.path.relpath(fpath, model_path)
            # Ignore patterns
            if ignore_patterns and any(Path(relpath).match(pat) for pat in ignore_patterns):
                continue
            # Allow patterns
            if allow_patterns and not any(Path(relpath).match(pat) for pat in allow_patterns):
                continue
            files_to_upload.append((fpath, relpath))

    tqdm.write(f"\nUploading {len(files_to_upload)} files...")
    api = HfApi()
    repo_url = str(api.create_repo(
        repo_id=repo_id,
        token=token,
        private=private,
        exist_ok=True,
        repo_type=repo_type,
    ))

    failed = 0
    for fpath, relpath in tqdm(files_to_upload, desc="Uploading files", unit="file"):
        try:
            upload_file(
                path_or_fileobj=fpath,
                path_in_repo=relpath if path_in_repo is None else os.path.join(path_in_repo, relpath),
                repo_id=repo_id,
                token=token,
                repo_type=repo_type,
                commit_message=commit_message,
                revision=revision,
            )
        except Exception as e:
            tqdm.write(f"Failed to upload {fpath}: {e}")
            failed += 1

    tqdm.write(f"\n✅ Upload finished!")
    tqdm.write(f"   Repository URL: {repo_url}")
    if failed:
        tqdm.write(f"   {failed} files failed to upload.")
    return repo_url


def main():
    """Command-line interface for uploading models to Hugging Face."""
    parser = argparse.ArgumentParser(
        description="Upload a model folder to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with token from environment variable HF_TOKEN
  python hf_utils.py --model_path ./checkpoints/best_model --repo_id username/model-name

  # Upload with explicit token
  python hf_utils.py --model_path ./model --repo_id username/model --token hf_xxxxx

    # Upload as private repository
  python hf_utils.py --model_path ./model --repo_id username/model --private

  # Upload with custom ignore patterns
  python hf_utils.py --model_path ./model --repo_id username/model --ignore "*.log" --ignore "tmp/*"

    # Upload a dataset instead of a model
    python hf_utils.py --model_path ./data/my_dataset --repo_id username/my-dataset --repo_type dataset
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the local model folder to upload"
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='Repository ID on HuggingFace (format: "username/repo-name")'
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN environment variable)"
    )
    
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )
    
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message for the upload"
    )
    
    parser.add_argument(
        "--ignore",
        action="append",
        dest="ignore_patterns",
        help="Pattern to ignore (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--allow",
        action="append",
        dest="allow_patterns",
        help="Pattern to include (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--create_pr",
        action="store_true",
        help="Create a pull request instead of committing to main"
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Git revision to commit to (e.g., 'main', 'dev')"
    )

    parser.add_argument(
        "--repo_type",
        type=str,
        choices=["model", "dataset"],
        default="model",
        help="Type of repository on the Hub (model or dataset)"
    )

    parser.add_argument(
        "--path_in_repo",
        type=str,
        default=None,
        help="Optional subdirectory inside the repo to upload into"
    )
    
    args = parser.parse_args()
    
    # Upload the model
    upload_model_to_hf(
        model_path=args.model_path,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
        ignore_patterns=args.ignore_patterns,
        create_pr=args.create_pr,
        revision=args.revision,
        allow_patterns=args.allow_patterns,
        repo_type=args.repo_type,
        path_in_repo=args.path_in_repo,
    )


if __name__ == "__main__":
    main()
