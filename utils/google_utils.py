"""
Utilities for uploading files to Google Drive.
"""

import os
import argparse
import pickle
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    from googleapiclient.errors import HttpError
except ImportError:
    print("Google Drive API libraries not found. Install with:")
    print("pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client")
    raise

# If modifying these scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive']


def get_drive_service(credentials_path: str = 'credentials.json', token_path: str = 'token.pickle'):
    """
    Authenticate and return Google Drive service.
    
    Args:
        credentials_path: Path to OAuth2 credentials JSON file
        token_path: Path to store/load authentication token
        
    Returns:
        Google Drive API service object
    """
    creds = None
    
    # Load token if exists
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            creds = pickle.load(token)
    
    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                raise FileNotFoundError(
                    f"Credentials file not found: {credentials_path}\n"
                    "Please download OAuth2 credentials from Google Cloud Console:\n"
                    "1. Go to https://console.cloud.google.com/\n"
                    "2. Enable Google Drive API\n"
                    "3. Create OAuth2 credentials\n"
                    "4. Download as 'credentials.json'"
                )
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save credentials for next time
        with open(token_path, 'wb') as token:
            pickle.dump(creds, token)
    
    return build('drive', 'v3', credentials=creds)


def create_folder(service, folder_name: str, parent_id: Optional[str] = None) -> str:
    """
    Create a folder in Google Drive.
    
    Args:
        service: Google Drive API service
        folder_name: Name of the folder to create
        parent_id: ID of parent folder (None for root)
        
    Returns:
        ID of created folder
    """
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    
    if parent_id:
        file_metadata['parents'] = [parent_id]
    
    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')


def find_folder(service, folder_name: str, parent_id: Optional[str] = None) -> Optional[str]:
    """
    Find a folder by name in Google Drive.
    
    Args:
        service: Google Drive API service
        folder_name: Name of the folder to find
        parent_id: ID of parent folder (None for root)
        
    Returns:
        ID of found folder or None
    """
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])
    
    return items[0]['id'] if items else None


def upload_file(
    service,
    file_path: str,
    folder_id: Optional[str] = None,
    file_name: Optional[str] = None
) -> str:
    """
    Upload a single file to Google Drive.
    
    Args:
        service: Google Drive API service
        file_path: Path to local file
        folder_id: ID of parent folder (None for root)
        file_name: Name for file in Drive (None to use local name)
        
    Returns:
        ID of uploaded file
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_metadata = {'name': file_name or file_path.name}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(str(file_path), resumable=True)
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    return file.get('id')


def upload_folder_to_drive(
    local_path: str,
    drive_folder_name: Optional[str] = None,
    parent_folder_id: Optional[str] = None,
    credentials_path: str = 'credentials.json',
    token_path: str = 'token.pickle',
    ignore_patterns: Optional[List[str]] = None,
) -> str:
    """
    Upload a folder and its contents to Google Drive with progress bar.
    
    Args:
        local_path: Path to local folder to upload
        drive_folder_name: Name for folder in Drive (None to use local name)
        parent_folder_id: ID of parent folder in Drive (None for root)
        credentials_path: Path to OAuth2 credentials JSON
        token_path: Path to authentication token
        ignore_patterns: List of patterns to ignore
        
    Returns:
        ID of created folder in Google Drive
        
    Example:
        >>> upload_folder_to_drive(
        ...     local_path="./data/my_dataset",
        ...     drive_folder_name="My Dataset",
        ...     ignore_patterns=["*.log", "*.tmp", "__pycache__"]
        ... )
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise ValueError(f"Path does not exist: {local_path}")
    
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
    
    # Authenticate
    print("🔐 Authenticating with Google Drive...")
    service = get_drive_service(credentials_path, token_path)
    print("✓ Authentication successful")
    
    # Create or find root folder
    folder_name = drive_folder_name or local_path.name
    
    # Check if folder exists
    existing_folder_id = find_folder(service, folder_name, parent_folder_id)
    if existing_folder_id:
        print(f"✓ Found existing folder: {folder_name}")
        root_folder_id = existing_folder_id
    else:
        print(f"📁 Creating folder: {folder_name}")
        root_folder_id = create_folder(service, folder_name, parent_folder_id)
        print(f"✓ Folder created: {folder_name}")
    
    # Collect files to upload
    files_to_upload = []
    folder_structure = {}  # Map relative paths to folder IDs
    
    if local_path.is_file():
        # Single file upload
        files_to_upload.append((local_path, local_path.name, root_folder_id))
    else:
        # Directory upload
        for root, dirs, files in os.walk(local_path):
            rel_root = os.path.relpath(root, local_path)
            
            # Skip ignored directories
            dirs[:] = [d for d in dirs if not any(
                Path(os.path.join(rel_root, d)).match(pat) for pat in ignore_patterns
            )]
            
            # Create folder structure in Drive
            if rel_root != '.':
                parent_rel = os.path.dirname(rel_root)
                parent_folder_id = folder_structure.get(parent_rel, root_folder_id)
                folder_name_current = os.path.basename(root)
                
                # Check if subfolder exists
                existing_subfolder_id = find_folder(service, folder_name_current, parent_folder_id)
                if existing_subfolder_id:
                    current_folder_id = existing_subfolder_id
                else:
                    current_folder_id = create_folder(service, folder_name_current, parent_folder_id)
                
                folder_structure[rel_root] = current_folder_id
            else:
                folder_structure[rel_root] = root_folder_id
            
            # Add files to upload list
            for fname in files:
                fpath = os.path.join(root, fname)
                relpath = os.path.relpath(fpath, local_path)
                
                # Check ignore patterns
                if any(Path(relpath).match(pat) for pat in ignore_patterns):
                    continue
                
                parent_rel = os.path.dirname(relpath) if os.path.dirname(relpath) else '.'
                parent_id = folder_structure.get(parent_rel, root_folder_id)
                
                files_to_upload.append((fpath, fname, parent_id))
    
    # Upload files with progress bar
    print(f"\n📤 Uploading {len(files_to_upload)} files to Google Drive...")
    
    failed = 0
    for file_path, file_name, parent_id in tqdm(files_to_upload, desc="Uploading files", unit="file"):
        try:
            upload_file(service, file_path, parent_id, file_name)
        except Exception as e:
            tqdm.write(f"Failed to upload {file_path}: {e}")
            failed += 1
    
    print(f"\n✅ Upload finished!")
    print(f"   Folder ID: {root_folder_id}")
    print(f"   View at: https://drive.google.com/drive/folders/{root_folder_id}")
    if failed:
        print(f"   {failed} files failed to upload.")
    
    return root_folder_id


def main():
    """Command-line interface for uploading to Google Drive."""
    parser = argparse.ArgumentParser(
        description="Upload files/folders to Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload a folder
  python google_utils.py --path ./data/my_dataset --folder-name "My Dataset"

  # Upload with custom credentials
  python google_utils.py --path ./model --credentials my_creds.json

  # Upload with ignore patterns
  python google_utils.py --path ./data --ignore "*.log" --ignore "*.tmp"

Setup:
  1. Go to https://console.cloud.google.com/
  2. Create a project and enable Google Drive API
  3. Create OAuth2 credentials (Desktop app)
  4. Download credentials as 'credentials.json'
  5. Run this script - it will open browser for authentication
        """
    )
    
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to local file or folder to upload"
    )
    
    parser.add_argument(
        "--folder-name",
        type=str,
        default=None,
        help="Name for folder in Google Drive (default: use local name)"
    )
    
    parser.add_argument(
        "--parent-id",
        type=str,
        default=None,
        help="ID of parent folder in Google Drive (default: root)"
    )
    
    parser.add_argument(
        "--credentials",
        type=str,
        default="credentials.json",
        help="Path to OAuth2 credentials JSON file"
    )
    
    parser.add_argument(
        "--token",
        type=str,
        default="token.pickle",
        help="Path to store authentication token"
    )
    
    parser.add_argument(
        "--ignore",
        action="append",
        dest="ignore_patterns",
        help="Pattern to ignore (can be specified multiple times)"
    )
    
    args = parser.parse_args()
    
    # Upload
    upload_folder_to_drive(
        local_path=args.path,
        drive_folder_name=args.folder_name,
        parent_folder_id=args.parent_id,
        credentials_path=args.credentials,
        token_path=args.token,
        ignore_patterns=args.ignore_patterns,
    )


if __name__ == "__main__":
    main()
