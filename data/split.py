import json
from pathlib import Path
import hashlib
from collections import defaultdict
import os

def md5_checksum(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.md5(data).hexdigest()

def split_data(input_file: Path):
    with open(input_file) as f:
        data = json.load(f)

    all_user_ids = {post["id"] for post in data}

    checksums = {}
    for post in data:
        sums = set()

        for profile_post in post["profile"]:
            sums.add(md5_checksum(profile_post["text"]))
        
        checksums[post["id"]] = sums

    checksum_to_users = defaultdict(set)

    for user_id, user_checksums in checksums.items():
        for checksum in user_checksums:
            checksum_to_users[checksum].add(user_id)

    user_duplicates = {}

    for user_id, user_checksums in checksums.items():
        duplicate_users = set()
        
        for checksum in user_checksums:
            # Get all users with this checksum
            users_with_same_checksum = checksum_to_users[checksum]
            # Add them (excluding the current user)
            duplicate_users.update(users_with_same_checksum - {user_id})
        
        if duplicate_users:  # Only add if there are duplicates
            user_duplicates[user_id] = list(duplicate_users)

    unique_users = all_user_ids - set(user_duplicates.keys())

    unique_rows = [post for post in data if post["id"] in unique_users]
    duplicate_rows = [post for post in data if post["id"] in user_duplicates]

    unique_save_path = input_file.parent / f"{input_file.stem}_public.json"
    duplicate_save_path = input_file.parent / f"{input_file.stem}.json"

    with open(unique_save_path, 'w') as f:
        json.dump(unique_rows, f, indent=2)
    
    with open(duplicate_save_path, 'w') as f:
        json.dump(duplicate_rows, f, indent=2)
    print(f"Unique data saved to {unique_save_path}")
    print(f"Duplicate data saved to {duplicate_save_path}")

if __name__ == "__main__":
    data_dir = Path("/scratch4/workspace/oyilmazel_umass_edu-lamp_dataset/dataset")
    files = os.listdir(data_dir)

    for file in files:
        input_file = data_dir / file
        split_data(input_file)
    