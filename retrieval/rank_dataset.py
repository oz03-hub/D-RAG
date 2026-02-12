import torch
from transformers import AutoModel, AutoTokenizer
import json
import tqdm
import argparse
import hashlib
from collections import defaultdict


sub_cats = {
    "Art_and_Entertainment": {
        "anime",
        "boardgames",
        "gaming",
        "literature",
        "movies",
        "music",
        "musicfans",
        "rpg",
        "scifi",
        "sound",
    },
    "Lifestyle_and_Personal_Development": {
        "bicycles",
        "cooking",
        "diy",
        "fitness",
        "freelancing",
        "gardening",
        "health",
        "lifehacks",
        "martialarts",
        "outdoors",
        "parenting",
        "pets",
        "sports",
        "sustainability",
        "travel",
        "woodworking",
        "workplace",
        "writers",
    },
    "Society_and_Culture": {
        "academia",
        "buddhism",
        "christianity",
        "english",
        "expatriates",
        "genealogy",
        "hermeneutics",
        "hinduism",
        "history",
        "interpersonal",
        "islam",
        "judaism",
        "law",
        "linguistics",
        "money",
        "philosophy",
        "politics",
        "skeptics",
        "vegetarianism",
    },
}


def calculate_checksum(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def batchify(lst, batch_size):
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def retrieve_top_k_with_contriver(
    contriver, tokenizer, corpus, profile, query, k, batch_size
):
    query_tokens = tokenizer(
        [query], padding=True, truncation=True, return_tensors="pt"
    ).to("cuda:0")
    output_query = contriver(**query_tokens)
    output_query = mean_pooling(
        output_query.last_hidden_state, query_tokens["attention_mask"]
    )
    scores = []
    batched_corpus = batchify(corpus, batch_size)
    with torch.no_grad():
        for batch in batched_corpus:
            tokens_batch = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda:0")
            outputs_batch = contriver(**tokens_batch)
            outputs_batch = mean_pooling(
                outputs_batch.last_hidden_state, tokens_batch["attention_mask"]
            )
            temp_scores = output_query.squeeze() @ outputs_batch.T
            scores.extend(temp_scores.tolist())
        topk_values, topk_indices = torch.topk(torch.tensor(scores), k)
    return [profile[m] for m in topk_indices.tolist()]


def retrieve_top_k_from_public(
    contriver, tokenizer, dataset, k, batch_size, same_checksum_users
):
    """
    Retrieve top-k relevant posts from public data for each query in the dataset.
    Uses chunked GPU processing for memory efficiency.
    """

    def get_domain(category):
        for domain, categories in sub_cats.items():
            if category in categories:
                return domain
        return None

    target_domain = get_domain(dataset[0].get("category", ""))

    public_posts = []
    public_corpus = []
    for item in dataset:
        for post in item.get("profile", []):
            if get_domain(post.get("category", "")) == target_domain:
                public_corpus.append(post["text"])
                public_posts.append(post)

    # Encode public corpus and keep embeddings on CPU
    batched_corpus = batchify(public_corpus, batch_size)
    public_embeddings = []

    with torch.no_grad():
        for batch in tqdm.tqdm(batched_corpus, desc="Encoding public corpus"):
            tokens_batch = tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda:0")
            outputs_batch = contriver(**tokens_batch)
            outputs_batch = mean_pooling(
                outputs_batch.last_hidden_state, tokens_batch["attention_mask"]
            )
            # Keep on CPU to save GPU memory
            public_embeddings.append(outputs_batch.cpu())

            # Clear GPU memory
            del tokens_batch, outputs_batch
            torch.cuda.empty_cache()

    # Concatenate on CPU
    public_embeddings = torch.cat(public_embeddings, dim=0)
    chunk_size = 1000

    # Process each query with chunked similarity computation
    with torch.no_grad():
        for data in tqdm.tqdm(dataset, desc="Retrieving from public"):
            query = data["question"]

            query_tokens = tokenizer(
                [query], padding=True, truncation=True, return_tensors="pt"
            ).to("cuda:0")

            output_query = contriver(**query_tokens)
            output_query = mean_pooling(
                output_query.last_hidden_state, query_tokens["attention_mask"]
            )

            # Compute scores in chunks on GPU
            all_scores = []
            num_chunks = (len(public_embeddings) + chunk_size - 1) // chunk_size

            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * chunk_size
                end_idx = min((chunk_idx + 1) * chunk_size, len(public_embeddings))

                # Load chunk to GPU
                chunk_embeddings = public_embeddings[start_idx:end_idx].to("cuda:0")

                # Compute similarity scores for this chunk
                chunk_scores = output_query.squeeze() @ chunk_embeddings.T

                # Move scores back to CPU to save GPU memory
                all_scores.append(chunk_scores.cpu())

                # Clear chunk from GPU
                del chunk_embeddings, chunk_scores
                torch.cuda.empty_cache()

            # Concatenate all scores on CPU
            scores = torch.cat(all_scores)

            # Get top-k
            topk_values, topk_indices = torch.topk(scores, len(public_corpus))
            ranked_posts = []
            i = 0
            while len(ranked_posts) < k and i < len(topk_indices):
                idx = topk_indices[i].item()
                if public_posts[idx]["pid"] not in same_checksum_users[data["id"]]:
                    ranked_posts.append(public_posts[idx])
                i += 1

            data["public_posts"] = ranked_posts

            # Clear GPU cache
            del query_tokens, output_query, all_scores, scores
            torch.cuda.empty_cache()

    return dataset


parser = argparse.ArgumentParser()
parser.add_argument("--input_dataset_addr", type=str, required=True)
parser.add_argument("--output_dataset_addr", type=str, required=True)
parser.add_argument("--model_name", type=str, default="facebook/contriever-msmarco")
parser.add_argument("--cache", type=str)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument(
    "--top_k_public",
    type=int,
    default=20,
    help="Number of top public posts to retrieve",
)

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.input_dataset_addr, "r") as file:
        dataset = json.load(file)

    # add id to each profile entry
    print("Adding profile IDs...")
    for obj in dataset:
        for entry in obj["profile"]:
            entry["pid"] = obj["id"]

    # Build inverted index: checksum -> set of user ids
    print("Calculating checksums and building inverted index...")
    checksum_to_users = defaultdict(set)
    for obj in dataset:
        user_id = obj["id"]
        for entry in obj["profile"]:
            checksum = calculate_checksum(entry["text"])
            checksum_to_users[checksum].add(user_id)

    # Build mapping from user id -> other users with same checksums
    print("Finding users with same checksums...")
    same_checksum_users = defaultdict(set)
    for users_with_checksum in checksum_to_users.values():
        # Only process if multiple users share this checksum
        if len(users_with_checksum) > 1:
            # For each user, add all other users with this checksum
            for user_id in users_with_checksum:
                same_checksum_users[user_id].update(users_with_checksum)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache)
    contriver = AutoModel.from_pretrained(args.model_name, cache_dir=args.cache).to(
        "cuda:0"
    )
    contriver.eval()

    # First, retrieve from personal profiles
    print("Starting personal profile retrieval...")
    dataset_new = []
    for data in tqdm.tqdm(dataset, desc="Retrieving from personal profiles"):
        profile = data["profile"]
        corpus = [x["text"] for x in profile]
        query = data["question"]

        ranked_profile = retrieve_top_k_with_contriver(
            contriver, tokenizer, corpus, profile, query, len(profile), args.batch_size
        )

        ranked_profile = [
            {
                "id": entry["id"],
                "pid": entry["pid"],
                "text": entry["text"],
                "category": entry.get("category", ""),
            }
            for entry in ranked_profile
        ]
        data["profile"] = ranked_profile
        dataset_new.append(data)

    # Then, retrieve as public posts
    print("Starting public retrieval...")
    print(
        f"Retrieving top-{args.top_k_public} posts from public data for each query..."
    )
    dataset_new = retrieve_top_k_from_public(
        contriver,
        tokenizer,
        dataset_new,
        args.top_k_public,
        args.batch_size,
        same_checksum_users,
    )

    with open(args.output_dataset_addr, "w") as file:
        json.dump(dataset_new, file, indent=4)
