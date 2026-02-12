from trl import SFTConfig, SFTTrainer
import argparse
from data.formetters import get_planner_formatter, get_cross_domain_planner_formatter, get_public_only_planner_formatter
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
import torch
import matplotlib.pyplot as plt
import os
import json

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


def get_domain(category):
    for domain, subcats in sub_cats.items():
        if category in subcats:
            return domain
    return None


def keep_target_domain_posts(dataset):
    for query_data in dataset:
        profile = query_data["profile"]
        category = query_data["category"]

        post_domain = get_domain(category)
        target_subcats = sub_cats[post_domain]

        new_profile = []
        for post in profile:
            if post["category"] in target_subcats:
                new_profile.append(post)

        query_data["profile"] = new_profile
    return dataset


def limit_target_domain_posts(dataset, limit):
    for query_data in dataset:
        profile = query_data["profile"]
        category = query_data["category"]

        post_domain = get_domain(category)
        target_subcats = sub_cats[post_domain]

        domain_post_count = 0
        new_profile = []
        for post in profile:
            if post["category"] in target_subcats:
                if domain_post_count < limit:
                    new_profile.append(post)
                    domain_post_count += 1
            else:
                new_profile.append(post)

        query_data["profile"] = new_profile
    return dataset


def prepare_prompts(dataset, formatter):
    reshaped_dataset = {
        "question": [],
        "target_subcat": [],
        "id": [],
        "profile": [],
        "public_posts": [],
        "narrative": [],
        "rubric_aspects": [],
    }
    for data in dataset:
        reshaped_dataset["question"].append(data["question"])
        reshaped_dataset["target_subcat"].append(data["category"])
        reshaped_dataset["id"].append(data["id"])
        reshaped_dataset["profile"].append(data["profile"])
        reshaped_dataset["public_posts"].append(data["public_posts"])
        reshaped_dataset["narrative"].append(data["narrative"])
        reshaped_dataset["rubric_aspects"].append(data["rubric_aspects"])
    return formatter(reshaped_dataset)


class LossLoggingCallback(TrainerCallback):
    def __init__(self, log_dir="logs", name="training_loss"):
        self.log_dir = log_dir
        self.losses = []
        self.steps = []
        self.name = name
        os.makedirs(log_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.losses.append(logs["loss"])
            self.steps.append(state.global_step)

            # Save loss data to JSON
            loss_data = {"steps": self.steps, "losses": self.losses}
            with open(os.path.join(self.log_dir, f"training_loss_{self.name}.json"), "w") as f:
                json.dump(loss_data, f)

            # Plot and save graph
            plt.figure(figsize=(10, 6))
            plt.plot(self.steps, self.losses, linewidth=2)
            plt.xlabel("Training Steps", fontsize=12)
            plt.ylabel("Loss", fontsize=12)
            plt.title("Training Loss Over Time", fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.log_dir, f"training_loss_{self.name}.png"), dpi=150)
            plt.close()


parser = argparse.ArgumentParser()

parser.add_argument("--inputs_addr", required=True)
parser.add_argument("--cache_dir", default="./cache")
parser.add_argument("--model_addr", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--num_context", type=int, default=10)
parser.add_argument("--per_device_train_batch_size", type=int, default=64)
parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--learning_rate", type=float, default=0.00005)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--max_steps", type=int, default=5000)
parser.add_argument("--save_steps", type=int, default=1000)
parser.add_argument("--warmup_steps", type=int, default=250)
parser.add_argument("--max_seq_length", type=int, default=4096)

parser.add_argument("--num_target_domain_contexts", type=int, default=0)
parser.add_argument("--method", type=str, choices=["rag", "aug2", "public_only"])
parser.add_argument(
    "--domain_adaptation",
    type=str,
    choices=["cross_domain", "in_domain", "multi_domain"],
)
parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                    help="Path to checkpoint directory to resume training from")


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    os.makedirs(args.output_dir, exist_ok=True)

    # dataset_orig = load_dataset(args.inputs_addr, cache_dir=args.cache_dir)
    with open(args.inputs_addr, "r") as f:
        dataset_orig = json.load(f)

    if args.domain_adaptation == "cross_domain":
        dataset_orig = limit_target_domain_posts(
            dataset_orig, args.num_target_domain_contexts
        )
    elif args.domain_adaptation == "in_domain":
        dataset_orig = keep_target_domain_posts(dataset_orig)
    elif args.domain_adaptation == "multi_domain":
        pass
    else:
        raise ValueError(f"Unknown domain adaptation: {args.domain_adaptation}")

    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_addr, cache_dir=args.cache_dir, use_flash_attention_2=True
    # )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_addr,
        cache_dir=args.cache_dir,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    # model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model_addr, cache_dir=args.cache_dir)

    if args.method == "rag":
        formatter = get_planner_formatter(tokenizer, args.num_context, train=True)
    elif args.method == "aug2":
        formatter = get_cross_domain_planner_formatter(
            tokenizer, args.num_context, train=True
        )
    elif args.method == "public_only":
        formatter = get_public_only_planner_formatter(
            tokenizer, args.num_context, train=True
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    dataset = prepare_prompts(dataset_orig, formatter)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        do_train=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_length=args.max_seq_length,
        save_steps=args.save_steps,
        save_only_model=True,
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        ddp_find_unused_parameters=False,
    )
    peft_config = LoraConfig(
        r=32, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    loss_callback = LossLoggingCallback(log_dir="logs", name=f"{args.method}_{args.domain_adaptation}")    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        callbacks=[loss_callback],
    )
    if args.resume_from_checkpoint is not None:
        print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()
