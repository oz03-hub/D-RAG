import random
import json
import datasets
import copy

_2_AUG_SYSTEM_PROMPT = """You are a helpful assistant that generates *personalized* answers to a user's current question.
## Your inputs:
1. **Publicly available posts** relevant to the user's current question (representing target-domain knowledge).
2. **The user's past posts** (questions + detailed descriptions) in other domains.
3. **The user's current question**.

## Your task:
- Use **public posts** as the factual and conceptual ground truth for the target domain.
- Use **past posts** to infer the user's preferences, style, and likely priorities — even if they are from different domains (asymmetric personalization).
- Weave these two sources together so that the answer is both **accurate** (public-grounded) and **personalized** (user-profile-grounded).
- When public and personal contexts conflict, **favor factual correctness from public posts**, but adapt tone, focus, and framing to the user's style/preferences.

## Output:
Return a valid JSON object inside a ```json``` block with:
- `personalized_answer`: The answer to the current question, incorporating the public posts for domain correctness and the user's past posts for personalization.
- Your answer should not reference the existence of past or public posts explicitly — integrate them seamlessly.
"""

_2_AUG_CATEGORIZED_SYSTEM_PROMPT = """You are a helpful assistant that generates *personalized* answers to a user's current question.

## Your inputs:
1. **Public Posts relevant to the question**, organized by sub-category.
2. **The User's Past Posts**, also organized by sub-category.
3. **The Target Sub-Category of the user's current question**, indicating which posts in the prompt are most directly relevant.

## Your task:
- Use **Public Posts** as the factual and conceptual ground truth for answering the current question.  
- Use **User Posts** to infer the user's preferences, style, and priorities — even if they are from domains different from the target (asymmetric personalization).  
- Weave these two sources together so the answer is both **accurate** (public-grounded) and **personalized** (user-profile-grounded).  
- When public and personal contexts conflict, **favor factual correctness from public posts**, but adapt tone, focus, and framing to the user's inferred preferences.

## Output:
Return a valid JSON object inside a ```json``` block with:
- `personalized_answer`: The answer to the current question, seamlessly integrating public posts for domain correctness and user posts for personalization.
- Do not explicitly mention the existence of sub-categories or posts in the output. Your answer should read as a natural, context-aware response.
- Your answer should not reference the existence of past or public posts explicitly — integrate them seamlessly.
"""


_RAG_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions.
# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences.
# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences. Your output should be a valid json object in ```json ``` block that contains the following fields:
    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

_RAG_WITH_PLAN_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions and detailed descriptions of these questions. Additionally, you are provided with the aspects that the user expects to see in the response to their question, which you can use to generate a personalized answer.

# Your input:
    - The user's current question from a post.
    - The user's past post questions and detailed descriptions of these questions.
    - The aspects that the user expects to see in the response to their question.
# Your task: Answer the user's current question in a personalized way by considering this user's past post questions and detailed descriptions of these questions, to learn about the user's preferences. Additionally, you should consider the aspects that the user expects to see in the response to their question.
# Your output: You should generate personalized answer to the user's current question by considering this user's past post questions and detailed descriptions of these questions to learn about user's preferences. Additionally, you should consider the aspects that the user expects to see in the response to their question. Your output should be a valid json object in ```json ``` block that contains the following fields: 
    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

_AUG2_WITH_PLAN_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering this user's past post questions, public posts, and detailed descriptions of these questions. Additionally, you are provided with the aspects that the user expects to see in the response to their question, which you can use to generate a personalized answer.

# Your input:
    - The user's current question from a post.
    - Public Posts relevant to the question, organized by sub-category.
    - The user's past post questions and detailed descriptions of these questions.
    - The aspects that the user expects to see in the response to their question.
# Your task: Answer the user's current question in a personalized way by considering this user's past posts and public questions and detailed descriptions of these questions, to learn about the user's preferences. Additionally, you should consider the aspects that the user expects to see in the response to their question.
# Your output: You should generate personalized answer to the user's current question by considering this user's past posts and public questions and detailed descriptions of these questions to learn about user's preferences. Additionally, you should consider the aspects that the user expects to see in the response to their question. Your output should be a valid json object in ```json ``` block that contains the following fields: 
    - personalized_answer: contains the personalized answer to the user's current question considering the this user's past post questions and detailed descriptions of these questions to learn about user's preferences.
"""

_PUBLIC_ONLY_WITH_PLAN_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering publicly available posts related to the user's current question.

# Your input:
    - Publicly available posts relevant to the current question.
    - The user's current question from a post.
    - The aspects that the user expects to see in the response to their question.

# Your task: Answer the user's current question in a personalized way by considering the publicly available post questions and detailed descriptions of these questions. Additionally, you should consider the aspects that the user expects to see in the response to their question.
# Your output: You should generate personalized answer to the user's current question by considering the publicly available post questions and detailed descriptions of these questions. Additionally, you should consider the aspects that the user expects to see in the response to their question. Your output should be a valid json object in ```json ``` block that contains the following fields: 
    - personalized_answer: contains the personalized answer to the user's current question considering the publicly available post questions and detailed descriptions of these questions.
"""

_PLANNER_PROMPT = """You are a helpful assistant designed to generate the topics that user might expect to see in a response to their question. Your task is to extract the important aspects that the user expects to see in a response to their question considering the previous questions asked by the same user and the detailed information need they provided in the post.
# Your input:
    - The user's current question.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Extract the important aspects that the user expects to see in a response to their question considering the previous questions asked by the same user and the detailed information need they provided in the post.
# Your output: You should generate a list of aspects that are important for the user to be included in the response to their question. 
"""

_PLANNER_PUBLIC_ONLY_PROMPT = """You are a helpful assistant designed to generate the topics that user might expect to see in a response to their question. Your task is to extract the important aspects that the user expects to see in a response to their question considering publicly available posts related to the question.
# Your input:
    - The user's current question.
    - The publicly available posts relevant to the current question.
# Your task: Extract the important aspects that the user expects to see in a response to their question considering the publicly available posts.
# Your output: You should generate a list of aspects that are important for the user to be included in the response to their question. 
"""


_PLANNER_CROSS_DOMAIN_PROMPT = """You are a helpful assistant designed to generate the topics that user might expect to see in a response to their question. Your task is to extract the important aspects that the user expects to see in a response to their question considering the previous questions asked by the same user and other questions asked by users with similar interests.
# Your input:
    - The user's current question.
    - Similar users' past post questions and detailed descriptions of these questions.
    - The user's past post questions and detailed descriptions of these questions.
# Your task: Extract the important aspects that the user expects to see in a response to their question considering the previous questions asked by the same user and other questions asked by users with similar interests.
# Your output: You should generate a list of aspects that are important for the user to be included in the response to their question. 
"""

_ONLY_PUBLIC_SYSTEM_PROMPT = """You are a helpful assistant designed to generate personalized responses to user questions. Your task is to answer a user's question from a post in a personalized way by considering publicly available posts related to the user's current question.

# Your input:
    - Publicly available posts relevant to the current question.
    - The user's current question from a post.
# Your task: Answer the user's current question in a personalized way by considering the publicly available post questions and detailed descriptions of these questions.
# Your output: You should generate personalized answer to the user's current question by considering the publicly available post questions and detailed descriptions of these questions. Your output should be a valid json object in ```json ``` block that contains the following fields: 
    - personalized_answer: contains the personalized answer to the user's current question considering the publicly available post questions and detailed descriptions of these questions.
"""

_ONLY_PUBLIC_USER_PROMPT = """
# Publicly available posts (relevant to the current questions and target domain):
{public_posts}

# Current question:
{question}
"""

_2_AUG_USER_PROMPT = """
# Publicly available posts (relevant to the current question and target domain):
{public_posts}

# User's past posts (questions + detailed descriptions):
{profile}

# Current question:
{question}
"""

_2_AUG_CATEGORIZED_USER_PROMPT = """
# Similar User Posts (domain knowledge for the current question) organized by sub-category:  
{public_posts}

# User Profile (past posts showing style, preferences, and priorities) organized by sub-category:  
{profile}

# Current Question, this question belongs to the target sub-category: {target_subcat}  
{question}
"""

_RAG_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
"""

_RAG_WITH_PLAN_USER_PROMPT = """
# Past post questions and detailed descriptions of these questions:
{profile}
# Current post question:
{question}
# Aspects expected in the response:
{aspects}
"""

_AUG2_WITH_PLAN_USER_PROMPT = """
# Similar User Posts (domain knowledge for the current question) organized by sub-category:  
{public_posts}

# User Profile (past posts showing style, preferences, and priorities) organized by sub-category:  
{profile}

# Current Question, this question belongs to the target sub-category: {target_subcat}  
{question}

# Aspects expected in the response:
{aspects}
"""

_PUBLIC_ONLY_WITH_PLAN_USER_PROMPT = """
# Publicly available posts (relevant to the current questions and target domain):
{public_posts}

# Current question:
{question}

# Aspects expected in the response:
{aspects}
"""


def apply_chat_template(
    conversation, tokenizer, tokenize=True, add_generation_prompt=False
):
    try:
        return tokenizer.apply_chat_template(
            conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
    except Exception as e:
        if e.message == "System role not supported":
            conversation_new = []
            system_prompt = conversation[0]["content"]
            user_prompt = conversation[1]["content"]
            new_user_prompt = system_prompt + "\n\n" + user_prompt
            conversation_new.append(
                {"role": "user", "content": new_user_prompt, "type": "text"}
            )
            return tokenizer.apply_chat_template(
                conversation_new,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
            )
        else:
            raise e


def get_baseline_no_rag_formatter(tokenizer, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant designed to generate personalized responses to user questions.  Your output should be a valid json object in ```json ``` block that contains the following fields:\n   - personalized_answer: contains the personalized answer to the user's current question.",
                    "type": "text",
                },
                {"role": "user", "content": user_prompt, "type": "text"},
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(
                    conversation, tokenizer, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)
        return texts

    return formatter


def get_baseline_2_aug_categorized_formatter(
    tokenizer, num_contexts, proprietary_llm=False, train=False
):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            target_subcat = data["target_subcat"][i]

            public_posts_by_category = {}
            for post in data["public_posts"][i][:num_contexts]:
                if post["category"] not in public_posts_by_category:
                    public_posts_by_category[post["category"]] = []
                public_posts_by_category[post["category"]].append(post["text"])

            public_text = ""
            for k, v in public_posts_by_category.items():
                public_text += f"{k.upper()} Category:\n\n"
                public_text += "\n".join(v)

            user_posts_by_category = {}
            for post in data["profile"][i][:num_contexts]:
                if post["category"] not in user_posts_by_category:
                    user_posts_by_category[post["category"]] = []
                user_posts_by_category[post["category"]].append(post["text"])

            user_posts_text = ""
            for k, v in user_posts_by_category.items():
                user_posts_text += f"{k.upper()} Category:\n\n"
                user_posts_text += "\n".join(v)

            conversation = [
                {
                    "role": "system",
                    "content": _2_AUG_CATEGORIZED_SYSTEM_PROMPT,
                    "type": "text",
                },
                {
                    "role": "user",
                    "content": _2_AUG_CATEGORIZED_USER_PROMPT.format(
                        public_posts=public_text,
                        profile=user_posts_text,
                        question=user_prompt,
                        target_subcat=target_subcat,
                    ),
                },
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expected_output = data["expected_output"][i]
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": expected_output,
                            "type": "text",
                        }
                    )
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(
                        conversation,
                        tokenizer,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter


def get_baseline_2_aug_formatter(
    tokenizer, num_contexts, proprietary_llm=False, train=False
):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            public_posts = "\n\n".join(
                [x["text"] for x in data["public_posts"][i][:num_contexts]]
            )
            profile = "\n\n".join(
                [x["text"] for x in data["profile"][i][:num_contexts]]
            )

            conversation = [
                {"role": "system", "content": _2_AUG_SYSTEM_PROMPT, "type": "text"},
                {
                    "role": "user",
                    "content": _2_AUG_USER_PROMPT.format(
                        public_posts=public_posts,
                        profile=profile,
                        question=user_prompt,
                    ),
                },
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expected_output = data["expected_output"][i]
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": expected_output,
                            "type": "text",
                        }
                    )
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(
                        conversation,
                        tokenizer,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter


def get_public_only_formatter(
    tokenizer, num_contexts, proprietary_llm=False, train=False
):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            public_posts = "\n\n".join(
                [x["text"] for x in data["profile"][i][:num_contexts]]
            )
            conversation = [
                {
                    "role": "system",
                    "content": _ONLY_PUBLIC_SYSTEM_PROMPT,
                    "type": "text",
                },
                {
                    "role": "user",
                    "content": _ONLY_PUBLIC_USER_PROMPT.format(
                        public_posts=public_posts,
                        question=user_prompt,
                    ),
                },
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expected_output = data["expected_output"][i]
                    conversation.append(
                        {
                            "role": "assistant",
                            "content": expected_output,
                            "type": "text",
                        }
                    )
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(
                        conversation,
                        tokenizer,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter


def get_baseline_rag_formatter(
    tokenizer, num_contexts, proprietary_llm=False, train=False
):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            profile = "\n\n".join(
                [x["text"] for x in data["profile"][i][:num_contexts]]
            )
            conversation = [
                {"role": "system", "content": _RAG_SYSTEM_PROMPT, "type": "text"},
                {
                    "role": "user",
                    "content": _RAG_USER_PROMPT.format(
                        profile=profile, question=user_prompt
                    ),
                    "type": "text",
                },
            ]
            if proprietary_llm:
                text = conversation
            else:
                if train:
                    expeted_output = data["expected_output"][i]
                    conversation.append(
                        {"role": "assistant", "content": expeted_output, "type": "text"}
                    )
                    text = {"messages": conversation}
                else:
                    text = apply_chat_template(
                        conversation,
                        tokenizer,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
            texts.append(text)
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter


def get_unsloth_formatter(tokenizer, num_contexts, train=False, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            profile = "\n\n".join(
                [x["text"] for x in data["profile"][i][:num_contexts]]
            )
            conversation = [
                {"role": "system", "content": _PLANNER_PROMPT, "type": "text"},
                {
                    "role": "user",
                    "content": _RAG_USER_PROMPT.format(
                        profile=profile, question=user_prompt
                    ),
                    "type": "text",
                },
            ]

            if not train:
                if proprietary_llm:
                    text = conversation
                else:
                    text = apply_chat_template(
                        conversation,
                        tokenizer,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                texts.append(text)
            else:
                aspects = ""
                for aspect in data["rubric_aspects"][i]:
                    aspects += "- " + aspect["aspect"] + "\n"
                conversation.append(
                    {
                        "role": "assistant",
                        "content": f"The user expects to see the following aspects in the response to their question:\n{aspects}",
                        "type": "text",
                    }
                )

                # Convert to text format using chat template
                formatted_text = tokenizer.apply_chat_template(
                    conversation,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(formatted_text)

        if train:
            # Return in the format Unsloth expects
            dataset = datasets.Dataset.from_dict({"text": texts})
            return dataset
        return texts

    return formatter


def get_cross_domain_planner_formatter(
    tokenizer, num_contexts, train=False, proprietary_llm=False
):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            target_subcat = data["target_subcat"][i]

            public_posts_by_category = {}
            for post in data["public_posts"][i][:num_contexts]:
                if post["category"] not in public_posts_by_category:
                    public_posts_by_category[post["category"]] = []
                public_posts_by_category[post["category"]].append(post["text"])

            public_text = ""
            for k, v in public_posts_by_category.items():
                public_text += f"{k.upper()} Category:\n\n"
                public_text += "\n".join(v)

            user_posts_by_category = {}
            for post in data["profile"][i][:num_contexts]:
                if post["category"] not in user_posts_by_category:
                    user_posts_by_category[post["category"]] = []
                user_posts_by_category[post["category"]].append(post["text"])

            user_posts_text = ""
            for k, v in user_posts_by_category.items():
                user_posts_text += f"{k.upper()} Category:\n\n"
                user_posts_text += "\n".join(v)

            user_prompt = data["question"][i]
            conversation = [
                {"role": "system", "content": _PLANNER_CROSS_DOMAIN_PROMPT, "type": "text"},
                {
                    "role": "user",
                    "content": _2_AUG_CATEGORIZED_USER_PROMPT.format(
                        public_posts=public_text,
                        profile=user_posts_text,
                        question=user_prompt,
                        target_subcat=target_subcat,
                    ),
                    "type": "text",
                },
            ]
            if not train:
                if proprietary_llm:
                    text = conversation
                else:
                    text = apply_chat_template(
                        conversation,
                        tokenizer,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                texts.append(text)
            else:
                aspects = ""
                for aspect in data["rubric_aspects"][i]:
                    aspects += "- " + aspect["aspect"] + "\n"
                conversation.append(
                    {
                        "role": "assistant",
                        "content": f"The user expects to see the following aspects in the response to their question:\n{aspects}",
                        "type": "text",
                    }
                )
                texts.append({"messages": conversation})
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter

def get_generation_with_plan_public_only_formatter(tokenizer, num_contexts, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            public_posts = "\n\n".join(
                [x["text"] for x in data["profile"][i][:num_contexts]]
            )
            plan = data["plan"][i]
            conversation = [
                {"role": "system", "content": _PUBLIC_ONLY_WITH_PLAN_SYSTEM_PROMPT, "type": "text"},
                {
                    "role": "user",
                    "content": _PUBLIC_ONLY_WITH_PLAN_USER_PROMPT.format(
                        public_posts=public_posts, question=user_prompt, aspects=plan
                    ),
                    "type": "text",
                },
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(
                    conversation, tokenizer, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)
        return texts

    return formatter

def get_public_only_planner_formatter(tokenizer, num_contexts, train=False, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            public_posts = "\n\n".join(
                [x["text"] for x in data["profile"][i][:num_contexts]]
            )
            conversation = [
                {"role": "system", "content": _PLANNER_PUBLIC_ONLY_PROMPT, "type": "text"},
                {
                    "role": "user",
                    "content": _ONLY_PUBLIC_USER_PROMPT.format(
                        public_posts=public_posts, question=user_prompt
                    ),
                    "type": "text",
                },
            ]
            if not train:
                if proprietary_llm:
                    text = conversation
                else:
                    text = apply_chat_template(
                        conversation,
                        tokenizer,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                texts.append(text)
            else:
                aspects = ""
                for aspect in data["rubric_aspects"][i]:
                    aspects += "- " + aspect["aspect"] + "\n"
                conversation.append(
                    {
                        "role": "assistant",
                        "content": f"The user expects to see the following aspects in the response to their question:\n{aspects}",
                        "type": "text",
                    }
                )
                texts.append({"messages": conversation})
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter


def get_planner_formatter(tokenizer, num_contexts, train=False, proprietary_llm=False):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            profile = "\n\n".join(
                [x["text"] for x in data["profile"][i][:num_contexts]]
            )
            conversation = [
                {"role": "system", "content": _PLANNER_PROMPT, "type": "text"},
                {
                    "role": "user",
                    "content": _RAG_USER_PROMPT.format(
                        profile=profile, question=user_prompt
                    ),
                    "type": "text",
                },
            ]
            if not train:
                if proprietary_llm:
                    text = conversation
                else:
                    text = apply_chat_template(
                        conversation,
                        tokenizer,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                texts.append(text)
            else:
                aspects = ""
                for aspect in data["rubric_aspects"][i]:
                    aspects += "- " + aspect["aspect"] + "\n"
                conversation.append(
                    {
                        "role": "assistant",
                        "content": f"The user expects to see the following aspects in the response to their question:\n{aspects}",
                        "type": "text",
                    }
                )
                texts.append({"messages": conversation})
        if train:
            dataset = datasets.Dataset.from_list(texts)
            return dataset
        return texts

    return formatter


def get_generation_with_plan_aug2_formatter(
    tokenizer, num_contexts, proprietary_llm=False
):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            target_subcat = data["target_subcat"][i]

            public_posts_by_category = {}
            for post in data["public_posts"][i][:num_contexts]:
                if post["category"] not in public_posts_by_category:
                    public_posts_by_category[post["category"]] = []
                public_posts_by_category[post["category"]].append(post["text"])

            public_text = ""
            for k, v in public_posts_by_category.items():
                public_text += f"{k.upper()} Category:\n\n"
                public_text += "\n".join(v)

            user_posts_by_category = {}
            for post in data["profile"][i][:num_contexts]:
                if post["category"] not in user_posts_by_category:
                    user_posts_by_category[post["category"]] = []
                user_posts_by_category[post["category"]].append(post["text"])

            user_posts_text = ""
            for k, v in user_posts_by_category.items():
                user_posts_text += f"{k.upper()} Category:\n\n"
                user_posts_text += "\n".join(v)

            user_prompt = data["question"][i]
            plan = data["plan"][i]
            conversation = [
                {
                    "role": "system",
                    "content": _AUG2_WITH_PLAN_SYSTEM_PROMPT,
                    "type": "text",
                },
                {
                    "role": "user",
                    "content": _AUG2_WITH_PLAN_USER_PROMPT.format(
                        public_posts=public_text,
                        profile=user_posts_text,
                        question=user_prompt,
                        target_subcat=target_subcat,
                        aspects=plan,
                    ),
                    "type": "text",
                },
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(
                    conversation, tokenizer, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)
        return texts

    return formatter


def get_generation_with_plan_rag_formatter(
    tokenizer, num_contexts, proprietary_llm=False
):
    def formatter(data):
        texts = []
        for i in range(len(data["question"])):
            user_prompt = data["question"][i]
            profile = "\n\n".join(
                [x["text"] for x in data["profile"][i][:num_contexts]]
            )
            plan = data["plan"][i]
            conversation = [
                {
                    "role": "system",
                    "content": _RAG_WITH_PLAN_SYSTEM_PROMPT,
                    "type": "text",
                },
                {
                    "role": "user",
                    "content": _RAG_WITH_PLAN_USER_PROMPT.format(
                        profile=profile, question=user_prompt, aspects=plan
                    ),
                    "type": "text",
                },
            ]
            if proprietary_llm:
                text = conversation
            else:
                text = apply_chat_template(
                    conversation, tokenizer, tokenize=False, add_generation_prompt=True
                )
            texts.append(text)
        return texts

    return formatter
