#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import accumulate, chain
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorForTokenClassification
)

from ..config import Config

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch, logits, scores
        torch.cuda.empty_cache()

    return output_scores


class PRM:
    def __init__(self, search_config: Config, **model_kwargs):
        self.search_config = search_config
        self.model, self.tokenizer = self.load_model_and_tokenizer(search_config.prm_path)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError


class MathShepherd(PRM):
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # For batched inference
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.search_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.model,
            self.tokenizer,
            inputs_for_prm,
            self.search_config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(
                output
            ), f"{len(output_score)} != {len(output)}"

        return output_scores


class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, model_path
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        model.config.use_cache = False # DO NOT use KV Cache
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
        batch_size=4,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size) # default batch_size: 8 -> 4
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        inputs2_batch[i, 1:] == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


class TRL(PRM):
    def load_model_and_tokenizer(self, model_path) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # For batched inference
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = 'right'
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        return model, tokenizer
    
    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        batch_size = self.search_config.prm_batch_size
        output_scores = []
        output_scores_flatten = []
        all_masks = []
        all_inputs = []
        
        # Prepare data
        for question, output in zip(questions, outputs):
            steps_input_ids = []
            steps_masks = []
            for solution in output:
                # process string
                steps = solution.split(self.search_config.separator)
                if steps[-1] == '':
                    steps.pop()
                convs = [
                    {"role": "system", "content": self.search_config.system_prompt},
                    {"role": "user", "content": question}, 
                ]
                templated_convs = self.tokenizer.apply_chat_template(
                    convs,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # tokenize sentences
                prompt_ids = self.tokenizer(templated_convs, add_special_tokens=False)["input_ids"]
                completions_ids = [
                    self.tokenizer(completion, add_special_tokens=False)["input_ids"] for completion in steps
                ]
                mask = [1] * len(completions_ids)
                
                # merge question and answer
                separator_ids = self.tokenizer.encode(self.search_config.separator, add_special_tokens=False)
                completions_ids = [completion + separator_ids for completion in completions_ids]
                mask = [[-100] * (len(completion) - 1) + [label] for completion, label in zip(completions_ids, mask)]
                
                completion_ids = list(chain(*completions_ids))
                mask = list(chain(*mask))
                
                input_ids = prompt_ids + completion_ids
                mask = [-100] * len(prompt_ids) + mask
                steps_input_ids.append(input_ids)
                steps_masks.append(mask)
            all_masks.append(steps_masks)
            all_inputs.append(steps_input_ids)

        all_inputs_flat = [item for sublist in all_inputs for item in sublist]
        all_masks_flat = [item for sublist in all_masks for item in sublist]
        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        # Calculate scores
        for i in range(0, len(all_inputs_flat), batch_size):
            batch_inputs = all_inputs_flat[i : i+batch_size]
            batch_masks = all_masks_flat[i: i+batch_size]
            
            batch = [{"input_ids": torch.tensor(seq, dtype=torch.long), "labels": torch.tensor(label, dtype=torch.long)} for seq, label in zip(batch_inputs, batch_masks)]
            collated_batch = data_collator(batch)
            
            with torch.no_grad():
                model_outputs = self.model(input_ids=collated_batch['input_ids'].to(self.model.device), attention_mask=collated_batch['attention_mask'].to(self.model.device))
                logits = model_outputs.logits

            # get scores
            logits = logits.softmax(dim=-1)
            for i in range(logits.shape[0]):
                mask = (collated_batch['labels'][i] == -100)
                scores = logits[i, ~mask, 1].cpu().tolist()
                output_scores_flatten.append(scores)

        # Reshape output_scores_flatten
        k = 0
        for output in outputs:
            tmp = []
            for i in range(len(output)):
                tmp.append(output_scores_flatten[k])
                k += 1
            output_scores.append(tmp)
        
        return output_scores


def load_prm(config: Config) -> PRM:
    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if "Llama3.1-8B-PRM-Deepseek-Data" in config.prm_path:
        return RLHFFlow(config)

    if "TRL-PRM" in config.prm_path:
        return TRL(config)

    raise NotImplementedError(f"PRM {config.prm_path} not implemented")
