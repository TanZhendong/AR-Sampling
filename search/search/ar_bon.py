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
import copy
import logging
import re
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from ..config import Config
from ..models.reward_models import PRM

from .utils import Beam, build_conv, generate_k_steps, last, SolutionPath, generate_multi_steps

logger = logging.getLogger()
from ..utils.score import aggregate_scores


def _ar_bon(batch_of_prompts, config: Config, llm: LLM, prm: PRM) -> list[SolutionPath]:
    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        stop=["\n\n"],
        include_stop_str_in_output=True,
        n=1,
    )

    # Init solution paths
    solutions: list[SolutionPath] = []
    prompt_index = 0
    for prompt in batch_of_prompts:
        for i in range(config.n):
            solutions.append(
                SolutionPath(
                    index=prompt_index,
                    prompt=prompt,
                    current_text="",
                    next_step=None,
                    scores=[],
                    agg_score=0.0,
                    stop_reasons=None,
                    completed=False,  
                    completion_tokens=0,
                )
            )
        prompt_index += 1
    completed_solutions: list[SolutionPath] = []

    # Iterative generation
    for i in tqdm(range(config.num_iterations), desc="Solution path generating"):
        if i == 0:
            active_paths = [b for b in solutions if not b.completed]
        else:
            active_paths = [b for b in active_paths if not b.completed]

        if i == config.num_iterations - 1:
            # Last iteration, generate to EOS
            sampling_params = SamplingParams(
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                n=1,
            )

        # Tokenize convs
        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active_paths
        ]
        continue_final_message = i > 0
        add_generation_prompt = i == 0

        tokenizer = llm.get_tokenizer()
        # TODO: Fix Mistral-7B chat_template
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template
        templated_convs = tokenizer.apply_chat_template(
            convs,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            tokenize=False,
        )
        gen_results = generate_multi_steps(
            templated_convs, config.max_steps, llm, sampling_params
        )

        prompts, completions = [], []
        for active_path, gen_result in zip(active_paths, gen_results, strict=True):
            active_path.current_text += gen_result.lookahead_text
            active_path.stop_reasons = gen_result.stop_reason
            # check finish
            if (
                active_path.stop_reasons == "EOS"
                or active_path.stop_reasons == "length"
            ):
                active_path.completed = True
                completed_solutions.append(active_path)
            prompts.append(active_path.prompt)
            completions.append([active_path.current_text])

        scores = prm.score(prompts, completions)
        for active_path, score in zip(active_paths, scores, strict=True):
            active_path.scores = score[0]
            active_path.agg_score = aggregate_scores(score[0], config.agg_strategy)

        active_paths = [p for p in active_paths if not p.completed]
        # Early stopping if all beams are completed
        if len(active_paths) == 0:
            break
        
        # check scores and add rethink prompt
        for active_path in active_paths:
            split_steps = active_path.current_text.split('\n\n')
            if split_steps[-1] == '':
                split_steps.pop()
            steps_window = split_steps[-config.max_steps:]
            scores_window = active_path.scores[-config.max_steps:]
            wrong_index = next((i for i, score in enumerate(scores_window) if score < config.p), None)

            # add rethink prompt
            if wrong_index is not None and active_path.rethink < config.max_rethink:
                wrong_step = steps_window[wrong_index]
                # match = re.search(r'## Step (\d+):', wrong_step)
                match = re.search(r'Step (\d+):', wrong_step) # cause qwen always forget ##
                if match:
                    step_number = match.group(1)
                    rethink_prompt = f"Wait! Maybe I made some mistakes in Step {step_number}. I need to rethink from it.\n## Step {step_number}: "
                    active_path.current_text += rethink_prompt
                    active_path.rethink += 1
            else: 
                active_path.rethink = 0

    completed_solutions = sorted(completed_solutions, key=lambda x: x.index, reverse=False)
    return completed_solutions


def ar_bon(examples, config: Config, llm: LLM, prm: PRM):
    problems = examples["problem"]
    solutions = _ar_bon(problems, config, llm, prm)

    completions = []
    pred = []
    completion_tokens = []
    scores = []

    for i in range(0, len(solutions), config.n):
        # add results
        completions.append([s.current_text for s in solutions[i : i+config.n]])
        completion_tokens.append([s.completion_tokens for s in solutions[i : i+config.n]])
        scores.append([s.scores for s in solutions[i : i+config.n]])

        candidates = completions[-1]
        agg_scores = [
            aggregate_scores(s, config.agg_strategy) for s in scores[-1]
        ]
        pred_results = candidates[np.argmax(agg_scores)]
        pred.append(pred_results)

    examples['completions'] = completions
    examples['pred'] = pred
    examples['scores'] = scores
    examples['completion_tokens'] = completion_tokens

    return examples
