# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Must be run with OMP_NUM_THREADS=1

import random
import argparse
import logging
import os
import threading
import time
import timeit
import traceback
import pprint
import typing

import torch
from torch import multiprocessing as mp
from torch import nn
from torch.nn import functional as F

import gym
import gym_minigrid.wrappers as wrappers

from torch.distributions.normal import Normal

from torchbeast.core import environment
from torchbeast.core import file_writer
from torchbeast.core import prof
from torchbeast.core import vtrace

from env_utils import Observation_WrapperSetup, FrameStack


# Some Global Variables
# We start t* at 7 steps.
generator_batch = dict()
generator_batch_aux = dict()
generator_current_target = 7.0
generator_count = 0

# yapf: disable
parser = argparse.ArgumentParser(description='PyTorch Scalable Agent')

parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0',
                    help='Gym environment.')
parser.add_argument('--mode', default='train',
                    choices=['train', 'test', 'test_render'],
                    help='Training or test mode.')
parser.add_argument('--xpid', default=None,
                    help='Experiment id (default: None).')

# Training settings.
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint.')
parser.add_argument('--savedir', default='./experimentsMinigrid',
                    help='Root dir where experiment data will be saved.')
parser.add_argument('--total_frames', default=5000000000, type=int, metavar='T',
                    help='Total environment frames to train for.')
parser.add_argument('--num_actors', default=4, type=int, metavar='N',
                    help='Number of actors (default: 4).')
parser.add_argument('--num_buffers', default=None, type=int,
                    metavar='N', help='Number of shared-memory buffers.')
parser.add_argument('--num_threads', default=4, type=int,
                    metavar='N', help='Number learner threads.')
parser.add_argument('--disable_cuda', action='store_true',
                    help='Disable CUDA.')

# Loss settings.
parser.add_argument('--entropy_cost', default=0.0005, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument('--generator_entropy_cost', default=0.05, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument('--baseline_cost', default=0.5, type=float,
                    help='Baseline cost/multiplier.')
parser.add_argument('--discounting', default=0.99, type=float,
                    help='Discounting factor.')
parser.add_argument('--reward_clipping', default='abs_one',
                    choices=['abs_one', 'soft_asymmetric', 'none'],
                    help='Reward clipping.')

# Optimizer settings.
parser.add_argument('--learning_rate', default=0.001, type=float,
                    metavar='LR', help='Learning rate.')
parser.add_argument('--generator_learning_rate', default=0.002, type=float,
                    metavar='LR', help='Learning rate.')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant.')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum.')
parser.add_argument('--epsilon', default=0.01, type=float,
                    help='RMSProp epsilon.')


# Other Hyperparameters
parser.add_argument('--batch_size', default=8, type=int, metavar='B',
                    help='Learner batch size (default: 4).')
parser.add_argument('--generator_batch_size', default=32, type=int, metavar='BB',
                    help='Learner batch size (default: 4).')
parser.add_argument('--unroll_length', default=100, type=int, metavar='T',
                    help='The unroll length (time dimension; default: 64).')
parser.add_argument('--goal_dim', default=10, type=int,
                    help='Size of Goal Embedding')
parser.add_argument('--state_embedding_dim', default=256, type=int,
                    help='Dimension of the state embedding representation used in the student')
parser.add_argument('--generator_reward_negative', default= -0.1, type=float,
                    help='Coefficient for the intrinsic reward')
parser.add_argument('--generator_threshold', default=0.1, type=float,
                    help='Threshold mean reward for wich scheduler increases difficulty')
parser.add_argument('--generator_counts', default=10, type=int,
                    help='Number of time before generator increases difficulty')
parser.add_argument('--generator_maximum', default=100, type=float,
                    help='Maximum difficulty')                    
parser.add_argument('--generator_reward_coef', default=1.0, type=float,
                    help='Coefficient for the generator reward')

# Map Layout 
parser.add_argument('--fix_seed', action='store_true',
                    help='Fix the environment seed so that it is \
                    no longer procedurally generated but rather a layout every time.')
parser.add_argument('--env_seed', default=1, type=int,
                    help='The seed to set for the env if we are using a single fixed seed.')
parser.add_argument('--inner', action='store_true',
                    help='Exlucde outer wall')
parser.add_argument('--num_input_frames', default=1, type=int,
                    help='Number of input frames to the model and state embedding including the current frame \
                    When num_input_frames > 1, it will also take the previous num_input_frames - 1 frames as input.')

# Ablations and other settings
parser.add_argument("--use_lstm", action="store_true",
                    help="Use LSTM in agent model.")
parser.add_argument('--num_lstm_layers', default=1, type=int,
                    help='Lstm layers.')
parser.add_argument('--disable_use_embedding', action='store_true',
                    help='Disable embeddings.')
parser.add_argument('--no_extrinsic_rewards', action='store_true',
                    help='Only intrinsic rewards.')
parser.add_argument('--no_generator', action='store_true',
                    help='Use vanilla policy-deprecated')
parser.add_argument('--intrinsic_reward_coef', default=1.0, type=float,
                    help='Coefficient for the intrinsic reward')
parser.add_argument('--random_agent', action='store_true',
                    help='Use a random agent to test the env.')
parser.add_argument('--novelty', action='store_true',
                    help='Discount rewards based on times goal has been proposed.')
parser.add_argument('--novelty_bonus', default=0.1, type=float,
                    help='Bonus you get for proposing objects if novelty')
parser.add_argument('--novelty_coef', default=0.3, type=float,
                    help='Modulates novelty bonus if novelty')
parser.add_argument('--restart_episode', action='store_true',
                    help='Restart Episode when reaching intrinsic goal.')
parser.add_argument('--modify', action='store_true',
                    help='Modify Goal instead of having to reach the goal')
parser.add_argument('--no_boundary_awareness', action='store_true',
                    help='Remove Episode Boundary Awareness')
parser.add_argument('--generator_loss_form', type=str, default='threshold',
                    help='[threshold,dummy,gaussian, linear]')
parser.add_argument('--generator_target', default=5.0, type=float,
                    help='Mean target for Gassian and Linear Rewards')
parser.add_argument('--target_variance', default=15.0, type=float,
                    help='Variance for the Gaussian Reward')
# yapf: enable

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    # Take the mean over batch, sum over time.
    return 0.5 * torch.sum(torch.mean(advantages ** 2, dim=1))


def compute_entropy_loss(logits):
    # Regularizing Entropy Loss
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    entropy_per_timestep = torch.sum(-policy * log_policy, dim=-1)
    return -torch.sum(torch.mean(entropy_per_timestep, dim=1))


def compute_policy_gradient_loss(logits, actions, advantages):
    # Main Policy Loss
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    advantages.requires_grad = False
    policy_gradient_loss_per_timestep = cross_entropy * advantages
    return torch.sum(torch.mean(policy_gradient_loss_per_timestep, dim=1))

def act(
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    generator_model,
    buffers: Buffers,
    initial_agent_state_buffers, flags):
    """Defines and generates IMPALA actors in multiples threads."""

    try:
        logging.info("Actor %i started.", actor_index)
        timings = prof.Timings()  # Keep track of how fast things are.
        gym_env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        gym_env.seed(seed)
        #gym_env = wrappers.FullyObsWrapper(gym_env)

        if flags.num_input_frames > 1:
            gym_env = FrameStack(gym_env, flags.num_input_frames)
        
        
        env = Observation_WrapperSetup(gym_env, fix_seed=flags.fix_seed, env_seed=flags.env_seed)
        env_output = env.initial()
        initial_frame = env_output['frame']

        
        agent_state = model.initial_state(batch_size=1)
        generator_output = generator_model(env_output)
        goal = generator_output["goal"]
        
        agent_output, unused_state = model(env_output, agent_state, goal)
        while True:
            index = free_queue.get()
            if index is None:
                break
            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for key in generator_output:
                buffers[key][index][0, ...] = generator_output[key]   
            buffers["initial_frame"][index][0, ...] = initial_frame     
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor

            # Do new rollout
            for t in range(flags.unroll_length):
                aux_steps = 0
                timings.reset()

                

                if flags.modify:
                    new_frame = torch.flatten(env_output['frame'], 2, 3)
                    old_frame = torch.flatten(initial_frame, 2, 3)
                    ans = new_frame == old_frame
                    ans = torch.sum(ans, 3) != 3  # Reached if the three elements of the frame are not the same.
                    reached_condition = torch.squeeze(torch.gather(ans, 2, torch.unsqueeze(goal.long(),2)))
                    
                else:
                    agent_location = torch.flatten(env_output['frame'], 2, 3)
                    agent_location = agent_location[:,:,:,0] 
                    agent_location = (agent_location == 10).nonzero() # select object id
                    agent_location = agent_location[:,2]
                    agent_location = agent_location.view(agent_output["action"].shape)
                    reached_condition = goal == agent_location
           

                if reached_condition:   # Generate new goal when reached intrinsic goal
                    if flags.restart_episode:
                        env_output = env.initial() 
                    else:
                        env.episode_step = 0    
                    initial_frame = env_output['frame']
                    with torch.no_grad():
                        generator_output = generator_model(env_output)
                    goal = generator_output["goal"]

                if env_output['done'][0] == 1:  # Generate a New Goal when episode finished
                    initial_frame = env_output['frame']
                    with torch.no_grad():
                        generator_output = generator_model(env_output)
                    goal = generator_output["goal"]


                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state, goal)
                
                    
                timings.time("model")

                env_output = env.step(agent_output["action"])

                    
                    

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]
                for key in generator_output:
                    buffers[key][index][t + 1, ...] = generator_output[key]  
                buffers["initial_frame"][index][t + 1, ...] = initial_frame     

                

                timings.time("write")
            full_queue.put(index)
        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    timings,
    lock=threading.Lock()):
    """Returns a Batch with the history."""

    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(t.to(device=flags.device, non_blocking=True)
                                for t in initial_agent_state)
    timings.time("device")

    return batch, initial_agent_state


def reached_goal_func(frames, goals, initial_frames = None, done_aux = None):
    """Auxiliary function which evaluates whether agent has reached the goal."""
    if flags.modify:
        new_frame = torch.flatten(frames, 2, 3)
        old_frame = torch.flatten(initial_frames, 2, 3)
        ans = new_frame == old_frame
        ans = torch.sum(ans, 3) != 3  # reached if the three elements are not the same
        reached = torch.squeeze(torch.gather(ans, 2, torch.unsqueeze(goals.long(),2)))
        if flags.no_boundary_awareness:
            reached = reached.float() * (1 - done_aux.float())
        return reached
    else:    
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() # select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(goals.shape)
        return (goals == agent_location).float()

def learn(
    actor_model, model, actor_generator_model, generator_model, batch, initial_agent_state, optimizer, generator_model_optimizer, scheduler, generator_scheduler, flags, max_steps=100.0, lock=threading.Lock()
):
    """Performs a learning (optimization) step for the policy, and for the generator whenever the generator batch is full."""
    with lock:

        # Loading Batch
        next_frame = batch['frame'][1:].float().to(device=flags.device)
        initial_frames = batch['initial_frame'][1:].float().to(device=flags.device)
        done_aux = batch['done'][1:].float().to(device=flags.device) 
        reached_goal = reached_goal_func(next_frame, batch['goal'][1:].to(device=flags.device), initial_frames = initial_frames, done_aux = done_aux)
        intrinsic_rewards = flags.intrinsic_reward_coef * reached_goal
        reached = reached_goal.type(torch.bool)
        intrinsic_rewards = intrinsic_rewards*(intrinsic_rewards - 0.9 * (batch["episode_step"][1:].float()/max_steps))

        learner_outputs, unused_state = model(batch, initial_agent_state, batch['goal'])
        bootstrap_value = learner_outputs["baseline"][-1]
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}
        rewards = batch["reward"]
        
        # Student Rewards
        if flags.no_generator:
            total_rewards = rewards
        elif flags.no_extrinsic_rewards:
            total_rewards = intrinsic_rewards 
        else:
            total_rewards = rewards + intrinsic_rewards

        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(total_rewards, -1, 1)
        elif flags.reward_clipping == "soft_asymmetric":
            squeezed = torch.tanh(total_rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = torch.where(total_rewards < 0, 0.3 * squeezed, squeezed) * 5.0
        elif flags.reward_clipping == "none":
            clipped_rewards = total_rewards
        discounts = (~batch["done"]).float() * flags.discounting
        clipped_rewards += 1.0 * (rewards>0.0).float()  



        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        # Student Loss
        # Compute loss as a weighted sum of the baseline loss, the policy
        # gradient loss and an entropy regularization term.
        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )
        
        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]

        if torch.isnan(torch.mean(episode_returns)):
            aux_mean_episode = 0.0
        else:
            aux_mean_episode = torch.mean(episode_returns).item()
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": aux_mean_episode, 
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "gen_rewards": None,  
            "gg_loss": None,
            "generator_baseline_loss": None,
            "generator_entropy_loss": None,
            "mean_intrinsic_rewards": None,
            "mean_episode_steps": None,
            "ex_reward": None,
            "generator_current_target": None,
        }

        if flags.no_generator:
            stats["gen_rewards"] = 0.0,  
            stats["gg_loss"] = 0.0,
            stats["generator_baseline_loss"] = 0.0,
            stats["generator_entropy_loss"] = 0.0,
            stats["mean_intrinsic_rewards"] = 0.0,
            stats["mean_episode_steps"] = 0.0,
            stats["ex_reward"] = 0.0,
            stats["generator_current_target"] = 0.0,

        scheduler.step()
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 40.0)
        optimizer.step()
        actor_model.load_state_dict(model.state_dict())

        # Generator:
        if not flags.no_generator:
            global generator_batch
            global generator_batch_aux
            global generator_current_target
            global generator_count
            global goal_count_dict

            # Loading Batch
            is_done = batch['done']==1
            reached = reached_goal.type(torch.bool)
            if 'frame' in generator_batch.keys():
                generator_batch['frame'] = torch.cat((generator_batch['frame'], batch['initial_frame'][is_done].float().to(device=flags.device)), 0) 
                generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][is_done].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat((generator_batch['episode_step'], batch['episode_step'][is_done].float().to(device=flags.device)), 0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'], batch['generator_logits'][is_done].float().to(device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'], torch.zeros(batch['goal'].shape)[is_done].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][is_done].float().to(device=flags.device)), 0)
                generator_batch['carried_obj'] = torch.cat((generator_batch['carried_obj'], batch['carried_obj'][is_done].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat((generator_batch['carried_col'], batch['carried_col'][is_done].float().to(device=flags.device)), 0)
                
                generator_batch['carried_obj'] = torch.cat((generator_batch['carried_obj'], batch['carried_obj'][reached].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat((generator_batch['carried_col'], batch['carried_col'][reached].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][reached].float().to(device=flags.device)), 0) 
                generator_batch['frame'] = torch.cat((generator_batch['frame'], batch['initial_frame'][reached].float().to(device=flags.device)), 0) 
                generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][reached].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat((generator_batch['episode_step'], batch['episode_step'][reached].float().to(device=flags.device)), 0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'], batch['generator_logits'][reached].float().to(device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'], torch.ones(batch['goal'].shape)[reached].float().to(device=flags.device)), 0)
            else:
                generator_batch['frame'] = (batch['initial_frame'][is_done]).float().to(device=flags.device) # Notice we use initial_frame from batch
                generator_batch['goal'] = (batch['goal'][is_done]).to(device=flags.device)
                generator_batch['episode_step'] = (batch['episode_step'][is_done]).float().to(device=flags.device)
                generator_batch['generator_logits'] = (batch['generator_logits'][is_done]).float().to(device=flags.device)
                generator_batch['reached'] = (torch.zeros(batch['goal'].shape)[is_done]).float().to(device=flags.device)
                generator_batch['ex_reward'] = (batch['reward'][is_done]).float().to(device=flags.device)
                generator_batch['carried_obj'] = (batch['carried_obj'][is_done]).float().to(device=flags.device)
                generator_batch['carried_col'] = (batch['carried_col'][is_done]).float().to(device=flags.device)

                generator_batch['carried_obj'] = torch.cat((generator_batch['carried_obj'], batch['carried_obj'][reached].float().to(device=flags.device)), 0)
                generator_batch['carried_col'] = torch.cat((generator_batch['carried_col'], batch['carried_col'][reached].float().to(device=flags.device)), 0)
                generator_batch['ex_reward'] = torch.cat((generator_batch['ex_reward'], batch['reward'][reached].float().to(device=flags.device)), 0) 
                generator_batch['frame'] = torch.cat((generator_batch['frame'], batch['initial_frame'][reached].float().to(device=flags.device)), 0) 
                generator_batch['goal'] = torch.cat((generator_batch['goal'], batch['goal'][reached].to(device=flags.device)), 0)
                generator_batch['episode_step'] = torch.cat((generator_batch['episode_step'], batch['episode_step'][reached].float().to(device=flags.device)), 0)
                generator_batch['generator_logits'] = torch.cat((generator_batch['generator_logits'], batch['generator_logits'][reached].float().to(device=flags.device)), 0)
                generator_batch['reached'] = torch.cat((generator_batch['reached'], torch.ones(batch['goal'].shape)[reached].float().to(device=flags.device)), 0)

    
            if generator_batch['frame'].shape[0] >= flags.generator_batch_size: # Run Gradient step, keep batch residual in batch_aux
                for key in generator_batch:
                    generator_batch_aux[key] = generator_batch[key][flags.generator_batch_size:]
                    generator_batch[key] =  generator_batch[key][:flags.generator_batch_size].unsqueeze(0)

            
                generator_outputs = generator_model(generator_batch)
                generator_bootstrap_value = generator_outputs["generator_baseline"][-1]
                
                # Generator Reward
                def distance2(episode_step, reached, targ=flags.generator_target): 
                    aux = flags.generator_reward_negative * torch.ones(episode_step.shape).to(device=flags.device)
                    aux += (episode_step >= targ).float() * reached
                    return aux             

                if flags.generator_loss_form == 'gaussian':
                    generator_target = flags.generator_target * torch.ones(generator_batch['episode_step'].shape).to(device=flags.device)
                    gen_reward = Normal(generator_target, flags.target_variance*torch.ones(generator_target.shape).to(device=flags.device))
                    generator_rewards = flags.generator_reward_coef * (2 + gen_reward.log_prob(generator_batch['episode_step']) - gen_reward.log_prob(generator_target)) * generator_batch['reached'] -1
                
                elif flags.generator_loss_form == 'linear':
                    generator_rewards = (generator_batch['episode_step']/flags.generator_target * (generator_batch['episode_step'] <= flags.generator_target).float() + \
                    torch.exp ((-generator_batch['episode_step'] + flags.generator_target)/20.0) * (generator_batch['episode_step'] > flags.generator_target).float()) * \
                    2*generator_batch['reached'] - 1

                
                elif flags.generator_loss_form == 'dummy':
                    generator_rewards = torch.tensor(distance2(generator_batch['episode_step'], generator_batch['reached'])).to(device=flags.device)

                elif flags.generator_loss_form == 'threshold':
                    generator_rewards = torch.tensor(distance2(generator_batch['episode_step'], generator_batch['reached'], targ=generator_current_target)).to(device=flags.device)        

                if torch.mean(generator_rewards).item() >= flags.generator_threshold:
                    generator_count += 1
                else:
                    generator_count = 0    
                
                if generator_count >= flags.generator_counts and generator_current_target<=flags.generator_maximum:
                    generator_current_target += 1.0
                    generator_count = 0
                    goal_count_dict *= 0.0


                if flags.novelty:
                    frames_aux = torch.flatten(generator_batch['frame'], 2, 3)
                    frames_aux = frames_aux[:,:,:,0] 
                    object_ids =torch.zeros(generator_batch['goal'].shape).long()
                    for i in range(object_ids.shape[1]):
                        object_ids[0,i] = frames_aux[0,i,generator_batch['goal'][0,i]]
                        goal_count_dict[object_ids[0,i]] += 1 
                    
                    bonus = (object_ids>2).float().to(device=flags.device)  * flags.novelty_bonus
                    generator_rewards += bonus 


                if flags.reward_clipping == "abs_one":
                    generator_clipped_rewards = torch.clamp(generator_rewards, -1, 1)


                if not flags.no_extrinsic_rewards:
                    generator_clipped_rewards = 1.0 * (generator_batch['ex_reward'] > 0).float() + generator_clipped_rewards * (generator_batch['ex_reward'] <= 0).float()


                generator_discounts = torch.zeros(generator_batch['episode_step'].shape).float().to(device=flags.device)

                
                goals_aux = generator_batch["goal"]
                if flags.inner:
                    goals_aux = goals_aux.float()
                    goals_aux -= 2 * (torch.floor(goals_aux/generator_model.height))
                    goals_aux -= generator_model.height -1
                    goals_aux = goals_aux.long()

                

                generator_vtrace_returns = vtrace.from_logits(
                    behavior_policy_logits=generator_batch["generator_logits"],
                    target_policy_logits=generator_outputs["generator_logits"],
                    actions=goals_aux,
                    discounts=generator_discounts,
                    rewards=generator_clipped_rewards,
                    values=generator_outputs["generator_baseline"],
                    bootstrap_value=generator_bootstrap_value,
                )   


                # Generator Loss
                gg_loss = compute_policy_gradient_loss(
                    generator_outputs["generator_logits"],
                    goals_aux,
                    generator_vtrace_returns.pg_advantages,
                )


                generator_baseline_loss = flags.baseline_cost * compute_baseline_loss(
                    generator_vtrace_returns.vs - generator_outputs["generator_baseline"]
                )

                generator_entropy_loss = flags.generator_entropy_cost * compute_entropy_loss(
                    generator_outputs["generator_logits"]
                )


                generator_total_loss = gg_loss + generator_entropy_loss +generator_baseline_loss 


                intrinsic_rewards_gen = generator_batch['reached']*(1- 0.9 * (generator_batch["episode_step"].float()/max_steps))
                stats["gen_rewards"] = torch.mean(generator_clipped_rewards).item()  
                stats["gg_loss"] = gg_loss.item() 
                stats["generator_baseline_loss"] = generator_baseline_loss.item() 
                stats["generator_entropy_loss"] = generator_entropy_loss.item() 
                stats["mean_intrinsic_rewards"] = torch.mean(intrinsic_rewards_gen).item()
                stats["mean_episode_steps"] = torch.mean(generator_batch["episode_step"]).item()
                stats["ex_reward"] = torch.mean(generator_batch['ex_reward']).item()
                stats["generator_current_target"] = generator_current_target
                
                
        
                generator_scheduler.step()
                generator_model_optimizer.zero_grad() 
                generator_total_loss.backward()
                
                nn.utils.clip_grad_norm_(generator_model.parameters(), 40.0)
                generator_model_optimizer.step()
                actor_generator_model.load_state_dict(generator_model.state_dict())
                

                if generator_batch_aux['frame'].shape[0]>0:
                    generator_batch = {key: tensor[:] for key, tensor in generator_batch_aux.items()}
                else:
                    generator_batch = dict()
                
        return stats


def create_buffers(obs_shape, num_actions, flags, width, height, logits_size) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        last_action=dict(size=(T + 1,), dtype=torch.int64),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        generator_baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        episode_win=dict(size=(T + 1,), dtype=torch.int32),
        generator_logits=dict(size=(T + 1, logits_size), dtype=torch.float32),
        goal=dict(size=(T + 1,), dtype=torch.int64),
        initial_frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        carried_col =dict(size=(T + 1,), dtype=torch.int64),
        carried_obj =dict(size=(T + 1,), dtype=torch.int64),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


def train(flags):  
    """Full training loop."""
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
    plogger = file_writer.FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")

    T = flags.unroll_length
    B = flags.batch_size
    

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)
    
    #env = wrappers.FullyObsWrapper(env)
    if flags.num_input_frames > 1:
        env = FrameStack(env, flags.num_input_frames)
    
    generator_model = Generator(env.observation_space.shape, env.width, env.height, num_input_frames=flags.num_input_frames)

    model = Net(env.observation_space.shape, env.action_space.n, state_embedding_dim=flags.state_embedding_dim, num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm, num_lstm_layers=flags.num_lstm_layers)
    global goal_count_dict
    goal_count_dict = torch.zeros(11).float().to(device=flags.device)

    
    if flags.inner:
        logits_size = (env.width-2)*(env.height-2)
    else:  
        logits_size = env.width * env.height  

    buffers = create_buffers(env.observation_space.shape, model.num_actions, flags, env.width, env.height, logits_size)

    model.share_memory()
    generator_model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        for t in state:
            t.share_memory_()
        initial_agent_state_buffers.append(state)

    actor_processes = []
    ctx = mp.get_context("fork")
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(i, free_queue, full_queue, model, generator_model, buffers,
                 initial_agent_state_buffers, flags))
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(env.observation_space.shape, env.action_space.n, state_embedding_dim=flags.state_embedding_dim, num_input_frames=flags.num_input_frames, use_lstm=flags.use_lstm, num_lstm_layers=flags.num_lstm_layers).to(
        device=flags.device
    )
    learner_generator_model = Generator(env.observation_space.shape, env.width, env.height, num_input_frames=flags.num_input_frames).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    generator_model_optimizer = torch.optim.RMSprop(
        learner_generator_model.parameters(),
        lr=flags.generator_learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_frames) / flags.total_frames

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    generator_scheduler = torch.optim.lr_scheduler.LambdaLR(generator_model_optimizer, lr_lambda)

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "gen_rewards",  
        "gg_loss",
        "generator_entropy_loss",
        "generator_baseline_loss",
        "mean_intrinsic_rewards",
        "mean_episode_steps",
        "ex_reward",
        "generator_current_target",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    frames, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal frames, stats
        timings = prof.Timings()
        while frames < flags.total_frames:
            timings.reset()
            batch, agent_state = get_batch(flags, free_queue, full_queue, buffers,
                initial_agent_state_buffers, timings)
            stats = learn(model, learner_model, generator_model, learner_generator_model, batch, agent_state, optimizer, generator_model_optimizer, scheduler, generator_scheduler, flags, env.max_steps)

            timings.time("learn")
            with lock:
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * B
        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "generator_model_state_dict": generator_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "generator_model_optimizer_state_dict": generator_model_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "generator_scheduler_state_dict": generator_scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        
        while frames < flags.total_frames:
            start_frames = frames
            start_time = timer()
            time.sleep(5) 
            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            fps = (frames - start_frames) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "After %i frames: loss %f @ %.1f fps. %sStats:\n%s",
                frames,
                total_loss,
                fps,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d frames.", frames)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


class Generator(nn.Module):
    """Constructs the Teacher Policy which takes an initial observation and produces a goal."""
    def __init__(self, observation_shape, width, height, num_input_frames, hidden_dim=256):
        super(Generator, self).__init__()
        self.observation_shape = observation_shape
        self.height = height
        self.width = width
        self.env_dim = self.width * self.height
        self.state_embedding_dim = 256
        

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim) * num_input_frames

        if flags.disable_use_embedding:
            print("not_using_embedding")
            self.num_channels = 3*num_input_frames

        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)


        K = self.num_channels  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = 4  # number of convnet layers
        E = 1 # output of last layer 

        in_channels = [K] + [M] * 4
        out_channels = [M] * 3 + [E] 

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]




        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        self.out_dim = self.env_dim * 16 + self.obj_dim + self.col_dim


        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))
        
        if flags.inner:
            self.aux_env_dim = (self.height-2) * (self.width-2)
        else:
            self.aux_env_dim = self.env_dim    

        self.baseline_teacher = init_(nn.Linear(self.aux_env_dim, 1))

    def _select(self, embed, x):
        """Efficient function to get embedding from an index."""
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape +(-1,))
        else:
            return embed(x)  

    def create_embeddings(self, x, id):
        """Generates compositional embeddings."""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def convert_inner(self, goals):
        """Transform environment if using inner flag."""
        goals = goals.float()       
        goals += 2*(1+torch.floor(goals/(self.height-2)))  
        goals += self.height - 1 
        goals = goals.long()
        return goals


    def agent_loc(self, frames):
        """Returns the location of an agent from an observation."""
        T, B, height, width, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() # select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(T,B,1)        
        return agent_location 

    def forward(self, inputs):
        """Main Function, takes an observation and returns a goal."""
        x = inputs["frame"]
        T, B, *_ = x.shape
        carried_col = inputs["carried_col"]
        carried_obj = inputs["carried_obj"]


        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        if flags.disable_use_embedding:
            x = x.float() 
            carried_obj = carried_obj.float()
            carried_col = carried_col.float()
        else:    
            x = x.long()
            carried_obj = carried_obj.long()
            carried_col = carried_col.long()
            x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2)], dim = 3)
            carried_obj_emb = self._select(self.embed_object, carried_obj)
            carried_col_emb = self._select(self.embed_color, carried_col)
              
        x = x.transpose(1, 3)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1)

        x = self.extract_representation(x)
        x = x.view(T * B, -1)

        generator_logits = x.view(T*B, -1)

        generator_baseline = self.baseline_teacher(generator_logits)
        
        goal = torch.multinomial(F.softmax(generator_logits, dim=1), num_samples=1)

        generator_logits = generator_logits.view(T, B, -1)
        generator_baseline = generator_baseline.view(T, B)
        goal = goal.view(T, B)

        if flags.inner:
            goal = self.convert_inner(goal)

        return dict(goal=goal, generator_logits=generator_logits, generator_baseline=generator_baseline)




class MinigridNet(nn.Module):
    """Constructs the Student Policy which takes an observation and a goal and produces an action."""
    def __init__(self, observation_shape, num_actions, state_embedding_dim=256, num_input_frames=1, use_lstm=False, num_lstm_layers=1):
        super(MinigridNet, self).__init__()
        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.state_embedding_dim = state_embedding_dim
        self.use_lstm = use_lstm
        self.num_lstm_layers = num_lstm_layers

        self.use_index_select = True
        self.obj_dim = 5
        self.col_dim = 3
        self.con_dim = 2
        self.goal_dim = flags.goal_dim
        self.agent_loc_dim = 10
        self.num_channels = (self.obj_dim + self.col_dim + self.con_dim + 1) * num_input_frames
        
        if flags.disable_use_embedding:
            print("not_using_embedding")
            self.num_channels = (3+1+1+1+1)*num_input_frames

        
        self.embed_object = nn.Embedding(11, self.obj_dim)
        self.embed_color = nn.Embedding(6, self.col_dim)
        self.embed_contains = nn.Embedding(4, self.con_dim)
        self.embed_goal = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.goal_dim)
        self.embed_agent_loc = nn.Embedding(self.observation_shape[0]*self.observation_shape[1] + 1, self.agent_loc_dim)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0), nn.init.calculate_gain('relu'))

                  
        self.feat_extract = nn.Sequential(
            init_(nn.Conv2d(in_channels=self.num_channels, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            init_(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=2, padding=1)),
                nn.ELU(),
            )
        

        self.fc = nn.Sequential(
            init_(nn.Linear(32 + self.obj_dim + self.col_dim, self.state_embedding_dim)),
            nn.ReLU(),
            init_(nn.Linear(self.state_embedding_dim, self.state_embedding_dim)),
            nn.ReLU(),
        )


        if use_lstm:
            self.core = nn.LSTM(self.state_embedding_dim, self.state_embedding_dim, self.num_lstm_layers)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                            constant_(x, 0))

        self.policy = init_(nn.Linear(self.state_embedding_dim, self.num_actions))
        self.baseline = init_(nn.Linear(self.state_embedding_dim, 1))


    def initial_state(self, batch_size):
        """Initializes LSTM."""
        if not self.use_lstm:
            return tuple()
        return tuple(torch.zeros(self.core.num_layers, batch_size, self.core.hidden_size) for _ in range(2))
  

    def create_embeddings(self, x, id):
        """Generates compositional embeddings."""
        if id == 0:
            objects_emb = self._select(self.embed_object, x[:,:,:,id::3])
        elif id == 1:
            objects_emb = self._select(self.embed_color, x[:,:,:,id::3])
        elif id == 2:
            objects_emb = self._select(self.embed_contains, x[:,:,:,id::3])
        embeddings = torch.flatten(objects_emb, 3, 4)
        return embeddings

    def _select(self, embed, x):
        """Efficient function to get embedding from an index."""
        if self.use_index_select:
            out = embed.weight.index_select(0, x.reshape(-1))
            # handle reshaping x to 1-d and output back to N-d
            return out.reshape(x.shape +(-1,))
        else:
            return embed(x) 

    def agent_loc(self, frames):
        """Returns the location of an agent from an observation."""
        T, B, *_ = frames.shape
        agent_location = torch.flatten(frames, 2, 3)
        agent_location = agent_location[:,:,:,0] 
        agent_location = (agent_location == 10).nonzero() # select object id
        agent_location = agent_location[:,2]
        agent_location = agent_location.view(T,B,1)
        return agent_location    



    def forward(self, inputs, core_state=(), goal=[]):
        """Main Function, takes an observation and a goal and returns and action."""

        # -- [unroll_length x batch_size x height x width x channels]
        x = inputs["frame"]
        T, B, h, w, *_ = x.shape
       
        # -- [unroll_length*batch_size x height x width x channels]
        x = torch.flatten(x, 0, 1)  # Merge time and batch.
        goal = torch.flatten(goal, 0, 1)

        # Creating goal_channel
        goal_channel = torch.zeros_like(x, requires_grad=False)
        goal_channel = torch.flatten(goal_channel, 1,2)[:,:,0]
        for i in range(goal.shape[0]):
            goal_channel[i,goal[i]] = 1.0
        goal_channel = goal_channel.view(T*B, h, w, 1)
        carried_col = inputs["carried_col"]
        carried_obj = inputs["carried_obj"]

        if flags.disable_use_embedding:
            x = x.float()
            goal = goal.float()
            carried_obj = carried_obj.float()
            carried_col = carried_col.float()
        else:    
            x = x.long()
            goal = goal.long()
            carried_obj = carried_obj.long()
            carried_col = carried_col.long()
            # -- [B x H x W x K]
            x = torch.cat([self.create_embeddings(x, 0), self.create_embeddings(x, 1), self.create_embeddings(x, 2), goal_channel.float()], dim = 3)
            carried_obj_emb = self._select(self.embed_object, carried_obj)
            carried_col_emb = self._select(self.embed_color, carried_col)

        if flags.no_generator:
            goal_emb = torch.zeros(goal_emb.shape, dtype=goal_emb.dtype, device=goal_emb.device, requires_grad = False) 

        
        x = x.transpose(1, 3)
        x = self.feat_extract(x)
        x = x.view(T * B, -1)
        carried_obj_emb = carried_obj_emb.view(T * B, -1)
        carried_col_emb = carried_col_emb.view(T * B, -1) 
        union = torch.cat([x, carried_obj_emb, carried_col_emb], dim=1)
        core_input = self.fc(union)

        if self.use_lstm:
            core_input = core_input.view(T, B, -1)
            core_output_list = []
            notdone = (~inputs["done"]).float()
            for input, nd in zip(core_input.unbind(), notdone.unbind()):
                nd = nd.view(1, -1, 1)
                core_state = tuple(nd * s for s in core_state)
                output, core_state = self.core(input.unsqueeze(0), core_state)
                core_output_list.append(output)
            core_output = torch.flatten(torch.cat(core_output_list), 0, 1)
        else:
            core_output = core_input
            core_state = tuple()

        policy_logits = self.policy(core_output)
        baseline = self.baseline(core_output)
        
        if self.training:
            action = torch.multinomial(F.softmax(policy_logits, dim=1), num_samples=1)
        else:
            action = torch.argmax(policy_logits, dim=1)

        policy_logits = policy_logits.view(T, B, self.num_actions)
        baseline = baseline.view(T, B)
        action = action.view(T, B)

        return dict(policy_logits=policy_logits, baseline=baseline, action=action), core_state

 

Net = MinigridNet
GeneratorNet = Generator


class Minigrid2Image(gym.ObservationWrapper):
    """Get MiniGrid observation to ignore language instruction."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, observation):
        return observation["image"]


def create_env(flags):
    return Minigrid2Image(wrappers.FullyObsWrapper(gym.make(flags.env)))

def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
