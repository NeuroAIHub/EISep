import collections
import numpy as np
import torch



Batch = collections.namedtuple(
	'Batch',
	['state', 'action', 'next_state', 'reward', 'done', 'task', 'next_task']
	)

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda:0'):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.task = np.zeros((max_size, 1))
		self.next_task = np.zeros((max_size, 1))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = device


	def add(self, state, action, next_state, reward, done, task, next_task):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done
		self.task[self.ptr] = task
		self.next_task[self.ptr] = next_task

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def add_batch(self, state, action, next_state, reward, done, task, next_task):
		batch_size = state.shape[0]
		if self.ptr + batch_size < self.max_size:
			self.state[self.ptr:self.ptr+batch_size] = state
			self.action[self.ptr:self.ptr+batch_size] = action
			self.next_state[self.ptr:self.ptr+batch_size] = next_state
			self.reward[self.ptr:self.ptr+batch_size] = reward
			self.done[self.ptr:self.ptr+batch_size] = done
			self.task[self.ptr:self.ptr+batch_size] = task
			self.next_task[self.ptr:self.ptr+batch_size] = next_task
			self.ptr += batch_size
			self.size = min(self.size + batch_size, self.max_size)
		elif self.ptr + batch_size == self.max_size:
			self.state[self.ptr:] = state
			self.action[self.ptr:] = action
			self.next_state[self.ptr:] = next_state
			self.reward[self.ptr:] = reward
			self.done[self.ptr:] = done
			self.task[self.ptr:] = task
			self.next_task[self.ptr:] = next_task
			self.ptr = 0
			self.size = self.max_size
		else:
			overflow = self.ptr + batch_size - self.max_size
			self.state[self.ptr:] = state[:-overflow]
			self.action[self.ptr:] = action[:-overflow]
			self.next_state[self.ptr:] = next_state[:-overflow]
			self.reward[self.ptr:] = reward[:-overflow]
			self.done[self.ptr:] = done[:-overflow]
			self.task[self.ptr:] = task[:-overflow]
			self.next_task[self.ptr:] = next_task[:-overflow]

			self.state[:overflow] = state[-overflow:]
			self.action[:overflow] = action[-overflow:]
			self.next_state[:overflow] = next_state[-overflow:]
			self.reward[:overflow] = reward[-overflow:]
			self.done[:overflow] = done[-overflow:]
			self.task[:overflow] = task[-overflow:]
			self.next_task[:overflow] = next_task[-overflow:]
			self.ptr = overflow
			self.size = self.max_size


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return Batch(
			state=torch.FloatTensor(self.state[ind]).to(self.device),
			action=torch.FloatTensor(self.action[ind]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[ind]).to(self.device),
			reward=torch.FloatTensor(self.reward[ind]).to(self.device),
			done=torch.FloatTensor(self.done[ind]).to(self.device),
			task=torch.FloatTensor(self.task[ind]).to(self.device),
			next_task=torch.FloatTensor(self.next_task[ind]).to(self.device)
		)
	def take(self, index):
		if isinstance(index, int):
			index = np.array([index])
		if np.max(index) >= self.size:
			raise ValueError('Index out of range')
		return Batch(
			state=torch.FloatTensor(self.state[index]).to(self.device),
			action=torch.FloatTensor(self.action[index]).to(self.device),
			next_state=torch.FloatTensor(self.next_state[index]).to(self.device),
			reward=torch.FloatTensor(self.reward[index]).to(self.device),
			done=torch.FloatTensor(self.done[index]).to(self.device),
			task=torch.FloatTensor(self.task[index]).to(self.device),
			next_task=torch.FloatTensor(self.next_task[index]).to(self.device)
		)
	
	def state_dict(self):
		return {
			'state': self.state[:self.size],
			'action': self.action[:self.size],
			'next_state': self.next_state[:self.size],
			'reward': self.reward[:self.size],
			'done': self.done[:self.size],
			'task': self.task[:self.size],
			'next_task': self.next_task[:self.size],
			'ptr': self.ptr,
			'size': self.size
		}
	def load_state_dict(self, state_dict):
		self.state = state_dict['state']
		self.action = state_dict['action']
		self.next_state = state_dict['next_state']
		self.reward = state_dict['reward']
		if 'task' in state_dict:
			self.task = state_dict['task']
			self.next_task = state_dict['next_task']
		self.done = state_dict['done']
		self.ptr = state_dict['ptr']
		self.size = state_dict['size']



