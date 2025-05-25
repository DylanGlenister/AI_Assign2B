import datetime
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import DataLoader, TensorDataset, random_split

import shared

# === Dataset ====================================================================
class SCATSDataset:
	def __init__(self, csv_file, sequence_length=24):
		self.sequence_length = sequence_length
		self.location_to_id = {}
		self.num_locations = 0
		self.training_data = None
		self.validation_data = None
		self.raw_series = {}
		self._prepare_data(csv_file)

	def _make_loc_str(self, scats, direction, day):
		return f'{scats}_{direction}_{int(day)}'

	def encode_location(self, scats, direction, day):
		key = self._make_loc_str(scats, direction, day)
		return self.location_to_id.get(key, -1)

	def _prepare_data(self, csv_file):
		print('Preparing data...')
		df = pd.read_csv(csv_file)

		scats_col, dir_col, day_col = df.columns[:3]
		time_cols = [c for c in df.columns if ':' in c]

		df['loc_str'] = df.apply(
			lambda r: self._make_loc_str(r[scats_col], r[dir_col], r[day_col]), axis=1
		)
		locs = df['loc_str'].unique().tolist()
		self.location_to_id = {loc: i for i, loc in enumerate(locs)}
		self.num_locations = len(locs)

		seq_len = self.sequence_length
		all_seqs = []
		all_locs = []
		all_tgts = []

		idxs = np.arange(seq_len) % 96
		sin_time = np.sin(2 * np.pi * idxs / 96)
		cos_time = np.cos(2 * np.pi * idxs / 96)

		for _, row in df.iterrows():
			series = row[time_cols].values.astype(np.float32)
			self.raw_series[row['loc_str']] = series
			if series.size < seq_len + 1:
				continue
			loc_id = self.location_to_id[row['loc_str']]
			dow = int(row[day_col])
			sin_dow = np.sin(2 * np.pi * dow / 7)
			cos_dow = np.cos(2 * np.pi * dow / 7)

			windows = sliding_window_view(series, window_shape=seq_len+1)
			inputs = windows[:, :seq_len]
			targets = windows[:, -1]

			n_win = inputs.shape[0]
			feats = np.empty((n_win, seq_len, 5), dtype=np.float32)
			feats[..., 0] = inputs
			feats[..., 1] = sin_time
			feats[..., 2] = cos_time
			feats[..., 3] = sin_dow
			feats[..., 4] = cos_dow

			all_seqs.append(feats)
			all_locs.extend([loc_id] * n_win)
			all_tgts.append(targets)

		X = torch.from_numpy(np.vstack(all_seqs))
		L = torch.tensor(all_locs, dtype=torch.long)
		y = torch.from_numpy(np.concatenate(all_tgts))
		full = TensorDataset(X, L, y)
		n = len(full)
		n_train = int(0.8 * n)
		self.training_data, self.validation_data = random_split(full, [n_train, n - n_train])

# === Model =====================================================================
class SCATSTrafficRNN(nn.Module):
	def __init__(self, num_locations, rnn_type='LSTM',
				 embedding_dim=32, hidden_size=128, num_layers=2, dropout=0.2):
		super().__init__()
		self.loc_embed = nn.Embedding(num_locations, embedding_dim)
		self.rnn_in = embedding_dim + 5
		if rnn_type == 'LSTM':
			self.rnn = nn.LSTM(self.rnn_in, hidden_size, num_layers,
								batch_first=True, dropout=dropout if num_layers > 1 else 0)
		elif rnn_type == 'GRU':
			self.rnn = nn.GRU(self.rnn_in, hidden_size, num_layers,
							   batch_first=True, dropout=dropout if num_layers > 1 else 0)
		else:
			print('not implemented')
			raise NotImplementedError(f'RNN type "{rnn_type}" is not supported')
			# TODO call third model class here

		self.drop = nn.Dropout(dropout)
		self.fc1 = nn.Linear(hidden_size, hidden_size//2)
		self.fc2 = nn.Linear(hidden_size//2, 1)
		self.act = nn.ReLU()

	def forward(self, x_time, x_loc, loc_dropout=0.1, is_training=True):
		B, T, _ = x_time.shape
		le = self.loc_embed(x_loc)
		if is_training and loc_dropout>0:
			mask = (torch.rand(B, device=le.device) > loc_dropout).float().unsqueeze(1)
			le = le * mask
		le = le.unsqueeze(1).expand(-1, T, -1)
		inp = torch.cat([x_time, le], dim=2)
		out, _ = self.rnn(inp)
		last = out[:, -1, :]
		h = self.drop(last)
		h = self.act(self.fc1(h))
		h = self.drop(h)
		return self.fc2(h).squeeze(-1)

class SCATSPredictor:
	'''This class contains the model and is used for saving, loading, training, and predicting.'''

	def __init__(self, dataset=shared.PATH_DATASET, type='LSTM'):
		MODEL_PATH = f'scats_{type}.pt'
		self.type = type
		self.SEQ_LEN = 24
		EPOCHS = 50
		BATCH = 64
		self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

		print(f'Using device: {self.DEVICE}')
		print('Creating dataset...')
		self.ds = SCATSDataset(dataset, sequence_length=self.SEQ_LEN)
		print('loading data...')
		print(f'Number of locations: {self.ds.num_locations}')

		train_ld = DataLoader(self.ds.training_data, batch_size=BATCH, shuffle=True)
		val_ld = DataLoader(self.ds.validation_data, batch_size=BATCH, shuffle=False)

		self.model = None

		if os.path.exists(MODEL_PATH):
			self.load(MODEL_PATH)
			print(f'Loaded model from {MODEL_PATH}')
		else:
			self.train_model(train_ld, val_ld, epochs=EPOCHS)
			self.save(MODEL_PATH)
			print(f'Training complete. Model saved to {MODEL_PATH}')

	# === Training ==================================================================
	def train_model(self, train_loader, val_loader, epochs=10, lr=1e-3):
		print('Training model...')

		self.model = SCATSTrafficRNN(num_locations=self.ds.num_locations, rnn_type=self.type)

		print(f'Using Model: {self.model.rnn.__class__.__name__}')
		print(f'starting training at {datetime.datetime.now()}')
		self.model.to(self.DEVICE)
		opt = optim.Adam(self.model.parameters(), lr=lr)
		sched = optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
		loss_fn = nn.MSELoss()

		for ep in range(1, epochs+1):
			self.model.train()
			tloss = 0.0
			for xt, xl, yt in train_loader:
				xt, xl, yt = xt.to(self.DEVICE), xl.to(self.DEVICE), yt.to(self.DEVICE)
				opt.zero_grad()
				yp = self.model(xt, xl, is_training=True)
				loss = loss_fn(yp, yt)
				loss.backward()
				nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
				opt.step()
				tloss += loss.item()
			vl = 0.0
			self.model.eval()
			with torch.no_grad():
				for xt, xl, yt in val_loader:
					xt, xl, yt = xt.to(self.DEVICE), xl.to(self.DEVICE), yt.to(self.DEVICE)
					yp = self.model(xt, xl, is_training=False)
					vl += loss_fn(yp, yt).item()
			print(f'Epoch {ep}/{epochs} — Train MSE: {tloss/len(train_loader):.4f}, Val MSE: {vl/len(val_loader):.4f}')
			print(f'completed epoch at {datetime.datetime.now()}')
			sched.step(vl/len(val_loader))
		print(f'Training complete at {datetime.datetime.now()}')
		# torch.save(model.state_dict(), f'scats_{model.rnn.__class__.__name__}.pt')

	# === Prediction Functions =======================================================
	def create_prediction_input(self, scats_ds, scats, direction, day, time_str):
		loc_str = f'{scats}_{direction}_{int(day)}'
		series = scats_ds.raw_series.get(loc_str, None)
		if series is None:
			raise ValueError(f'No data for {loc_str}')

		hour, minute = map(int, time_str.split(':'))
		idx = (hour*60 + minute) // 15

		start = idx - self.SEQ_LEN
		if start < 0:
			history = np.concatenate([np.zeros(-start, dtype=np.float32),
									series[0:idx].astype(np.float32)])
		else:
			history = series[start:idx].astype(np.float32)

		sin_t = np.sin(2*np.pi*idx/96)
		cos_t = np.cos(2*np.pi*idx/96)
		sin_d = np.sin(2*np.pi*day/7)
		cos_d = np.cos(2*np.pi*day/7)

		feats = np.zeros((self.SEQ_LEN, 5), dtype=np.float32)
		feats[:, 0] = history
		feats[:, 1] = sin_t
		feats[:, 2] = cos_t
		feats[:, 3] = sin_d
		feats[:, 4] = cos_d

		t = torch.tensor(np.array(feats), dtype=torch.float32)
		l = torch.tensor([scats_ds.encode_location(scats, direction, day)], dtype=torch.long)
		return t, l

	def predict_traffic(self, scats, direction, day, time_str):
		if self.model is None:
			raise ValueError('No model has been trained or loaded')

		self.model.to(self.DEVICE).eval()
		t, l = self.create_prediction_input(self.ds, scats, direction, day, time_str)
		t, l = t.unsqueeze(0).to(self.DEVICE), l.to(self.DEVICE)
		with torch.no_grad():
			return self.model(t, l, is_training=False).item()

	def load(self, path):
		self.model = SCATSTrafficRNN(num_locations=self.ds.num_locations, rnn_type=self.type)
		self.model.load_state_dict(torch.load(path, map_location=self.DEVICE))
		self.model.to(self.DEVICE)

	def save(self, path):
		if self.model is None:
			raise ValueError('No model has been trained or loaded')

		torch.save(self.model.state_dict(), path)

class PreLoadedPredictor:
	def __init__(self, model, day_of_week, time_of_day):
		self.day_of_week = day_of_week
		self.time_of_day = time_of_day
		self.predictor = SCATSPredictor(type=model)

	def query(self, scats, direction):
		return self.predictor.predict_traffic(scats, direction, self.day_of_week, self.time_of_day)

# === Main =====================================================================
if __name__ == '__main__':

	day_of_week = 2
	time_of_day = '05:45'

	predictor = PreLoadedPredictor('LSTM', day_of_week, time_of_day)

	# Example prediction
	scats_id = '3804'
	direction = 'W'
	pred = predictor.query(scats_id, direction)
	print('Predicted traffic:', pred)
