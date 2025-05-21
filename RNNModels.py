import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

class SCATSDataset(Dataset):
    def __init__(self, csv_file, sequence_length=24):
        self.sequence_length = sequence_length
        self.prepare_data(csv_file)
    
    def prepare_data(self, csv_file):
        print("Preparing data...")
        df = pd.read_csv(csv_file)
        
        time_columns = [col for col in df.columns if ':' in col]  
        
        df_long = df.melt(
            id_vars=['SCATS Number', 'Location', 'NB_LATITUDE', 'NB_LONGITUDE', 'Date'],
            value_vars=time_columns,
            var_name='Time',
            value_name='Traffic_Count'
        )
        
        df_long['DateTime'] = pd.to_datetime(df_long['Date'] + ' ' + df_long['Time'], format='%d/%m/%Y %H:%M')
        
        df_long = df_long.sort_values(['SCATS Number', 'DateTime']).reset_index(drop=True)
        
        df_long['hour'] = df_long['DateTime'].dt.hour
        df_long['day_of_week'] = df_long['DateTime'].dt.dayofweek
        
        df_long['hour_sin'] = np.sin(2 * np.pi * df_long['hour'] / 24)
        df_long['hour_cos'] = np.cos(2 * np.pi * df_long['hour'] / 24)
        df_long['dow_sin'] = np.sin(2 * np.pi * df_long['day_of_week'] / 7)
        df_long['dow_cos'] = np.cos(2 * np.pi * df_long['day_of_week'] / 7)
        
        unique_locations = sorted(df_long['SCATS Number'].unique())
        self.location_to_id = {loc: idx for idx, loc in enumerate(unique_locations)}
        self.id_to_location = {idx: loc for idx, loc in enumerate(unique_locations)}
        df_long['location_encoded'] = df_long['SCATS Number'].map(self.location_to_id)
        self.num_locations = len(unique_locations)
        
        self.traffic_min = df_long['Traffic_Count'].min()
        self.traffic_max = df_long['Traffic_Count'].max()
        df_long['traffic_scaled'] = (df_long['Traffic_Count'] - self.traffic_min) / (self.traffic_max - self.traffic_min)
        
        self.sequences = []
        self.targets = []
        self.locations = []
        
        for location_id, group in df_long.groupby('location_encoded'):
            if len(group) > self.sequence_length:
                group = group.sort_values('DateTime').reset_index(drop=True)
                
                for i in range(len(group) - self.sequence_length):
                    time_features = group.iloc[i:i+self.sequence_length][
                        ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
                    ].values
                    
                    target = group.iloc[i+self.sequence_length]['traffic_scaled']
                    
                    self.sequences.append(time_features)
                    self.targets.append(target)
                    self.locations.append(location_id)
        
        self.sequences = torch.FloatTensor(np.array(self.sequences))
        self.targets = torch.FloatTensor(np.array(self.targets))
        self.locations = torch.LongTensor(np.array(self.locations))
    
    def encode_location(self, location):
        print("Encoding location...")   
        return self.location_to_id.get(location, -1)
    
    def denormalize_traffic(self, scaled_traffic):
        print("Denormalizing traffic...")
        return scaled_traffic * (self.traffic_max - self.traffic_min) + self.traffic_min
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.locations[idx], self.targets[idx]

class SCATSTrafficRNN(nn.Module):
    def __init__(self, 
                 time_features_size=4,  # Only hour_sin, hour_cos, dow_sin, dow_cos
                 num_locations=100,
                 embedding_dim=32,
                 hidden_size=128, 
                 num_layers=2, 
                 rnn_type='LSTM',
                 dropout=0.2):
        super(SCATSTrafficRNN, self).__init__()
    
        self.location_embedding = nn.Embedding(num_locations, embedding_dim)
        self.rnn_input_size = time_features_size + embedding_dim
        
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.rnn_input_size, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(self.rnn_input_size, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, time_features, locations, location_dropout_rate=0.1, is_training=True):
        batch_size, seq_len, _ = time_features.shape

        location_embed = self.location_embedding(locations)  

        if is_training and location_dropout_rate > 0:
            mask = (torch.rand(location_embed.size(0), device=location_embed.device) > location_dropout_rate).float().unsqueeze(1)
            location_embed = location_embed * mask 
        location_embed = location_embed.unsqueeze(1).repeat(1, seq_len, 1) 
        rnn_input = torch.cat([time_features, location_embed], dim=2)
        rnn_out, _ = self.rnn(rnn_input)
        last_output = rnn_out[:, -1, :]
        out = self.dropout(last_output)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out.squeeze(-1)


def train_model(model, train_loader, val_loader, num_epochs=2, lr=0.001, device='cpu'):
    print("Training model...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    model.to(device)
    
    for epoch in range(num_epochs):

        model.train()
        total_train_loss = 0
        
        for batch_time_features, batch_locations, batch_targets in train_loader:
            batch_time_features = batch_time_features.to(device)
            batch_locations = batch_locations.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_time_features, batch_locations, is_training=True)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch_time_features, batch_locations, batch_targets in val_loader:
                batch_time_features = batch_time_features.to(device)
                batch_locations = batch_locations.to(device)
                batch_targets = batch_targets.to(device)
                
                outputs = model(batch_time_features, batch_locations)
                loss = criterion(outputs, batch_targets)
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        scheduler.step(avg_val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    torch.save(model.state_dict(), 'scats_model_' + model.rnn_type + '.pt')

    return train_losses, val_losses

def predict_traffic(model, dataset, time_features_sequence, location, device='cpu'):
    print("Predicting traffic count...")
    model.eval()
    model.to(device)
    
    location_encoded = dataset.encode_location(location)
    
    if location_encoded == -1:
        raise ValueError(f"Unknown location: {location}")

    time_features_tensor = torch.FloatTensor(time_features_sequence).unsqueeze(0).to(device)
    location_tensor = torch.LongTensor([location_encoded]).to(device)
    
    with torch.no_grad():
        prediction = model(time_features_tensor, location_tensor)
        prediction_denormalized = dataset.denormalize_traffic(prediction.cpu().numpy()[0])
    
    return prediction_denormalized

def create_prediction_input(dataset, location, target_datetime, sequence_length=24):
    print("Creating prediction input...")

    time_features = []
    for i in range(sequence_length):
        dt = target_datetime - pd.Timedelta(minutes=15 * (sequence_length - 1 - i))
        
        hour = dt.hour
        day_of_week = dt.dayofweek
        
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)
        
        features = [hour_sin, hour_cos, dow_sin, dow_cos]
        time_features.append(features)
    
    return np.array(time_features)

def evaluate_model(x_val, y_val, model):
    model.eval()
    with torch.no_grad():
        y_pred = model(x_val).squeeze()
        y_true = y_val.squeeze()
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        print("\nSample Predictions vs Actual:")
        for i in range(10):
            print(f"Predicted: {y_pred_np[i]:.4f}, Actual: {y_true_np[i]:.4f}")

        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mse = mean_squared_error(y_true_np, y_pred_np)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = mse ** 0.5
        print(f"\nValidation RMSE: {rmse:.4f}")
        print(f"Validation MAE: {mae:.4f}")


if __name__ == "__main__":
    print("Starting SCATS Traffic Prediction...")
    csv_file = 'scats_data.csv' 
    model_type = 'GRU'
    sequence_length = 24
    dataset = SCATSDataset(csv_file, sequence_length=sequence_length)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SCATSTrafficRNN(
        num_locations=dataset.num_locations,
        rnn_type=model_type,
        time_features_size=4
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                         num_epochs=2, device=device)
    
    print("Training completed!")
    


    location = 970
    target_datetime = pd.Timestamp('2006-10-28 12:00')

    time_features = create_prediction_input(dataset, location, target_datetime)
    prediction = predict_traffic(model, dataset, time_features, location, device)
    print(f"Predicted traffic count: {prediction}")
