import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GaussianNLLLoss(nn.Module):
    """
    Negative Log-Likelihood Loss for Gaussian Distribution.
    
    This loss function trains the model to output both the mean (μ) and 
    variance (σ²) of a Gaussian distribution, allowing the model to 
    express uncertainty in its predictions.
    
    The NLL for a Gaussian is:
        NLL = 0.5 * [log(σ²) + (y - μ)² / σ²] + 0.5 * log(2π)
    
    We minimize this to find the μ and σ² that best explain the data.
    
    Args:
        eps: Small value for numerical stability (default: 1e-6)
        reduction: Specifies the reduction to apply: 'none', 'mean', 'sum'
    """
    
    def __init__(self, eps: float = 1e-6, reduction: str = 'mean'):
        super(GaussianNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
    
    def forward(self, mu: torch.Tensor, variance: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the Negative Log-Likelihood loss.
        
        Args:
            mu: Predicted mean (batch_size, output_size)
            variance: Predicted variance σ² (batch_size, output_size) - must be positive
            target: Ground truth values (batch_size, output_size)
            
        Returns:
            NLL loss value
        """
        # Ensure variance is positive for numerical stability
        variance = variance + self.eps
        
        # Compute NLL: 0.5 * [log(σ²) + (y - μ)² / σ²]
        # We omit the constant term 0.5 * log(2π) as it doesn't affect optimization
        nll = 0.5 * (torch.log(variance) + (target - mu) ** 2 / variance)
        
        if self.reduction == 'mean':
            return nll.mean()
        elif self.reduction == 'sum':
            return nll.sum()
        else:  # 'none'
            return nll


class ProbabilisticLSTM(nn.Module):
    """
    Probabilistic LSTM that outputs parameters of a Gaussian distribution.
    
    Instead of predicting a single point value, this model outputs:
    - μ (mu): The mean of the predicted distribution
    - σ² (variance): The variance of the predicted distribution
    
    This allows the model to express uncertainty in its predictions.
    High variance indicates the model is uncertain about its prediction.
    
    Args:
        input_size: Number of input features
        hidden_size: Number of hidden units in LSTM
        output_size: Number of output features (predictions per sample)
        num_layers: Number of LSTM layers (default: 2)
        dropout: Dropout probability between LSTM layers (default: 0.2)
        variance_activation: Activation for variance output - 'softplus' or 'exp' (default: 'softplus')
        min_variance: Minimum variance value for numerical stability (default: 1e-4)
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        output_size: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
        variance_activation: str = 'softplus',
        min_variance: float = 1e-4
    ):
        super(ProbabilisticLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.variance_activation = variance_activation
        self.min_variance = min_variance
        
        # LSTM layer for sequence processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Separate output heads for mean and variance
        # Using separate heads allows each to specialize
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the probabilistic network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Tuple of (mu, variance):
                - mu: Mean predictions (batch_size, output_size)
                - variance: Variance predictions (batch_size, output_size), always positive
        """
        # Pass through LSTM
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the last hidden state
        # last_hidden shape: (batch_size, hidden_size)
        last_hidden = lstm_out[:, -1, :]
        
        # Predict mean (μ) - can be any real number
        mu = self.fc_mu(last_hidden)
        
        # Predict variance (σ²) - must be positive
        var_raw = self.fc_var(last_hidden)
        
        if self.variance_activation == 'softplus':
            # Softplus: log(1 + exp(x)) - smooth approximation to ReLU
            variance = F.softplus(var_raw) + self.min_variance
        else:  # 'exp'
            # Exponential: exp(x) - can lead to very large values
            variance = torch.exp(var_raw) + self.min_variance
        
        return mu, variance
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        confidence_level: float = 0.95
    ) -> dict[str, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            confidence_level: Confidence level for interval (default: 0.95 for 95% CI)
            
        Returns:
            Dictionary containing:
                - 'mean': Point predictions (μ)
                - 'variance': Predicted variance (σ²)
                - 'std': Standard deviation (σ)
                - 'lower': Lower bound of confidence interval
                - 'upper': Upper bound of confidence interval
        """
        self.eval()
        with torch.no_grad():
            mu, variance = self.forward(x)
            std = torch.sqrt(variance)
            
            # Z-score for confidence interval (e.g., 1.96 for 95% CI)
            # Using inverse of standard normal CDF
            z_score = torch.tensor(
                self._get_z_score(confidence_level),
                device=x.device, 
                dtype=x.dtype
            )
            
            lower = mu - z_score * std
            upper = mu + z_score * std
            
            return {
                'mean': mu,
                'variance': variance,
                'std': std,
                'lower': lower,
                'upper': upper
            }
    
    def sample(self, x: torch.Tensor, n_samples: int = 100) -> torch.Tensor:
        """
        Generate samples from the predicted distribution.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            n_samples: Number of samples to draw
            
        Returns:
            Samples tensor of shape (n_samples, batch_size, output_size)
        """
        self.eval()
        with torch.no_grad():
            mu, variance = self.forward(x)
            std = torch.sqrt(variance)
            
            # Create normal distribution and sample
            dist = torch.distributions.Normal(mu, std)
            samples = dist.sample((n_samples,))
            
            return samples
    
    @staticmethod
    def _get_z_score(confidence_level: float) -> float:
        """Get z-score for a given confidence level."""
        # Common confidence levels
        z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576
        }
        return z_scores.get(confidence_level, 1.960)


class HeteroscedasticLoss(nn.Module):
    """
    Heteroscedastic loss that learns input-dependent uncertainty.
    
    This is an alternative formulation that directly optimizes for
    calibrated uncertainty by treating the variance as a learnable
    function of the input.
    
    The loss combines:
    1. Reconstruction error weighted by uncertainty
    2. Regularization term to prevent trivially large uncertainties
    
    Args:
        eps: Small value for numerical stability
    """
    
    def __init__(self, eps: float = 1e-6):
        super(HeteroscedasticLoss, self).__init__()
        self.eps = eps
    
    def forward(
        self, 
        mu: torch.Tensor, 
        log_variance: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute heteroscedastic loss using log-variance for stability.
        
        Args:
            mu: Predicted mean
            log_variance: Log of predicted variance (more numerically stable)
            target: Ground truth values
            
        Returns:
            Loss value
        """
        # Compute precision (inverse variance)
        precision = torch.exp(-log_variance)
        
        # Weighted squared error + regularization
        loss = precision * (target - mu) ** 2 + log_variance
        
        return 0.5 * loss.mean()


def train_probabilistic_lstm(
    model: ProbabilisticLSTM,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader = None,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    early_stopping_patience: int = 10,
    verbose: bool = True
) -> dict:
    """
    Training function for Probabilistic LSTM with NLL loss.
    
    Args:
        model: ProbabilisticLSTM model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to train on ('cpu' or 'cuda')
        early_stopping_patience: Epochs to wait before early stopping
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing training history
    """
    model = model.to(device)
    criterion = GaussianNLLLoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=verbose
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_rmse': [],
        'val_rmse': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        n_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            mu, variance = model(batch_x)
            loss = criterion(mu, variance, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            train_mse += ((mu - batch_y) ** 2).sum().item()
            n_train += batch_x.size(0)
        
        train_loss /= n_train
        train_rmse = math.sqrt(train_mse / n_train)
        history['train_loss'].append(train_loss)
        history['train_rmse'].append(train_rmse)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            n_val = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    mu, variance = model(batch_x)
                    loss = criterion(mu, variance, batch_y)
                    
                    val_loss += loss.item() * batch_x.size(0)
                    val_mse += ((mu - batch_y) ** 2).sum().item()
                    n_val += batch_x.size(0)
            
            val_loss /= n_val
            val_rmse = math.sqrt(val_mse / n_val)
            history['val_loss'].append(val_loss)
            history['val_rmse'].append(val_rmse)
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
        else:
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


# Example usage and demonstration
if __name__ == "__main__":
    # Create synthetic data for demonstration
    torch.manual_seed(42)
    
    batch_size = 32
    seq_length = 50
    input_size = 14
    hidden_size = 64
    output_size = 1
    
    # Create model
    model = ProbabilisticLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=2,
        dropout=0.2
    )
    
    print("=" * 60)
    print("Probabilistic LSTM Model Architecture")
    print("=" * 60)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Forward pass
    mu, variance = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Mean (μ) shape: {mu.shape}")
    print(f"Variance (σ²) shape: {variance.shape}")
    
    # Test prediction with uncertainty
    predictions = model.predict_with_uncertainty(x, confidence_level=0.95)
    print(f"\n95% Confidence Interval:")
    print(f"  Lower bound shape: {predictions['lower'].shape}")
    print(f"  Upper bound shape: {predictions['upper'].shape}")
    
    # Test sampling
    samples = model.sample(x, n_samples=100)
    print(f"\nSampled predictions shape: {samples.shape}")
    
    # Demonstrate loss computation
    target = torch.randn(batch_size, output_size)
    criterion = GaussianNLLLoss()
    loss = criterion(mu, variance, target)
    print(f"\nNLL Loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("The model is ready for training on RUL prediction!")
    print("=" * 60)

