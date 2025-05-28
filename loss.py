import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

def max_voting(frame_probs, frame_labels=None, strategy='prob_avg'):
    """
    Perform max voting to aggregate frame-level predictions to video-level
    Args:
        frame_probs: (frame_num, 2) - frame-level softmax probabilities
        frame_labels: (frame_num,) - frame-level labels (optional)
        strategy: 'prob_avg', 'majority_vote', 'weighted_avg'
    Returns:
        video_prob: scalar - video-level prediction probability
        video_label: scalar - video-level ground truth label (if frame_labels provided)
    """
    if strategy == 'prob_avg':
        # Probability averaging - take mean of positive class probabilities
        video_prob = torch.mean(frame_probs[:, 1])
    elif strategy == 'majority_vote':
        # Hard voting - majority vote based on predicted labels
        frame_preds = torch.argmax(frame_probs, dim=1)
        video_pred = torch.mode(frame_preds).values.float()
        video_prob = video_pred
    elif strategy == 'weighted_avg':
        # Weighted averaging - give higher weights to more confident frames
        confidences = torch.max(frame_probs, dim=1).values
        weights = F.softmax(confidences, dim=0)
        video_prob = torch.sum(weights * frame_probs[:, 1])
    else:
        raise ValueError(f"Unknown voting strategy: {strategy}")
    
    # Label voting
    if frame_labels is not None:
        video_label = torch.mode(frame_labels).values.float()
        # Alternative: video_label = torch.round(torch.mean(frame_labels.float()))
        return video_prob, video_label
    else:
        return video_prob, None


class lossAV(nn.Module):
    def __init__(self, voting_strategy='prob_avg'):
        super(lossAV, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)
        self.voting_strategy = voting_strategy
        
    def forward(self, x, labels=None, r=1):    
        x = x.squeeze(1)  # (frame_num, 128)
        x = self.FC(x)    # (frame_num, 2)
        
        if labels == None:
            # Inference phase
            frame_probs = F.softmax(x, dim=-1)
            video_prob, _ = max_voting(frame_probs, None, self.voting_strategy)
            predScore = video_prob.detach().cpu().numpy()
            return predScore
        else:
            # Training phase
            x1 = x / r
            frame_probs = F.softmax(x1, dim=-1)
            
            # Perform max voting to get video-level prediction and label
            video_prob, video_label = max_voting(frame_probs, labels, self.voting_strategy)
            
            # Calculate video-level loss
            nloss = self.criterion(video_prob, video_label)
            
            # Prediction results for statistics
            predScore = F.softmax(x, dim=-1)  # frame-level prediction probabilities
            predLabel = torch.round(video_prob)  # video-level prediction label
            correctNum = (predLabel == video_label).float()  # whether prediction is correct
            
            return nloss, predScore, predLabel, correctNum


class lossV(nn.Module):
    def __init__(self, voting_strategy='prob_avg'):
        super(lossV, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)
        self.voting_strategy = voting_strategy

    def forward(self, x, labels, r=1):    
        x = x.squeeze(1)  # (frame_num, 128)
        x = self.FC(x)    # (frame_num, 2)
        
        x = x / r
        frame_probs = F.softmax(x, dim=-1)
        
        # Perform max voting
        video_prob, video_label = max_voting(frame_probs, labels, self.voting_strategy)
        
        # Calculate video-level loss
        nloss = self.criterion(video_prob, video_label)
        
        return nloss