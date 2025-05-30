import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

def voting(frame_probs, frame_labels=None, strategy='weighted_avg'):
    """
    Perform voting to aggregate frame-level predictions to video-level
    Args:
        frame_probs: (frame_num, 2) - frame-level softmax probabilities
        frame_labels: (frame_num,) - frame-level labels (optional)
        strategy: 'prob_avg', 'majority_vote', 'weighted_avg', 'majority_frames_only'
    Returns:
        video_prob: tensor scalar - video-level prediction probability
        video_label: tensor scalar - video-level ground truth label (if frame_labels provided)
        majority_mask: tensor (only for majority_frames_only strategy)
        majority_probs: tensor (only for majority_frames_only strategy)
        majority_labels: tensor (only for majority_frames_only strategy)
    """
    if strategy == 'prob_avg':
        # Probability averaging - take mean of positive class probabilities
        video_prob = torch.mean(frame_probs[:, 1])
    elif strategy == 'majority_vote':
        # Hard voting - majority vote based on predicted labels (differentiable version)
        frame_preds = torch.argmax(frame_probs, dim=1)  # Get hard predictions
        # Use vote ratio as probability (more differentiable than hard 0/1)
        video_prob = torch.mean(frame_preds.float())
    elif strategy == 'weighted_avg':
        # Weighted averaging - give higher weights to more confident frames
        confidences = torch.max(frame_probs, dim=1).values
        weights = F.softmax(confidences, dim=0)
        video_prob = torch.sum(weights * frame_probs[:, 1])
    elif strategy == 'majority_frames_only':
        # Hard majority voting but return info about majority frames for selective loss
        frame_preds = torch.argmax(frame_probs, dim=1)  # [0, 1, 1, 0, 1, ...]
        votes_for_class_0 = torch.sum(frame_preds == 0).float()
        votes_for_class_1 = torch.sum(frame_preds == 1).float()
        
        # Determine winning class
        winning_class = 1 if votes_for_class_1 > votes_for_class_0 else 0
        video_prob = torch.tensor(float(winning_class), 
                                device=frame_probs.device, requires_grad=True)
        
        # Create mask for majority frames
        majority_mask = (frame_preds == winning_class)
        
        # Extract majority frame probabilities
        majority_probs = frame_probs[majority_mask, winning_class]  # Only winning class probs
        
        # Handle labels
        if frame_labels is not None:
            # Video-level label (majority vote on labels)
            label_votes_0 = torch.sum(frame_labels == 0).float()
            label_votes_1 = torch.sum(frame_labels == 1).float()
            video_label = torch.tensor(1.0 if label_votes_1 > label_votes_0 else 0.0,
                                     device=frame_labels.device, dtype=torch.float32)
            
            # Labels for majority frames
            majority_labels = frame_labels[majority_mask].float()
            
            return video_prob, video_label, majority_mask, majority_probs, majority_labels
        else:
            return video_prob, None, majority_mask, majority_probs, None
    else:
        raise ValueError(f"Unknown voting strategy: {strategy}")
    
    # Label voting for non-majority_frames_only strategies
    if frame_labels is not None:
        if strategy == 'majority_vote':
            # Hard majority vote for labels
            votes_1 = torch.sum(frame_labels == 1).float()
            votes_0 = torch.sum(frame_labels == 0).float()
            video_label = torch.tensor(1.0 if votes_1 > votes_0 else 0.0, 
                                     device=frame_labels.device, dtype=torch.float32)
        else:
            # Average voting for labels
            video_label = torch.round(torch.mean(frame_labels.float()))
            
        return video_prob, video_label
    else:
        return video_prob, None


class lossAV(nn.Module):
    def __init__(self, voting_strategy='weighted_avg'):
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
            if self.voting_strategy == 'majority_frames_only':
                video_prob, _, _, _, _ = voting(frame_probs, None, self.voting_strategy)
            else:
                video_prob, _ = voting(frame_probs, None, self.voting_strategy)
            predScore = video_prob.detach().cpu().numpy()
            return predScore
        else:
            # Training phase
            x1 = x / r
            frame_probs = F.softmax(x1, dim=-1)
            
            if self.voting_strategy == 'majority_frames_only':
                # Use majority frames only strategy
                voting_result = voting(frame_probs, labels, self.voting_strategy)
                video_prob, video_label, majority_mask, majority_probs, majority_labels = voting_result
                
                # Calculate loss only on majority frames
                if majority_probs.numel() > 0:  # Check if there are majority frames
                    # Clamp probabilities to avoid numerical issues
                    majority_probs_clamped = torch.clamp(majority_probs, min=1e-7, max=1.0-1e-7)
                    # Loss on majority frames only
                    nloss = self.criterion(majority_probs_clamped, majority_labels)
                else:
                    # Fallback: if no clear majority, use video-level loss
                    video_prob_clamped = torch.clamp(video_prob, min=1e-7, max=1.0-1e-7)
                    nloss = self.criterion(video_prob_clamped, video_label)
            else:
                # Use standard voting strategies
                video_prob, video_label = voting(frame_probs, labels, self.voting_strategy)
                
                # Ensure video_prob requires gradient and is in correct range
                video_prob = torch.clamp(video_prob, min=1e-7, max=1.0-1e-7)
                
                # Calculate video-level loss
                nloss = self.criterion(video_prob, video_label)
            
            # Prediction results for statistics (detach for non-gradient computations)
            with torch.no_grad():
                predScore = F.softmax(x, dim=-1)  # frame-level prediction probabilities
                predLabel = torch.round(video_prob.detach())  # video-level prediction label
                correctNum = (predLabel == video_label).float()  # whether prediction is correct
            
            return nloss, predScore, predLabel, correctNum


class lossV(nn.Module):
    def __init__(self, voting_strategy='weighted_avg'):
        super(lossV, self).__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)
        self.voting_strategy = voting_strategy

    def forward(self, x, labels, r=1):    
        x = x.squeeze(1)  # (frame_num, 128)
        x = self.FC(x)    # (frame_num, 2)
        
        x = x / r
        frame_probs = F.softmax(x, dim=-1)
        
        if self.voting_strategy == 'majority_frames_only':
            # Use majority frames only strategy
            voting_result = voting(frame_probs, labels, self.voting_strategy)
            video_prob, video_label, majority_mask, majority_probs, majority_labels = voting_result
            
            # Calculate loss only on majority frames
            if majority_probs.numel() > 0:
                majority_probs_clamped = torch.clamp(majority_probs, min=1e-7, max=1.0-1e-7)
                nloss = self.criterion(majority_probs_clamped, majority_labels)
            else:
                # Fallback
                video_prob_clamped = torch.clamp(video_prob, min=1e-7, max=1.0-1e-7)
                nloss = self.criterion(video_prob_clamped, video_label)
        else:
            # Use standard voting strategies
            video_prob, video_label = voting(frame_probs, labels, self.voting_strategy)
            
            # Ensure video_prob requires gradient and is in correct range
            video_prob = torch.clamp(video_prob, min=1e-7, max=1.0-1e-7)
            
            # Calculate video-level loss
            nloss = self.criterion(video_prob, video_label)
        
        return nloss
