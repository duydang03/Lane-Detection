class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1.0 - dice.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        probs = probs.contiguous().view(probs.size(0), -1)
        targets = targets.contiguous().view(targets.size(0), -1)

        tp = (probs * targets).sum(dim=1)
        fp = ((1 - targets) * probs).sum(dim=1)
        fn = (targets * (1 - probs)).sum(dim=1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        return 1.0 - tversky.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LovaszLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def lovasz_grad(self, gt_sorted):

        p = len(gt_sorted)
        gts = gt_sorted.sum()
        if p == 0:
            return gt_sorted

        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union

        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]

        return jaccard

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        targets = targets.float()

        if probs.numel() == 0:
            return probs * 0.0

        probs = probs.view(-1)
        targets = targets.view(-1)

        errors = (targets - probs).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        targets_sorted = targets[perm]

        grad = self.lovasz_grad(targets_sorted)
        loss = torch.dot(errors_sorted, grad)

        return loss

class ComboLoss(nn.Module):
    def __init__(self, focal_weight=0.4, dice_weight=0.2, tversky_weight=0.2, lovasz_weight=0.2):
        super().__init__()
        self.focal = FocalLoss(alpha=0.25, gamma=2.0)
        self.dice = DiceLoss(smooth=1.0)
        self.tversky = TverskyLoss(alpha=0.5, beta=0.5)
        self.lovasz = LovaszLoss()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        tversky_loss = self.tversky(logits, targets)
        lovasz_loss = self.lovasz(logits, targets)

        total_loss = (self.focal_weight * focal_loss +
                     self.dice_weight * dice_loss +
                     self.tversky_weight * tversky_loss +
                     self.lovasz_weight * lovasz_loss)

        return total_loss
        
criterion = ComboLoss().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=7,factor=0.3,min_lr=1e-7,verbose=True)