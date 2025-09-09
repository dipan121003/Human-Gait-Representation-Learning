import math, sys
import torch
import utils as ut
from math import inf
import numpy as np
import time
import numpy as np

class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler('cuda')

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def train_one_epoch(model, data_loader, optimizer, device, epoch, 
                    loss_scaler, log_writer=None, config=None, start_time=None, model_without_ddp=None, 
                    img_feature_extractor=None, preprocess=None):
    model.train(True)
    optimizer.zero_grad()
    total_loss = []
    total_cor = []
    accum_iter = config.accum_iter
    batch = 0

    for data_iter_step, samples in enumerate(data_loader):

        if data_iter_step % accum_iter == 0:
            ut.adjust_learning_rate(optimizer, data_iter_step / config.steps_per_epoch + epoch, config)

        samples = samples.to(device) #[S, T, C]
        # print("✅ samples_shape: ", samples.shape)

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', enabled=True):
            '''print("✅ Checking sample before model forward")
            print("Sample shape:", samples.shape)
            print("NaNs:", torch.isnan(samples).any().item())
            print("Infs:", torch.isinf(samples).any().item())'''

            loss, pred, _ = model(samples, mask_ratio=config.mask_ratio)
            # print("✅ OUT OF MODEL")
            # print("✅ pred_shape: ", pred.shape)
            '''print("✅ Checking loss value")
            print("Loss:", loss.item() if not torch.isnan(loss) else "NaN ❌")'''

            '''if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Loss became NaN or Inf — skipping this batch")
                continue'''

        
        loss_value = loss.item()
        print("Epoch : ",epoch+1," step: ",data_iter_step+1,f" Loss : {loss_value:.4f}")
        batch += 1

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training at step {data_iter_step} epoch {epoch}")
            sys.exit(1)

        # pred = pred.to('cpu').detach()  # [B, L, patch_size*channels]
        # recon = model_without_ddp.mae.unpatchify(pred)  # [B, C, T]

        # Project input (original signal) before MAE
        # projected = model_without_ddp.project(samples.permute(0, 2, 1))  # [B, C, T]

        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=config.clip_grad)

        '''# Compute correlation channel-wise (first 3 channels for example)
        r = model_without_ddp.unpatchify(pred).cpu().detach()  # [B, 128, 128]
        o = model_without_ddp.project(samples.permute(0, 2, 1)).cpu().detach()  #[S, C, T]
        o = o.permute(0, 2, 1) #[S, T, C]

        cors = []
        for r_sample, o_sample in zip(r, o):  # loop through batch
            for ch in range(r_sample.shape[0]):  # loop through all channels
                corr = torch.corrcoef(torch.stack([r_sample[ch], o_sample[ch]]))[0, 1].item()
                cors.append(corr)
        cor = np.mean(cors)'''



        optimizer.zero_grad()

        total_loss.append(loss_value)
        # total_cor.append(cor)

        if device == torch.device('cuda:0'):
            lr = optimizer.param_groups[0]["lr"]
            print('train_loss_step:', np.mean(total_loss), 'lr:', lr, 'cor', np.mean(total_cor))

    if log_writer is not None:
        lr = optimizer.param_groups[0]["lr"]
        log_writer.log('train_loss_step', np.mean(total_loss), step=epoch)
        log_writer.log('lr', lr, step=epoch)
        log_writer.log('cor', np.mean(total_cor), step=epoch)
        if start_time is not None:
            log_writer.log('time (min)', (time.time() - start_time) / 60.0, step=epoch)

    if config.local_rank == 0:
        print(f'[Epoch {epoch+1}] loss: {np.mean(total_loss)}')

    return np.mean(total_loss)

@torch.no_grad()
def evaluate(model, data_loader, device, config):
    """
    Evaluate model on validation set.
    Returns average loss.
    """
    model.eval()
    total_loss = []
    
    for samples in data_loader:
        samples = samples.to(device)
        loss, _, _ = model(samples, mask_ratio=config.mask_ratio)
        total_loss.append(loss.item())

    avg_loss = float(np.mean(total_loss))
    print(f"[Val] Loss: {avg_loss:.4f}")
    
    return avg_loss

