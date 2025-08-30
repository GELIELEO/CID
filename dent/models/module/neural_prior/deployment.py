import os
import torch
from model import SpikeModel

model_path = '/home/ibasaw/causal_ws/WorldModel/disentangling-correlated-factors/dent/models/module/checkpoints'
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

num_units = 10
epoch = 654
step = 20305

model = SpikeModel(num_units).to(device)
state_dict = torch.load(os.path.join(model_path, f'epoch={epoch}-step={step}.ckpt'), map_location=device)['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace('model.', '')] = v
model.load_state_dict(new_state_dict)

print(model.flow.flows[-1].quantize_enabled)
# model.flow.flows[-1].quantize_enabled = False


if model.flow.flows[-1].quantize_enabled:
    fake_input = torch.ones(1, num_units)*22
else:
    fake_input = torch.rand(1, num_units)

ldj_0 = torch.zeros(fake_input.shape[0])
out_f, ldj_f = model.inverse_and_log_det(fake_input)
ldj_0 = torch.zeros(fake_input.shape[0])
out_r, ldj_r = model.forward_and_log_det(out_f)
print(fake_input, out_f, out_r, ldj_f, ldj_r)
sample = model.sample(1)
log_p = model.log_prob(sample[0])
print(sample, log_p)