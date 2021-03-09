import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from contextlib import contextmanager

from einops import reduce

# helpers

def exists(val):
    return val is not None

@contextmanager
def null_context():
    yield

def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]

# function wrapper for determinism on backwards

class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng = False, set_rng = False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)

# reversible self attention block

class ReversibleSelfAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)        

    def forward(self, x, m, mask = None, msa_mask = None, seq_shape = None, msa_shape = None, _reverse = True):
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        with context():
            y1 = x1 + self.f(x2, shape = seq_shape, record_rng = record_rng, mask = mask)
            y2 = x2 + self.g(y1, shape = seq_shape, record_rng = record_rng)
            n1 = m1 + self.j(m2, shape = msa_shape, record_rng = record_rng, mask = msa_mask)
            n2 = m2 + self.k(n1, record_rng = record_rng)

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)

    def backward_pass(self, y, n, dy, dn, mask = None, msa_mask = None, seq_shape = None, msa_shape = None):
        y1, y2 = torch.chunk(y, 2, dim = 2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        del dy

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, shape = seq_shape, set_rng = True)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, shape = seq_shape, set_rng = True, mask = mask)
            torch.autograd.backward(fx2, dx1, retain_graph = True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim = 2)
            dx = torch.cat([dx1, dx2], dim = 2)

        n1, n2 = torch.chunk(n, 2, dim = 2)
        del n

        dn1, dn2 = torch.chunk(dn, 2, dim = 2)
        del dn

        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.k(n1, set_rng = True)
            torch.autograd.backward(gn1, dn2)

        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1

            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None

        with torch.enable_grad():
            m2.requires_grad = True
            fm2 = self.j(m2, shape = msa_shape, set_rng = True, mask = msa_mask)
            torch.autograd.backward(fm2, dm1, retain_graph=True)

        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2

            dm2 = dn2 + m2.grad
            del dn2
            m2.grad = None

            m = torch.cat([m1, m2.detach()], dim = 2)
            dm = torch.cat([dm1, dm2], dim = 2)

        return x, m, dx, dm

# reversible cross attention block

class ReversibleCrossAttnBlock(nn.Module):
    def __init__(self, f, g, j, k):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)
        self.j = Deterministic(j)
        self.k = Deterministic(k)

    def forward(self, x, m, mask = None, msa_mask = None, seq_shape = None, msa_shape = None, _reverse = True, **kwargs):
        x1, x2 = torch.chunk(x, 2, dim = 2)
        m1, m2 = torch.chunk(m, 2, dim = 2)
        y1, y2, n1, n2 = None, None, None, None

        context = torch.no_grad if _reverse else null_context
        record_rng = self.training and _reverse

        with context():
            y1 = x1 + self.f(x2, m2, record_rng = record_rng, mask = mask, context_mask = msa_mask, shape = seq_shape, context_shape = msa_shape)
            y2 = x2 + self.g(y1, record_rng = record_rng)
            n1 = m1 + self.j(m2, y2, record_rng = record_rng, mask = msa_mask, context_mask = mask, shape = msa_shape, context_shape = seq_shape)
            n2 = m2 + self.k(n1, record_rng = record_rng)

        return torch.cat((y1, y2), dim = 2), torch.cat((n1, n2), dim = 2)

    def backward_pass(self, y, n, dy, dn, mask = None, msa_mask = None, seq_shape = None, msa_shape = None, **kwargs):
        n1, n2 = torch.chunk(n, 2, dim = 2)
        del n

        dn1, dn2 = torch.chunk(dn, 2, dim = 2)
        del dn

        y1, y2 = torch.chunk(y, 2, dim = 2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim = 2)
        del dy

        with torch.enable_grad():
            n1.requires_grad = True
            gn1 = self.k(n1, set_rng = True)
            torch.autograd.backward(gn1, dn2)

        with torch.no_grad():
            m2 = n2 - gn1
            del n2, gn1

            dm1 = dn1 + n1.grad
            del dn1
            n1.grad = None

        with torch.enable_grad():
            m2.requires_grad = True
            y2.requires_grad = True
            fm2 = self.j(m2, y2, set_rng=True, mask = msa_mask, context_mask = mask, shape = msa_shape, context_shape = seq_shape)
            torch.autograd.backward(fm2, dm1)

        with torch.no_grad():
            m1 = n1 - fm2
            del n1, fm2

            dm2 = dn2 + m2.grad
            dx2 = dy2 + y2.grad
            del dn2
            del dy2
            m2.grad = None
            y2.grad = None

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng = True)
            torch.autograd.backward(gy1, dx2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            m2.requires_grad = True
            fx2 = self.f(x2, m2, set_rng = True, mask = mask, context_mask = msa_mask, shape = seq_shape, context_shape = msa_shape)
            torch.autograd.backward(fx2, dx1)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dx2 + x2.grad
            dm2 = dm2 + m2.grad
            x2.grad = None
            m2.grad = None

        with torch.no_grad():
            m = torch.cat([m1, m2.detach()], dim = 2)
            dm = torch.cat([dm1, dm2], dim = 2)

            x = torch.cat([x1, x2.detach()], dim = 2)
            dx = torch.cat([dx1, dx2], dim = 2)

        return x, m, dx, dm

# reverse and non reverse functions

class ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, inp, ind, blocks, kwargs):
        x, m = split_at_index(1, ind, inp)

        for block in blocks:
            x, m = block(x, m, _reverse = True, **kwargs)

        ctx.blocks = blocks
        ctx.kwargs = kwargs
        ctx.ind = ind
        ctx.save_for_backward(x.detach(), m.detach())
        return torch.cat((x, m), dim = 1)

    @staticmethod
    def backward(ctx, d):
        ind = ctx.ind
        blocks = ctx.blocks
        kwargs = ctx.kwargs
        dy, dn = split_at_index(1, ind, d)
        y, n = ctx.saved_tensors

        for block in blocks[::-1]:
            y, n, dy, dn = block.backward_pass(y, n, dy, dn, **kwargs)

        d = torch.cat((dy, dn), dim = 1)
        return d, None, None, None

reversible_apply = ReversibleFunction.apply

def irreversible_apply(inputs, ind, blocks, kwargs):
    x, m = split_at_index(1, ind, inputs)
    for block in blocks:
        x, m = block(x, m, _reverse = False, **kwargs)
    return torch.cat((x, m), dim = 1)

# main reversible sequence class

class ReversibleSequence(nn.Module):
    def __init__(self, input_blocks):
        super().__init__()
        blocks = nn.ModuleList([])

        for self_attn_block, cross_attn_block in zip(*[iter(input_blocks)] * 2):
            blocks.append(ReversibleSelfAttnBlock(*self_attn_block))
            blocks.append(ReversibleCrossAttnBlock(*cross_attn_block))

        self.blocks = blocks

    def forward(self, seq, msa, seq_shape = None, msa_shape = None, mask = None, msa_mask = None, reverse = True):
        assert exists(msa), 'reversibility does not work with no MSA sequences yet'
        
        blocks = self.blocks
        seq, msa = list(map(lambda t: torch.cat((t, t), dim = -1), (seq, msa)))
        kwargs = {'mask': mask, 'msa_mask': msa_mask, 'seq_shape': seq_shape, 'msa_shape': msa_shape}

        fn = reversible_apply if reverse else irreversible_apply
        ind = seq.shape[1]
        inp = torch.cat((seq, msa), dim = 1)
        out = fn(inp, ind, blocks, kwargs)
        seq, msa  = split_at_index(1, ind, out)
        return list(map(lambda t: reduce(t, 'b n (c d) -> b n d', 'mean', c = 2), (seq, msa)))
