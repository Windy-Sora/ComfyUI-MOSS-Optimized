import argparse,importlib.util,sys
from pathlib import Path
parser=argparse.ArgumentParser(description='Verify storage dtype stays separate from MOSS activation dtype')
parser.add_argument('comfy_root',type=Path)
parser.add_argument('--cuda',action='store_true')
args=parser.parse_args()
sys.path.insert(0,str(args.comfy_root.resolve()))
sys.argv=[sys.argv[0],'--cpu','--disable-xformers']
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.ops
spec=importlib.util.spec_from_file_location('moss_dtype_test',Path(__file__).resolve().parents[1]/'model.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_grad_enabled(False)
torch.manual_seed(42)
device='cuda' if args.cuda else 'cpu'
compute=torch.bfloat16 if args.cuda else torch.float32
x=torch.randn(2,128,3,2,device=device,dtype=compute)
context=torch.randn(2,4,4096,device=device,dtype=compute)
mask=torch.tensor([[1,1,0,0],[1,0,1,1]],device=device)
times=torch.tensor([500.,250.],device=device)

def make_model(dtype):
    return m.MOSSImageTransformer(dim=64,ffn_dim=128,num_heads=1,num_layers=1,dtype=dtype,
        operations=comfy.ops.manual_cast,device=device)

for storage in (torch.float32,torch.bfloat16,torch.float8_e4m3fn,torch.float8_e5m2):
    model=make_model(storage)
    reference=make_model(compute)
    for param in model.parameters():
        param.copy_((torch.randn(param.shape,device=device)*0.02).to(storage))
    reference.load_state_dict(model.state_dict())
    actual=model(x,times,context,attention_mask=mask)
    expected=reference(x,times,context,attention_mask=mask)
    assert actual.dtype==compute,(storage,actual.dtype,compute)
    assert actual.shape==x.shape and torch.isfinite(actual).all()
    torch.testing.assert_close(actual,expected,rtol=0.02 if args.cuda else 1e-5,atol=2e-4 if args.cuda else 1e-7)
    print('PASS:',device,storage,'storage ->',actual.dtype,'activations; reference values match')
