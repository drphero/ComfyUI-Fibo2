import sys, ast, importlib.util, types, inspect, argparse, hashlib, json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("--reference", type=Path, required=True, help="Directory containing the pinned transformer_bria_fibo2.py reference source")
parser.add_argument("--comfyui", type=Path, default=Path.cwd())
args=parser.parse_args()
package=Path(__file__).resolve().parents[1]
source=args.reference / "transformer_bria_fibo2.py"
assert hashlib.sha256(source.read_bytes()).hexdigest() == "c03cf2bc94ca6c5b09903e24cbb20334e871b743b75129738fde09ee2a8dac0a", "Use the pinned reference source in README.md"
import torch
from torch import nn
from torch.nn import functional as F
sys.argv=['test', '--use-pytorch-cross-attention', '--disable-comfy-compiler']
sys.path.insert(0,str(args.comfyui))
spec=importlib.util.spec_from_file_location('fibo2_test',str(package/'__init__.py'),submodule_search_locations=[str(package)])
pkg=importlib.util.module_from_spec(spec);sys.modules[spec.name]=pkg;spec.loader.exec_module(pkg)
from fibo2_test.model import Fibo2
import comfy.ops

def dispatch(q,k,v,attn_mask=None,**kwargs):
 return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=attn_mask).transpose(1,2)
tree=ast.parse(source.read_text())
for node in tree.body:
 if isinstance(node,ast.ClassDef):
  node.decorator_list=[]
  if node.bases:
   node.bases=[ast.Attribute(value=ast.Name(id='nn',ctx=ast.Load()),attr='Module',ctx=ast.Load())]
  for f in node.body:
   if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef)):
    f.decorator_list=[d for d in f.decorator_list if isinstance(d,ast.Name) and d.id=='staticmethod']
keep=[n for n in tree.body if not isinstance(n,(ast.ImportFrom,ast.Import))]
ns=dict(torch=torch,nn=nn,F=F,math=__import__('math'),RMSNorm=nn.RMSNorm,dispatch_attention_fn=dispatch,Transformer2DModelOutput=object)
exec(compile(ast.fix_missing_locations(ast.Module(body=keep,type_ignores=[])),'reference','exec'),ns)
# AttentionMixin normally installs this method; the math below is unchanged.
ns['BriaFibo2Attention'].set_processor=lambda self,p:setattr(self,'processor',p)
config=dict(in_channels=4,patch_size=2,dim=128,n_layers=3,n_refiner_layers=2,n_heads=4,cap_feat_dim=48,injection_layer_ids=(0,2),perceiver_num_layers=2,min_num_gist_tokens=4,max_num_gist_tokens=8,gist_step=2,gist_min_text_len=8,gist_max_text_len=32,axes_dims=(8,12,12))
reference=ns['BriaFibo2Transformer2DModel'](**config)
full={k:v.default for k,v in inspect.signature(ns['BriaFibo2Transformer2DModel'].__init__).parameters.items() if k!='self'}
reference.config=types.SimpleNamespace(**(full|config))
native=Fibo2(**config,operations=comfy.ops.disable_weight_init,dtype=torch.float32,device='cpu')
print(native.load_state_dict(reference.state_dict()),flush=True)
def pack(z):
 b,c,h,w=z.shape
 return z.view(b,c,h//2,2,w//2,2).permute(0,1,3,5,2,4).reshape(b,c*4,h//2,w//2)
torch.manual_seed(1729)
for batch,refs,masked in [(1,0,False),(1,1,False),(2,2,True)]:
 x=torch.randn(batch,4,8,12)
 text=torch.randn(batch,6,24,48)
 ts=torch.tensor([0.7]*batch)
 context=[torch.randn(batch,4,6,8),torch.randn(batch,4,4,10)][:refs]
 mask=None
 if masked:
  mask=torch.ones(batch,24,dtype=torch.bool);mask[0,5:]=False
 expected=reference(x,ts,text,encoder_attention_mask=mask,context_latents=context,return_dict=False)[0]
 actual=native(pack(x),ts,text.movedim(1,2).flatten(2),reference_latents=[pack(z) for z in context],attention_mask=mask)
 error=(actual-pack(expected)).abs()
 print(dict(batch=batch,refs=refs,masked=masked,max_error=error.max().item(),mean_error=error.mean().item()),flush=True)
 torch.testing.assert_close(actual,pack(expected),rtol=2e-5,atol=2e-5)
print('REFERENCE PARITY PASSED',flush=True)

from fibo2_test.nodes import Fibo2Scheduler, crop_reference
from fibo2_test.convert import local_file
from fibo2_test.prompt import Prompt
from fibo2_test.model import gist_count

expected=torch.tensor([1.0,0.9045307667386396,0.7595109169491111,0.5128441015091338,0.0])
actual=Fibo2Scheduler.execute({"samples":torch.empty(1,128,64,64)},4)[0]
torch.testing.assert_close(actual,expected,rtol=0,atol=0)
assert [gist_count(x) for x in [0,256,257,2048,10000]] == [256,256,320,512,512]
marker="<|vision_start|><|image_pad|><|vision_end|>"
assert Prompt._get_edit_prompt("Make it blue",1)==json.dumps({"image":marker,"edit_instruction":"Make it blue"},separators=(",",":"))
caption=json.dumps({"short_description":"a blue bicycle","objects":[],"aesthetics":{"mood":"calm","aesthetic_score":"high"},"edit_instruction":"Paint <image_1> like <image_2>"})
assert json.loads(Prompt._get_edit_prompt(caption,2))=={"image_1":marker,"image_2":marker,"structured_caption":{"short_description":"a blue bicycle","aesthetics":{"mood":"calm"}},"edit_instruction":"Paint <image_1> like <image_2>"}
try:
 local_file(package,"../../outside.safetensors")
except ValueError:
 pass
else:
 raise AssertionError("Loader allowed a model component outside its directory")
source_pixels=torch.ones(8,8,3)
mask=torch.zeros(8,8);mask[2:6,2:6]=1
crop=crop_reference(source_pixels,8,8,mask)
assert crop.getpixel((0,0))==(255,255,255)
assert crop.getpixel((3,3))==(128,128,128)
assert "diffusers" not in sys.modules
print("SCHEDULE, PROMPTS, MASK, PATH CONTAINMENT PASSED; no diffusers imported",flush=True)
