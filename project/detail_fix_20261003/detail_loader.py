"""Compile legacy formulas into an isolated context, without editing originals.

The compiler emits inspectable migrated Python sources and a change log. File
writes are redirected. Each module has local API bindings, never global patches.
"""
import ast,builtins,hashlib,importlib.util,json,types
from pathlib import Path
import runtime as rt

class Migrate(ast.NodeTransformer):
    def __init__(self):self.function='';self.changes=[]
    def note(self,node,kind):self.changes.append({'line':node.lineno,'kind':kind,'function':self.function or '<module>'})
    def visit_FunctionDef(self,node):
        old=self.function;self.function=node.name
        node=self.generic_visit(node);self.function=old;return node
    def visit_Import(self,node):
        result=[]
        for a in node.names:
            if a.name in {'cv2','numpy'}:
                self.note(node,'local_raster_binding')
                result.append(ast.ImportFrom(module='runtime',names=[ast.alias(name='cv' if a.name=='cv2' else 'np',asname=a.asname or a.name)],level=0))
            else:result.append(ast.Import(names=[a]))
        return result
    def visit_ImportFrom(self,node):
        if node.module=='PIL':
            self.note(node,'local_PIL_binding');node.module='runtime'
        return node
    def visit_Subscript(self,node):
        if ast.unparse(node.value)=='np.mgrid':
            self.note(node,'logical_sampling_grid')
            raw=self.function in {'styled','style','style_actor','posters','mouth_alpha'} or '.shape' in ast.unparse(node.slice)
            slices=node.slice.elts if isinstance(node.slice,ast.Tuple) else [node.slice]
            args=[]
            for sl in slices:
                if not isinstance(sl,ast.Slice):return self.generic_visit(node)
                args.append(ast.Call(func=ast.Name(id='slice',ctx=ast.Load()),args=[sl.lower or ast.Constant(None),sl.upper or ast.Constant(None),sl.step or ast.Constant(None)],keywords=[]))
            flag=ast.Call(func=ast.Attribute(ast.Name('rt',ast.Load()),'grid_is_screen',ast.Load()),args=[ast.Call(ast.Name('locals',ast.Load()),[],[])],keywords=[]) if raw else ast.Constant(True)
            return ast.copy_location(ast.Call(func=ast.Attribute(ast.Name('rt',ast.Load()),'grid',ast.Load()),args=[ast.Tuple(args,ast.Load())],keywords=[ast.keyword(arg='screen',value=flag)]),node)
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node);name=ast.unparse(node.func)
        if name=='next' and node.args and '.glob(' in ast.unparse(node.args[0]) and '*.mp4' in ast.unparse(node.args[0]):
            self.note(node,'explicit_source');return ast.copy_location(ast.Attribute(ast.Name('rt',ast.Load()),'SOURCE',ast.Load()),node)
        if name in {'np.zeros','np.ones','np.empty','np.full'} and node.args:
            sh=node.args[0]
            if isinstance(sh,ast.Tuple) and len(sh.elts)>=2:
                tiny=all(isinstance(x,ast.Constant) and isinstance(x.value,int) and x.value<100 for x in sh.elts[:2])
                native=self.function in {'styled','style','style_actor','posters','mouth_alpha','_floor_mask'}
                if not tiny and not native:
                    self.note(node,'dense_screen_allocation');node.args.insert(0,ast.Constant(name[3:]));node.func=ast.Attribute(ast.Name('np',ast.Load()),'screen',ast.Load())
        elif name=='np.indices':
            self.note(node,'logical_indices');node.func=ast.Attribute(ast.Name('rt',ast.Load()),'indices',ast.Load())
        elif name=='np.meshgrid':
            self.note(node,'logical_meshgrid');node.func=ast.Attribute(ast.Name('rt',ast.Load()),'meshgrid',ast.Load())
        elif name=='np.arange' and self.function=='actor':
            self.note(node,'logical_gradient_samples');node.func=ast.Attribute(ast.Name('rt',ast.Load()),'screen_arange',ast.Load())
        elif name=='cv2.resize' and len(node.args)>1:
            sz=ast.unparse(node.args[1])
            if isinstance(node.args[1],ast.Tuple) and all(isinstance(x,ast.Subscript) and '.shape' in ast.unparse(x) for x in node.args[1].elts):
                self.note(node,'native_resize_domain');node.keywords.append(ast.keyword(arg='output_space',value=ast.Constant('native')))
        elif name in {'cv2.warpAffine','cv2.warpPerspective'} and len(node.args)>2:
            if '.shape' in ast.unparse(node.args[2]):
                self.note(node,'native_registration_domain');node.keywords.append(ast.keyword(arg='output_space',value=ast.Constant('native')))
        elif name.endswith('.write_text'):
            self.note(node,'redirect_write');node.args.insert(0,node.func.value);node.func=ast.Attribute(ast.Name('rt',ast.Load()),'write_text',ast.Load())
        elif name.endswith('.read_text'):
            self.note(node,'record_config_dependency');node.args.insert(0,node.func.value);node.func=ast.Attribute(ast.Name('rt',ast.Load()),'read_text',ast.Load())
        elif name.endswith('.mkdir'):
            self.note(node,'redirect_mkdir');node.args.insert(0,node.func.value);node.func=ast.Attribute(ast.Name('rt',ast.Load()),'mkdir',ast.Load())
        elif name=='sys.path.insert':return ast.copy_location(ast.Constant(None),node)
        elif name=='importlib.util.spec_from_file_location':node.func=ast.Attribute(ast.Name('loader',ast.Load()),'spec',ast.Load())
        return node

class ModuleLoader:
    def __init__(self):self.cache={};self.logs=[]
    def load(self,path):
        path=Path(path).resolve();key=str(path)
        if key in self.cache:return self.cache[key]
        assert rt.ROOT in path.parents
        src=path.read_text(encoding='utf-8-sig');rt.record(path)
        from fixes import edit_source, install
        src=edit_source(path,src)
        t=Migrate();tree=t.visit(ast.parse(src));ast.fix_missing_locations(tree)
        module=types.ModuleType('migrated_'+hashlib.sha256(key.encode()).hexdigest()[:12]);module.__file__=key
        module.__dict__.update({'rt':rt,'loader':self,'__builtins__':dict(vars(builtins),__import__=self.importer(path))})
        self.cache[key]=module
        target=rt.WORK/'migrated'/path.relative_to(rt.ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(ast.unparse(tree),encoding='utf-8')
        self.logs.append({'source':path.relative_to(rt.ROOT).as_posix(),'sha256':hashlib.sha256(src.encode()).hexdigest(),'migrated':target.relative_to(rt.WORK).as_posix(),'changes':t.changes})
        exec(compile(tree,str(target),'exec'),module.__dict__)
        install(path,module)
        if path.name=='seedance_clip.py':
            original=module.render
            module.render=lambda *a,**kw:rt.cv.resize(original(*a,**kw),(960,540),interpolation=rt.cv.INTER_CUBIC)
        return module
    def importer(self,current):
        def imp(name,globals=None,locals=None,fromlist=(),level=0):
            if name in {'render','render_v2','render_v3','render_v4','seeded_masks','partwise_masks','renderer_v3'}:
                candidates=[current.parent/(name+'.py'),current.parent.parent/(name+'.py'),rt.ROOT/'analysis/segment_30_60/mask_lab'/(name+'.py')]
                if name=='render' and current.parent.name in {'opening_00_05','segment_05_21','segment_21_29'}:candidates[0],candidates[1]=candidates[1],candidates[0]
                p=next((p for p in candidates if p.is_file()),None)
                if p is not None:return self.load(p)
            return builtins.__import__(name,globals,locals,fromlist,level)
        return imp
    def spec(self,name,path):
        owner=self
        class LocalExec:
            def create_module(self,spec):return owner.load(path)
            def exec_module(self,module):pass
        return importlib.util.spec_from_loader(name,LocalExec())
    def save_log(self):
        dest=rt.WORK/'logs'/f'{rt.JOB}_migration.json';dest.parent.mkdir(exist_ok=True)
        dest.write_text(json.dumps(self.logs,ensure_ascii=False,indent=2),encoding='utf-8')
