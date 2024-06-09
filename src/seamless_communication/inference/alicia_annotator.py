import torch.nn as nn
import torch.profiler as profiler

# from src.model.transformer import _InnerAttention

# exclude_types = [_InnerAttention]
exclude_types = []
def get_children_modules(model):
    children_modules = []
    for child in model.children():
        
        if any(isinstance(child,t) for t in exclude_types) and (type(child) == t for t in exclude_types):
            # children_modules.append(child)
            continue

        else:
            if len(list(child.children())) > 0:
                children_modules.extend(get_children_modules(child))
            else:
                children_modules.append(child)
            # if isinstance(child, nn.ModuleList) or isinstance(child, BasicTransformerBlock) or isinstance(child, FeedForward) or isinstance(child, Transformer2DModel):
            #     children_modules.extend(get_children_modules(child))
            # elif len(list(child.children())) > 0:
            #     children_modules.extend(get_children_modules(child))
            # else:
            #     children_modules.append(child)

    return children_modules

class ModelAnnotator:
    def __init__(self):
        self.ref = None
        self.name_counts = {}
    
    def annotate_modules(self, module, input):
        module_name = str(module)
        module_name = module_name[:module_name.find("(")]
        name = "MODULE_"+module_name+"_AG"

        if name in self.name_counts.keys():
            self.name_counts[name] = self.name_counts[name] + 1
        else:
            self.name_counts[name] = 1


        # with profiler.record_function(name):
        self.ref = profiler.record_function(name)
        self.ref.__enter__()
        return
    
    def stop_annotation(self, module, input, output):
        self.ref.__exit__(None, None, None)
        return

    def print_name_counts(self, model_name):
        print(model_name, " self.name_counts = ", self.name_counts)
        # print(model_name, " self.modules = ", self.name_counts.keys())

        return list(self.name_counts.keys())