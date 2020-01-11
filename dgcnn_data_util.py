import pickle
import os
import re
import collections
import numpy as np
import networkx as nx
import random
import math
import argparse

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='DGCNN', help='gnn model to use')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of discrete node feature (maximum node tag)')
cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge features')
cmd_opt.add_argument('-num_class', type=int, default=0, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-test_number', type=int, default=0,
                     help='if specified, will overwrite -fold and use the last -test_number graphs as testing data')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=str, default='64', help='dimension(s) of latent layers')
cmd_opt.add_argument('-sortpooling_k', type=float, default=30, help='number of nodes kept after SortPooling')
cmd_opt.add_argument('-conv1d_activation', type=str, default='ReLU', help='which nn activation layer to use')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='graph embedding output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of mlp hidden layer')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-dropout', type=bool, default=False, help='whether add dropout after dense layer')
cmd_opt.add_argument('-printAUC', type=bool, default=False,
                     help='whether to print AUC (for binary classification only)')
cmd_opt.add_argument('-extract_features', type=bool, default=False, help='whether to extract final graph features')

cmd_args, _ = cmd_opt.parse_known_args()
cmd_args.latent_dim = [int(x) for x in cmd_args.latent_dim.split('-')]
if len(cmd_args.latent_dim) == 1:
    cmd_args.latent_dim = cmd_args.latent_dim[0]

EMB_PATH = r'embeddings/emb.p'
DICT_PATH = r'embeddings/dic_pickle'
PARALLEL_DATA_FOLDER = r"data/Parallel_test/1/"
UNPARALLEL_DATA_FOLDER = r"data/Parallel_test/2/"
unknown_token = '!UNK'  # !开头确保排序结果
########################################################################################################################
# 所有stmt标签信息
########################################################################################################################
llvm_IR_stmt_families = [
    # ["tag level 1",                  "tag level 2",            "tag level 3",              "regex"                    ]
    ["unknown token", "unknown token", "unknown token", '!UNK'],
    ["integer arithmetic", "addition", "add integers", "<%ID> = add .*"],
    ["integer arithmetic", "subtraction", "subtract integers", "<%ID> = sub .*"],
    ["integer arithmetic", "multiplication", "multiply integers", "<%ID> = mul .*"],
    ["integer arithmetic", "division", "unsigned integer division", "<%ID> = udiv .*"],
    ["integer arithmetic", "division", "signed integer division", "<%ID> = sdiv .*"],
    ["integer arithmetic", "remainder", "remainder of signed div", "<%ID> = srem .*"],
    ["integer arithmetic", "remainder", "remainder of unsigned div.", "<%ID> = urem .*"],

    ["floating-point arithmetic", "addition", "add floats", "<%ID> = fadd .*"],
    ["floating-point arithmetic", "subtraction", "subtract floats", "<%ID> = fsub .*"],
    ["floating-point arithmetic", "multiplication", "multiply floats", "<%ID> = fmul .*"],
    ["floating-point arithmetic", "division", "divide floats", "<%ID> = fdiv .*"],

    ["bitwise arithmetic", "and", "and", "<%ID> = and .*"],
    ["bitwise arithmetic", "or", "or", "<%ID> = or .*"],
    ["bitwise arithmetic", "xor", "xor", "<%ID> = xor .*"],
    ["bitwise arithmetic", "shift left", "shift left", "<%ID> = shl .*"],
    ["bitwise arithmetic", "arithmetic shift right", "ashr", "<%ID> = ashr .*"],
    ["bitwise arithmetic", "logical shift right", "logical shift right", "<%ID> = lshr .*"],

    ["comparison operation", "compare integers", "compare integers", "<%ID> = icmp .*"],
    ["comparison operation", "compare floats", "compare floats", "<%ID> = fcmp .*"],
    ["conversion operation", "bitcast", "bitcast single val",
     '<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque) .* to .*'],
    ["conversion operation", "bitcast", "bitcast single val*",
     '<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\* .* to .*'],
    ["conversion operation", "bitcast", "bitcast single val**",
     '<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\* .* to .*'],
    ["conversion operation", "bitcast", "bitcast single val***",
     '<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\*\* .* to .*'],
    ["conversion operation", "bitcast", "bitcast single val****",
     '<%ID> = bitcast (i\d+|float|double|x86_fp80|opaque)\*\*\*\* .* to .*'],
    ["conversion operation", "bitcast", "bitcast array", '<%ID> = bitcast \[\d.* to .*'],
    ["conversion operation", "bitcast", "bitcast vector", '<%ID> = bitcast <\d.* to .*'],
    ["conversion operation", "bitcast", "bitcast structure", '<%ID> = bitcast (%"|<{|<%|{).* to .*'],
    ["conversion operation", "bitcast", "bitcast void", '<%ID> = bitcast void '],
    ["conversion operation", "extension/truncation", "extend float", "<%ID> = fpext .*"],
    ["conversion operation", "extension/truncation", "truncate floats", "<%ID> = fptrunc .*"],
    ["conversion operation", "extension/truncation", "sign extend ints", "<%ID> = sext .*"],
    ["conversion operation", "extension/truncation", "truncate int to ... ", "<%ID> = trunc .* to .*"],
    ["conversion operation", "extension/truncation", "zero extend integers", "<%ID> = zext .*"],
    ["conversion operation", "convert", "convert signed integers to... ", "<%ID> = sitofp .*"],
    ["conversion operation", "convert", "convert unsigned integer to... ", "<%ID> = uitofp .*"],
    ["conversion operation", "convert int to ptr", "convert int to ptr", "<%ID> = inttoptr .*"],
    ["conversion operation", "convert ptr to int", "convert ptr to int", "<%ID> = ptrtoint .*"],
    ["conversion operation", "convert floats", "convert float to sint", "<%ID> = fptosi .*"],
    ["conversion operation", "convert floats", "convert float to uint", "<%ID> = fptoui .*"],
    ["control flow", "phi", "phi", "<%ID> = phi .*"],
    ["control flow", "switch", "jump table line", "i\d{1,2} <(INT|FLOAT)>, label <%ID>"],
    ["control flow", "select", "select", "<%ID> = select .*"],
    ["control flow", "invoke", "invoke and ret type", "<%ID> = invoke .*"],
    ["control flow", "invoke", "invoke void", "invoke (fastcc )?void .*"],
    ["control flow", "branch", "branch conditional", "br i1 .*"],
    ["control flow", "branch", "branch unconditional", "br label .*"],
    ["control flow", "branch", "branch indirect", "indirectbr .*"],
    ["control flow", "control flow", "switch", "switch .*"],
    ["control flow", "return", "return", "ret .*"],
    ["control flow", "resume", "resume", "resume .*"],
    ["control flow", "unreachable", "unreachable", "unreachable.*"],
    ["control flow", "exception handling", "catch block", "catch .*"],
    ["control flow", "exception handling", "cleanup clause", "cleanup"],
    ["control flow", "exception handling", "landingpad for exceptions", "<%ID> = landingpad ."],
    ["function", "function call", "sqrt (llvm-intrinsic)",
     "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>) @(llvm|llvm\..*)\.sqrt.*"],
    ["function", "function call", "fabs (llvm-intr.)",
     "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.fabs.*"],
    ["function", "function call", "max (llvm-intr.)",
     "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.max.*"],
    ["function", "function call", "min (llvm-intr.)",
     "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.min.*"],
    ["function", "function call", "fma (llvm-intr.)",
     "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.fma.*"],
    ["function", "function call", "phadd (llvm-intr.)",
     "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.phadd.*"],
    ["function", "function call", "pabs (llvm-intr.)",
     "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.pabs.*"],
    ["function", "function call", "pmulu (llvm-intr.)",
     "<%ID> = (tail |musttail |notail )?call (fast |)?(i\d+|float|double|x86_fp80|<%ID>|<\d x float>|<\d x double>|<\d x i\d+>) @(llvm|llvm\..*)\.pmulu.*"],
    ["function", "function call", "umul (llvm-intr.)", "<%ID> = (tail |musttail |notail )?call {.*} @llvm\.umul.*"],
    ["function", "function call", "prefetch (llvm-intr.)", "(tail |musttail |notail )?call void @llvm\.prefetch.*"],
    ["function", "function call", "trap (llvm-intr.)", "(tail |musttail |notail )?call void @llvm\.trap.*"],
    ["function", "func decl / def", "function declaration", "declare .*"],
    ["function", "func decl / def", "function definition", "define .*"],
    ["function", "function call", "function call void",
     "(tail |musttail |notail )?call( \w+)? void [\w\)\(\}\{\.\,\*\d\[\]\s<>%]*(<[@%]ID>\(|.*bitcast )"],
    ["function", "function call", "function call mem lifetime",
     "(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.lifetime.*"],
    ["function", "function call", "function call mem copy",
     "(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.memcpy\..*"],
    ["function", "function call", "function call mem set",
     "(tail |musttail |notail )?call( \w+)? void ([\w)(\.\,\*\d ])*@llvm\.memset\..*"],
    ["function", "function call", "function call single val",
     '<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80|<\d+ x (i\d+|float|double)>) (.*<[@%]ID>\(|(\(.*\) )?bitcast ).*'],
    ["function", "function call", "function call single val*",
     '<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80)\* (.*<[@%]ID>\(|\(.*\) bitcast ).*'],
    ["function", "function call", "function call single val**",
     '<%ID> = (tail |musttail |notail )?call[^{]* (i\d+|float|double|x86_fp80)\*\* (.*<[@%]ID>\(|\(.*\) bitcast ).*'],
    ["function", "function call", "function call array",
     '<%ID> = (tail |musttail |notail )?call[^{]* \[.*\] (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )'],
    ["function", "function call", "function call array*",
     '<%ID> = (tail |musttail |notail )?call[^{]* \[.*\]\* (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )'],
    ["function", "function call", "function call array**",
     '<%ID> = (tail |musttail |notail )?call[^{]* \[.*\]\*\* (\(.*\) )?(<[@%]ID>\(|\(.*\) bitcast )'],
    ["function", "function call", "function call structure",
     '<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>) (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )'],
    ["function", "function call", "function call structure*",
     '<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )'],
    ["function", "function call", "function call structure**",
     '<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\*\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )'],
    ["function", "function call", "function call structure***",
     '<%ID> = (tail |musttail |notail )?call[^{]* (\{ .* \}[\w\_]*|<?\{ .* \}>?|opaque|\{\}|<%ID>)\*\*\* (\(.*\)\*? )?(<[@%]ID>\(|\(.*\) bitcast )'],
    ["function", "function call", "function call asm value", '<%ID> = (tail |musttail |notail )?call.* asm .*'],
    ["function", "function call", "function call asm void", '(tail |musttail |notail )?call void asm .*'],
    ["function", "function call", "function call function",
     '<%ID> = (tail |musttail |notail )?call[^{]* void \([^\(\)]*\)\** <[@%]ID>\('],
    ["global variables", "glob. var. definition", "???", "<@ID> = (?!.*constant)(?!.*alias).*"],
    ["global variables", "constant definition", "???", "<@ID> = .*constant .*"],
    ["memory access", "load from memory", "load structure", '<%ID> = load (\w* )?(%"|<\{|\{ <|\{ \[|\{ |<%|opaque).*'],
    ["memory access", "load from memory", "load single val", '<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)[, ].*'],
    ["memory access", "load from memory", "load single val*",
     '<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*[, ].*'],
    ["memory access", "load from memory", "load single val**",
     '<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*[, ].*'],
    ["memory access", "load from memory", "load single val***",
     '<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*[, ].*'],
    ["memory access", "load from memory", "load single val****",
     '<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*[, ].*'],
    ["memory access", "load from memory", "load single val*****",
     '<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*[, ].*'],
    ["memory access", "load from memory", "load single val******",
     '<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*\*[, ].*'],
    ["memory access", "load from memory", "load single val*******",
     '<%ID> = load (\w* )?(i\d+|float|double|x86_fp80)\*\*\*\*\*\*\*[, ].*'],
    ["memory access", "load from memory", "load vector", '<%ID> = load <\d+ x .*'],
    ["memory access", "load from memory", "load array", '<%ID> = load \[\d.*'],
    ["memory access", "load from memory", "load fction ptr", '<%ID> = load void \('],
    ["memory access", "store", "store", 'store.*'],
    ["memory addressing", "GEP", "GEP", "<%ID> = getelementptr .*"],
    ["memory allocation", "allocate on stack", "allocate structure", '<%ID> = alloca (%"|<{|<%|{ |opaque).*'],
    ["memory allocation", "allocate on stack", "allocate vector", "<%ID> = alloca <\d.*"],
    ["memory allocation", "allocate on stack", "allocate array", "<%ID> = alloca \[\d.*"],
    ["memory allocation", "allocate on stack", "allocate single value", "<%ID> = alloca (double|float|i\d{1,3})\*?.*"],
    ["memory allocation", "allocate on stack", "allocate void", "<%ID> = alloca void \(.*"],
    ["memory atomics", "atomic memory modify", "atomicrw xchg", "<%ID> = atomicrmw.* xchg .*"],
    ["memory atomics", "atomic memory modify", "atomicrw add", "<%ID> = atomicrmw.* add .*"],
    ["memory atomics", "atomic memory modify", "atomicrw sub", "<%ID> = atomicrmw.* sub .*"],
    ["memory atomics", "atomic memory modify", "atomicrw or", "<%ID> = atomicrmw.* or .*"],
    ["memory atomics", "atomic compare exchange", "cmpxchg single val",
     "<%ID> = cmpxchg (weak )?(i\d+|float|double|x86_fp80)\*"],
    ["non-instruction", "label", "label declaration", "; <label>:.*(\s+; preds = <LABEL>)?"],
    ["non-instruction", "label", "label declaration", "<LABEL>:( ; preds = <LABEL>)?"],
    ["value aggregation", "extract value", "extract value", "<%ID> = extractvalue .*"],
    ["value aggregation", "insert value", "insert value", "<%ID> = insertvalue .*"],
    ["vector operation", "insert element", "insert element", "<%ID> = insertelement .*"],
    ["vector operation", "extract element", "extract element", "<%ID> = extractelement .*"],
    ["vector operation", "shuffle vector", "shuffle vector", "<%ID> = shufflevector .*"]
]


########################################################################################################################
# 图类
########################################################################################################################
class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(node_tags)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())

        if len(g.edges()) != 0:
            x, y = zip(*g.edges())
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])

        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert (type(edge_features.values()[0]) == np.ndarray)
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in edge_features.items()}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)


########################################################################################################################
# 获取所有stmt标签,key=node_tag,value=idx
########################################################################################################################
def get_tag_dict():
    tag_dict = collections.OrderedDict()
    list_tags = list()
    for fam in llvm_IR_stmt_families[1:]:
        list_tags.append(fam[1])
    list_tags = sorted(set(list_tags))
    # 编号从1开始，！UNK为0
    for i in range(1, len(list_tags) + 1):
        tag_dict[list_tags[i - 1]] = i
    return tag_dict


########################################################################################################################
# 获取所有stmt标签正则,key=regex,value=node_tag
########################################################################################################################
def get_regex_dict():
    regex_dic = collections.OrderedDict()
    for fam in llvm_IR_stmt_families:
        regex_dic[fam[3]] = fam[1]
    return regex_dic


########################################################################################################################
# 读取向量和字典,key=...value=idx
########################################################################################################################
def get_embeddings_dict():
    with open(EMB_PATH, 'rb') as f:
        embedding_matrix = pickle.load(f)
    with open(DICT_PATH, 'rb') as f:
        stmt_dict = pickle.load(f)
    return embedding_matrix, stmt_dict


########################################################################################################################
#                       从文件加载一张图，加公图信息
########################################################################################################################
def load_graph(data_folder, graph_file_name, lable, regex_dict, tag_dict, embedding_matrix, stmt_dict):
    g = nx.Graph()
    node_dict = {}
    with open(os.path.join(data_folder, graph_file_name), 'rb') as f:
        XFG = pickle.load(f)
    # 給节点编号
    node_idx = 0
    for node in XFG.nodes():
        node_dict[node] = node_idx
        g.add_node(node_idx)
        node_idx += 1
    # 处理边信息,将边转换为数值对
    # G = GNNGraph(XFG, lable, node_tags, node_features)
    edge_pair_1, edge_pair_2 = zip(*XFG.edges())
    for x, y in zip(edge_pair_1, edge_pair_2):
        x = node_dict[x]
        y = node_dict[y]
        g.add_edge(x, y)
    # 处理节点信息
    # type(node)=str
    # 先通过遍历regx_dict,得到node_tag,然后根据node得到node_features
    # CFG = collections.namedtuple('CFG', 'graph lable node_tags node_features')

    node_tags = []
    node_features = []

    for node in XFG.nodes():
        # 处理node_features
        stmt = node.split('§')[0]
        if stmt_dict.__contains__(stmt):
            node_features.append(embedding_matrix[stmt_dict[stmt]])
        else:
            node_features.append(embedding_matrix[stmt_dict['!UNK']])
        # 处理node_tags
        flag = False
        for regx in regex_dict.keys():
            res = re.match(regx, node)
            if res:
                node_tags.append(tag_dict[regex_dict[regx]])
                flag = True
                break
        if not flag:
            node_tags.append(0)

    G = GNNGraph(g, lable, node_tags, np.stack(np.array(node_features)))
    return G


# e, s = get_embeddings_dict()
# G = load_graph(PARALLEL_DATA_FOLDER, 'test1.p', 1, get_regex_dict(), get_tag_dict(), e, s)
# print(G.node_tags)


########################################################################################################################
#                       从文件夹加载所有图并切分
########################################################################################################################
def load_data(regex_dict, tag_dict, embedding_matrix, stmt_dict):
    print('loading data')
    random.seed(100)
    parallel_g_list = []
    unparallel_g_list = []
    parallel_file_list = [f for f in os.listdir(PARALLEL_DATA_FOLDER)]
    unparallel_file_list = [f for f in os.listdir(UNPARALLEL_DATA_FOLDER)]
    for f1 in parallel_file_list:
        G1 = load_graph(PARALLEL_DATA_FOLDER, f1, 0, regex_dict, tag_dict, embedding_matrix, stmt_dict)
        parallel_g_list.append(G1)
    for f2 in unparallel_file_list:
        G2 = load_graph(UNPARALLEL_DATA_FOLDER, f2, 1, regex_dict, tag_dict, embedding_matrix, stmt_dict)
        unparallel_g_list.append(G2)
    # 切分数据7：3
    split_i = math.ceil(min(len(parallel_file_list), len(unparallel_file_list)) * 0.7)
    g_train = parallel_g_list[:split_i]
    g_train.extend(unparallel_g_list[:split_i])
    g_test = parallel_g_list[split_i:]
    g_test.extend(unparallel_g_list[split_i:])
    random.shuffle(g_train)
    random.shuffle(g_test)
    cmd_args.num_class = 2
    cmd_args.feat_dim = 46  # maximum node label (tag)
    cmd_args.edge_feat_dim = 0
    cmd_args.attr_dim = 200  # dim of node features (attributes)

    print('# classes: %d' % cmd_args.num_class)
    print('# maximum node tag: %d' % cmd_args.feat_dim)
    return g_train, g_test
