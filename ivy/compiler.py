# global
import ivy
import queue
import inspect
import importlib

# local
from ivy.wrapper import _wrap_or_unwrap_methods, NON_WRAPPED_METHODS, ARRAYLESS_RET_METHODS

op_logging = False


ARRAY_BUILTINS = ['__neg__', '__pow__', '__rpow__', '__add__', '__radd__', '__iadd__', '__sub__', '__rsub__',
                  '__isub__', '__mul__', '__rmul__', '__imul__', '__truediv__', '__rtruediv__', '__itruediv__',
                  '__floordiv__', '__rfloordiv__', '__ifloordiv__', '__abs__', '__lt__', '__le__', '__eq__', '__ne__',
                  '__gt__', '__ge__', '__and__', '__rand__', '__or__', '__ror__', '__invert__', '__xor__', '__rxor__']

CLASSES_TO_WRAP = {'numpy': [],
                   'jax': [],
                   'tensorflow': [],
                   'torch': [('torch', 'Tensor')],
                   'mxnet': []}

class Param:

    def __init__(self):
        self._count = 0
        self._param_stack = list()

    def set(self, val):
        self._param_stack = [val]*self._count

    def set_count(self, count):
        self._count = count

    def get(self):
        return self._param_stack.pop()

    def __repr__(self):
        return '<Param, count={}, current={}>'.format(self._count, len(self._param_stack))

    def __len__(self):
        return len(self._param_stack)

    @property
    def count(self):
        return self._count


class Graph:

    # noinspection PyProtectedMember
    def __init__(self, fn, *args, num_workers=1, **kwargs):

        # function being compiled into a graph
        self._fn = fn

        # positional args
        self._args = args
        self._arg_nest_idxs = ivy.nested_indices_where(args, lambda x: ivy.is_array(x))
        self._arg_param_ids = [id(x) for x in ivy.multi_index_nest(list(args), self._arg_nest_idxs)]

        # key-word args
        self._kwargs = kwargs
        self._kwarg_nest_idxs = ivy.nested_indices_where(kwargs, lambda x: ivy.is_array(x))
        self._kwarg_param_ids = [id(x) for x in ivy.multi_index_nest(kwargs, self._kwarg_nest_idxs)]

        # output idxs
        self._output_param_ids = list()

        # graph storage
        self._param_dict = dict()
        self._functions_dict = dict()
        self._functions = list()

        # multiprocessing
        self._timeout = ivy.queue_timeout()
        self._num_workers = num_workers

    # Multiprocessing #
    # ----------------#

    def _initialize_multiprocessing(self):

        # prevent redundant workers by limiting to graph width
        self._num_workers = min(self._num_workers, self._max_graph_width)

        if self._num_workers <= 1:
            return

        # multiprocessing module
        multiprocessing = ivy.multiprocessing('fork')

        # initialize multi dict
        self._param_dict_multi = multiprocessing.Manager().dict(self._param_dict)

        # create workers
        self._input_queues = list()
        self._output_queues = list()
        self._workers = list()
        for i in range(self._num_workers):
            input_queue = multiprocessing.Queue()
            output_queue = multiprocessing.Queue()
            worker = multiprocessing.Process(
                target=self._worker_fn, args=(
                    input_queue, output_queue, ivy.default_device(), self._functions[i::self._num_workers],
                    ivy.current_framework_str()))
            worker.start()
            self._input_queues.append(input_queue)
            self._output_queues.append(output_queue)
            self._workers.append(worker)

    def _worker_fn(self, input_queue, output_queue, dev_str, functions, framework_str):
        ivy.set_framework(framework_str)
        ivy.set_default_device(dev_str)
        while True:
            try:
                input_queue.get(timeout=self._timeout)
            except queue.Empty:
                continue
            for fn in functions:
                arg_vals = [self.get_param_multi(pid) for pid in fn.arg_param_ids]
                kwarg_vals = [self.get_param_multi(pid) for pid in fn.kwarg_param_ids]
                ret = fn(arg_vals, kwarg_vals)
                [self.set_param_multi(pid, ivy.index_nest(ret, idx))
                 for pid, idx in zip(fn.output_param_ids, fn.output_nest_idxs)]
            output_queue.put(True)

    # Foward with Op Logging #
    # -----------------------#

    # noinspection PyProtectedMember
    def log_all_ops(self):

        global op_logging
        op_logging = True

        ret = self._fn(*self._args, **self._kwargs)
        if not isinstance(ret, tuple):
            ret = (ret,)
        output_nest_idxs = ivy.nested_indices_where(ret, lambda x: ivy.is_array(x))
        self._output_param_ids = [id(x) for x in ivy.multi_index_nest(list(ret), output_nest_idxs)]

        op_logging = False

    # Getters and Setters #
    # --------------------#

    # inference

    def get_param(self, pid):
        return self._param_dict[pid].get()

    def set_param(self, pid, value):
        self._param_dict[pid].set(value)

    # multiprocessing inference

    def get_param_multi(self, pid):
        # ToDo: make this more efficient
        while True:
            try:
                return self._param_dict_multi[pid].get()
            except IndexError:
                pass

    def set_param_multi(self, pid, value):
        # ToDo: make this more efficient
        param = self._param_dict_multi[pid]
        param.set(value)
        self._param_dict_multi[pid] = param

    # compiling

    def add_param(self, pid):
        self._param_dict[pid] = Param()

    def increment_param_count(self, pid):
        self._param_dict[pid].set_count(self._param_dict[pid].count + 1)

    def add_fn_to_dict(self, pid, fn):
        self._functions_dict[pid] = fn

    def get_param_recursive(self, pid, depth, receiving_fn=None):
        if pid in self._param_dict:
            return
        if pid in self._functions_dict:
            fn = self._functions_dict[pid]
        else:
            if not ivy.exists(receiving_fn):
                raise Exception('passing created tensors directly to output function is not yet supported.')
            if pid in receiving_fn.arg_param_ids:
                idx = receiving_fn.arg_param_ids.index(pid)
                del receiving_fn.arg_param_ids[idx]
                del receiving_fn.arg_array_idxs[idx]
            if pid in receiving_fn.kwarg_param_ids:
                idx = receiving_fn.kwarg_param_ids.index(pid)
                del receiving_fn.kwarg_param_ids[idx]
                del receiving_fn.kwarg_array_idxs[idx]
            return
        fn.tree_depth = depth
        self._functions.append(fn)
        [self.get_param_recursive(pid, depth + 1, fn) for pid in fn.arg_param_ids]
        [self.get_param_recursive(pid, depth + 1, fn) for pid in fn.kwarg_param_ids]
        [self.increment_param_count(pid) for pid in fn.arg_param_ids + fn.kwarg_param_ids]
        [self.add_param(pid) for pid in fn.output_param_ids]
        return

    # debugging

    def params_all_empty(self):
        return min([len(param) == 0 for param in self._param_dict.values()]) is True

    # Function creation #
    # ------------------#

    def _chain_functions(self):

        # add input params to param dict
        [self.add_param(pid) for pid in self._arg_param_ids]
        [self.add_param(pid) for pid in self._kwarg_param_ids]

        # recursively chain the graph via backward traversal
        [self.get_param_recursive(pid, depth=0) for pid in self._output_param_ids]
        [self.increment_param_count(pid) for pid in self._output_param_ids]

        # function for storing function heights
        def store_fn_heights(fn):
            heights_in = [store_fn_heights(fn_in) for fn_in in fn.fns_in if fn_in in self._functions]
            if heights_in:
                _height = max(heights_in) + 1
            else:
                _height = 0
            fn.tree_height = _height
            return _height

        # store function heights
        [store_fn_heights(self._functions_dict[pid]) for pid in self._output_param_ids]

        # find the height of the tree
        max_tree_height = max([fn.tree_height for fn in self._functions])

        # group the functions based on their height in the tree from the starting leaf nodes
        grouped_functions = list()
        for height in range(0, max_tree_height+1):
            fns = [fn for fn in self._functions if fn.tree_height == height]
            if height == 0:
                fns = sorted(fns, key=lambda x: len(x.fns_in))
            else:
                fns_hm1 = grouped_functions[-1]
                # noinspection PyUnresolvedReferences
                leftmost_idxs =\
                    [max(enumerate([fn in fn_hm1.fns_out for fn_hm1 in fns_hm1]), key=lambda x: x[1])[0] for fn in fns]
                fns = [fn for fn, _ in sorted(zip(fns, leftmost_idxs), key=lambda x: x[1])]
            grouped_functions.append(fns)

        # stack functions in the best order
        self._functions = [i for sl in grouped_functions for i in sl]

        # compute maximum width of the graph
        self._max_graph_width = max([len(fns) for fns in grouped_functions])

    def _call(self, *args, **kwargs):
        # ToDo: make this as efficient as possible; this is performed at runtime
        [self.set_param(pid, ivy.index_nest(args, idx))
         for pid, idx in zip(self._arg_param_ids, self._arg_nest_idxs)]
        [self.set_param(pid, ivy.index_nest(kwargs, idx))
         for pid, idx in zip(self._kwarg_param_ids, self._kwarg_nest_idxs)]
        for fn in self._functions:
            arg_vals = [self.get_param(pid) for pid in fn.arg_param_ids]
            kwarg_vals = [self.get_param(pid) for pid in fn.kwarg_param_ids]
            ret = fn(arg_vals, kwarg_vals)
            [self.set_param(pid, ivy.index_nest(ret, idx))
             for pid, idx in zip(fn.output_param_ids, fn.output_nest_idxs)]
        ret = [self.get_param(pid) for pid in self._output_param_ids]
        if len(ret) == 1:
            return ret[0]
        return ret

    def _multi_call(self, *args, **kwargs):
        # ToDo: make this as efficient as possible; this is performed at runtime
        [self.set_param_multi(pid, ivy.index_nest(args, idx))
         for pid, idx in zip(self._arg_param_ids, self._arg_nest_idxs)]
        [self.set_param_multi(pid, ivy.index_nest(kwargs, idx))
         for pid, idx in zip(self._kwarg_param_ids, self._kwarg_nest_idxs)]
        [q.put(True) for q in self._input_queues]
        [q.get(timeout=None) for q in self._output_queues]
        ret = [self.get_param_multi(pid) for pid in self._output_param_ids]
        if len(ret) == 1:
            return ret[0]
        return ret

    def compiled(self):
        self._chain_functions()
        self._initialize_multiprocessing()
        if self._num_workers <= 1:
            return self._call
        return self._multi_call

    def __del__(self):
        if self._num_workers <= 1:
            return
        # noinspection PyBroadException
        try:
            for i, w in enumerate(self._workers):
                self._input_queues[i].put(None)
                w.join(timeout=0.25)
            for q in self._input_queues:
                q.cancel_join_thread()
                q.close()
            for q in self._output_queues:
                q.cancel_join_thread()
                q.close()
        except Exception:
            pass
        finally:
            for w in self._workers:
                if w.is_alive():
                    w.terminate()

    # Clearing #
    # ---------#

    def clear(self):
        self._param_dict.clear()
        self._functions.clear()


# Methods #

def _wrap_method_for_compiling(fn, graph):

    if inspect.isclass(fn) or (hasattr(fn, '__name__') and
            ((fn.__name__[0] == '_' and fn.__name__ not in ARRAY_BUILTINS) or
             fn.__name__ in NON_WRAPPED_METHODS + ARRAYLESS_RET_METHODS)) or\
            (hasattr(fn, 'wrapped_for_compiling') and fn.wrapped_for_compiling):
        return fn

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def _method_wrapped(*args, **kwargs):

        # immutable tuple to mutable list
        args = list(args)

        # get array idxs for positional args
        arg_array_idxs = ivy.nested_indices_where(args, lambda x: ivy.is_array(x))
        args_cont = ivy.Container(args, types_to_iteratively_nest=(list, tuple))
        arg_param_ids_cont = args_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()
        arg_param_ids = list(arg_param_ids_cont.to_iterator_values())

        # get array idxs for key-word args
        kwarg_array_idxs = ivy.nested_indices_where(kwargs, lambda x: ivy.is_array(x))
        kwargs_cont = ivy.Container(kwargs, types_to_iteratively_nest=(list, tuple))
        kwarg_param_ids_cont = kwargs_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()
        kwarg_param_ids = list(kwarg_param_ids_cont.to_iterator_values())

        # compute the return
        ret_raw = fn(*args, **kwargs)
        if fn.__name__[0:3] == '__i':
            # clone the return values if the function is in-place, to ensure output ids are unique from input ids
            ret_raw = tuple([ivy.array(ivy.to_numpy(x)) if ivy.is_array(x) else x for x in ret_raw]) \
                if isinstance(ret_raw, tuple) else ivy.array(ivy.to_numpy(ret_raw))
        ret = ret_raw if isinstance(ret_raw, tuple) else (ret_raw,)

        # get array idxs for return
        ret_nest_idxs = ivy.nested_indices_where(ret, lambda x: ivy.is_array(x))
        ret_cont = ivy.Container(ret, types_to_iteratively_nest=(list, tuple))
        ret_params_ids_cont = ret_cont.map(lambda x, kc: id(x) if ivy.is_array(x) else None).prune_empty()
        ret_param_ids = list(ret_params_ids_cont.to_iterator_values())

        # wrap the function
        def new_fn(arg_array_vals, kwarg_array_vals):
            # ToDo: make this as efficient as possible; this is performed at runtime
            ivy.set_nest_at_indices(args, arg_array_idxs, arg_array_vals)
            ivy.set_nest_at_indices(kwargs, kwarg_array_idxs, kwarg_array_vals)
            ret_ = fn(*args, **kwargs)
            if not isinstance(ret_, tuple):
                ret_ = (ret_,)
            return ret_

        # add function attributes which inform about the input idxs
        new_fn.arg_param_ids = arg_param_ids
        new_fn.kwarg_param_ids = kwarg_param_ids
        new_fn.output_param_ids = ret_param_ids
        new_fn.output_nest_idxs = ret_nest_idxs
        new_fn.arg_array_idxs = arg_array_idxs
        new_fn.kwarg_array_idxs = kwarg_array_idxs
        fns_in = [graph._functions_dict[pid]
                  for pid in arg_param_ids + kwarg_param_ids if pid in graph._functions_dict]
        for fn_in in fns_in:
            if not hasattr(fn_in, 'fns_out'):
                fn_in.fns_out = list()
            if new_fn not in fn_in.fns_out:
                fn_in.fns_out.append(new_fn)

        new_fn.fns_in = fns_in

        if hasattr(fn, '__name__'):
            new_fn.__name__ = fn.__name__

        # add to graph if compiling
        if op_logging:

            # add this function to the graph for each output pid
            [graph.add_fn_to_dict(pid, new_fn) for pid in ret_param_ids]

        # return the function output
        return ret_raw

    if hasattr(fn, '__name__'):
        _method_wrapped.__name__ = fn.__name__
    _method_wrapped.wrapped_for_compiling = True
    _method_wrapped.inner_fn = fn
    return _method_wrapped


def _unwrap_method_from_compiling(method_wrapped):
    if not hasattr(method_wrapped, 'wrapped_for_compiling') or not method_wrapped.wrapped_for_compiling:
        return method_wrapped
    return method_wrapped.inner_fn


def _wrap_methods_for_op_logging(graph):
    classes_to_wrap = [getattr(importlib.import_module(ctw[0]), ctw[1])
                       for ctw in CLASSES_TO_WRAP[ivy.current_framework_str()]]
    return _wrap_or_unwrap_methods(
        lambda fn: _wrap_method_for_compiling(fn, graph), classes_to_wrap=classes_to_wrap, native=True)


def _unwrap_methods_from_op_logging():
    classes_to_wrap = [getattr(importlib.import_module(ctw[0]), ctw[1])
                       for ctw in CLASSES_TO_WRAP[ivy.current_framework_str()]]
    return _wrap_or_unwrap_methods(
        lambda fn: _unwrap_method_from_compiling(fn), classes_to_wrap=classes_to_wrap, native=True)


def compile_ivy(fn, *args, num_workers=1, **kwargs):
    graph = Graph(fn, *args, **kwargs, num_workers=num_workers)
    _wrap_methods_for_op_logging(graph)
    graph.log_all_ops()
    _unwrap_methods_from_op_logging()
    return graph.compiled()
