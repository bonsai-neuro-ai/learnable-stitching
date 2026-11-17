from nn_lib.models import get_pretrained_model, GraphModulePlus


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print model metadata to stdout")
    parser.add_argument("model", help="Model name", type=str)
    parser.add_argument("--layers", type=str, help="Print layer names")
    #parser.add_argument("--")
    args = parser.parse_args()

    model = get_pretrained_model(args.model)
    module_graph = GraphModulePlus.new_from_trace(model)

    node_list = []
    for node in module_graph.graph.nodes:
        node_list.append(node)
        print(node.name)
